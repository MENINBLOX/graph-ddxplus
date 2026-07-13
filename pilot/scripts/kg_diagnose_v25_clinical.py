#!/usr/bin/env python3
"""진단 v25: 임상 보고서 형식 + few-shot 없이 명확한 지시.

DDXPlus 49 질환 중 LLM이 어떤 변별 단서를 활용하는지 확인.
"""
from __future__ import annotations
import ast, csv, json, os, re, time
import math
from collections import Counter, defaultdict
from pathlib import Path
import ahocorasick
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")
KG_CACHE = Path("pilot/results/kg_v3_cache.json")
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}
STOPWORDS = {'does','have','your','you','the','and','for','are','with','that','this','from','been','were','being','which','their','than','other','about','into','over','some','only','very','also','just','more','most','such','much','will','would','could','should','make','like','time','when','what','where','how','who','all','each','every','both','few','any','not','can','may','her','his','its','our','they','them','then','had','has','him','but','one','two','way','day','did','get','got','let','say','she','too','use','yes','yet','now','new','old','see','own','why','try','ask','set','related','reason','consulting','significant','measured','thermometer','either','believe','racing','missing','beat','fast','irregularly','problems','situation','associated','inability','speak','trouble','keeping','opening','raising','annoying','else','body','somewhere','anywhere','nowhere','recently','currently','usually','often','sometimes','worse','better'}

PROMPT = """## Clinical Case

**Patient demographics:** {age} years old, {sex}
**Chief complaint:** {chief}

**Presenting symptoms:**
{profile}

**Differential diagnosis (consider these conditions):**
{candidates}

## Task

You are an experienced physician. Based on the clinical presentation, identify the MOST LIKELY diagnosis.

Key considerations:
- Age and sex (e.g., children rarely have COPD or angina)
- Symptom pattern (acute vs chronic, location, character)
- Risk factors and history
- Pathognomonic findings

Respond with ONLY: ANSWER: <number 1-{n}>"""


def main():
    print("="*80, flush=True)
    print("진단 v25: 임상 보고서 형식 re-ranking", flush=True)
    print("="*80, flush=True)

    print("\n[1] 로드...", flush=True)
    can = defaultdict(set); cp = {}
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG":
                can[p[0]].add(p[14].strip())
                if p[2] == "P" and p[0] not in cp: cp[p[0]] = p[14].strip()

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f: ev_fr = json.load(f)
    with open(KG_CACHE) as f: cache = json.load(f)

    pc = Counter()
    for k, v in cache["pair_counts"]: pc[tuple(k)] = v

    diseases = {}; fr2cui = {}; cui2name = {}
    for dn, info in cond.items():
        if dn not in icd_map: continue
        dc = icd_map[dn]["cui"]; diseases[dn] = {"cui": dc}
        fr2cui[info.get("cond-name-fr", "")] = dc; cui2name[dc] = dn
    dcs = set(d["cui"] for d in diseases.values())

    ev_info = {}
    for eid, info in ev_fr.items():
        ev_info[eid] = {"question_en": info.get("question_en", ""), "is_antecedent": info.get("is_antecedent", False), "value_en": {}}
        vm = info.get("value_meaning", {})
        if isinstance(vm, dict):
            for k, v in vm.items():
                if isinstance(v, dict) and v.get("en"): ev_info[eid]["value_en"][k] = v["en"]

    ds = defaultdict(dict); scuis = set()
    for (a, b), cnt in pc.items():
        if a in NOISE or b in NOISE: continue
        if a in dcs: ds[a][b] = cnt; scuis.add(b)
        if b in dcs: ds[b][a] = cnt; scuis.add(a)
    ds = dict(ds)

    aho = ahocorasick.Automaton()
    for cui in scuis:
        for name in can.get(cui, set()):
            lo = name.lower().strip()
            if len(lo) < 4 or lo in STOPWORDS: continue
            try: aho.add_word(lo, (lo, cui))
            except: pass
    aho.make_automaton()
    all_s = set()
    for syms in ds.values(): all_s.update(syms.keys())

    def text_match(evidences):
        cuis = set()
        for ev in evidences:
            parts = ev.split("_@_"); base = parts[0]; value = parts[1] if len(parts) > 1 else None
            info = ev_info.get(base, {})
            if info.get("is_antecedent"): continue
            terms = []
            bc = re.sub(r"_.*", "", base); bc = re.sub(r"xx$", "", bc)
            if len(bc) >= 3 and bc not in STOPWORDS: terms.append(bc)
            q = info.get("question_en", "")
            if q:
                text = re.sub(r"\(.*?\)", "", q); text = re.sub(r"[?.,;:!]", "", text)
                terms.extend(w.lower() for w in text.split() if w.lower() not in STOPWORDS and len(w) >= 3)
            if value:
                val_en = info.get("value_en", {}).get(value, "")
                if val_en and val_en.lower() not in ("na", "nowhere", "n"):
                    vc = re.sub(r"\([rl]\)", "", val_en.lower()).strip()
                    if vc and vc not in STOPWORDS:
                        terms.append(vc)
                        if "pain" in q.lower(): terms.append(f"{vc} pain")
            pt = " . ".join(terms)
            for ei, (n, cui) in aho.iter(pt):
                si = ei - len(n) + 1
                if si > 0 and pt[si-1].isalpha(): continue
                if ei+1 < len(pt) and pt[ei+1].isalpha(): continue
                cuis.add(cui)
        return cuis

    def patient_clinical_report(evidences):
        """임상 보고서 형식으로 환자 증상 정리."""
        sec = defaultdict(list)
        for ev in evidences:
            parts = ev.split("_@_"); base = parts[0]; value = parts[1] if len(parts) > 1 else None
            info = ev_info.get(base, {})
            q = info.get("question_en", "")
            val_en = info.get("value_en", {}).get(value, "") if value else ""
            if val_en and val_en.lower() in ("na", "nowhere", "n"): val_en = ""
            q_clean = re.sub(r"Do you |Are you |Have you |Did you |Is your ", "", q).rstrip("?").strip() if q else ""
            entry = q_clean + (f" → {val_en}" if val_en else "")
            if info.get("is_antecedent"): sec["history"].append(entry); continue
            if "pain" in q.lower() or "douleur" in base: sec["pain"].append(entry)
            elif "skin" in q.lower() or "rash" in q.lower(): sec["skin"].append(entry)
            elif any(k in q.lower() for k in ["breath","cough","wheez","dyspn","sputum"]): sec["resp"].append(entry)
            elif any(k in q.lower() for k in ["heart","palpit","chest","syncope","consciousness"]): sec["cardiac"].append(entry)
            elif any(k in q.lower() for k in ["nausea","vomit","diarr","stool","abdom","swallow"]): sec["gi"].append(entry)
            elif any(k in q.lower() for k in ["dizz","numb","weak","paralys","vision","speech","seizure"]): sec["neuro"].append(entry)
            else: sec["constitutional"].append(entry)
        out = []
        labels = {"pain":"Pain","skin":"Skin/Rash","resp":"Respiratory","cardiac":"Cardiovascular","gi":"Gastrointestinal","neuro":"Neurological","constitutional":"Constitutional","history":"Past medical history"}
        for k in ["pain","skin","resp","cardiac","gi","neuro","constitutional","history"]:
            if sec[k]: out.append(f"- **{labels[k]}**: " + "; ".join(sec[k]))
        return "\n".join(out)

    print("\n[2] 테스트...", flush=True)
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            patients.append({"evidences": ast.literal_eval(row["EVIDENCES"]),
                             "pathology": row["PATHOLOGY"], "age": row.get("AGE", "30"),
                             "sex": row.get("SEX", "M"), "initial": row.get("INITIAL_EVIDENCE", "")})
    print(f"  {len(patients):,}명", flush=True)

    print("\n[3] Bayesian top-10...", flush=True)
    candidates = []; baseline = 0; n = 0
    for p in patients:
        tdc = fr2cui.get(p["pathology"])
        if not tdc: continue
        n += 1
        ps = text_match(p["evidences"])
        sc = {}
        for dc in dcs:
            s = ds.get(dc, {})
            if not s: sc[dc] = -1e6; continue
            tw = sum(s.values()) + len(all_s) * 0.1
            sc[dc] = sum(math.log((s[x]+0.1)/tw+1e-10) if x in s else math.log(0.1/tw+1e-10) for x in ps)
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        topk = [dc for dc, _ in ranked[:10]]
        candidates.append({"patient": p, "true_dc": tdc, "topk": topk})
        if topk and topk[0] == tdc: baseline += 1
    print(f"  Baseline @1={100*baseline/n:.1f}%", flush=True)

    print("\n[4] 임상 프롬프트...", flush=True)
    prompts = []
    for c in candidates:
        p = c["patient"]
        report = patient_clinical_report(p["evidences"])
        ie = p.get("initial", "")
        chief = ev_info.get(ie, {}).get("question_en", ie) if ie else "—"
        chief = re.sub(r"Do you |Have you |Are you ", "", chief).rstrip("?").strip()
        cands = "\n".join(f"{i+1}. {cui2name.get(dc, dc)}" for i, dc in enumerate(c["topk"]))
        prompts.append(PROMPT.format(
            age=p["age"], sex="Male" if p["sex"] == "M" else "Female",
            chief=chief, profile=report, candidates=cands, n=len(c["topk"]),
        ))

    print("\n[5] vLLM batch...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=32)
    convs = [[{"role": "user", "content": p}] for p in prompts]
    t0 = time.time()
    outputs = llm.chat(convs, sampling)
    print(f"  완료: {time.time()-t0:.0f}초", flush=True)

    t1 = t3 = t5 = pf = 0
    for c, out in zip(candidates, outputs):
        text = out.outputs[0].text.strip()
        m = re.search(r"ANSWER:\s*(\d+)", text, re.IGNORECASE)
        if not m: m = re.search(r"\b(\d+)\b", text)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(c["topk"]):
                reranked = list(c["topk"]); chosen = reranked.pop(idx); reranked.insert(0, chosen)
            else: reranked = c["topk"]; pf += 1
        else: reranked = c["topk"]; pf += 1
        tdc = c["true_dc"]
        if reranked[0] == tdc: t1 += 1
        if tdc in reranked[:3]: t3 += 1
        if tdc in reranked[:5]: t5 += 1

    nt = len(candidates)
    print(f"\n  Re-ranked: @1={100*t1/nt:.1f}% @3={100*t3/nt:.1f}% @5={100*t5/nt:.1f}%", flush=True)
    print(f"  parse_fail={pf:,}", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"v25 GTPA@1 = {100*t1/nt:.1f}%", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
