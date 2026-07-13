#!/usr/bin/env python3
"""진단 v26: 의학적 용어 정제 + 정밀한 환자 보고서.

DDXPlus 영어 번역의 부정확성 보정:
  - "haunting" → "shooting/stabbing pain"
  - "tugging" → "pulling/dragging"
  - "Characterize your pain" → "Pain character"
  - 기타 어색한 번역을 표준 의학 용어로

LLM에 의학적으로 명확한 환자 보고서 전달.
"""
from __future__ import annotations
import ast, csv, json, math, os, re, time
from collections import Counter, defaultdict
from pathlib import Path
import ahocorasick
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")
KG_CACHE = Path("pilot/results/kg_v3_cache.json")
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}
STOPWORDS = {'does','have','your','you','the','and','for','are','with','that','this','from','been','were','being','which','their','than','other','about','into','over','some','only','very','also','just','more','most','such','much','will','would','could','should','make','like','time','when','what','where','how','who','all','each','every','both','few','any','not','can','may','her','his','its','our','they','them','then','had','has','him','but','one','two','way','day','did','get','got','let','say','she','too','use','yes','yet','now','new','old','see','own','why','try','ask','set','related','reason','consulting','significant','measured','thermometer','either','believe','racing','missing','beat','fast','irregularly','problems','situation','associated','inability','speak','trouble','keeping','opening','raising','annoying','else','body','somewhere','anywhere','nowhere','recently','currently','usually','often','sometimes','worse','better'}

# 부정확한 번역 → 의학 용어
TRANSLATION_FIX = {
    "haunting": "shooting/stabbing",
    "tugging": "pulling/dragging",
    "tightness or heavy feeling": "pressure-like/heavy",
    "a lump in my throat": "globus sensation",
    "heart attack like": "heart attack-like",
    "burning sensation or hot": "burning",
    "exhausting": "exhausting/throbbing",
    "splitting": "splitting/sharp",
    "characterize your pain": "Pain character",
    "do you feel pain somewhere": "Pain location",
    "how intense is the pain": "Pain intensity",
    "does the pain radiate to another location": "Pain radiation",
    "how fast did the pain appear": "Pain onset speed",
    "how precisely is the pain located": "Pain precision",
}

# Question keyword → 의학적 표현
QUESTION_REWRITE = {
    "do you have pain somewhere, related to your reason for consulting": "Pain present",
    "do you have a fever (either felt or measured with a thermometer)": "Fever",
    "do you have a cough": "Cough",
    "are you experiencing shortness of breath or difficulty breathing in a significant way": "Dyspnea",
    "do you feel your heart is beating fast (racing), irregularly (missing a beat) or do you feel palpitations": "Palpitations",
    "did you lose consciousness": "Loss of consciousness",
    "do you have a sore throat": "Sore throat",
    "have you noticed any change in the color of your stool": "Stool color change",
    "have you been coughing up blood": "Hemoptysis",
    "do you have nausea or do you feel like vomiting": "Nausea/vomiting",
    "do you have diarrhea or an increase in stool frequency": "Diarrhea",
    "are you sweating more than usual when sleeping": "Night sweats",
    "have you had an involuntary weight loss over the last 3 months": "Recent weight loss",
}


def medical_translate(text):
    """부정확한 번역을 의학 용어로 보정."""
    text_lower = text.lower().strip()
    for bad, good in TRANSLATION_FIX.items():
        if bad in text_lower:
            text_lower = text_lower.replace(bad, good)
    return text_lower


def rewrite_question(q):
    """질문을 간결한 의학 용어로 재작성."""
    q_lower = q.lower().strip().rstrip("?")
    for orig, rewritten in QUESTION_REWRITE.items():
        if orig in q_lower:
            return rewritten
    # 기본 정리
    q_clean = re.sub(r"do you |are you |have you |did you |is your |does the ", "", q_lower)
    return q_clean.strip().rstrip("?")


def main():
    print("="*80, flush=True)
    print("진단 v26: 의학 용어 정제 + 정밀 환자 보고서", flush=True)
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
                    val_en_fixed = medical_translate(val_en)  # 의학 용어 보정
                    vc = re.sub(r"\([rl]\)", "", val_en_fixed.lower()).strip()
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

    def medical_report(evidences, age, sex):
        """의학적으로 정제된 환자 보고서."""
        # 카테고리별 정리
        sections = defaultdict(list)
        pain_chars = []
        pain_locs = []
        pain_intens = None

        for ev in evidences:
            parts = ev.split("_@_"); base = parts[0]; value = parts[1] if len(parts) > 1 else None
            info = ev_info.get(base, {})
            q = info.get("question_en", "")
            val_en = info.get("value_en", {}).get(value, "") if value else ""
            if val_en and val_en.lower() in ("na", "nowhere", "n"): val_en = ""
            val_fixed = medical_translate(val_en) if val_en else ""

            # 통증 세부정보 통합
            if "douleurxx_carac" in base and val_fixed:
                pain_chars.append(val_fixed)
                continue
            if "douleurxx_endroitducorps" in base and val_fixed:
                pain_locs.append(re.sub(r"\([rl]\)", "", val_fixed).strip())
                continue
            if "douleurxx_intens" in base and value:
                pain_intens = value
                continue
            if "douleurxx_irrad" in base and val_fixed:
                if val_fixed and "nowhere" not in val_fixed:
                    sections["pain_radiation"].append(val_fixed)
                continue
            if "douleurxx_soudain" in base and value:
                sections["pain_onset"].append(f"{value}/10")
                continue
            if "douleurxx_precis" in base and value:
                sections["pain_precision"].append(f"{value}/10")
                continue
            if base == "douleurxx":
                # main pain question - skip, will be summarized
                continue

            # 일반 증상
            if info.get("is_antecedent"):
                rewritten = rewrite_question(q)
                if val_fixed: sections["history"].append(f"{rewritten}: {val_fixed}")
                else: sections["history"].append(rewritten)
                continue

            rewritten = rewrite_question(q)
            entry = rewritten + (f": {val_fixed}" if val_fixed else "")

            ql = q.lower()
            if any(k in ql for k in ["breath","cough","wheez","dyspn","sputum","throat"]):
                sections["resp"].append(entry)
            elif any(k in ql for k in ["heart","palpit","chest","syncope","consciousness"]):
                sections["cardiac"].append(entry)
            elif any(k in ql for k in ["nausea","vomit","diarr","stool","abdom","swallow"]):
                sections["gi"].append(entry)
            elif any(k in ql for k in ["dizz","numb","weak","paralys","vision","speech","seizure","faint"]):
                sections["neuro"].append(entry)
            elif any(k in ql for k in ["skin","rash","lesion","itch","red"]):
                sections["skin"].append(entry)
            else:
                sections["other"].append(entry)

        # 통증 통합 보고
        out = []
        if pain_locs or pain_chars or pain_intens:
            pain_desc = []
            if pain_locs: pain_desc.append(f"location: {', '.join(pain_locs[:5])}")
            if pain_chars: pain_desc.append(f"character: {', '.join(set(pain_chars))}")
            if pain_intens: pain_desc.append(f"intensity: {pain_intens}/10")
            if sections.get("pain_radiation"): pain_desc.append(f"radiation: {', '.join(sections['pain_radiation'][:3])}")
            if sections.get("pain_onset"): pain_desc.append(f"onset speed: {sections['pain_onset'][0]}")
            out.append(f"**PAIN**: " + "; ".join(pain_desc))

        labels = {"resp":"RESPIRATORY","cardiac":"CARDIOVASCULAR","gi":"GASTROINTESTINAL","neuro":"NEUROLOGICAL","skin":"DERMATOLOGICAL","other":"OTHER FINDINGS","history":"PAST MEDICAL HISTORY"}
        for k in ["resp","cardiac","gi","neuro","skin","other","history"]:
            if sections[k]: out.append(f"**{labels[k]}**: " + "; ".join(sections[k]))
        return "\n".join(out)

    print("\n[2] 테스트...", flush=True)
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            patients.append({"evidences": ast.literal_eval(row["EVIDENCES"]),
                             "pathology": row["PATHOLOGY"], "age": row.get("AGE", "30"),
                             "sex": row.get("SEX", "M"), "initial": row.get("INITIAL_EVIDENCE", "")})

    # Sample
    print("\n[Sample 환자 보고서]:", flush=True)
    for i in range(2):
        p = patients[i]
        print(f"\n  {p['pathology']}, {p['age']}yo {p['sex']}:", flush=True)
        print(medical_report(p["evidences"], p["age"], p["sex"])[:500], flush=True)

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

    print("\n[4] 의학 보고서 프롬프트...", flush=True)
    prompts = []
    for c in candidates:
        p = c["patient"]
        report = medical_report(p["evidences"], p["age"], p["sex"])
        ie = p.get("initial", "")
        chief = ev_info.get(ie, {}).get("question_en", ie) if ie else "—"
        chief = rewrite_question(chief)
        cands = "\n".join(f"{i+1}. {cui2name.get(dc, dc)}" for i, dc in enumerate(c["topk"]))
        prompt = f"""## Clinical Case

**Patient:** {p['age']}yo {'Male' if p['sex']=='M' else 'Female'}
**Chief complaint:** {chief}

**Clinical findings:**
{report}

**Differential:**
{cands}

Most likely diagnosis (number 1-{len(c['topk'])} only):"""
        prompts.append(prompt)

    print(f"  프롬프트: {len(prompts):,}", flush=True)

    print("\n[5] vLLM batch...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=16)
    convs = [[{"role": "user", "content": p}] for p in prompts]
    t0 = time.time()
    outputs = llm.chat(convs, sampling)
    print(f"  완료: {time.time()-t0:.0f}초", flush=True)

    t1 = t3 = t5 = pf = 0
    for c, out in zip(candidates, outputs):
        text = out.outputs[0].text.strip()
        m = re.search(r"(\d+)", text)
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
    print(f"v26 GTPA@1 = {100*t1/nt:.1f}%", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
