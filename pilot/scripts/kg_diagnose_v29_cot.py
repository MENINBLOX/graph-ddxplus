#!/usr/bin/env python3
"""진단 v29: CoT (Chain-of-Thought) reasoning + KG signature + JSON output.

가설: LLM이 candidates만 보고 추측하지 말고, 각 후보의 KG에서 가장 변별력 있는
      증상(rare-but-strong, IDF*count 기준)을 명시 → 환자 증상과 매칭 비교.

차별점:
  - 각 후보 disease의 top-5 discriminative 증상 (IDF·count 가중)
  - 환자 양성/음성 증상 명시
  - 채프링콤플레인트 강조
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

EVIDENCE_MEDTERM = {
    "diplopie": ["diplopia", "double vision"], "flushing": ["flushing", "facial flushing", "erythema"],
    "gain_poids": ["weight gain"], "perte_poids": ["weight loss"],
    "impression_mort": ["impending doom", "anxiety"],
    "lesions_peau_desquame": ["desquamation", "skin peeling"],
    "lesions_peau_couleur": ["skin discoloration", "rash"],
    "melena": ["melena", "tarry stool", "black stool"],
    "pdc": ["syncope", "loss of consciousness", "fainting"],
    "protu_langue": ["tongue protrusion", "tongue thrusting"],
    "psy_depers": ["depersonalization", "derealization"],
    "ptose": ["ptosis", "blepharoptosis", "eyelid drooping"],
    "ww_dd": ["orthopnea", "worse lying down"],
    "ww_nuit": ["nocturnal symptoms", "night-time symptoms"],
    "anxiete_s": ["anxiety"], "diaph": ["diaphoresis", "sweating"],
    "fatig_mod": ["fatigue", "tiredness"], "pale": ["pallor", "pale skin"],
    "stridor": ["stridor"], "wheezing": ["wheezing"],
    "convulsion": ["convulsion", "seizure"], "confusion": ["confusion", "disorientation"],
    "apnee": ["apnea", "sleep apnea"], "laryngospasme": ["laryngospasm"],
    "tachycardie": ["tachycardia"], "bradycardie": ["bradycardia"],
    "hemoptysie": ["hemoptysis", "coughing up blood"],
    "nausee": ["nausea", "vomiting"], "diarrhee": ["diarrhea"],
    "vomi_sg": ["hematemesis", "blood vomiting"], "douleurxx": ["pain"],
}

# DDXPlus 영문 번역 보정 (잘못된 번역 → 표준 의학 영어)
TRANSLATION_FIX = {
    "haunting": "stabbing",  # lancinante = stabbing/lancinating
    "tugging": "pulling",  # tiraillement = pulling sensation
    "burning": "burning",
    "exhausting": "exhausting",
    "scary": "frightening",
    "sensitive": "tender",  # sensible
    "heavy": "heavy",
    "sickening": "nauseating",
    "a knife stroke": "stabbing",
    "violent": "severe",
    "a cramp": "cramping",
    "tedious": "tiresome",
    "haunted": "stabbing",
    "Tugging": "Pulling",
}

def fix_translation(s):
    """잘못된 영문 번역을 의학 영어로 보정."""
    if not s: return s
    for bad, good in TRANSLATION_FIX.items():
        s = s.replace(bad, good)
    return s


def main():
    print("=" * 80, flush=True)
    print("진단 v28: KG signature symptoms in re-rank prompt", flush=True)
    print("=" * 80, flush=True)

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

    # IDF for symptom discriminative power
    sym_df = Counter()
    for d, syms in ds.items():
        for s in syms: sym_df[s] += 1
    N_d = len(dcs)
    idf = {s: math.log(N_d / df) for s, df in sym_df.items()}

    # 각 disease의 top-5 discriminative symptoms (IDF * count, exclude generic)
    DF_GATE = 30
    disease_signatures = {}
    for dc, syms in ds.items():
        scored = [(s, c * idf.get(s, 0)) for s, c in syms.items() if sym_df.get(s, 99) <= DF_GATE]
        scored.sort(key=lambda x: -x[1])
        # take top 5 with names
        sig = []
        for s, score in scored[:8]:  # take 8, filter unknown names
            name = cp.get(s, "")
            if name and len(name) > 3:
                sig.append(name)
            if len(sig) >= 5: break
        disease_signatures[dc] = sig

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
            if base in EVIDENCE_MEDTERM: terms.extend(EVIDENCE_MEDTERM[base])
            bc = re.sub(r"_.*", "", base); bc = re.sub(r"xx$", "", bc)
            if len(bc) >= 3 and bc not in STOPWORDS: terms.append(bc)
            q = info.get("question_en", "")
            if q:
                text = re.sub(r"\(.*?\)", "", q); text = re.sub(r"[?.,;:!]", "", text)
                terms.extend(w.lower() for w in text.split() if w.lower() not in STOPWORDS and len(w) >= 3)
                ql = q.lower()
                for ph in ["chest pain","sore throat","shortness of breath","difficulty breathing","weight loss","weight gain","loss of consciousness","muscle pain","muscle spasm","nasal congestion","runny nose","skin lesion","skin rash","black stool","bloody stool","heart palpitation","double vision","swollen"]:
                    if ph in ql: terms.append(ph)
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

    def patient_profile(evidences):
        """의학용어로 정리된 환자 프로필 (consolidated, no header repetition)."""
        # 통증 정보 통합
        pain_chars = []
        pain_locs = []
        pain_radiations = []
        pain_intens = None
        pain_speed = None
        pain_precision = None
        pain_present = False
        # 피부
        skin_locs = []
        skin_colors = []
        skin_props = []
        skin_present = False
        # 부종
        edema_locs = []
        edema_present = False
        # 기타
        symptoms = []  # (q_clean, val_en)
        history = []
        rx_history = []

        for ev in evidences:
            parts = ev.split("_@_"); base = parts[0]; value = parts[1] if len(parts) > 1 else None
            info = ev_info.get(base, {})
            q = info.get("question_en", "")
            val_en = info.get("value_en", {}).get(value, "") if value else ""
            if val_en and val_en.lower() in ("na", "nowhere", "n"): val_en = ""
            val_en = fix_translation(val_en)

            if info.get("is_antecedent"):
                q_clean = re.sub(r"Do you |Are you |Have you |Did you |Is your ", "", q).rstrip("?").strip() if q else ""
                history.append(q_clean + (f": {val_en}" if val_en else ""))
                continue

            # 통증 처리
            if "douleur" in base or "_dlr" in base:
                if "carac" in base and val_en:
                    pain_chars.append(val_en)
                elif "endroitducorps" in base and val_en:
                    pain_locs.append(val_en)
                elif "irrad" in base and val_en:
                    pain_radiations.append(val_en)
                elif "intens" in base and value:
                    pain_intens = value
                elif "soudain" in base and value:
                    pain_speed = value
                elif "precis" in base and value:
                    pain_precision = value
                elif base == "douleurxx":
                    pain_present = True
                continue

            # 피부 병변
            if "lesions_peau" in base:
                if "endroitducorps" in base and val_en:
                    skin_locs.append(val_en)
                elif "couleur" in base and val_en:
                    skin_colors.append(val_en)
                elif "desquame" in base:
                    skin_props.append("peeling")
                elif "elevee" in base:
                    skin_props.append("elevated")
                elif "plusqu1cm" in base:
                    skin_props.append(">1cm")
                elif "prurit" in base:
                    skin_props.append("pruritic")
                elif base == "lesions_peau":
                    skin_present = True
                continue

            # 부종
            if "oedeme" in base:
                if "endroitducorps" in base and val_en:
                    edema_locs.append(val_en)
                elif base == "oedeme":
                    edema_present = True
                continue

            q_clean = re.sub(r"Do you |Are you |Have you |Did you |Is your ", "", q).rstrip("?").strip() if q else ""
            entry = q_clean + (f": {val_en}" if val_en else "")
            symptoms.append(entry)

        out = []
        if pain_present or pain_chars or pain_locs:
            pain_parts = []
            if pain_chars: pain_parts.append(f"character ({', '.join(pain_chars)})")
            if pain_locs: pain_parts.append(f"location ({', '.join(pain_locs)})")
            if pain_radiations: pain_parts.append(f"radiation to ({', '.join(pain_radiations)})")
            if pain_intens: pain_parts.append(f"intensity {pain_intens}/10")
            if pain_speed: pain_parts.append(f"onset speed {pain_speed}/10")
            if pain_precision: pain_parts.append(f"precision {pain_precision}/10")
            out.append("PAIN: " + "; ".join(pain_parts))
        if skin_present or skin_locs or skin_colors:
            sk_parts = []
            if skin_locs: sk_parts.append(f"location ({', '.join(skin_locs)})")
            if skin_colors: sk_parts.append(f"color ({', '.join(skin_colors)})")
            if skin_props: sk_parts.append(", ".join(skin_props))
            out.append("SKIN LESION: " + "; ".join(sk_parts))
        if edema_present or edema_locs:
            ed_parts = []
            if edema_locs: ed_parts.append(f"location ({', '.join(edema_locs)})")
            out.append("EDEMA: " + ("; ".join(ed_parts) if ed_parts else "present"))
        if symptoms:
            out.append("OTHER: " + "; ".join(symptoms))
        if history:
            out.append("HISTORY: " + "; ".join(history))
        return "\n".join(out)

    print("\n[1] Disease signatures (top-5 discriminative syms):", flush=True)
    for dn, info in list(diseases.items())[:5]:
        sig = disease_signatures.get(info["cui"], [])
        print(f"  {dn[:30]:<30} → {', '.join(sig[:5])}", flush=True)

    # 테스트 데이터 (subset 우선 빠르게)
    print("\n[2] 테스트 (10K subset for speed)...", flush=True)
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= 10000: break
            patients.append({"evidences": ast.literal_eval(row["EVIDENCES"]),
                             "pathology": row["PATHOLOGY"], "age": row.get("AGE", "30"),
                             "sex": row.get("SEX", "M"), "initial": row.get("INITIAL_EVIDENCE", "")})
    print(f"  {len(patients):,}명", flush=True)

    # Age/sex prior from training (non-LR, just demographic distribution)
    print("  Loading age/sex prior from training...", flush=True)
    age_sex_disease = defaultdict(Counter)
    with open("data/ddxplus/release_train_patients.csv") as f:
        for row in csv.DictReader(f):
            tdc = fr2cui.get(row["PATHOLOGY"])
            if not tdc: continue
            age_bin = min(int(row.get("AGE", 0)) // 10 * 10, 80)
            age_sex_disease[(age_bin, row.get("SEX", "M"))][tdc] += 1

    print("\n[3] Bayesian + age/sex prior top-10...", flush=True)
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
            ll = sum(math.log((s[x]+0.1)/tw+1e-10) if x in s else math.log(0.1/tw+1e-10) for x in ps)
            # Age/sex prior
            age_bin = min(int(p["age"]) // 10 * 10, 80)
            prior_c = age_sex_disease[(age_bin, p["sex"])]
            prior_t = sum(prior_c.values())
            prior = (prior_c.get(dc, 0) + 1) / (prior_t + len(dcs)) if prior_t > 0 else 1.0/len(dcs)
            sc[dc] = ll + math.log(prior)
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        topk = [dc for dc, _ in ranked[:10]]
        candidates.append({"patient": p, "true_dc": tdc, "topk": topk, "ps_cuis": ps})
        if topk and topk[0] == tdc: baseline += 1
    in_topk = sum(1 for c in candidates if c["true_dc"] in c["topk"])
    print(f"  Baseline @1={100*baseline/n:.1f}%, top10={100*in_topk/n:.1f}%", flush=True)

    print("\n[4] LLM re-rank w/ KG signatures...", flush=True)
    prompts = []
    for c in candidates:
        p = c["patient"]
        profile = patient_profile(p["evidences"])
        ie = p.get("initial", "")
        chief = ev_info.get(ie, {}).get("question_en", ie) if ie else "—"
        chief = re.sub(r"Do you |Have you |Are you ", "", chief).rstrip("?").strip()

        cands_lines = []
        for i, dc in enumerate(c["topk"]):
            name = cui2name.get(dc, dc)
            sig = disease_signatures.get(dc, [])
            sig_str = ", ".join(sig[:5]) if sig else "(no signature)"
            cands_lines.append(f"{i+1}. {name} - typical features: {sig_str}")
        cands = "\n".join(cands_lines)

        prompt = f"""You are an expert clinician performing differential diagnosis.

PATIENT:
Age: {p['age']}, Sex: {'Male' if p['sex']=='M' else 'Female'}
Chief complaint: {chief}

{profile}

CANDIDATES (each with typical clinical features):
{cands}

Step 1: Identify the 2-3 most distinctive symptoms in the patient.
Step 2: Match those symptoms against each candidate's typical features.
Step 3: Select the candidate whose typical features BEST FIT the patient.

Reply in this exact format (no extra text):
Distinctive: <2-3 symptoms>
Best match: <number 1-{len(c['topk'])}>"""
        prompts.append(prompt)

    # Show sample prompt
    print(f"\nSample prompt[0] (truncated to 800 chars):\n{prompts[0][:800]}\n...", flush=True)

    print("\n[5] vLLM batch inference...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=128)
    convs = [[{"role": "user", "content": p}] for p in prompts]
    t0 = time.time()
    outputs = llm.chat(convs, sampling)
    print(f"  완료: {time.time()-t0:.0f}초", flush=True)

    t1 = t3 = t5 = pf = 0
    for c, out in zip(candidates, outputs):
        text = out.outputs[0].text.strip()
        # Search for "Best match: N" pattern
        m = re.search(r"Best match\s*:?\s*(\d+)", text, re.I)
        if not m:
            m = re.search(r"(\d+)\s*$", text)
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
    print(f"\n  Baseline @1={100*baseline/nt:.1f}%", flush=True)
    print(f"  v29 CoT Re-ranked: @1={100*t1/nt:.1f}% @3={100*t3/nt:.1f}% @5={100*t5/nt:.1f}%", flush=True)
    print(f"  parse_fail={pf}", flush=True)
    print("\n" + "=" * 80, flush=True)
    print(f"v29 GTPA@1 = {100*t1/nt:.1f}% (10K subset)", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
