#!/usr/bin/env python3
"""진단 v85: v82 + 0-1000 scale (instead of 0-100) → fewer ties.

LLM 출력은 보통 round numbers (0, 50, 80, 90...) → 27.8% top-1 ties 발생.
0-1000 스케일이면 LLM이 더 미세하게 표현 가능 → ties 감소 기대.
"""
from __future__ import annotations
import ast, csv, json, math, os, re, time, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")
KG_CACHE = Path("pilot/results/kg_v3_cache.json")
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}
GENERIC_TERMS = {'symptom', 'sign', 'pain', 'patient', 'disease', 'syndrome', 'condition',
                 'signs and symptoms', 'physical findings', 'feeling relief', 'perceived quality of life',
                 'intensity and distress 5', 'functional disorder', 'reduced', 'abnormal',
                 'discomfort', 'severe (severity modifier)', 'severe', 'mild', 'moderate',
                 'pressure (finding)', 'pressure', 'pressure - action', 'inflammation'}

TRANSLATION_FIX = {
    "haunting": "stabbing", "tugging": "pulling", "sensitive": "tender",
    "a knife stroke": "stabbing", "a cramp": "cramping", "haunted": "stabbing",
    "sickening": "nauseating", "tedious": "tiresome", "scary": "frightening",
    "violent": "severe",
}
DISEASE_NAME_FIX = {
    "URTI": "Upper respiratory tract infection (URTI)",
    "Larygospasm": "Laryngospasm",
    "GERD": "Gastroesophageal reflux disease (GERD)",
    "PSVT": "Paroxysmal supraventricular tachycardia (PSVT)",
    "SLE": "Systemic lupus erythematosus (SLE)",
    "Boerhaave": "Boerhaave syndrome (esophageal rupture)",
    "Possible NSTEMI / STEMI": "NSTEMI or STEMI (myocardial infarction)",
    "HIV (initial infection)": "Acute HIV infection (acute retroviral syndrome)",
    "Localized edema": "Localized edema",
    "Bronchospasm / acute asthma exacerbation": "Acute asthma exacerbation",
    "Acute COPD exacerbation / infection": "Acute COPD exacerbation",
    "Pulmonary embolism": "Pulmonary embolism (PE)",
    "Atrial fibrillation": "Atrial fibrillation (AFib)",
    "Whooping cough": "Whooping cough (pertussis)",
    "Pulmonary neoplasm": "Pulmonary neoplasm (lung cancer)",
    "Pancreatic neoplasm": "Pancreatic neoplasm (pancreatic cancer)",
    "Influenza": "Influenza (flu)",
    "Tuberculosis": "Tuberculosis (TB)",
    "Allergic sinusitis": "Allergic rhinitis/sinusitis",
}


def fix_translation(s):
    if not s: return s
    for bad, good in TRANSLATION_FIX.items():
        s = s.replace(bad, good)
    return s

def disease_full_name(short):
    return DISEASE_NAME_FIX.get(short, short)


def main():
    print("="*80, flush=True)
    print("진단 v85: TF-IDF + 0-1000 scale", flush=True)
    print("="*80, flush=True)

    cp = {}
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG" and p[2] == "P" and p[0] not in cp:
                cp[p[0]] = p[14].strip()

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
    dcs_list = sorted(dcs)

    ev_info = {}
    for eid, info in ev_fr.items():
        ev_info[eid] = {"question_en": info.get("question_en", ""), "is_antecedent": info.get("is_antecedent", False), "value_en": {}}
        vm = info.get("value_meaning", {})
        if isinstance(vm, dict):
            for k, v in vm.items():
                if isinstance(v, dict) and v.get("en"): ev_info[eid]["value_en"][k] = v["en"]

    ds = defaultdict(dict)
    for (a, b), cnt in pc.items():
        if a in NOISE or b in NOISE: continue
        if a in dcs: ds[a][b] = cnt
        if b in dcs: ds[b][a] = cnt

    feat_disease_count = Counter()
    for dc, feats in ds.items():
        for cui in feats:
            feat_disease_count[cui] += 1

    N_DISEASES = len(dcs_list)
    TOP_K_FEATURES = 8
    disease_features = {}
    for dc in dcs_list:
        feats = ds.get(dc, {})
        total = sum(feats.values()) or 1
        scored = []
        for cui, cnt in feats.items():
            tf = cnt / total
            df = feat_disease_count[cui]
            idf = math.log(N_DISEASES / df) if df > 0 else 0
            scored.append((cui, cnt, tf * idf))
        scored.sort(key=lambda x: -x[2])
        names = []
        seen = set()
        for cui, cnt, sc in scored:
            n_ = cp.get(cui, cui)
            nl = n_.lower().strip()
            if not nl or nl in seen or nl in GENERIC_TERMS: continue
            if len(nl) < 3 or len(nl) > 60: continue
            if nl == cui2name.get(dc, "").lower(): continue
            seen.add(nl); names.append(n_)
            if len(names) >= TOP_K_FEATURES: break
        disease_features[dc] = ", ".join(names) if names else "—"

    def patient_profile(evidences):
        pain_chars=[]; pain_locs=[]; pain_radiations=[]
        pain_intens=None; pain_speed=None; pain_present=False
        symptoms=[]; history=[]
        for ev in evidences:
            parts = ev.split("_@_"); base=parts[0]; value=parts[1] if len(parts)>1 else None
            info = ev_info.get(base, {}); q = info.get("question_en", "")
            val_en = info.get("value_en", {}).get(value, "") if value else ""
            if val_en and val_en.lower() in ("na","nowhere","n"): val_en=""
            val_en = fix_translation(val_en)
            if info.get("is_antecedent"):
                q_clean = re.sub(r"Do you |Are you |Have you |Did you |Is your ", "", q).rstrip("?").strip() if q else ""
                history.append(q_clean + (f": {val_en}" if val_en else "")); continue
            if "douleur" in base or "_dlr" in base:
                if "carac" in base and val_en: pain_chars.append(val_en)
                elif "endroitducorps" in base and val_en: pain_locs.append(val_en)
                elif "irrad" in base and val_en: pain_radiations.append(val_en)
                elif "intens" in base and value: pain_intens=value
                elif "soudain" in base and value: pain_speed=value
                elif base=="douleurxx": pain_present=True
                continue
            q_clean = re.sub(r"Do you |Are you |Have you |Did you |Is your ", "", q).rstrip("?").strip() if q else ""
            symptoms.append(q_clean + (f": {val_en}" if val_en else ""))
        out=[]
        if pain_present or pain_chars:
            pp=[]
            if pain_chars: pp.append(f"character ({', '.join(pain_chars)})")
            if pain_locs: pp.append(f"location ({', '.join(pain_locs)})")
            if pain_radiations: pp.append(f"radiation to ({', '.join(pain_radiations)})")
            if pain_intens: pp.append(f"intensity {pain_intens}/10")
            if pain_speed: pp.append(f"onset {pain_speed}/10")
            out.append("PAIN: " + "; ".join(pp))
        if symptoms: out.append("OTHER: " + "; ".join(symptoms))
        if history: out.append("HISTORY: " + "; ".join(history))
        return "\n".join(out)

    SUBSET = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    print(f"\n[2] 테스트 (SUBSET={SUBSET})...", flush=True)
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= SUBSET: break
            patients.append({"evidences": ast.literal_eval(row["EVIDENCES"]),
                             "pathology": row["PATHOLOGY"], "age": row.get("AGE", "30"),
                             "sex": row.get("SEX", "M"), "initial": row.get("INITIAL_EVIDENCE", "")})
    print(f"  {len(patients):,}명", flush=True)

    candidates = []; n = 0
    for p in patients:
        tdc = fr2cui.get(p["pathology"])
        if not tdc: continue
        n += 1
        candidates.append({"patient": p, "true_dc": tdc})

    print(f"\n[3] vLLM init...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})

    sampling = SamplingParams(temperature=0, max_tokens=8)
    convs = []
    pair_meta = []
    for c_idx, c in enumerate(candidates):
        p = c["patient"]
        profile = patient_profile(p["evidences"])
        ie = p.get("initial", "")
        chief = ev_info.get(ie, {}).get("question_en", ie) if ie else "—"
        chief = re.sub(r"Do you |Have you |Are you ", "", chief).rstrip("?").strip()
        age_sex = f"{p['age']}yo {'Male' if p['sex']=='M' else 'Female'}"
        for d_idx, dc in enumerate(dcs_list):
            name = disease_full_name(cui2name.get(dc, dc))
            kg_feats = disease_features[dc]
            prompt = f"""Patient: {age_sex}
Chief complaint: {chief}
{profile}

Diagnosis hypothesis: {name}
Discriminative features (medical literature): {kg_feats}

Score how well the patient matches this diagnosis. Use a fine-grained 0-1000 integer scale (0=no match, 1000=perfect textbook match). Reply with ONLY the integer score."""
            convs.append([{"role": "user", "content": prompt}])
            pair_meta.append((c_idx, d_idx))

    print(f"  Total LLM calls: {len(convs)}", flush=True)
    CHUNK = 5000
    score_matrix = np.zeros((n, len(dcs_list)), dtype=np.float32)
    t0 = time.time()
    for chunk_start in range(0, len(convs), CHUNK):
        chunk = convs[chunk_start:chunk_start+CHUNK]
        outs = llm.chat(chunk, sampling)
        for i, out in enumerate(outs):
            c_idx, d_idx = pair_meta[chunk_start + i]
            text = out.outputs[0].text.strip()
            m = re.search(r"(\d+)", text)
            score = int(m.group(1)) if m else 0
            score = min(score, 1000)
            score_matrix[c_idx, d_idx] = score
        print(f"  진행: {chunk_start+len(chunk)}/{len(convs)} ({time.time()-t0:.0f}초)", flush=True)
    print(f"  완료: {time.time()-t0:.0f}초", flush=True)
    np.save("pilot/results/v85_score_matrix.npy", score_matrix)

    t1 = t3 = t5 = t10 = 0
    for c_idx, c in enumerate(candidates):
        ranked = np.argsort(-score_matrix[c_idx])
        if dcs_list[ranked[0]] == c["true_dc"]: t1 += 1
        if c["true_dc"] in [dcs_list[i] for i in ranked[:3]]: t3 += 1
        if c["true_dc"] in [dcs_list[i] for i in ranked[:5]]: t5 += 1
        if c["true_dc"] in [dcs_list[i] for i in ranked[:10]]: t10 += 1

    # Tie statistics
    tied = sum(1 for c_idx in range(n) if (score_matrix[c_idx] == score_matrix[c_idx].max()).sum() >= 2)
    print(f"\n  Top-1 ties: {tied}/{n} ({100*tied/n:.1f}%)", flush=True)
    print(f"  v85 (TF-IDF + 0-1000) @1={100*t1/n:.1f}% @3={100*t3/n:.1f}% @5={100*t5/n:.1f}% @10={100*t10/n:.1f}%", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"v85 GTPA@1 = {100*t1/n:.1f}% (SUBSET={n})", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
