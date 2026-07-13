#!/usr/bin/env python3
"""진단 v71: v59 stage1 score를 candidate별 z-score로 정규화.

가설: LLM이 특정 disease(예: Pneumonia)에 자주 높은 점수 → bias.
candidate별 mean/std로 normalize → 환자에 진짜 specific한 candidate 부각.
TF-IDF 같은 discriminative 변환.
"""
from __future__ import annotations
import ast, csv, json, math, os, re, time, sys
from collections import Counter, defaultdict
from pathlib import Path
import ahocorasick
import numpy as np
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
    print("진단 v71: per-candidate z-score normalization", flush=True)
    print("="*80, flush=True)

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
    dcs_list = sorted(dcs)

    ev_info = {}
    for eid, info in ev_fr.items():
        ev_info[eid] = {"question_en": info.get("question_en", ""), "is_antecedent": info.get("is_antecedent", False), "value_en": {}}
        vm = info.get("value_meaning", {})
        if isinstance(vm, dict):
            for k, v in vm.items():
                if isinstance(v, dict) and v.get("en"): ev_info[eid]["value_en"][k] = v["en"]

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

    # Stage 1: per-candidate scoring (49 candidates) - same as v59
    print("\n[4] Stage 1: Per-candidate 0-100 (all 49)...", flush=True)
    sampling = SamplingParams(temperature=0, max_tokens=8)

    convs = []
    pair_meta = []  # (c_idx, d_idx)
    for c_idx, c in enumerate(candidates):
        p = c["patient"]
        profile = patient_profile(p["evidences"])
        ie = p.get("initial", "")
        chief = ev_info.get(ie, {}).get("question_en", ie) if ie else "—"
        chief = re.sub(r"Do you |Have you |Are you ", "", chief).rstrip("?").strip()
        age_sex = f"{p['age']}yo {'Male' if p['sex']=='M' else 'Female'}"

        for d_idx, dc in enumerate(dcs_list):
            name = disease_full_name(cui2name.get(dc, dc))
            prompt = f"""Patient: {age_sex}
Chief complaint: {chief}
{profile}

Diagnosis hypothesis: {name}

How likely is this hypothesis given the patient's presentation? Reply with ONLY a percentage 0-100 (0=impossible, 100=certain)."""
            convs.append([{"role": "user", "content": prompt}])
            pair_meta.append((c_idx, d_idx))

    print(f"  Stage 1 calls: {len(convs)}", flush=True)
    CHUNK = 5000
    # score_matrix[c_idx, d_idx]
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
            score = min(score, 100)
            score_matrix[c_idx, d_idx] = score
        print(f"  진행: {chunk_start+len(chunk)}/{len(convs)} ({time.time()-t0:.0f}초)", flush=True)
    print(f"  Stage 1 완료: {time.time()-t0:.0f}초", flush=True)

    # Save matrix for re-use
    np.save("pilot/results/v71_score_matrix.npy", score_matrix)

    # Raw v59 results
    t1_raw = sum(1 for c_idx, c in enumerate(candidates)
                 if dcs_list[int(np.argmax(score_matrix[c_idx]))] == c["true_dc"])
    print(f"\n  Raw v59 @1={100*t1_raw/n:.1f}%", flush=True)

    # ===== Variant 1: per-candidate z-score across ALL patients =====
    cand_mean = score_matrix.mean(axis=0)  # (49,)
    cand_std = score_matrix.std(axis=0) + 1e-6
    z_matrix = (score_matrix - cand_mean) / cand_std  # high z = unusually high for this candidate

    t1_z = 0
    for c_idx, c in enumerate(candidates):
        idx = int(np.argmax(z_matrix[c_idx]))
        if dcs_list[idx] == c["true_dc"]: t1_z += 1
    print(f"  v71-z (cand z-score) @1={100*t1_z/n:.1f}%", flush=True)

    # ===== Variant 2: subtract candidate mean only =====
    centered = score_matrix - cand_mean
    t1_c = 0
    for c_idx, c in enumerate(candidates):
        idx = int(np.argmax(centered[c_idx]))
        if dcs_list[idx] == c["true_dc"]: t1_c += 1
    print(f"  v71-c (mean centered) @1={100*t1_c/n:.1f}%", flush=True)

    # ===== Variant 3: rank-based (per-candidate percentile) =====
    # For each candidate, what's this patient's rank vs other patients?
    rank_matrix = np.zeros_like(score_matrix)
    for d_idx in range(len(dcs_list)):
        # rank patients by score for this disease (high score = high rank)
        order = np.argsort(-score_matrix[:, d_idx])
        ranks = np.empty(n, dtype=np.float32)
        for r, idx in enumerate(order):
            ranks[idx] = 1.0 - r / max(n - 1, 1)  # 1 = best, 0 = worst
        rank_matrix[:, d_idx] = ranks
    t1_r = 0
    for c_idx, c in enumerate(candidates):
        idx = int(np.argmax(rank_matrix[c_idx]))
        if dcs_list[idx] == c["true_dc"]: t1_r += 1
    print(f"  v71-r (rank percentile) @1={100*t1_r/n:.1f}%", flush=True)

    # ===== Variant 4: combined raw + z (additive) =====
    # raw normalized to [0,1] + z normalized
    raw_norm = score_matrix / 100.0
    z_clip = np.clip(z_matrix, -3, 3) / 6 + 0.5
    combo = 0.5 * raw_norm + 0.5 * z_clip
    t1_combo = 0
    for c_idx, c in enumerate(candidates):
        idx = int(np.argmax(combo[c_idx]))
        if dcs_list[idx] == c["true_dc"]: t1_combo += 1
    print(f"  v71-combo (raw+z) @1={100*t1_combo/n:.1f}%", flush=True)

    # ===== Variant 5: per-PATIENT z-score (within patient) =====
    pat_mean = score_matrix.mean(axis=1, keepdims=True)
    pat_std = score_matrix.std(axis=1, keepdims=True) + 1e-6
    pat_z = (score_matrix - pat_mean) / pat_std
    t1_pz = 0
    for c_idx, c in enumerate(candidates):
        idx = int(np.argmax(pat_z[c_idx]))
        if dcs_list[idx] == c["true_dc"]: t1_pz += 1
    print(f"  v71-pz (patient z) @1={100*t1_pz/n:.1f}% (should == raw)", flush=True)

    # Output @3, @5 for best variant
    def topk_acc(M, k):
        hits = 0
        for c_idx, c in enumerate(candidates):
            top = np.argsort(-M[c_idx])[:k]
            if any(dcs_list[i] == c["true_dc"] for i in top): hits += 1
        return 100 * hits / n

    best_label = "raw"
    best_M = score_matrix
    best_t1 = t1_raw
    for label, M, t1 in [("z", z_matrix, t1_z), ("c", centered, t1_c),
                         ("r", rank_matrix, t1_r), ("combo", combo, t1_combo)]:
        if t1 > best_t1:
            best_t1, best_M, best_label = t1, M, label
    print(f"\n  Best variant: {best_label} @1={100*best_t1/n:.1f}% @3={topk_acc(best_M,3):.1f}% @5={topk_acc(best_M,5):.1f}%", flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"v71 BEST GTPA@1 = {100*best_t1/n:.1f}% ({best_label}, SUBSET={n})", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
