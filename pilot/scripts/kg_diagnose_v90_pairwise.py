#!/usr/bin/env python3
"""진단 v90: pairwise tie-break (size-2 ties only).

927/1388 = 67% of ties are size-2. For these, pairwise A vs B comparison.
For size 3+, listwise CoT (v87 style).

Hypothesis: pairwise might be more accurate than listwise for size-2 ties.
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
GENERIC_TERMS = {'symptom', 'sign', 'pain', 'patient', 'disease', 'syndrome', 'condition'}

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
    print("진단 v90: pairwise tie-break (size-2 ties)", flush=True)
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
    cui_to_idx = {dc: i for i, dc in enumerate(dcs_list)}

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

    disease_features = {}
    TOP_K_FEATURES = 8
    for dc in dcs_list:
        feats = ds.get(dc, {})
        top_cuis = sorted(feats.items(), key=lambda x: -x[1])[:TOP_K_FEATURES * 3]
        names = []; seen = set()
        for cui, cnt in top_cuis:
            n_ = cp.get(cui, cui)
            nl = n_.lower().strip()
            if not nl or nl in seen or nl in GENERIC_TERMS: continue
            if len(nl) < 3 or len(nl) > 50: continue
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

    score_matrix = np.load("pilot/results/v79_stage1.npy")
    n = score_matrix.shape[0]
    print(f"v79 stage1: {n} patients", flush=True)

    candidates = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if len(candidates) >= n: break
            tdc = fr2cui.get(row["PATHOLOGY"])
            if not tdc: continue
            candidates.append({
                "evidences": ast.literal_eval(row["EVIDENCES"]),
                "true_dc": tdc,
                "age": row.get("AGE", "30"),
                "sex": row.get("SEX", "M"),
                "initial": row.get("INITIAL_EVIDENCE", "")
            })

    patient_data = []
    for c in candidates:
        profile = patient_profile(c["evidences"])
        ie = c.get("initial", "")
        chief = ev_info.get(ie, {}).get("question_en", ie) if ie else "—"
        chief = re.sub(r"Do you |Have you |Are you ", "", chief).rstrip("?").strip()
        age_sex = f"{c['age']}yo {'Male' if c['sex']=='M' else 'Female'}"
        patient_data.append({"profile": profile, "chief": chief, "age_sex": age_sex})

    t1_s1 = sum(1 for c_idx, c in enumerate(candidates)
                if dcs_list[int(np.argmax(score_matrix[c_idx]))] == c["true_dc"])
    print(f"  Stage 1 baseline @1 = {100*t1_s1/n:.2f}%", flush=True)

    # Categorize ties
    size2_ties = []  # (c_idx, [a, b])
    larger_ties = []  # (c_idx, [a, b, c, ...])
    for c_idx in range(n):
        scores = score_matrix[c_idx]
        max_s = scores.max()
        ti = np.where(scores == max_s)[0]
        if len(ti) == 2:
            size2_ties.append((c_idx, [dcs_list[i] for i in ti]))
        elif 3 <= len(ti) <= 10:
            larger_ties.append((c_idx, [dcs_list[i] for i in ti]))
    print(f"  Size-2 ties: {len(size2_ties)}, larger 3-10 ties: {len(larger_ties)}", flush=True)

    print(f"\n[vLLM init]...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    cot_sampling = SamplingParams(temperature=0, max_tokens=400)

    chosen_dc = {}

    # ---- Pairwise for size-2 ties (in BOTH orders to remove position bias) ----
    print(f"\n[Stage 2a] Pairwise tie-break (size-2)...", flush=True)
    pw_convs = []; pw_meta = []  # (c_idx, [a, b], order)  order: 'ab' or 'ba'
    for c_idx, (a, b) in [(c, td) for c, td in size2_ties]:
        pd = patient_data[c_idx]
        for order in ['ab', 'ba']:
            x, y = (a, b) if order == 'ab' else (b, a)
            name_x = disease_full_name(cui2name.get(x, x))
            name_y = disease_full_name(cui2name.get(y, y))
            kgx = disease_features[x]
            kgy = disease_features[y]
            prompt = f"""Patient: {pd['age_sex']}
Chief complaint: {pd['chief']}
{pd['profile']}

Two diagnoses are equally likely so far. Which is MORE likely for this patient?

(A) {name_x} — typical features: {kgx}
(B) {name_y} — typical features: {kgy}

Briefly explain (1-2 sentences) which patient features support A or B more, then answer.

Format:
REASON: <brief>
ANSWER: <A or B>"""
            pw_convs.append([{"role": "user", "content": prompt}])
            pw_meta.append((c_idx, [a, b], order))

    print(f"  Pairwise calls: {len(pw_convs)} (=2x size-2 ties for both orders)", flush=True)
    pw_votes = defaultdict(lambda: defaultdict(int))  # c_idx -> dc -> votes
    CHUNK = 5000
    t2 = time.time()
    sample_pw = None
    for chunk_start in range(0, len(pw_convs), CHUNK):
        outs = llm.chat(pw_convs[chunk_start:chunk_start+CHUNK], cot_sampling)
        if sample_pw is None: sample_pw = outs[:2]
        for i, out in enumerate(outs):
            c_idx, (a, b), order = pw_meta[chunk_start + i]
            text = out.outputs[0].text.strip()
            m = re.search(r"ANSWER\s*:?\s*\(?([AB])\)?", text, re.IGNORECASE)
            if not m: m = re.search(r"\b([AB])\s*$", text.strip(), re.IGNORECASE)
            if m:
                pick_letter = m.group(1).upper()
                if order == 'ab':
                    picked = a if pick_letter == 'A' else b
                else:
                    picked = b if pick_letter == 'A' else a
                pw_votes[c_idx][picked] += 1
        print(f"  Pairwise: {chunk_start+len(outs)}/{len(pw_convs)} ({time.time()-t2:.0f}초)", flush=True)
    print(f"  Pairwise 완료: {time.time()-t2:.0f}초", flush=True)

    # Aggregate pairwise: pick the disease with more votes
    for c_idx, votes_dict in pw_votes.items():
        if not votes_dict: continue
        # Find max votes
        max_votes = max(votes_dict.values())
        winners = [d for d, v in votes_dict.items() if v == max_votes]
        # Tie-break of tie-break: keep first-listed (= dcs_list ordering)
        for c, td in size2_ties:
            if c == c_idx:
                for cand in td:
                    if cand in winners:
                        chosen_dc[c_idx] = cand
                        break
                break

    if sample_pw:
        print(f"\n샘플 pairwise:")
        for o in sample_pw[:1]:
            print(f"  {o.outputs[0].text.strip()[:300]}")

    # ---- Listwise for size 3-10 (v87 style) ----
    print(f"\n[Stage 2b] Listwise CoT for size 3-10...", flush=True)
    lw_convs = []; lw_meta = []
    for c_idx, tied_dcs in larger_ties:
        pd = patient_data[c_idx]
        lines = []
        for i, dc in enumerate(tied_dcs):
            name = disease_full_name(cui2name.get(dc, dc))
            kg_feats = disease_features[dc]
            lines.append(f"({i+1}) {name} — typical features: {kg_feats}")
        disease_list = "\n".join(lines)
        prompt = f"""Patient: {pd['age_sex']}
Chief complaint: {pd['chief']}
{pd['profile']}

The following candidates are equally likely so far. Carefully evaluate each and pick the SINGLE most likely.

Candidates:
{disease_list}

Format:
EVAL:
(1) brief evaluation
(2) brief evaluation
...
PICK: <number>"""
        lw_convs.append([{"role": "user", "content": prompt}])
        lw_meta.append((c_idx, tied_dcs))

    if lw_convs:
        print(f"  Listwise calls: {len(lw_convs)}", flush=True)
        t3 = time.time()
        for chunk_start in range(0, len(lw_convs), CHUNK):
            outs = llm.chat(lw_convs[chunk_start:chunk_start+CHUNK], cot_sampling)
            for i, out in enumerate(outs):
                c_idx, tied_dcs = lw_meta[chunk_start + i]
                text = out.outputs[0].text.strip()
                m = re.search(r"PICK\s*:\s*\(?(\d+)\)?", text, re.IGNORECASE)
                if not m: m = re.search(r"\(?(\d+)\)?\s*$", text.strip())
                if m:
                    pick = int(m.group(1)) - 1
                    if 0 <= pick < len(tied_dcs):
                        chosen_dc[c_idx] = tied_dcs[pick]
                if c_idx not in chosen_dc:
                    chosen_dc[c_idx] = tied_dcs[0]
            print(f"  Listwise: {chunk_start+len(outs)}/{len(lw_convs)} ({time.time()-t3:.0f}초)", flush=True)

    # Build final score
    final_score = score_matrix.copy()
    for c_idx, picked_dc in chosen_dc.items():
        chosen_idx = cui_to_idx[picked_dc]
        max_s = score_matrix[c_idx].max()
        final_score[c_idx, chosen_idx] = max_s + 1.0

    t1c = t3c = t5c = t10c = 0
    for c_idx, c in enumerate(candidates):
        ranked = np.argsort(-final_score[c_idx])
        if dcs_list[ranked[0]] == c["true_dc"]: t1c += 1
        if c["true_dc"] in [dcs_list[i] for i in ranked[:3]]: t3c += 1
        if c["true_dc"] in [dcs_list[i] for i in ranked[:5]]: t5c += 1
        if c["true_dc"] in [dcs_list[i] for i in ranked[:10]]: t10c += 1

    print(f"\n  v90 (pairwise + listwise tie-break) @1={100*t1c/n:.2f}% @3={100*t3c/n:.1f}% @5={100*t5c/n:.1f}% @10={100*t10c/n:.1f}%", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"v90 GTPA@1 = {100*t1c/n:.2f}% (SUBSET={n})", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
