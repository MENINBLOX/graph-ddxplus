#!/usr/bin/env python3
"""진단 v75: 4 prompt variants ensemble per-candidate scoring.

각 (patient, candidate) 쌍을 4개 다른 framing으로 score → 평균.
다양한 bias 상쇄.
"""
from __future__ import annotations
import ast, csv, json, math, os, re, time, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")

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
    print("진단 v75: 3 prompt variants ensemble", flush=True)
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

    sampling = SamplingParams(temperature=0, max_tokens=8)
    PROMPT_TEMPLATES = [
        # A: original v54 phrasing
        """Patient: {age_sex}
Chief complaint: {chief}
{profile}

Diagnosis hypothesis: {name}

How likely is this hypothesis given the patient's presentation? Reply with ONLY a percentage 0-100 (0=impossible, 100=certain).""",

        # B: probability framing
        """Clinical case:
Patient: {age_sex}
Chief complaint: {chief}
{profile}

Question: What is the probability that this patient has "{name}"?
Answer with ONLY an integer 0-100.""",

        # C: rank-as-DDx framing
        """Patient: {age_sex}
Chief complaint: {chief}
{profile}

Considering the differential diagnosis, rate how strongly the patient's presentation supports the diagnosis "{name}".
Reply with ONLY a number 0-100 (0=does not fit, 100=textbook match).""",
    ]

    n_templates = len(PROMPT_TEMPLATES)
    print(f"\n[4] {n_templates}-prompt ensemble scoring...", flush=True)
    convs = []
    pair_meta = []  # (c_idx, d_idx, t_idx)
    for c_idx, c in enumerate(candidates):
        p = c["patient"]
        profile = patient_profile(p["evidences"])
        ie = p.get("initial", "")
        chief = ev_info.get(ie, {}).get("question_en", ie) if ie else "—"
        chief = re.sub(r"Do you |Have you |Are you ", "", chief).rstrip("?").strip()
        age_sex = f"{p['age']}yo {'Male' if p['sex']=='M' else 'Female'}"
        for d_idx, dc in enumerate(dcs_list):
            name = disease_full_name(cui2name.get(dc, dc))
            for t_idx, tmpl in enumerate(PROMPT_TEMPLATES):
                prompt = tmpl.format(age_sex=age_sex, chief=chief, profile=profile, name=name)
                convs.append([{"role": "user", "content": prompt}])
                pair_meta.append((c_idx, d_idx, t_idx))

    print(f"  Total LLM calls: {len(convs)}", flush=True)
    CHUNK = 5000
    score_matrix = np.zeros((n, len(dcs_list), n_templates), dtype=np.float32)
    t0 = time.time()
    for chunk_start in range(0, len(convs), CHUNK):
        chunk = convs[chunk_start:chunk_start+CHUNK]
        outs = llm.chat(chunk, sampling)
        for i, out in enumerate(outs):
            c_idx, d_idx, t_idx = pair_meta[chunk_start + i]
            text = out.outputs[0].text.strip()
            m = re.search(r"(\d+)", text)
            score = int(m.group(1)) if m else 0
            score = min(score, 100)
            score_matrix[c_idx, d_idx, t_idx] = score
        elapsed = time.time() - t0
        eta = elapsed / max(chunk_start+len(chunk), 1) * (len(convs) - chunk_start - len(chunk))
        print(f"  진행: {chunk_start+len(chunk)}/{len(convs)} ({elapsed:.0f}초, ETA {eta:.0f}초)", flush=True)
    print(f"  완료: {time.time()-t0:.0f}초", flush=True)

    np.save("pilot/results/v75_score_matrix.npy", score_matrix)

    # Variant 1: average across templates
    avg = score_matrix.mean(axis=2)
    t1 = sum(1 for c_idx, c in enumerate(candidates) if dcs_list[int(np.argmax(avg[c_idx]))] == c["true_dc"])
    print(f"\n  v75-avg (mean ensemble) @1={100*t1/n:.1f}%", flush=True)

    # Variant 2: max across templates
    mx = score_matrix.max(axis=2)
    t1m = sum(1 for c_idx, c in enumerate(candidates) if dcs_list[int(np.argmax(mx[c_idx]))] == c["true_dc"])
    print(f"  v75-max (max ensemble) @1={100*t1m/n:.1f}%", flush=True)

    # Variant 3: rank-fusion (Borda count)
    borda = np.zeros((n, len(dcs_list)), dtype=np.float32)
    for t_idx in range(n_templates):
        for c_idx in range(n):
            order = np.argsort(-score_matrix[c_idx, :, t_idx])
            for r, idx in enumerate(order):
                borda[c_idx, idx] += (len(dcs_list) - r)
    t1b = sum(1 for c_idx, c in enumerate(candidates) if dcs_list[int(np.argmax(borda[c_idx]))] == c["true_dc"])
    print(f"  v75-borda (rank fusion) @1={100*t1b/n:.1f}%", flush=True)

    # Each template alone
    for t_idx in range(n_templates):
        t1_t = sum(1 for c_idx, c in enumerate(candidates) if dcs_list[int(np.argmax(score_matrix[c_idx, :, t_idx]))] == c["true_dc"])
        print(f"  v75-T{t_idx} alone @1={100*t1_t/n:.1f}%", flush=True)

    # Topk for best variant
    def topk_acc(M, k):
        hits = 0
        for c_idx, c in enumerate(candidates):
            top = np.argsort(-M[c_idx])[:k]
            if any(dcs_list[i] == c["true_dc"] for i in top): hits += 1
        return 100 * hits / n

    best_t1 = max(t1, t1m, t1b)
    if best_t1 == t1: best_M = avg; best_label = "avg"
    elif best_t1 == t1m: best_M = mx; best_label = "max"
    else: best_M = borda; best_label = "borda"

    print(f"\n  Best: {best_label} @1={100*best_t1/n:.1f}% @3={topk_acc(best_M,3):.1f}% @5={topk_acc(best_M,5):.1f}%", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"v75 BEST GTPA@1 = {100*best_t1/n:.1f}% ({best_label}, SUBSET={n})", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
