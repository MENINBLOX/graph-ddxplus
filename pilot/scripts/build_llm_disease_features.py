#!/usr/bin/env python3
"""Generate disease features using gemma-4-E4B's pretrained medical knowledge.

For each DDXPlus disease, ask LLM: list 8 most distinctive clinical features.
Save as JSON for use in v101.
"""
from __future__ import annotations
import json, os, re, time
from pathlib import Path
from vllm import LLM, SamplingParams

DISEASE_NAME_FIX = {
    "URTI": "Upper respiratory tract infection (URTI / common cold)",
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


def main():
    print("="*80, flush=True)
    print("Generate LLM-derived disease features", flush=True)
    print("="*80, flush=True)

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    diseases = {}
    for dn, info in cond.items():
        if dn not in icd_map: continue
        diseases[dn] = icd_map[dn]["cui"]

    print(f"\nDiseases: {len(diseases)}")

    print("\nvLLM init...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})

    sampling = SamplingParams(temperature=0, max_tokens=200)
    convs = []
    disease_list = list(diseases.items())
    for dn, dc in disease_list:
        full_name = DISEASE_NAME_FIX.get(dn, dn)
        prompt = f"""List the 8 MOST DISTINCTIVE clinical features (symptoms and signs) of {full_name}.

Output ONLY a comma-separated list of 8 specific features, no explanation.
Each feature should be a short medical term (2-5 words).

Features:"""
        convs.append([{"role": "user", "content": prompt}])

    print(f"\nQuerying {len(convs)} diseases...", flush=True)
    outs = llm.chat(convs, sampling)

    disease_features = {}
    for i, out in enumerate(outs):
        dn, dc = disease_list[i]
        text = out.outputs[0].text.strip()
        # Parse comma-separated features
        features = [f.strip().rstrip('.,;:') for f in text.split(',') if f.strip()]
        # Remove enumeration prefixes (1., 2., etc.)
        features = [re.sub(r"^\d+[\.\)]\s*", "", f) for f in features]
        # Filter empty
        features = [f for f in features if f and len(f) > 2 and len(f) < 60][:10]
        disease_features[dc] = features
        if i < 5:
            print(f"  {dn}: {features}", flush=True)

    # Save
    with open("pilot/results/llm_disease_features.json", "w") as f:
        json.dump(disease_features, f, indent=2)

    print(f"\nSaved to pilot/results/llm_disease_features.json")
    print(f"Total: {len(disease_features)} diseases")


if __name__ == "__main__":
    main()
