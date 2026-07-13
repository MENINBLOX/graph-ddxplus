#!/usr/bin/env python3
"""KG features를 자가인지 가능(Y) vs 의학검사 필요(N)로 LLM 분류.

DDXPlus 환자 evidence는 모두 자가인지 가능한 증상이므로, KG features 중
검사로만 알 수 있는 것 (X-ray의 lymphadenopathy, ECG의 atrial fibrillation 등)을
필터링하면 매칭 신호가 정확해진다.

Output: pilot/results/feature_patient_reportable.json
  cui -> 'Y' (자가인지 가능) | 'N' (검사 필요)
"""
from __future__ import annotations
import json, os, re
from collections import Counter, defaultdict
from pathlib import Path
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")
KG_CACHE = Path("pilot/results/kg_v3_cache.json")
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}


def main():
    print("="*80, flush=True)
    print("Classify KG features: patient-reportable vs clinical-only", flush=True)
    print("="*80, flush=True)

    cp = {}
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG" and p[2] == "P" and p[0] not in cp:
                cp[p[0]] = p[14].strip()

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open(KG_CACHE) as f: cache = json.load(f)

    pc = Counter()
    for k, v in cache["pair_counts"]: pc[tuple(k)] = v

    dcs = set()
    for dn, info in cond.items():
        if dn not in icd_map: continue
        dcs.add(icd_map[dn]["cui"])

    # Get all unique features in top-30 of any disease
    all_features = set()
    ds = defaultdict(dict)
    for (a, b), cnt in pc.items():
        if a in NOISE or b in NOISE: continue
        if a in dcs: ds[a][b] = cnt
        if b in dcs: ds[b][a] = cnt
    for dc, feats in ds.items():
        top = sorted(feats.items(), key=lambda x: -x[1])[:30]
        for cui, cnt in top:
            all_features.add(cui)

    # Build (cui, name) list
    feats_list = []
    for cui in sorted(all_features):
        n = cp.get(cui, cui)
        if not n or len(n) > 80: continue
        feats_list.append((cui, n))
    print(f"Features to classify: {len(feats_list)}")

    # vLLM
    print("\nvLLM init...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})

    sampling = SamplingParams(temperature=0, max_tokens=1500)

    BATCH = 30
    convs = []
    batch_metas = []
    for i in range(0, len(feats_list), BATCH):
        batch = feats_list[i:i+BATCH]
        items = "\n".join(f"{j+1}. {name}" for j, (cui, name) in enumerate(batch))
        prompt = f"""For each medical term below, classify whether a typical patient WITHOUT medical training can self-report or notice this:

- Y: Patient can self-report (e.g., cough, fever, pain, rash, dizziness, vomiting, headache, runny nose)
- N: Requires medical examination (e.g., lymphadenopathy on imaging, ST segment depression on ECG, granuloma on biopsy, crackles on auscultation, blood test results, X-ray findings)

Items:
{items}

Output ONLY a numbered list of Y/N (no explanation):
1. Y
2. N
..."""
        convs.append([{"role": "user", "content": prompt}])
        batch_metas.append(batch)

    print(f"\nClassifying {len(feats_list)} features in {len(convs)} batches...", flush=True)
    outs = llm.chat(convs, sampling)

    classification = {}
    parse_errors = 0
    for batch_idx, out in enumerate(outs):
        text = out.outputs[0].text.strip()
        batch = batch_metas[batch_idx]
        # Parse "1. Y\n2. N\n..."
        lines = text.split('\n')
        labels = {}
        for line in lines:
            m = re.match(r"\s*(\d+)\s*[\.\)]\s*([YN])", line.strip())
            if m:
                idx = int(m.group(1)) - 1
                labels[idx] = m.group(2)
        for j, (cui, name) in enumerate(batch):
            if j in labels:
                classification[cui] = labels[j]
            else:
                classification[cui] = 'N'  # default to safe (filter out)
                parse_errors += 1

    print(f"\nClassified: {len(classification)} (parse errors: {parse_errors})")
    y_count = sum(1 for v in classification.values() if v == 'Y')
    n_count = sum(1 for v in classification.values() if v == 'N')
    print(f"  Patient-reportable (Y): {y_count}")
    print(f"  Clinical-only (N): {n_count}")

    # Save
    with open("pilot/results/feature_patient_reportable.json", "w") as f:
        json.dump(classification, f, indent=2)

    # Show samples
    print("\n샘플 patient-reportable (Y):")
    for cui, label in classification.items():
        if label == 'Y':
            print(f"  {cp.get(cui, cui)[:50]}")
        if sum(1 for c, l in classification.items() if l == 'Y' and list(classification.keys()).index(c) <= list(classification.keys()).index(cui)) > 15: break

    print("\n샘플 clinical-only (N):")
    cnt = 0
    for cui, label in classification.items():
        if label == 'N':
            print(f"  {cp.get(cui, cui)[:50]}")
            cnt += 1
        if cnt > 15: break


if __name__ == "__main__":
    main()
