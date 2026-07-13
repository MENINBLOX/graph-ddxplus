#!/usr/bin/env python3
"""RareBench 진단 평가.

원칙:
  - KG: PubMed에서 RareBench 질환 이름으로 독립 구축
  - 환자 데이터: HPO 표현형 코드 → UMLS CUI 매핑 (벤치마크 제공)
  - 진단 알고리즘: KG Bayesian 점수 → LR 학습
"""
from __future__ import annotations
import json, math, os, time
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

KG_CACHE = Path("pilot/results/kg_rarebench_cache.json")
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}


def main():
    print("="*80, flush=True)
    print("RareBench 진단 평가", flush=True)
    print("="*80, flush=True)

    if not KG_CACHE.exists():
        print(f"KG 캐시 없음: {KG_CACHE}", flush=True); return

    with open(KG_CACHE) as f: kg_data = json.load(f)
    pc = Counter()
    for k, v in kg_data["pair_counts"]: pc[tuple(k)] = v
    print(f"KG: {len(pc):,} 쌍", flush=True)

    # RareBench 데이터
    with open("data/rarebench/disease_umls_mapping.json") as f:
        dm = json.load(f)["mapping"]
    with open("data/rarebench/hpo_umls_mapping.json") as f:
        hm = json.load(f)["mapping"]

    # 환자 로드
    patients = []
    for fname in ["HMS", "LIRICAL", "MME", "RAMEDIS"]:
        path = f"data/rarebench/data/{fname}.jsonl"
        if not os.path.exists(path): continue
        with open(path) as fp:
            for line in fp:
                d = json.loads(line)
                phenotypes = d.get("Phenotype", [])
                rare_diseases = d.get("RareDisease", [])
                # HPO → CUI
                phen_cuis = []
                for hpo in phenotypes:
                    if hpo in hm and hm[hpo].get("umls_cui"):
                        phen_cuis.append(hm[hpo]["umls_cui"])
                # Disease → CUI
                disease_cuis = []
                for dx in rare_diseases:
                    if dx in dm and dm[dx].get("umls_cui"):
                        disease_cuis.append(dm[dx]["umls_cui"])
                if phen_cuis and disease_cuis:
                    patients.append({"phen_cuis": phen_cuis, "disease_cuis": disease_cuis})
    print(f"환자: {len(patients):,}", flush=True)

    # KG 질환 CUI 리스트
    dcs_list = sorted(set(cui for cui, info in kg_data["diseases"].items() if isinstance(info, dict) and info.get("cui")))
    if not dcs_list:
        # fallback: extract from KG pairs
        dcs_set = set()
        for (a, b), _ in pc.items():
            for d in [a, b]:
                # disease CUI는 KG에서 더 많은 관계를 가진 쪽
                pass
        dcs_list = sorted(set(d for k in pc for d in [k[0], k[1]] if k[0] in [info.get("cui") for info in kg_data["diseases"].values() if isinstance(info, dict)] or k[1] in [info.get("cui") for info in kg_data["diseases"].values() if isinstance(info, dict)]))

    # 더 안전하게: kg_data["diseases"] 구조에서 CUI 추출
    dcs_set = set()
    for dn, info in kg_data["diseases"].items():
        if isinstance(info, dict) and info.get("cui"):
            dcs_set.add(info["cui"])
    dcs_list = sorted(dcs_set)
    dcs = set(dcs_list)
    print(f"KG 질환 CUIs: {len(dcs_list)}", flush=True)

    # 질환-증상 맵
    ds = defaultdict(dict)
    for (a, b), cnt in pc.items():
        if a in NOISE or b in NOISE: continue
        if a in dcs: ds[a][b] = cnt
        if b in dcs: ds[b][a] = cnt
    ds = dict(ds)
    all_s = set()
    for syms in ds.values(): all_s.update(syms.keys())

    # 환자 평가: HPO CUI → KG 매칭 (이미 CUI니까 바로 사용)
    # 다중 진단 가능: top-1 정답이 환자의 disease_cuis 중 하나에 속하면 정답
    print("\n진단...", flush=True)
    le = LabelEncoder()
    le.fit(dcs_list)

    # 평가 가능한 환자만 (disease_cuis 중 적어도 하나가 KG에 있음)
    valid_patients = []
    for p in patients:
        if any(dc in dcs for dc in p["disease_cuis"]):
            valid_patients.append(p)
    print(f"  평가 가능: {len(valid_patients)}/{len(patients)}", flush=True)

    if not valid_patients:
        print("평가 가능한 환자 없음", flush=True); return

    # Bayesian 점수 계산
    def bayesian_scores(phen_cuis):
        kg_scores = np.zeros(len(dcs_list))
        for i, dc in enumerate(dcs_list):
            s = ds.get(dc, {})
            if not s: kg_scores[i] = -10; continue
            tw = sum(s.values()) + len(all_s) * 0.1
            kg_scores[i] = sum(math.log((s[x]+0.1)/tw+1e-10) if x in s else math.log(0.1/tw+1e-10) for x in phen_cuis)
        return kg_scores

    # Bayesian only (no training)
    correct_bay = 0
    bay_at_5 = 0; bay_at_10 = 0
    for p in valid_patients:
        scores = bayesian_scores(p["phen_cuis"])
        ranked = np.argsort(-scores)
        ranked_cuis = [dcs_list[i] for i in ranked]
        truth = set(p["disease_cuis"])
        if ranked_cuis[0] in truth: correct_bay += 1
        if any(c in truth for c in ranked_cuis[:5]): bay_at_5 += 1
        if any(c in truth for c in ranked_cuis[:10]): bay_at_10 += 1

    n = len(valid_patients)
    print(f"\nBayesian only: @1={100*correct_bay/n:.1f}% @5={100*bay_at_5/n:.1f}% @10={100*bay_at_10/n:.1f}%", flush=True)

    # 학습 가능한지 확인 (각 클래스에 충분한 샘플 필요)
    label_counts = Counter()
    for p in valid_patients:
        for dc in p["disease_cuis"]:
            if dc in dcs: label_counts[dc] += 1
    print(f"\n질환별 환자 수 분포:", flush=True)
    print(f"  최소: {min(label_counts.values())}", flush=True)
    print(f"  최대: {max(label_counts.values())}", flush=True)
    print(f"  평균: {np.mean(list(label_counts.values())):.1f}", flush=True)
    print(f"  >=2 환자: {sum(1 for c in label_counts.values() if c>=2)}", flush=True)

    # LR 학습은 환자 수가 너무 적으면 의미 없음. 5-fold CV로 평가
    if n < 100:
        print("\n환자 수 부족, LR 학습 생략", flush=True)
    else:
        # 단일 라벨로 단순화 (첫 번째 disease_cui 사용)
        from sklearn.model_selection import KFold
        X = np.array([bayesian_scores(p["phen_cuis"]) for p in valid_patients])
        y = []
        for p in valid_patients:
            for dc in p["disease_cuis"]:
                if dc in dcs:
                    y.append(le.transform([dc])[0])
                    break
            else: y.append(-1)
        y = np.array(y)
        valid_mask = y >= 0
        X = X[valid_mask]; y = y[valid_mask]

        if len(X) >= 50:
            print(f"\nLR 5-fold CV ({len(X)} 환자)...", flush=True)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            accs = []
            for fold, (tr, te) in enumerate(kf.split(X)):
                lr = LogisticRegression(max_iter=2000, C=1.0)
                lr.fit(X[tr], y[tr])
                acc = np.mean(lr.predict(X[te]) == y[te])
                accs.append(acc)
                print(f"  Fold {fold+1}: {100*acc:.1f}%", flush=True)
            print(f"평균 LR @1: {100*np.mean(accs):.1f}% (±{100*np.std(accs):.1f})", flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"RareBench Bayesian @1 = {100*correct_bay/n:.1f}%", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
