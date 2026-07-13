#!/usr/bin/env python3
"""SymCat 진단 평가 (DDXPlus와 동일 파이프라인).

SymCat은 환자 데이터셋이 없으므로 질환-증상 빈도 분포에서 환자 시뮬레이션.
원칙: KG는 PubMed 독립 구축. SymCat의 disease_symptom_pairs는 평가용 ground truth.
"""
from __future__ import annotations
import json, math, re, time, random
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import ahocorasick
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

UMLS_DIR = Path("data/umls_extracted")
KG_CACHE = Path("pilot/results/kg_symcat_cache.json")
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}
STOPWORDS = {'a','an','the','and','or','of','for','with','that','this','your','you'}

random.seed(42)
np.random.seed(42)


def main():
    print("="*80, flush=True)
    print("SymCat 진단 평가", flush=True)
    print("="*80, flush=True)

    # Load KG
    if not KG_CACHE.exists():
        print(f"KG 캐시 없음: {KG_CACHE}", flush=True)
        return

    with open(KG_CACHE) as f: kg_data = json.load(f)
    pc = Counter()
    for k, v in kg_data["pair_counts"]: pc[tuple(k)] = v
    diseases = kg_data["diseases"]
    print(f"KG: {len(pc):,} 쌍, {len(diseases)} 질환", flush=True)

    # SymCat data (질환-증상 빈도)
    with open("data/symcat/symcat_parsed.json") as f:
        sc = json.load(f)
    pairs = sc["disease_symptom_pairs"]  # {disease: [[symptom, freq%], ...]}

    # 질환 CUI 리스트
    dcs_list = []
    name_to_cui = {}
    for dn in sorted(diseases):
        if diseases[dn]["cui"]:
            dcs_list.append(diseases[dn]["cui"])
            name_to_cui[dn] = diseases[dn]["cui"]
    dcs = set(dcs_list)
    print(f"질환 CUI: {len(dcs_list)}", flush=True)

    # UMLS load
    print("\nUMLS 로드...", flush=True)
    can = defaultdict(set); cp = {}
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG":
                can[p[0]].add(p[14].strip())
                if p[2] == "P" and p[0] not in cp: cp[p[0]] = p[14].strip()

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

    # Generate simulated patients
    print("\n환자 시뮬레이션...", flush=True)
    train_patients = []
    test_patients = []
    for dn in sorted(diseases):
        if dn not in pairs: continue
        if dn not in name_to_cui: continue
        sym_list = pairs[dn]  # [[symptom, freq%], ...]
        if not sym_list: continue
        true_dc = name_to_cui[dn]

        # Generate patients: each symptom is included with its frequency probability
        for split, n in [("train", 200), ("test", 50)]:
            for _ in range(n):
                patient_syms = []
                for sym, freq in sym_list:
                    if random.random() * 100 < freq:
                        patient_syms.append(sym)
                if patient_syms:
                    if split == "train":
                        train_patients.append((true_dc, patient_syms))
                    else:
                        test_patients.append((true_dc, patient_syms))
    print(f"  Train: {len(train_patients):,}, Test: {len(test_patients):,}", flush=True)

    # Match symptoms to KG (text matching)
    def patient_to_kg(symptoms):
        cuis = set()
        text = " . ".join(s.lower() for s in symptoms)
        for ei, (n, cui) in aho.iter(text):
            si = ei - len(n) + 1
            if si > 0 and text[si-1].isalpha(): continue
            if ei+1 < len(text) and text[ei+1].isalpha(): continue
            cuis.add(cui)

        kg_scores = np.zeros(len(dcs_list))
        for i, dc in enumerate(dcs_list):
            s = ds.get(dc, {})
            if not s: kg_scores[i] = -10; continue
            tw = sum(s.values()) + len(all_s) * 0.1
            kg_scores[i] = sum(math.log((s[x]+0.1)/tw+1e-10) if x in s else math.log(0.1/tw+1e-10) for x in cuis)
        return kg_scores

    le = LabelEncoder()
    le.fit(dcs_list)

    print("\n특성 추출...", flush=True)
    t0 = time.time()
    train_X = np.array([patient_to_kg(syms) for _, syms in train_patients])
    train_y = np.array([le.transform([dc])[0] for dc, _ in train_patients])
    test_X = np.array([patient_to_kg(syms) for _, syms in test_patients])
    test_y = np.array([le.transform([dc])[0] for dc, _ in test_patients])
    print(f"  Train: {train_X.shape}, Test: {test_X.shape} ({time.time()-t0:.0f}s)", flush=True)

    # Bayesian only
    bay_pred = np.argmax(test_X, axis=1)
    bay_acc = np.mean(bay_pred == test_y)
    print(f"\nBayesian argmax: @1={100*bay_acc:.1f}%", flush=True)

    # LR
    print("\nLR 학습...", flush=True)
    t0 = time.time()
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    lr.fit(train_X, train_y)
    lr_pred = lr.predict(test_X)
    lr_acc = np.mean(lr_pred == test_y)
    lr_proba = lr.predict_proba(test_X)
    print(f"LR: @1={100*lr_acc:.1f}%", flush=True)
    for k in [3, 5, 10]:
        topk = np.argsort(-lr_proba, axis=1)[:, :k]
        topk_acc = np.mean([test_y[i] in topk[i] for i in range(len(test_y))])
        print(f"  @{k}={100*topk_acc:.1f}%", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"SymCat GTPA@1 = {100*lr_acc:.1f}%", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
