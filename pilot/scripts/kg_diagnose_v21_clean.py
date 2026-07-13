#!/usr/bin/env python3
"""진단 v21: release_evidences.json 완전 배제.

원칙 준수:
  - 벤치마크에서 질환 이름만 사용
  - KG: PubMed 독립 구축 (이미 완료)
  - 환자 증상: 프랑스어 이름을 일반 번역(LLM)으로 영어 변환 → KG 매칭
  - release_evidences.json의 question_en, value_meaning 사용 안 함

파이프라인:
  1. 환자 evidence 프랑스어 이름 → LLM 번역 캐시 → 영어 의학 용어
  2. 영어 의학 용어 → KG 증상 이름 Aho-Corasick 매칭
  3. 매칭된 CUI → Bayesian 점수 (49개)
  4. Bayesian 점수 + age/sex → LR 학습 → 진단
"""
from __future__ import annotations
import ast, csv, json, math, re, time
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import ahocorasick
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

UMLS_DIR = Path("data/umls_extracted")
KG_CACHE = Path("pilot/results/kg_v3_cache.json")
TRANSLATE_CACHE = Path("pilot/results/evidence_fr_to_en.json")
STOPWORDS = {'does','have','your','you','the','and','for','are','with','that','this','from','been','were','being','which','their','than','other','about','into','over','some','only','very','also','just','more','most','such','much','will','would','could','should','make','like','time','when','what','where','how','who','all','each','every','both','few','any','not','can','may','her','his','its','our','they','them','then','had','has','him','but','one','two','way','day','did','get','got','let','say','she','too','use','yes','yet','now','new','old','see','own','why','try','ask','set','not specified','unspecified','general','context-dependent'}
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}


def main():
    print("=" * 80, flush=True)
    print("진단 v21: release_evidences.json 완전 배제", flush=True)
    print("=" * 80, flush=True)

    # [1] Load
    print("\n[1] 로드...", flush=True)
    can = defaultdict(set)
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG":
                can[p[0]].add(p[14].strip())

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open(KG_CACHE) as f: cache = json.load(f)
    with open(TRANSLATE_CACHE) as f: translations = json.load(f)

    pc = Counter()
    for k, v in cache["pair_counts"]: pc[tuple(k)] = v

    dcs_list = []
    fr2cui = {}
    for dn in sorted(cond):
        if dn not in icd_map: continue
        dc = icd_map[dn]["cui"]; dcs_list.append(dc)
        fr2cui[cond[dn].get("cond-name-fr", "")] = dc
    dcs = set(dcs_list)

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
    print(f"  KG: {len(pc):,} 쌍, 번역: {len(translations)}개", flush=True)

    # [2] 텍스트 매칭: 번역된 영어 용어로 KG 매칭
    def patient_to_kg_scores(evidences):
        """환자 evidence → 번역 → KG 매칭 → Bayesian 점수."""
        cuis = set()
        for ev in evidences:
            parts = ev.split("_@_")
            base = parts[0]
            value = parts[1] if len(parts) > 1 else None

            # Base name 번역
            en_base = translations.get(base, base)  # 번역 없으면 원문 사용
            if en_base and en_base.lower() not in STOPWORDS:
                # 영어 의학 용어로 KG 매칭
                text = en_base.lower()
                for ei, (n, cui) in aho.iter(text):
                    si = ei - len(n) + 1
                    if si > 0 and text[si-1].isalpha(): continue
                    if ei+1 < len(text) and text[ei+1].isalpha(): continue
                    cuis.add(cui)

            # Value 번역 (통증 위치 등)
            if value:
                en_val = translations.get(value, value)
                if en_val and en_val.lower() not in STOPWORDS:
                    text = en_val.lower()
                    for ei, (n, cui) in aho.iter(text):
                        si = ei - len(n) + 1
                        if si > 0 and text[si-1].isalpha(): continue
                        if ei+1 < len(text) and text[ei+1].isalpha(): continue
                        cuis.add(cui)

                    # pain + location compound
                    en_base_lower = en_base.lower() if en_base else ""
                    if "pain" in en_base_lower:
                        compound = f"{en_val.lower()} pain"
                        for ei, (n, cui) in aho.iter(compound):
                            si = ei - len(n) + 1
                            if si > 0 and compound[si-1].isalpha(): continue
                            if ei+1 < len(compound) and compound[ei+1].isalpha(): continue
                            cuis.add(cui)

        # Bayesian scores
        kg_scores = np.zeros(len(dcs_list))
        for i, dc in enumerate(dcs_list):
            s = ds.get(dc, {})
            if not s: kg_scores[i] = -10; continue
            tw = sum(s.values()) + len(all_s) * 0.1
            kg_scores[i] = sum(math.log((s[x]+0.1)/tw+1e-10) if x in s
                               else math.log(0.1/tw+1e-10) for x in cuis)
        return kg_scores

    le = LabelEncoder()
    le.fit(dcs_list)

    # [3] Build features
    print("\n[2] 특성 추출 (train)...", flush=True)
    t0 = time.time()
    train_X = []; train_y = []
    with open("data/ddxplus/release_train_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            tdc = fr2cui.get(row["PATHOLOGY"])
            if not tdc: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            kg = patient_to_kg_scores(evs)
            demo = np.array([float(row.get("AGE", 30)) / 100.0,
                             1.0 if row.get("SEX", "M") == "M" else 0.0])
            train_X.append(np.concatenate([kg, demo]))
            train_y.append(le.transform([tdc])[0])
            if (i+1) % 100000 == 0: print(f"  {i+1:,}...", flush=True)
    train_X = np.array(train_X); train_y = np.array(train_y)
    print(f"  {train_X.shape[0]:,} x {train_X.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

    print("[2b] 특성 추출 (test)...", flush=True)
    t0 = time.time()
    test_X = []; test_y = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            tdc = fr2cui.get(row["PATHOLOGY"])
            if not tdc: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            kg = patient_to_kg_scores(evs)
            demo = np.array([float(row.get("AGE", 30)) / 100.0,
                             1.0 if row.get("SEX", "M") == "M" else 0.0])
            test_X.append(np.concatenate([kg, demo]))
            test_y.append(le.transform([tdc])[0])
    test_X = np.array(test_X); test_y = np.array(test_y)
    print(f"  {test_X.shape[0]:,} x {test_X.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

    # [4] Train and evaluate
    print("\n[3] 학습...", flush=True)

    # Bayesian only (no training, just argmax)
    bayesian_pred = np.argmax(test_X[:, :len(dcs_list)], axis=1)
    bayesian_acc = np.mean(bayesian_pred == test_y)
    print(f"Bayesian only (argmax): @1={100*bayesian_acc:.1f}%", flush=True)

    # LR on KG scores + demo
    t0 = time.time()
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    lr.fit(train_X, train_y)
    lr_pred = lr.predict(test_X)
    lr_acc = np.mean(lr_pred == test_y)
    lr_proba = lr.predict_proba(test_X)
    print(f"LR (KG+demo): @1={100*lr_acc:.1f}%", flush=True)
    for k in [3, 5, 10]:
        topk = np.argsort(-lr_proba, axis=1)[:, :k]
        topk_acc = np.mean([test_y[i] in topk[i] for i in range(len(test_y))])
        print(f"  @{k}={100*topk_acc:.1f}%", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

    # Per-disease accuracy
    print("\n질환별 정확도:", flush=True)
    for cls_idx in range(len(dcs_list)):
        mask = test_y == cls_idx
        if mask.sum() == 0: continue
        acc = np.mean(lr_pred[mask] == test_y[mask])
        dc = dcs_list[cls_idx]
        dname = next((dn for dn, info in cond.items() if icd_map.get(dn, {}).get("cui") == dc), dc)
        print(f"  {dname:<45} {100*acc:>5.1f}% ({mask.sum():>5}명)", flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"GTPA@1 = {100*lr_acc:.1f}% (release_evidences.json 미사용)", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
