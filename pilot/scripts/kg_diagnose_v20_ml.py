#!/usr/bin/env python3
"""진단 v20: KG 특성 + evidence 특성 → ML 분류기.

KG 구축: PubMed 독립 (원칙 유지)
진단 알고리즘: training data로 학습 (정당한 접근)

특성:
  - 223개 evidence 이진 (환자가 가진 증상)
  - 49개 KG Bayesian 점수 (PubMed KG 기반)
  - 2개 인구통계 (age, sex)
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
STOPWORDS = {'does','have','your','you','the','and','for','are','with','that','this','from','been','were','being','which','their','than','other','about','into','over','some','only','very','also','just','more','most','such','much','will','would','could','should','make','like','time','when','what','where','how','who','all','each','every','both','few','any','not','can','may','her','his','its','our','they','them','then','had','has','him','but','one','two','way','day','did','get','got','let','say','she','too','use','yes','yet','now','new','old','see','own','why','try','ask','set','related','reason','consulting','significant','measured','thermometer','either','believe','racing','missing','beat','fast','irregularly','problems','situation','associated','inability','speak','trouble','keeping','opening','raising','annoying','else','body','somewhere','anywhere','nowhere','recently','currently','usually','often','sometimes','worse','better'}
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}

def main():
    print("=" * 80, flush=True)
    print("진단 v20: KG + evidence → ML 분류기", flush=True)
    print("=" * 80, flush=True)

    # Load UMLS
    print("\n[1] UMLS...", flush=True)
    can = defaultdict(set)
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG": can[p[0]].add(p[14].strip())

    # Load DDXPlus
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f: ev_fr = json.load(f)
    with open(KG_CACHE) as f: cache = json.load(f)

    pc = Counter()
    for k, v in cache["pair_counts"]: pc[tuple(k)] = v

    dcs_list = []
    fr2cui = {}
    for dn in sorted(cond):
        if dn not in icd_map: continue
        dc = icd_map[dn]["cui"]; dcs_list.append(dc)
        fr2cui[cond[dn].get("cond-name-fr", "")] = dc
    dcs = set(dcs_list)

    ev_bases_list = sorted(ev_fr.keys())
    ev_to_idx = {e: i for i, e in enumerate(ev_bases_list)}
    ev_info = {}
    for eid, info in ev_fr.items():
        ev_info[eid] = {"question_en": info.get("question_en", ""),
                        "is_antecedent": info.get("is_antecedent", False), "value_en": {}}
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
    print(f"  KG: {len(pc):,} 쌍, {len(scuis):,} 증상", flush=True)

    def tm(evidences):
        cuis = set()
        for ev in evidences:
            parts = ev.split("_@_"); base = parts[0]; value = parts[1] if len(parts) > 1 else None
            info = ev_info.get(base, {})
            if info.get("is_antecedent"): continue
            terms = []
            bc = re.sub(r"_.*", "", base); bc = re.sub(r"xx$", "", bc)
            if len(bc) >= 3 and bc not in STOPWORDS: terms.append(bc)
            q = info.get("question_en", "")
            if q:
                text = re.sub(r"\(.*?\)", "", q); text = re.sub(r"[?.,;:!]", "", text)
                terms.extend(w.lower() for w in text.split() if w.lower() not in STOPWORDS and len(w) >= 3)
                ql = q.lower()
                for ph in ["chest pain","sore throat","shortness of breath","difficulty breathing",
                            "weight loss","weight gain","loss of consciousness","muscle pain",
                            "muscle spasm","nasal congestion","runny nose","skin lesion","skin rash",
                            "black stool","bloody stool","heart palpitation","double vision","swollen"]:
                    if ph in ql: terms.append(ph)
            if value:
                val_en = info.get("value_en", {}).get(value, "")
                if val_en and val_en.lower() not in ("na", "nowhere", "n"):
                    vc = re.sub(r"\([rl]\)", "", val_en.lower()).strip()
                    if vc and vc not in STOPWORDS:
                        terms.append(vc)
                        if "pain" in q.lower():
                            terms.append(f"{vc} pain")
                            for part in vc.split():
                                if part not in STOPWORDS and len(part) >= 4: terms.append(f"{part} pain")
            pt = " . ".join(terms)
            for ei, (nm, cui) in aho.iter(pt):
                si = ei - len(nm) + 1
                if si > 0 and pt[si - 1].isalpha(): continue
                if ei + 1 < len(pt) and pt[ei + 1].isalpha(): continue
                cuis.add(cui)
        return cuis

    def extract_features(evidences, age, sex):
        ev_feat = np.zeros(len(ev_bases_list))
        for ev in evidences:
            base = ev.split("_@_")[0]
            if base in ev_to_idx: ev_feat[ev_to_idx[base]] = 1
        ps = tm(evidences)
        kg_scores = np.zeros(len(dcs_list))
        for i, dc in enumerate(dcs_list):
            s = ds.get(dc, {})
            if not s: kg_scores[i] = -10; continue
            tw = sum(s.values()) + len(all_s) * 0.1
            kg_scores[i] = sum(math.log((s[x]+0.1)/tw+1e-10) if x in s
                               else math.log(0.1/tw+1e-10) for x in ps)
        demo = np.array([float(age) / 100.0, 1.0 if sex == "M" else 0.0])
        return np.concatenate([ev_feat, kg_scores, demo])

    le = LabelEncoder()
    le.fit(dcs_list)
    n_ev = len(ev_bases_list)

    # Build features
    print("\n[2] 특성 추출 (train)...", flush=True)
    t0 = time.time()
    train_X = []; train_y = []
    with open("data/ddxplus/release_train_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            tdc = fr2cui.get(row["PATHOLOGY"])
            if not tdc: continue
            feat = extract_features(ast.literal_eval(row["EVIDENCES"]),
                                    row.get("AGE", "30"), row.get("SEX", "M"))
            train_X.append(feat)
            train_y.append(le.transform([tdc])[0])
            if (i + 1) % 100000 == 0: print(f"  {i+1:,}...", flush=True)
    train_X = np.array(train_X); train_y = np.array(train_y)
    print(f"  {train_X.shape[0]:,} x {train_X.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

    print("[2b] 특성 추출 (test)...", flush=True)
    t0 = time.time()
    test_X = []; test_y = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            tdc = fr2cui.get(row["PATHOLOGY"])
            if not tdc: continue
            feat = extract_features(ast.literal_eval(row["EVIDENCES"]),
                                    row.get("AGE", "30"), row.get("SEX", "M"))
            test_X.append(feat)
            test_y.append(le.transform([tdc])[0])
    test_X = np.array(test_X); test_y = np.array(test_y)
    print(f"  {test_X.shape[0]:,} x {test_X.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

    # Models
    print("\n[3] 학습...", flush=True)

    # A. Evidence + KG + demo
    t0 = time.time()
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", n_jobs=-1)
    lr.fit(train_X, train_y)
    lr_pred = lr.predict(test_X)
    lr_acc = np.mean(lr_pred == test_y)
    lr_proba = lr.predict_proba(test_X)
    print(f"\nLR (evidence+KG+demo): @1={100*lr_acc:.1f}%", flush=True)
    for k in [3, 5, 10]:
        topk = np.argsort(-lr_proba, axis=1)[:, :k]
        topk_acc = np.mean([test_y[i] in topk[i] for i in range(len(test_y))])
        print(f"  @{k}={100*topk_acc:.1f}%", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

    # B. Evidence + demo only (no KG)
    t0 = time.time()
    ev_demo_train = np.hstack([train_X[:, :n_ev], train_X[:, -2:]])
    ev_demo_test = np.hstack([test_X[:, :n_ev], test_X[:, -2:]])
    lr2 = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", n_jobs=-1)
    lr2.fit(ev_demo_train, train_y)
    lr2_acc = np.mean(lr2.predict(ev_demo_test) == test_y)
    print(f"LR (evidence+demo only): @1={100*lr2_acc:.1f}% ({time.time()-t0:.0f}s)", flush=True)

    # C. KG + demo only
    t0 = time.time()
    kg_demo_train = train_X[:, n_ev:]
    kg_demo_test = test_X[:, n_ev:]
    lr3 = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", n_jobs=-1)
    lr3.fit(kg_demo_train, train_y)
    lr3_acc = np.mean(lr3.predict(kg_demo_test) == test_y)
    print(f"LR (KG+demo only): @1={100*lr3_acc:.1f}% ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Evidence only: {100*lr2_acc:.1f}%", flush=True)
    print(f"KG only:       {100*lr3_acc:.1f}%", flush=True)
    print(f"Evidence + KG: {100*lr_acc:.1f}%", flush=True)
    print(f"KG 기여:       +{100*(lr_acc-lr2_acc):.1f}%p", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
