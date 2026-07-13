#!/usr/bin/env python3
"""DDXPlus 한 케이스 의학적 상세 분석.

seed로 랜덤 patient 추출 → 모든 evidence를 임상적으로 해석.
IE 단계 / KG matching / 진단 추론 흐름을 사례로 검증.
"""
from __future__ import annotations
import sys, csv, ast, json, random, argparse, pickle, math
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--graph", default="pilot/data/onlykg_graph_v95_full_s3.pkl")
    ap.add_argument("--n_pool", type=int, default=10000,
                    help="patient pool size to sample from (read N patients then random pick)")
    args = ap.parse_args()

    ev_meta = json.load(open("data/ddxplus/release_evidences.json"))
    value_cuis = json.load(open(MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"))
    icd = json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    cond = json.load(open("data/ddxplus/release_conditions_en.json"))

    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    fr2en = {info.get("cond-name-fr",""): dn for dn, info in cond.items()}
    cui2en = {fr2cui[fr]: fr2en.get(fr, fr) for fr in fr2cui}

    # Read pool
    rng = random.Random(args.seed)
    pool = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            pool.append(row)
            if len(pool) >= args.n_pool: break
    case = rng.choice(pool)

    print("="*78)
    print(f"DDXPlus Case (seed={args.seed}, pool={len(pool)}, idx selected from pool)")
    print("="*78)
    age = case["AGE"]; sex = case["SEX"]
    truth_fr = case["PATHOLOGY"]
    truth_en = fr2en.get(truth_fr, truth_fr)
    truth_cui = fr2cui.get(truth_fr, "?")
    init_ev = case["INITIAL_EVIDENCE"]
    init_q = ev_meta.get(init_ev, {}).get("question_en", "?")

    print(f"\n📋 Demographics")
    print(f"  Age: {age} years")
    print(f"  Sex: {sex} ({'Female' if sex=='F' else 'Male'})")

    print(f"\n🎯 Ground truth (PATHOLOGY)")
    print(f"  French: {truth_fr}")
    print(f"  English: {truth_en}")
    print(f"  UMLS CUI: {truth_cui}")

    print(f"\n🩺 INITIAL_EVIDENCE (chief complaint, 환자가 처음 호소한 것)")
    print(f"  Token: {init_ev}")
    print(f"  Q (English): {init_q}")

    # Parse evidence list
    evs = ast.literal_eval(case["EVIDENCES"])
    print(f"\n📝 EVIDENCES — 의사의 문진 흐름 (총 {len(evs)} tokens)")
    print(f"  의사가 {len(evs)}개 질문을 통해 추가 정보 수집")

    # Group by base evidence
    by_base = defaultdict(list)
    for ev in evs:
        if "_@_" in ev:
            base, val = ev.split("_@_", 1)
            by_base[base].append(val)
        else:
            by_base[ev].append("_BINARY_YES_")

    # Print each base with all answers + clinical interpretation
    n_ev_section = 1
    for base, vals in by_base.items():
        m = ev_meta.get(base, {})
        en_q = m.get("question_en", "?")
        dtype = m.get("data_type", "?")
        type_name = {"B":"Binary Yes/No", "M":"Multi-choice (1개)", "C":"Multi-select (복수)"}.get(dtype, dtype)

        print(f"\n  [{n_ev_section}] {base}  ({type_name})")
        print(f"      Q: {en_q}")
        n_ev_section += 1

        # CUI lookup
        base_cuis = value_cuis.get(base, {}).get("_question", [])
        is_numeric = False
        try:
            num_vals = [int(v) for v in vals if v != "_BINARY_YES_"]
            if num_vals and all(0 <= n <= 10 for n in num_vals):
                is_numeric = True
        except: pass

        for v in vals:
            if v == "_BINARY_YES_":
                print(f"      답: ✓ YES")
            else:
                vm = m.get("value_meaning", {}).get(v, {})
                en_v = vm.get("en", v)
                print(f"      답: {v} ({en_v})")
                v_cuis = value_cuis.get(base, {}).get(v, [])
                extra = set(v_cuis) - set(base_cuis)
                if extra:
                    print(f"          ↳ extra CUI: {sorted(extra)}")

        if is_numeric:
            print(f"      ⚠️ NUMERIC FORM — value-별 CUI 구분 없음 (모두 {base_cuis}만)")

        print(f"      Base CUI: {base_cuis}")

    # Differential diagnosis (의사의 differential ground truth)
    print(f"\n🔬 DIFFERENTIAL_DIAGNOSIS (의사 differential, ground truth 분포)")
    ddx = ast.literal_eval(case["DIFFERENTIAL_DIAGNOSIS"])
    for d, p in ddx:
        en = fr2en.get(d, d)
        mark = " ⭐ TRUTH" if d == truth_fr else ""
        print(f"  {p*100:5.1f}%  {d:30s} ({en}){mark}")

    # KG matching
    print(f"\n🔍 KG matching analysis (v95_full KG)")
    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open("pilot/data/pr_universe.json")))
    binary_evs = {e for e, mm in ev_meta.items()
                  if mm.get("data_type") == "B" and mm.get("default_value") == 0}

    # patient evidence CUIs
    pos_cuis = set()
    for ev in evs:
        if "_@_" in ev:
            base, val = ev.split("_@_", 1)
            mm = value_cuis.get(base, {})
            for k in ("_question", val):
                vv = mm.get(k, [])
                if isinstance(vv, list): pos_cuis.update(vv)
        else:
            pos_cuis.update(value_cuis.get(ev, {}).get("_question", []))
    print(f"  Patient evidence CUI set: {len(pos_cuis)} unique CUIs")

    # Truth disease profile in KG
    allowed = {"patient_reportable","history","demographic"}
    profile = defaultdict(float)
    if truth_cui in G:
        for _, p, ed in G.out_edges(truth_cui, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            cat = ed.get("category")
            if cat is None and p not in pr: continue
            if cat is not None and cat not in allowed: continue
            profile[p] += ed.get("weight", 0)
    matched = pos_cuis & set(profile.keys())
    missing = pos_cuis - set(profile.keys())
    print(f"  Truth disease profile size: {len(profile)} CUIs")
    print(f"  Patient ∩ Truth profile: {len(matched)} CUIs (good match!)")
    print(f"  Patient \\ Truth profile: {len(missing)} CUIs (patient has but disease profile doesn't)")

    if missing:
        print(f"\n  Missing-from-truth CUIs (potential false signals):")
        for c in list(missing)[:8]:
            ng = G.nodes[c].get("name", c) if c in G else c
            print(f"    {c} ({ng[:60]})")

    print()
    print("="*78)
    print("의학적 해석 (case-specific narrative)")
    print("="*78)
    print(f"""
이 환자는 {age}세 {('여성' if sex=='F' else '남성')}이며 chief complaint은 \"{init_q[:80]}\"입니다.
의사는 {len(evs)}개의 후속 질문(검사 항목)을 통해 {len(by_base)}개의 다른 base evidence를 평가했습니다.
최종 truth disease는 \"{truth_en}\" ({truth_cui})로 라벨됨.

DDXPlus의 ground truth differential은 disease별 확률 분포로 주어지는데, top-1이 항상 truth인
것은 아닙니다 (이 케이스 truth 순위: {[i+1 for i,(d,_) in enumerate(ddx) if d==truth_fr][0] if any(d==truth_fr for d,_ in ddx) else 'N/A'} / {len(ddx)}).
의학적으로는 differential 자체가 의사의 추론이고, top-1은 그 시점의 best guess이지만
disease confidence를 모두 명시한 dist임.

KG matching: 환자의 {len(pos_cuis)} unique CUI 중 {len(matched)}개가 truth disease profile에
존재. 나머지 {len(missing)}개는 truth disease와는 직접 매칭되지 않는 CUI들로, 다른 disease를
가리킬 수 있는 noisy signal이거나 KG에서 누락된 phenotype edge.
""")


if __name__ == "__main__":
    main()
