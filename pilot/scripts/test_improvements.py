#!/usr/bin/env python3
"""개선 방법 테스트: 기존 V2C 데이터 재처리.

Quick wins (LLM 재호출 불필요):
  1. 동의어 필터: MRREL SY 관계인 쌍 제거
  2. CUI 정규화: 동의어 CUI를 대표 CUI로 통합 후 집계
  3. 관계 유형 단순화: manifestation-of → 의미 재분류
  4. 전파 없이 평가 vs 전파 평가
  5. 1+2+3+4 조합
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

UMLS_DIR = Path("data/umls_extracted")
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}
ALLOWED_C = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049",
             "T033", "T031", "T040"}


def load_parent_map():
    parents = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[3] in ("PAR", "RB"):
                parents[p[0]].add(p[4])
    return dict(parents)


def load_synonym_map():
    """MRREL에서 SY(synonym) 관계 로드 → 동의어 그룹."""
    syn = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[3] in ("SY", "RO", "RQ"):
                # SY=synonym, RO=has relationship, RQ=related and possibly synonymous
                if p[3] == "SY":
                    syn[p[0]].add(p[4])
                    syn[p[4]].add(p[0])
    return dict(syn)


def load_cui_canonical():
    """CUI를 대표 CUI로 정규화하는 맵 구축.
    같은 개념의 여러 CUI를 하나로 통합.
    MRREL PAR/CHD에서 직접 부모-자식이면 부모로 정규화.
    """
    # 방법: 각 CUI의 1-level parent 중 가장 일반적인 것을 canonical로 사용
    parents = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[3] in ("PAR", "RB"):
                parents[p[0]].add(p[4])

    canonical = {}
    # parent가 1개뿐인 경우 parent를 canonical로
    for cui, pars in parents.items():
        if len(pars) == 1:
            canonical[cui] = next(iter(pars))
    return canonical


def load_cui_stys():
    r = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            r[p[0]].add(p[1])
    return dict(r)


def load_cui_names():
    names = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[1] == "ENG" and (p[0] not in names or p[2] == "P"):
                names[p[0]] = p[14]
    return names


def prepare_gold():
    with open("data/ddxplus/release_conditions_en.json") as f:
        conditions = json.load(f)
    with open("data/ddxplus/release_evidences_en.json") as f:
        ev_en = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f:
        ev_fr = json.load(f)
    with open("data/ddxplus/umls_mapping.json") as f:
        umap = json.load(f)["mapping"]
    with open("data/ddxplus/disease_umls_mapping.json") as f:
        disease_map = json.load(f)["mapping"]

    eid_to_fr = {}
    for eid, en_info in ev_en.items():
        for fr_name, fr_info in ev_fr.items():
            if en_info.get("question_en") == fr_info.get("question_en") and en_info.get("question_en"):
                eid_to_fr[eid] = fr_name
                break

    gold_pairs = set()
    for disease_name, info in conditions.items():
        d_cui = disease_map.get(disease_name, {}).get("umls_cui")
        if not d_cui:
            continue
        for eid in info.get("symptoms", {}):
            if ev_en.get(eid, {}).get("is_antecedent", False):
                continue
            fr_name = eid_to_fr.get(eid)
            if fr_name and fr_name in umap:
                cui = umap[fr_name].get("cui")
                if cui:
                    gold_pairs.add(tuple(sorted([d_cui, cui])))
    return gold_pairs


def cui_match(a, b, parent_map):
    if a == b:
        return True
    return b in parent_map.get(a, set()) or a in parent_map.get(b, set())


def evaluate(our_pairs, gold_pairs, parent_map):
    mg, mo = set(), set()
    for op in our_pairs:
        for gp in gold_pairs:
            if ((cui_match(op[0], gp[0], parent_map) and cui_match(op[1], gp[1], parent_map)) or
                    (cui_match(op[0], gp[1], parent_map) and cui_match(op[1], gp[0], parent_map))):
                mg.add(gp)
                mo.add(op)
    p = len(mo) / len(our_pairs) if our_pairs else 0
    r = len(mg) / len(gold_pairs) if gold_pairs else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return {"P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4),
            "matched": len(mg), "our": len(our_pairs), "gold": len(gold_pairs)}


def main():
    print("=" * 80)
    print("개선 방법 테스트 (V2C 데이터 재처리)")
    print("=" * 80)

    # 데이터 로드
    print("\n[1] 데이터 로드...")
    parent_map = load_parent_map()
    synonym_map = load_synonym_map()
    cui_stys = load_cui_stys()
    cui_names = load_cui_names()
    gold_pairs = prepare_gold()

    with open("pilot/data/kg_v2c_checkpoint.json") as f:
        ckpt = json.load(f)
    cls = ckpt["classifications"]
    print(f"  V2C 관계: {len(cls):,}건, Gold: {len(gold_pairs)}쌍")

    # 질환 CUI 목록
    with open("data/ddxplus/disease_umls_mapping.json") as f:
        dm = json.load(f)["mapping"]
    disease_cuis = {v["umls_cui"] for v in dm.values() if v.get("umls_cui")}

    # ============================================================
    # Baseline: V2C 현재 결과 (MC=5, CUI 전파)
    # ============================================================
    print("\n[2] Baseline 재확인...")

    pair_counts_base = Counter()
    for c in cls:
        pair = tuple(sorted([c["disease_cui"], c["cui"]]))
        pair_counts_base[pair] += 1

    def build_and_eval(pair_counts, mc, use_propagation=True, label=""):
        kg = {p for p, cnt in pair_counts.items() if cnt >= mc}
        if use_propagation:
            expanded = set(kg)
            for (a, b) in list(kg):
                for pa in parent_map.get(a, set()):
                    if cui_stys.get(pa, set()) & ALLOWED_C and pa not in BLACKLIST:
                        expanded.add(tuple(sorted([pa, b])))
                for pb in parent_map.get(b, set()):
                    if cui_stys.get(pb, set()) & ALLOWED_C and pb not in BLACKLIST:
                        expanded.add(tuple(sorted([a, pb])))
        else:
            expanded = kg
        ev = evaluate(expanded, gold_pairs, parent_map)
        prop = "prop" if use_propagation else "noprop"
        print(f"  {label:<40} MC={mc} {prop:<6} edges={len(expanded):>6,} "
              f"P={ev['P']:.3f} R={ev['R']:.3f} F1={ev['F1']:.3f} match={ev['matched']}/{ev['gold']}")
        return ev

    print("\n  --- Baseline (V2C 그대로) ---")
    for mc in [1, 2, 3, 5]:
        build_and_eval(pair_counts_base, mc, True, "baseline")

    # ============================================================
    # 개선 1: 동의어 필터
    # ============================================================
    print("\n[3] 개선 1: 동의어 쌍 제거...")

    # 동의어인 쌍 식별
    syn_removed = 0
    cls_no_syn = []
    for c in cls:
        dc = c["disease_cui"]
        cui = c["cui"]
        # 질환과 동의어인 CUI 제거
        is_syn = cui in synonym_map.get(dc, set()) or dc in synonym_map.get(cui, set())
        # 또는 parent-child 관계 (같은 개념의 구체/추상)
        is_parent = cui in parent_map.get(dc, set()) or dc in parent_map.get(cui, set())
        if is_syn or is_parent:
            syn_removed += 1
            continue
        cls_no_syn.append(c)

    print(f"  동의어/부모자식 제거: {syn_removed}건 ({100*syn_removed/len(cls):.1f}%)")
    print(f"  남은 관계: {len(cls_no_syn):,}건")

    pair_counts_1 = Counter()
    for c in cls_no_syn:
        pair = tuple(sorted([c["disease_cui"], c["cui"]]))
        pair_counts_1[pair] += 1

    for mc in [1, 2, 3, 5]:
        build_and_eval(pair_counts_1, mc, True, "1_syn_filter")

    # ============================================================
    # 개선 2: manifestation-of 관계 제거
    # ============================================================
    print("\n[4] 개선 2: manifestation-of 관계 제거...")

    cls_no_manif = [c for c in cls if c["relation"] != "manifestation-of"]
    print(f"  manifestation-of 제거: {len(cls) - len(cls_no_manif)}건")
    print(f"  남은 관계: {len(cls_no_manif):,}건")

    pair_counts_2 = Counter()
    for c in cls_no_manif:
        pair = tuple(sorted([c["disease_cui"], c["cui"]]))
        pair_counts_2[pair] += 1

    for mc in [1, 2, 3, 5]:
        build_and_eval(pair_counts_2, mc, True, "2_no_manifestation")

    # ============================================================
    # 개선 3: 동의어 필터 + manifestation-of 제거
    # ============================================================
    print("\n[5] 개선 3: 1+2 조합 (동의어 + manifestation-of 제거)...")

    cls_3 = [c for c in cls_no_syn if c["relation"] != "manifestation-of"]
    print(f"  남은 관계: {len(cls_3):,}건")

    pair_counts_3 = Counter()
    for c in cls_3:
        pair = tuple(sorted([c["disease_cui"], c["cui"]]))
        pair_counts_3[pair] += 1

    for mc in [1, 2, 3, 5]:
        build_and_eval(pair_counts_3, mc, True, "3_syn+manif_filter")

    # ============================================================
    # 개선 4: CUI 정규화 (동의어 CUI 통합 후 집계)
    # ============================================================
    print("\n[6] 개선 4: CUI 정규화 (부모 CUI로 통합)...")

    # 각 CUI를 1-level parent가 있으면 parent로 정규화
    def normalize_cui(cui):
        pars = parent_map.get(cui, set())
        # disease CUI는 정규화하지 않음
        if cui in disease_cuis:
            return cui
        # parent가 1개이고 같은 STY면 parent로
        if len(pars) == 1:
            par = next(iter(pars))
            if cui_stys.get(par, set()) & cui_stys.get(cui, set()):
                return par
        return cui

    pair_counts_4 = Counter()
    for c in cls:
        dc = c["disease_cui"]
        cui = normalize_cui(c["cui"])
        pair = tuple(sorted([dc, cui]))
        pair_counts_4[pair] += 1

    print(f"  정규화 전 고유 쌍: {len(pair_counts_base):,}")
    print(f"  정규화 후 고유 쌍: {len(pair_counts_4):,}")

    for mc in [1, 2, 3, 5]:
        build_and_eval(pair_counts_4, mc, True, "4_cui_normalize")

    # ============================================================
    # 개선 5: 전파 없이 평가
    # ============================================================
    print("\n[7] 개선 5: CUI 전파 없이 (baseline vs 개선3)...")

    for mc in [1, 2, 3, 5]:
        build_and_eval(pair_counts_base, mc, False, "baseline_noprop")
    for mc in [1, 2, 3, 5]:
        build_and_eval(pair_counts_3, mc, False, "3_noprop")

    # ============================================================
    # 개선 6: 모든 개선 조합 (1+2+4)
    # ============================================================
    print("\n[8] 개선 6: 전체 조합 (동의어필터 + manif제거 + CUI정규화)...")

    pair_counts_6 = Counter()
    for c in cls_3:  # 동의어 + manifestation 필터 적용된 것
        dc = c["disease_cui"]
        cui = normalize_cui(c["cui"])
        pair = tuple(sorted([dc, cui]))
        pair_counts_6[pair] += 1

    for mc in [1, 2, 3, 5]:
        build_and_eval(pair_counts_6, mc, True, "6_all_improvements")
    for mc in [1, 2, 3, 5]:
        build_and_eval(pair_counts_6, mc, False, "6_all_noprop")

    # ============================================================
    # 요약
    # ============================================================
    print("\n" + "=" * 80)
    print("요약 (MC=5, CUI 전파 적용)")
    print("=" * 80)

    results = [
        ("baseline", pair_counts_base),
        ("1_syn_filter", pair_counts_1),
        ("2_no_manifestation", pair_counts_2),
        ("3_syn+manif", pair_counts_3),
        ("4_cui_normalize", pair_counts_4),
        ("6_all_combined", pair_counts_6),
    ]

    print(f"\n  {'방법':<30} {'edges':>7} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'-'*60}")

    for name, pc in results:
        kg = {p for p, cnt in pc.items() if cnt >= 5}
        expanded = set(kg)
        for (a, b) in list(kg):
            for pa in parent_map.get(a, set()):
                if cui_stys.get(pa, set()) & ALLOWED_C and pa not in BLACKLIST:
                    expanded.add(tuple(sorted([pa, b])))
            for pb in parent_map.get(b, set()):
                if cui_stys.get(pb, set()) & ALLOWED_C and pb not in BLACKLIST:
                    expanded.add(tuple(sorted([a, pb])))
        ev = evaluate(expanded, gold_pairs, parent_map)
        print(f"  {name:<30} {len(expanded):>7,} {ev['P']:>7.3f} {ev['R']:>7.3f} {ev['F1']:>7.3f}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
