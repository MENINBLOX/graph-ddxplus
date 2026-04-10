#!/usr/bin/env python3
"""Cypher 튜닝 스크립트.

DDXPlus 케이스별로 KG 쿼리를 분석하고 튜닝한다.
"""

import ast
import json
import sys

import pandas as pd
from neo4j import GraphDatabase


def load_data() -> tuple:
    """데이터 로드."""
    df = pd.read_csv("data/ddxplus/release_test_patients.csv")

    with open("data/ddxplus/release_conditions.json") as f:
        conditions = json.load(f)

    # category = severity
    pathology_to_cat = {
        cond_info["cond-name-fr"]: cond_info["severity"]
        for cond_name, cond_info in conditions.items()
    }

    df["CATEGORY"] = df["PATHOLOGY"].map(pathology_to_cat)

    with open("data/ddxplus/umls_mapping.json") as f:
        umls_map = json.load(f)["mapping"]

    with open("data/ddxplus/disease_umls_mapping.json") as f:
        disease_data = json.load(f)
        disease_map = disease_data["mapping"]

    # French name -> English name
    fr_to_eng = {}
    for eng, info in disease_map.items():
        fr_to_eng[info.get("name_fr", "")] = eng

    return df, umls_map, disease_map, fr_to_eng, conditions


def analyze_case(
    row: pd.Series,
    umls_map: dict,
    disease_map: dict,
    fr_to_eng: dict,
    driver,
) -> dict:
    """단일 케이스 분석."""
    result = {
        "age": row["AGE"],
        "sex": row["SEX"],
        "pathology_fr": row["PATHOLOGY"],
    }

    # Ground truth disease
    pathology_eng = fr_to_eng.get(row["PATHOLOGY"])
    if pathology_eng:
        result["pathology_eng"] = pathology_eng
        result["gt_cui"] = disease_map[pathology_eng]["umls_cui"]
    else:
        result["pathology_eng"] = None
        result["gt_cui"] = None

    # Initial evidence
    initial = row["INITIAL_EVIDENCE"]
    initial_info = umls_map.get(initial, {})
    result["initial"] = initial
    result["initial_cui"] = initial_info.get("cui")
    result["initial_name"] = initial_info.get("name")

    # All evidences
    evidences_raw = ast.literal_eval(row["EVIDENCES"])
    confirmed_cuis = []
    for ev in evidences_raw:
        if ev.startswith("_"):
            sym_key = ev.split("@_")[0]
        else:
            sym_key = ev
        sym_info = umls_map.get(sym_key, {})
        if sym_info.get("cui"):
            confirmed_cuis.append(sym_info["cui"])

    result["confirmed_cuis"] = list(set(confirmed_cuis))
    result["n_symptoms"] = len(evidences_raw)
    result["n_mapped"] = len(result["confirmed_cuis"])

    # Differential diagnosis
    dd_raw = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
    gt_dd_cuis = []
    for dd_name, dd_prob in dd_raw:
        dd_eng = fr_to_eng.get(dd_name)
        if dd_eng:
            gt_dd_cuis.append(disease_map[dd_eng]["umls_cui"])
    result["gt_dd_cuis"] = gt_dd_cuis
    result["n_dd"] = len(dd_raw)
    result["n_dd_mapped"] = len(gt_dd_cuis)

    # KG 쿼리 결과
    with driver.session() as session:
        # Check initial symptom in KG
        r = session.run(
            """
            MATCH (s:Symptom {cui: $cui})
            RETURN s.name as name, s.is_antecedent as is_antecedent
            """,
            cui=result["initial_cui"],
        ).single()
        result["initial_in_kg"] = r is not None

        # Check GT disease in KG
        r = session.run(
            """
            MATCH (d:Disease {cui: $cui})
            RETURN d.name as name
            """,
            cui=result["gt_cui"],
        ).single()
        result["gt_in_kg"] = r is not None

        # Check how many symptoms are in KG
        records = session.run(
            """
            UNWIND $cuis AS cui
            OPTIONAL MATCH (s:Symptom {cui: cui})
            RETURN cui, s.name AS name
            """,
            cuis=result["confirmed_cuis"],
        )
        result["symptoms_in_kg"] = sum(1 for r in records if r["name"])

        # Current KG diagnosis query (v7 additive) - 범용 버전
        records = session.run(
            """
            MATCH (d:Disease)
            OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
            WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis,
                 count(DISTINCT s) AS total_symptoms

            WITH d, disease_symptom_cuis, total_symptoms,
                 [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched

            WHERE size(matched) > 0

            WITH d, total_symptoms,
                 size(matched) AS confirmed_count

            WITH d, confirmed_count, total_symptoms,
                 toFloat(confirmed_count) AS raw_score

            WITH collect({
                cui: d.cui,
                name: d.name,
                raw_score: raw_score,
                confirmed_count: confirmed_count,
                total_symptoms: total_symptoms
            }) AS all_candidates

            WITH all_candidates,
                 reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score

            UNWIND all_candidates AS c
            WITH c, total_score,
                 CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS prob

            RETURN c.cui AS cui,
                   c.name AS name,
                   prob AS score,
                   c.confirmed_count AS confirmed_count,
                   c.total_symptoms AS total_symptoms
            ORDER BY score DESC
            LIMIT 30
            """,
            confirmed_cuis=result["confirmed_cuis"],
        )

        kg_results = [
            {
                "cui": r["cui"],
                "name": r["name"],
                "score": r["score"],
                "confirmed_count": r["confirmed_count"],
            }
            for r in records
        ]
        result["kg_results"] = kg_results

        # Calculate metrics (raw - no cutoff)
        kg_cuis_raw = [r["cui"] for r in kg_results]
        result["gt_rank_raw"] = (
            kg_cuis_raw.index(result["gt_cui"]) + 1
            if result["gt_cui"] in kg_cuis_raw
            else -1
        )

        # Apply min_prob cutoff
        total_score = sum(r["score"] for r in kg_results)
        if total_score > 0:
            min_prob = 0.03
            kg_filtered = [r for r in kg_results if r["score"] / total_score >= min_prob]
        else:
            kg_filtered = kg_results

        result["kg_filtered"] = kg_filtered
        result["n_candidates_raw"] = len(kg_results)
        result["n_candidates_filtered"] = len(kg_filtered)

        kg_cuis = [r["cui"] for r in kg_filtered]
        result["gt_rank"] = (
            kg_cuis.index(result["gt_cui"]) + 1
            if result["gt_cui"] in kg_cuis
            else -1
        )

        # DDR: How many GT DD are in KG results (filtered)
        dd_found = sum(1 for cui in result["gt_dd_cuis"] if cui in kg_cuis)
        result["ddr"] = dd_found / len(result["gt_dd_cuis"]) if result["gt_dd_cuis"] else 0

        # DDP: How many KG results are in GT DD
        kg_correct = sum(1 for cui in kg_cuis if cui in result["gt_dd_cuis"])
        result["ddp"] = kg_correct / len(kg_cuis) if kg_cuis else 0

        # DDF1
        if result["ddr"] + result["ddp"] > 0:
            result["ddf1"] = 2 * result["ddr"] * result["ddp"] / (result["ddr"] + result["ddp"])
        else:
            result["ddf1"] = 0

        # Also compute raw metrics (no cutoff)
        dd_found_raw = sum(1 for cui in result["gt_dd_cuis"] if cui in kg_cuis_raw)
        result["ddr_raw"] = dd_found_raw / len(result["gt_dd_cuis"]) if result["gt_dd_cuis"] else 0
        kg_correct_raw = sum(1 for cui in kg_cuis_raw if cui in result["gt_dd_cuis"])
        result["ddp_raw"] = kg_correct_raw / len(kg_cuis_raw) if kg_cuis_raw else 0
        if result["ddr_raw"] + result["ddp_raw"] > 0:
            result["ddf1_raw"] = 2 * result["ddr_raw"] * result["ddp_raw"] / (result["ddr_raw"] + result["ddp_raw"])
        else:
            result["ddf1_raw"] = 0

    return result


def print_case_analysis(case: dict, verbose: bool = True) -> None:
    """케이스 분석 출력."""
    print("=" * 80)
    print(f"환자: Age {case['age']}, {case['sex']}")
    print(f"정답 질환: {case['pathology_fr']} ({case['pathology_eng']})")
    print(f"정답 CUI: {case['gt_cui']} (KG 존재: {case['gt_in_kg']})")
    print()
    print(f"주호소: {case['initial']} → {case['initial_name']} ({case['initial_cui']})")
    print(f"  KG 존재: {case['initial_in_kg']}")
    print()
    print(f"증상 수: {case['n_symptoms']} (매핑: {case['n_mapped']}, KG 존재: {case['symptoms_in_kg']})")
    print(f"정답 DD 수: {case['n_dd']} (매핑: {case['n_dd_mapped']})")
    print()
    print("KG 진단 결과:")
    for i, r in enumerate(case["kg_results"][:10], 1):
        is_gt = "✓" if r["cui"] == case["gt_cui"] else ""
        is_dd = "★" if r["cui"] in case["gt_dd_cuis"] else ""
        print(f"  {i:2}. {r['name'][:40]:40} {r['score']:.1%} ({r['confirmed_count']} syms) {is_gt}{is_dd}")
    print()
    print(f"정답 순위: {case['gt_rank']}")
    print(f"DDR: {case['ddr']:.1%}, DDP: {case['ddp']:.1%}, DDF1: {case['ddf1']:.1%}")


def run_batch_analysis(n_cases: int = 100, category: int = 2) -> None:
    """배치 분석."""
    df, umls_map, disease_map, fr_to_eng, conditions = load_data()
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))

    # Filter by category
    if category:
        df_filtered = df[df["CATEGORY"] == category].head(n_cases)
    else:
        df_filtered = df.head(n_cases)

    print(f"분석 대상: {len(df_filtered)} cases (category={category})")
    print()

    results = []
    for idx, (_, row) in enumerate(df_filtered.iterrows()):
        case = analyze_case(row, umls_map, disease_map, fr_to_eng, driver)
        results.append(case)

        if idx < 3:  # 처음 3개만 상세 출력
            print_case_analysis(case)

    driver.close()

    # 통계 요약
    print("\n" + "=" * 80)
    print("통계 요약")
    print("=" * 80)

    gt_in_kg = sum(1 for r in results if r["gt_in_kg"])
    gt_top1_raw = sum(1 for r in results if r["gt_rank_raw"] == 1)
    gt_top1 = sum(1 for r in results if r["gt_rank"] == 1)
    gt_top5 = sum(1 for r in results if 1 <= r["gt_rank"] <= 5)
    gt_missing = sum(1 for r in results if r["gt_rank"] == -1)

    avg_ddr_raw = sum(r["ddr_raw"] for r in results) / len(results)
    avg_ddp_raw = sum(r["ddp_raw"] for r in results) / len(results)
    avg_ddf1_raw = sum(r["ddf1_raw"] for r in results) / len(results)

    avg_ddr = sum(r["ddr"] for r in results) / len(results)
    avg_ddp = sum(r["ddp"] for r in results) / len(results)
    avg_ddf1 = sum(r["ddf1"] for r in results) / len(results)

    avg_n_raw = sum(r["n_candidates_raw"] for r in results) / len(results)
    avg_n_filtered = sum(r["n_candidates_filtered"] for r in results) / len(results)

    print(f"정답 질환 KG 존재: {gt_in_kg}/{len(results)} ({100*gt_in_kg/len(results):.1f}%)")
    print(f"정답 Top-1 (raw): {gt_top1_raw}/{len(results)} ({100*gt_top1_raw/len(results):.1f}%)")
    print(f"정답 Top-1 (filtered): {gt_top1}/{len(results)} ({100*gt_top1/len(results):.1f}%)")
    print(f"정답 Top-5 (filtered): {gt_top5}/{len(results)} ({100*gt_top5/len(results):.1f}%)")
    print(f"정답 누락 (filtered): {gt_missing}/{len(results)} ({100*gt_missing/len(results):.1f}%)")
    print()
    print(f"평균 후보 수: raw={avg_n_raw:.1f}, filtered={avg_n_filtered:.1f}")
    print()
    print("Raw (no cutoff):")
    print(f"  DDR: {avg_ddr_raw:.1%}, DDP: {avg_ddp_raw:.1%}, DDF1: {avg_ddf1_raw:.1%}")
    print()
    print("Filtered (min_prob=0.03):")
    print(f"  DDR: {avg_ddr:.1%}, DDP: {avg_ddp:.1%}, DDF1: {avg_ddf1:.1%}")

    # 문제 케이스 분석
    print("\n" + "=" * 80)
    print("문제 케이스 분석")
    print("=" * 80)

    # GT missing cases
    missing_cases = [r for r in results if r["gt_rank"] == -1]
    if missing_cases:
        print(f"\n정답이 KG 결과에 없는 케이스 ({len(missing_cases)}개):")
        for case in missing_cases[:5]:
            print(f"  - {case['pathology_eng']}: GT_in_KG={case['gt_in_kg']}, symptoms_in_kg={case['symptoms_in_kg']}/{case['n_mapped']}")

    # Low DDF1 cases
    low_ddf1 = sorted(results, key=lambda r: r["ddf1"])[:5]
    print(f"\nDDF1이 낮은 케이스 (하위 5개):")
    for case in low_ddf1:
        print(f"  - {case['pathology_eng']}: DDF1={case['ddf1']:.1%}, DDR={case['ddr']:.1%}, DDP={case['ddp']:.1%}")


def interactive_tuning(case_idx: int = 0, category: int = 2) -> None:
    """대화형 튜닝."""
    df, umls_map, disease_map, fr_to_eng, conditions = load_data()
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))

    # Filter by category
    df_filtered = df[df["CATEGORY"] == category]
    row = df_filtered.iloc[case_idx]

    case = analyze_case(row, umls_map, disease_map, fr_to_eng, driver)
    print_case_analysis(case)

    # 추가 분석
    print("\n추가 분석:")
    print(f"Confirmed CUIs: {case['confirmed_cuis']}")
    print(f"GT DD CUIs: {case['gt_dd_cuis']}")

    driver.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "batch":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            run_batch_analysis(n_cases=n)
        elif sys.argv[1] == "case":
            idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            interactive_tuning(case_idx=idx)
    else:
        run_batch_analysis(n_cases=10)
