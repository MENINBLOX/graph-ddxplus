#!/usr/bin/env python3
"""Cutoff 전략 최적화 스크립트."""

import ast
import json

import pandas as pd
from neo4j import GraphDatabase


def load_data():
    """데이터 로드."""
    df = pd.read_csv("data/ddxplus/release_test_patients.csv")
    with open("data/ddxplus/release_conditions.json") as f:
        conditions = json.load(f)
    pathology_to_cat = {
        cond_info["cond-name-fr"]: cond_info["severity"]
        for cond_name, cond_info in conditions.items()
    }
    df["CATEGORY"] = df["PATHOLOGY"].map(pathology_to_cat)

    with open("data/ddxplus/umls_mapping.json") as f:
        umls_map = json.load(f)["mapping"]
    with open("data/ddxplus/disease_umls_mapping.json") as f:
        disease_map = json.load(f)["mapping"]

    fr_to_eng = {info.get("name_fr", ""): eng for eng, info in disease_map.items()}
    return df, umls_map, disease_map, fr_to_eng


def get_case_results(row, umls_map, disease_map, fr_to_eng, driver):
    """단일 케이스의 KG 결과 가져오기."""
    # Get confirmed symptoms
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
    confirmed_cuis = list(set(confirmed_cuis))

    # Get GT info
    pathology_eng = fr_to_eng.get(row["PATHOLOGY"])
    if not pathology_eng:
        return None
    gt_cui = disease_map[pathology_eng]["umls_cui"]

    dd_raw = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
    gt_dd_cuis = []
    for dd_name, dd_prob in dd_raw:
        dd_eng = fr_to_eng.get(dd_name)
        if dd_eng:
            gt_dd_cuis.append(disease_map[dd_eng]["umls_cui"])

    # Get KG results
    with driver.session() as session:
        records = session.run(
            """
            MATCH (d:Disease)
            OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
            WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis
            WITH d, [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched
            WHERE size(matched) > 0
            WITH d, size(matched) AS confirmed_count
            WITH d, confirmed_count, toFloat(confirmed_count) AS raw_score
            WITH collect({cui: d.cui, score: raw_score, count: confirmed_count}) AS all_candidates
            WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.score) AS total_score
            UNWIND all_candidates AS c
            WITH c, CASE WHEN total_score > 0 THEN c.score / total_score ELSE 0.0 END AS prob
            RETURN c.cui AS cui, prob AS score, c.count AS count
            ORDER BY score DESC
            LIMIT 50
            """,
            confirmed_cuis=confirmed_cuis,
        )
        kg_results = [(r["cui"], r["score"], r["count"]) for r in records]

    return {
        "gt_cui": gt_cui,
        "gt_dd_cuis": gt_dd_cuis,
        "kg_results": kg_results,
    }


def evaluate_cutoff(cases, min_prob=0.0, top_k=None, cumulative_prob=None):
    """특정 cutoff 전략 평가."""
    total_ddr_num, total_ddr_den = 0, 0
    total_ddp_num, total_ddp_den = 0, 0
    correct_top1 = 0

    for case in cases:
        if case is None:
            continue

        kg_results = case["kg_results"]
        gt_cui = case["gt_cui"]
        gt_dd_cuis = set(case["gt_dd_cuis"])

        # Apply cutoff
        if kg_results:
            total_score = sum(s for _, s, _ in kg_results)
            if total_score > 0:
                filtered = []
                cumsum = 0.0
                for cui, score, count in kg_results:
                    prob = score / total_score if total_score > 0 else 0

                    # Apply cutoffs
                    if min_prob > 0 and prob < min_prob:
                        continue
                    if top_k and len(filtered) >= top_k:
                        break
                    if cumulative_prob:
                        if cumsum >= cumulative_prob:
                            break
                        cumsum += prob

                    filtered.append(cui)
            else:
                filtered = [cui for cui, _, _ in kg_results]
        else:
            filtered = []

        # Ensure at least 1 candidate
        if not filtered and kg_results:
            filtered = [kg_results[0][0]]

        # Calculate metrics
        kg_cuis = set(filtered)

        # Top-1 accuracy
        if filtered and filtered[0] == gt_cui:
            correct_top1 += 1

        # DDR: GT DD 중 KG에 포함된 비율
        intersection = len(gt_dd_cuis & kg_cuis)
        total_ddr_num += intersection
        total_ddr_den += len(gt_dd_cuis) if gt_dd_cuis else 1

        # DDP: KG 후보 중 GT DD에 포함된 비율
        total_ddp_num += intersection
        total_ddp_den += len(kg_cuis) if kg_cuis else 1

    n = len([c for c in cases if c])
    ddr = total_ddr_num / total_ddr_den if total_ddr_den > 0 else 0
    ddp = total_ddp_num / total_ddp_den if total_ddp_den > 0 else 0
    ddf1 = 2 * ddr * ddp / (ddr + ddp) if (ddr + ddp) > 0 else 0
    gtpa1 = correct_top1 / n if n > 0 else 0
    avg_candidates = total_ddp_den / n if n > 0 else 0

    return {
        "gtpa1": gtpa1,
        "ddr": ddr,
        "ddp": ddp,
        "ddf1": ddf1,
        "avg_candidates": avg_candidates,
    }


def main():
    df, umls_map, disease_map, fr_to_eng = load_data()
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))

    # Load cases
    cat2 = df[df["CATEGORY"] == 2].head(500)
    print(f"분석 케이스 수: {len(cat2)}")

    cases = []
    for _, row in cat2.iterrows():
        case = get_case_results(row, umls_map, disease_map, fr_to_eng, driver)
        cases.append(case)
    driver.close()

    print("\n" + "=" * 90)
    print("Cutoff 전략 비교")
    print("=" * 90)
    print(f"{'Strategy':30} {'GTPA@1':>8} {'DDR':>8} {'DDP':>8} {'DDF1':>8} {'Avg N':>8}")
    print("-" * 90)

    strategies = [
        ("Raw (no cutoff)", {"min_prob": 0}),
        ("min_prob=0.01", {"min_prob": 0.01}),
        ("min_prob=0.02", {"min_prob": 0.02}),
        ("min_prob=0.03", {"min_prob": 0.03}),
        ("min_prob=0.04", {"min_prob": 0.04}),
        ("min_prob=0.05", {"min_prob": 0.05}),
        ("top_k=5", {"top_k": 5}),
        ("top_k=8", {"top_k": 8}),
        ("top_k=10", {"top_k": 10}),
        ("top_k=15", {"top_k": 15}),
        ("top_k=20", {"top_k": 20}),
        ("cumulative=0.90", {"cumulative_prob": 0.90}),
        ("cumulative=0.95", {"cumulative_prob": 0.95}),
        ("cumulative=0.99", {"cumulative_prob": 0.99}),
        # Combined strategies
        ("min=0.02 + top=15", {"min_prob": 0.02, "top_k": 15}),
        ("min=0.01 + top=20", {"min_prob": 0.01, "top_k": 20}),
    ]

    best_ddf1 = 0
    best_strategy = None

    for name, params in strategies:
        result = evaluate_cutoff(cases, **params)
        print(
            f"{name:30} {result['gtpa1']:>7.1%} {result['ddr']:>7.1%} "
            f"{result['ddp']:>7.1%} {result['ddf1']:>7.1%} {result['avg_candidates']:>7.1f}"
        )
        if result["ddf1"] > best_ddf1:
            best_ddf1 = result["ddf1"]
            best_strategy = name

    print("-" * 90)
    print(f"Best DDF1: {best_strategy} ({best_ddf1:.1%})")


if __name__ == "__main__":
    main()
