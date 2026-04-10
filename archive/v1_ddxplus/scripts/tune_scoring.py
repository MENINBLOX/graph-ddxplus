#!/usr/bin/env python3
"""Cypher 스코어링 공식 튜닝 스크립트."""

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


SCORING_QUERIES = {
    "v7_additive": """
        // v7: confirmed_count (additive)
        MATCH (d:Disease)
        OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
        WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis
        WITH d, [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched
        WHERE size(matched) > 0
        WITH d, size(matched) AS confirmed_count
        WITH d, toFloat(confirmed_count) AS raw_score
        WITH collect({cui: d.cui, score: raw_score}) AS all_candidates
        WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.score) AS total_score
        UNWIND all_candidates AS c
        WITH c.cui AS cui, CASE WHEN total_score > 0 THEN c.score / total_score ELSE 0.0 END AS prob
        RETURN cui, prob AS score
        ORDER BY score DESC
        LIMIT 50
    """,
    "v8_coverage_ratio": """
        // v8: confirmed_count / total_symptoms (coverage ratio)
        MATCH (d:Disease)
        OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
        WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis, count(DISTINCT s) AS total_syms
        WITH d, [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched, total_syms
        WHERE size(matched) > 0 AND total_syms > 0
        WITH d, toFloat(size(matched)) / toFloat(total_syms) AS raw_score
        WITH collect({cui: d.cui, score: raw_score}) AS all_candidates
        WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.score) AS total_score
        UNWIND all_candidates AS c
        WITH c.cui AS cui, CASE WHEN total_score > 0 THEN c.score / total_score ELSE 0.0 END AS prob
        RETURN cui, prob AS score
        ORDER BY score DESC
        LIMIT 50
    """,
    "v9_combined": """
        // v9: count * (count / total_symptoms) = count² / total_symptoms
        MATCH (d:Disease)
        OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
        WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis, count(DISTINCT s) AS total_syms
        WITH d, [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched, total_syms
        WHERE size(matched) > 0 AND total_syms > 0
        WITH d, toFloat(size(matched) * size(matched)) / toFloat(total_syms) AS raw_score
        WITH collect({cui: d.cui, score: raw_score}) AS all_candidates
        WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.score) AS total_score
        UNWIND all_candidates AS c
        WITH c.cui AS cui, CASE WHEN total_score > 0 THEN c.score / total_score ELSE 0.0 END AS prob
        RETURN cui, prob AS score
        ORDER BY score DESC
        LIMIT 50
    """,
    "v10_sqrt": """
        // v10: sqrt(count) * (count / total_symptoms) - 더 균형잡힌
        MATCH (d:Disease)
        OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
        WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis, count(DISTINCT s) AS total_syms
        WITH d, [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched, total_syms
        WHERE size(matched) > 0 AND total_syms > 0
        WITH d, sqrt(toFloat(size(matched))) * (toFloat(size(matched)) / toFloat(total_syms)) AS raw_score
        WITH collect({cui: d.cui, score: raw_score}) AS all_candidates
        WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.score) AS total_score
        UNWIND all_candidates AS c
        WITH c.cui AS cui, CASE WHEN total_score > 0 THEN c.score / total_score ELSE 0.0 END AS prob
        RETURN cui, prob AS score
        ORDER BY score DESC
        LIMIT 50
    """,
    "v11_log": """
        // v11: log(count+1) * (count / total_symptoms)
        MATCH (d:Disease)
        OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
        WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis, count(DISTINCT s) AS total_syms
        WITH d, [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched, total_syms
        WHERE size(matched) > 0 AND total_syms > 0
        WITH d, log(toFloat(size(matched)) + 1.0) * (toFloat(size(matched)) / toFloat(total_syms)) AS raw_score
        WITH collect({cui: d.cui, score: raw_score}) AS all_candidates
        WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.score) AS total_score
        UNWIND all_candidates AS c
        WITH c.cui AS cui, CASE WHEN total_score > 0 THEN c.score / total_score ELSE 0.0 END AS prob
        RETURN cui, prob AS score
        ORDER BY score DESC
        LIMIT 50
    """,
    "v12_idf_weighted": """
        // v12: sum(1/disease_count_per_symptom) - IDF 가중치
        MATCH (d:Disease)
        OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
        WITH d, collect(DISTINCT s.cui) AS disease_symptom_cuis
        WITH d, [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched
        WHERE size(matched) > 0
        // 각 매칭된 증상의 질환 연결 수 계산
        UNWIND matched AS sym_cui
        OPTIONAL MATCH (s2:Symptom {cui: sym_cui})-[:INDICATES]->(d2:Disease)
        WITH d, sym_cui, count(DISTINCT d2) AS disease_count
        WITH d, sum(1.0 / toFloat(disease_count + 1)) AS idf_score
        WITH collect({cui: d.cui, score: idf_score}) AS all_candidates
        WITH all_candidates, reduce(total = 0.0, c IN all_candidates | total + c.score) AS total_score
        UNWIND all_candidates AS c
        WITH c.cui AS cui, CASE WHEN total_score > 0 THEN c.score / total_score ELSE 0.0 END AS prob
        RETURN cui, prob AS score
        ORDER BY score DESC
        LIMIT 50
    """,
}


def get_case_kg_results(row, umls_map, disease_map, fr_to_eng, driver, query):
    """단일 케이스의 KG 결과."""
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
        try:
            records = session.run(query, confirmed_cuis=confirmed_cuis)
            kg_results = [(r["cui"], r["score"]) for r in records]
        except Exception as e:
            print(f"Query error: {e}")
            return None

    return {
        "gt_cui": gt_cui,
        "gt_dd_cuis": gt_dd_cuis,
        "kg_results": kg_results,
    }


def evaluate_scoring(cases, min_prob=0.02):
    """스코어링 평가."""
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
            total_score = sum(s for _, s in kg_results)
            if total_score > 0:
                filtered = [cui for cui, score in kg_results if score / total_score >= min_prob]
            else:
                filtered = [cui for cui, _ in kg_results]
        else:
            filtered = []

        if not filtered and kg_results:
            filtered = [kg_results[0][0]]

        kg_cuis = set(filtered)

        # Top-1 accuracy
        if filtered and filtered[0] == gt_cui:
            correct_top1 += 1

        # DDR
        intersection = len(gt_dd_cuis & kg_cuis)
        total_ddr_num += intersection
        total_ddr_den += len(gt_dd_cuis) if gt_dd_cuis else 1

        # DDP
        total_ddp_num += intersection
        total_ddp_den += len(kg_cuis) if kg_cuis else 1

    n = len([c for c in cases if c])
    ddr = total_ddr_num / total_ddr_den if total_ddr_den > 0 else 0
    ddp = total_ddp_num / total_ddp_den if total_ddp_den > 0 else 0
    ddf1 = 2 * ddr * ddp / (ddr + ddp) if (ddr + ddp) > 0 else 0
    gtpa1 = correct_top1 / n if n > 0 else 0
    avg_candidates = total_ddp_den / n if n > 0 else 0

    return {"gtpa1": gtpa1, "ddr": ddr, "ddp": ddp, "ddf1": ddf1, "avg_n": avg_candidates}


def main():
    df, umls_map, disease_map, fr_to_eng = load_data()
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))

    cat2 = df[df["CATEGORY"] == 2].head(300)
    print(f"분석 케이스 수: {len(cat2)}")

    print("\n" + "=" * 90)
    print("스코어링 공식 비교 (min_prob=0.02 cutoff 적용)")
    print("=" * 90)
    print(f"{'Scoring':20} {'GTPA@1':>8} {'DDR':>8} {'DDP':>8} {'DDF1':>8} {'Avg N':>8}")
    print("-" * 90)

    best_ddf1 = 0
    best_scoring = None

    for name, query in SCORING_QUERIES.items():
        cases = []
        for _, row in cat2.iterrows():
            case = get_case_kg_results(row, umls_map, disease_map, fr_to_eng, driver, query)
            cases.append(case)

        result = evaluate_scoring(cases)
        print(
            f"{name:20} {result['gtpa1']:>7.1%} {result['ddr']:>7.1%} "
            f"{result['ddp']:>7.1%} {result['ddf1']:>7.1%} {result['avg_n']:>7.1f}"
        )

        if result["ddf1"] > best_ddf1:
            best_ddf1 = result["ddf1"]
            best_scoring = name

    driver.close()
    print("-" * 90)
    print(f"Best DDF1: {best_scoring} ({best_ddf1:.1%})")


if __name__ == "__main__":
    main()
