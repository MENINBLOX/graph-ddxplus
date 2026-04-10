#!/usr/bin/env python3
"""H-DDx 지표 계산 (Hierarchical DDx Recall/Precision/F1).

ICD-10 계층 구조를 활용하여 부분 점수 부여.
"""

import ast
import json
from collections import defaultdict

import pandas as pd
from neo4j import GraphDatabase


def get_icd10_ancestors(icd10_code: str) -> set[str]:
    """ICD-10 코드의 조상 노드 반환 (H-DDx 논문 방식).

    H-DDx 논문: "augments the set S by adding all ancestral nodes
    for each diagnosis in S, from its immediate parent up to the
    chapter level in the ICD-10 taxonomy."

    ICD-10 계층 구조:
    - Chapter: 1자리 (예: J = Diseases of the respiratory system)
    - Category: 3자리 (예: J93 = Pneumothorax)
    - Subcategory: 3자리 + 소수점 + 추가 (예: J93.1, J93.11)

    예: J93.1 -> {J93.1, J93, J}

    참고: https://arxiv.org/abs/2510.03700
    """
    if not icd10_code:
        return set()

    code = icd10_code.upper().strip()
    ancestors = {code}

    # 1. Subcategory → Category (소수점 이전 부분)
    if "." in code:
        category = code.split(".")[0]
        ancestors.add(category)
    else:
        category = code

    # 2. Category → Chapter (첫 글자)
    if len(category) >= 1:
        chapter = category[0]
        ancestors.add(chapter)

    return ancestors


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
        disease_data = json.load(f)
        disease_map = disease_data["mapping"]

    # French name -> info
    fr_to_info = {}
    for eng, info in disease_map.items():
        fr_name = info.get("name_fr", "")
        if fr_name:
            fr_to_info[fr_name] = info

    return df, umls_map, disease_map, fr_to_info


def calculate_hddx_metrics(gt_icd10_codes: list[str], pred_icd10_codes: list[str]) -> dict:
    """단일 케이스의 H-DDx 지표 계산.

    Args:
        gt_icd10_codes: Ground truth ICD-10 코드 리스트
        pred_icd10_codes: 예측 ICD-10 코드 리스트

    Returns:
        {"hdr": float, "hdp": float, "hdf1": float}
    """
    # ICD-10 조상까지 확장
    gt_expanded = set()
    for code in gt_icd10_codes:
        gt_expanded.update(get_icd10_ancestors(code))

    pred_expanded = set()
    for code in pred_icd10_codes:
        pred_expanded.update(get_icd10_ancestors(code))

    # 교집합
    intersection = len(gt_expanded & pred_expanded)

    # HDR = |C ∩ Ĉ| / |C|
    hdr = intersection / len(gt_expanded) if gt_expanded else 0

    # HDP = |C ∩ Ĉ| / |Ĉ|
    hdp = intersection / len(pred_expanded) if pred_expanded else 0

    # HDF1
    hdf1 = 2 * hdr * hdp / (hdr + hdp) if (hdr + hdp) > 0 else 0

    return {"hdr": hdr, "hdp": hdp, "hdf1": hdf1}


def main():
    df, umls_map, disease_map, fr_to_info = load_data()
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))

    # Category 2 케이스
    cat2 = df[df["CATEGORY"] == 2].head(1000)
    print(f"분석 케이스: {len(cat2)}")

    # 결과 저장
    results = []

    for idx, (_, row) in enumerate(cat2.iterrows()):
        # Ground truth DD의 ICD-10 코드
        dd_raw = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
        gt_icd10 = []
        for dd_name, _ in dd_raw:
            info = fr_to_info.get(dd_name)
            if info and info.get("icd10_original"):
                gt_icd10.append(info["icd10_original"])

        # KG 예측 (confirmed symptoms 기반)
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

        # KG 쿼리
        with driver.session() as session:
            records = session.run(
                """
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
                WHERE prob >= 0.02
                RETURN cui, prob AS score
                ORDER BY score DESC
                LIMIT 30
                """,
                confirmed_cuis=confirmed_cuis,
            )
            kg_cuis = [r["cui"] for r in records]

        # KG CUI -> ICD-10 변환
        pred_icd10 = []
        for cui in kg_cuis:
            for eng, info in disease_map.items():
                if info.get("umls_cui") == cui:
                    if info.get("icd10_original"):
                        pred_icd10.append(info["icd10_original"])
                    break

        # Ground truth pathology
        gt_pathology = row["PATHOLOGY"]
        gt_info = fr_to_info.get(gt_pathology)
        gt_cui = gt_info.get("umls_cui") if gt_info else None

        # H-DDx 지표 계산
        metrics = calculate_hddx_metrics(gt_icd10, pred_icd10)

        # Flat metrics (DDR, DDP, DDF1)
        gt_dd_cuis = set()
        for dd_name, _ in dd_raw:
            info = fr_to_info.get(dd_name)
            if info and info.get("umls_cui"):
                gt_dd_cuis.add(info["umls_cui"])

        pred_cuis = set(kg_cuis)
        intersection = len(gt_dd_cuis & pred_cuis)
        metrics["ddr"] = intersection / len(gt_dd_cuis) if gt_dd_cuis else 0
        metrics["ddp"] = intersection / len(pred_cuis) if pred_cuis else 0
        if metrics["ddr"] + metrics["ddp"] > 0:
            metrics["ddf1"] = 2 * metrics["ddr"] * metrics["ddp"] / (metrics["ddr"] + metrics["ddp"])
        else:
            metrics["ddf1"] = 0

        # Top-k accuracy
        metrics["top1"] = 1 if kg_cuis and kg_cuis[0] == gt_cui else 0
        metrics["top3"] = 1 if gt_cui in kg_cuis[:3] else 0
        metrics["top5"] = 1 if gt_cui in kg_cuis[:5] else 0
        metrics["top10"] = 1 if gt_cui in kg_cuis[:10] else 0

        results.append(metrics)

        if (idx + 1) % 100 == 0:
            print(f"  진행: {idx + 1}/{len(cat2)}")

    driver.close()

    # 평균 계산
    n = len(results)
    avg_hdr = sum(r["hdr"] for r in results) / n
    avg_hdp = sum(r["hdp"] for r in results) / n
    avg_hdf1 = sum(r["hdf1"] for r in results) / n

    avg_ddr = sum(r["ddr"] for r in results) / n
    avg_ddp = sum(r["ddp"] for r in results) / n
    avg_ddf1 = sum(r["ddf1"] for r in results) / n

    top1_acc = sum(r["top1"] for r in results) / n
    top3_acc = sum(r["top3"] for r in results) / n
    top5_acc = sum(r["top5"] for r in results) / n
    top10_acc = sum(r["top10"] for r in results) / n

    print("\n" + "=" * 70)
    print(f"전체 결과 (n={n})")
    print("=" * 70)

    print("\n[Hierarchical Metrics - H-DDx 스타일]")
    print(f"  HDR  (Hierarchical DDx Recall):    {avg_hdr:.1%}")
    print(f"  HDP  (Hierarchical DDx Precision): {avg_hdp:.1%}")
    print(f"  HDF1 (Hierarchical DDx F1):        {avg_hdf1:.1%}")

    print("\n[Flat Metrics - 기존 스타일]")
    print(f"  DDR  (DDx Recall):    {avg_ddr:.1%}")
    print(f"  DDP  (DDx Precision): {avg_ddp:.1%}")
    print(f"  DDF1 (DDx F1):        {avg_ddf1:.1%}")

    print("\n[Top-k Accuracy]")
    print(f"  Top-1:  {top1_acc:.1%}")
    print(f"  Top-3:  {top3_acc:.1%}")
    print(f"  Top-5:  {top5_acc:.1%}")
    print(f"  Top-10: {top10_acc:.1%}")

    print("\n" + "=" * 70)
    print("H-DDx 논문 DDXPlus 결과 비교 (참고용)")
    print("=" * 70)
    print(f"{'Model':<25} {'Top-5':>8} {'HDF1':>8} {'DDR':>8} {'DDF1':>8}")
    print("-" * 70)
    print("H-DDx 논문 기준선 (상용 LLM - 본 연구 미사용):")
    print(f"  {'Claude-Sonnet-4':<23} {'83.9%':>8} {'36.7%':>8} {'-':>8} {'-':>8}")
    print(f"  {'GPT-4o':<23} {'80.4%':>8} {'35.0%':>8} {'-':>8} {'-':>8}")
    print(f"  {'MediPhi (3.8B)':<23} {'66.6%':>8} {'35.3%':>8} {'-':>8} {'-':>8}")
    print("-" * 70)
    print(f"{'Ours (Small LLM + KG)':<25} {top5_acc:>7.1%} {avg_hdf1:>7.1%} {avg_ddr:>7.1%} {avg_ddf1:>7.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
