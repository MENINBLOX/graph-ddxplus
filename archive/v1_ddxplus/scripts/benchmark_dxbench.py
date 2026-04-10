#!/usr/bin/env python3
"""DxBench Complete Profile 평가.

DxBench (2,788 cases, 461 diseases) + Dxy (104 cases) + Muzhi (142 cases)에서
GraphTrace의 scoring function으로 최종 진단 정확도를 측정.

증상이 구조화되어 있으므로 (symptom_name, True/False) →
confirmed/denied CUI로 직접 변환 가능.

Usage:
    uv run python scripts/benchmark_dxbench.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


UMLS_DIR = Path("data/umls_extracted")
OUTPUT_DIR = Path("results")


def load_dxbench_data() -> List[dict]:
    """DxBench 3개 config 모두 로드."""
    from datasets import load_dataset

    all_cases = []
    for config in ["DxBench", "Dxy", "Muzhi"]:
        ds = load_dataset("FreedomIntelligence/DxBench", config, split="en")
        for entry in ds:
            explicit = entry.get("explicit_symptoms", [])
            implicit = entry.get("implicit_symptoms", [])
            all_cases.append({
                "id": entry["id"],
                "source": config,
                "disease": entry["disease"],
                "candidate_diseases": entry.get("candidate_diseases", []),
                "department": entry.get("department", ""),
                "explicit_symptoms": explicit,
                "implicit_symptoms": implicit,
            })
        print(f"  {config}: {len(ds)} cases")

    print(f"  Total: {len(all_cases)} cases")
    return all_cases


def build_umls_name_index() -> Dict[str, Tuple[str, str]]:
    """MRCONSO에서 영어 이름 → (CUI, preferred_name) 인덱스."""
    print("  MRCONSO 인덱싱...")
    name_to_cui = {}
    preferred = {}
    with open(UMLS_DIR / "MRCONSO.RRF", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            cui, lang, ts, name = parts[0], parts[1], parts[2], parts[14]
            if lang != "ENG":
                continue
            lower = name.lower().strip()
            if lower not in name_to_cui:
                name_to_cui[lower] = (cui, name)
            if ts == "P" and cui not in preferred:
                preferred[cui] = name
    print(f"    {len(name_to_cui)} names, {len(preferred)} CUIs")
    return name_to_cui, preferred


def map_symptoms_to_cuis(
    cases: List[dict],
    name_to_cui: Dict[str, Tuple[str, str]],
) -> Tuple[List[dict], dict]:
    """증상/질병 이름 → UMLS CUI 매핑."""
    mapped_cases = []
    stats = {"total_symptoms": 0, "mapped_symptoms": 0, "total_diseases": 0, "mapped_diseases": 0}

    for case in cases:
        confirmed_cuis = set()
        denied_cuis = set()
        unmapped_symptoms = []

        # explicit + implicit 증상 처리
        all_symptoms = case["explicit_symptoms"] + case["implicit_symptoms"]
        for sym_pair in all_symptoms:
            if len(sym_pair) < 2:
                continue
            sym_name, value = sym_pair[0], sym_pair[1]
            stats["total_symptoms"] += 1

            lower = sym_name.lower().strip()
            result = name_to_cui.get(lower)
            if result:
                cui = result[0]
                stats["mapped_symptoms"] += 1
                if value == "True":
                    confirmed_cuis.add(cui)
                else:
                    denied_cuis.add(cui)
            else:
                unmapped_symptoms.append(sym_name)

        # 질병 매핑
        stats["total_diseases"] += 1
        disease_lower = case["disease"].lower().strip()
        disease_cui = None
        disease_result = name_to_cui.get(disease_lower)
        if disease_result:
            disease_cui = disease_result[0]
            stats["mapped_diseases"] += 1

        # 후보 질병 CUI 매핑
        candidate_cuis = {}
        for cd in case.get("candidate_diseases", []):
            cd_result = name_to_cui.get(cd.lower().strip())
            if cd_result:
                candidate_cuis[cd] = cd_result[0]

        mapped_cases.append({
            **case,
            "confirmed_cuis": confirmed_cuis,
            "denied_cuis": denied_cuis,
            "disease_cui": disease_cui,
            "candidate_cuis": candidate_cuis,
            "unmapped_symptoms": unmapped_symptoms,
        })

    return mapped_cases, stats


def build_symptom_disease_index(name_to_cui: Dict) -> Dict[str, Set[str]]:
    """MRREL에서 symptom CUI → disease CUI 관계 추출."""
    print("  MRREL에서 증상-질병 관계 추출...")

    # Semantic types
    disease_stys = {"T047", "T019", "T191", "T046"}
    symptom_stys = {"T184", "T033", "T034", "T048", "T037"}

    cui_stys = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            cui_stys[parts[0]].add(parts[1])

    is_disease = lambda cui: bool(cui_stys.get(cui, set()) & disease_stys)
    is_symptom = lambda cui: bool(cui_stys.get(cui, set()) & symptom_stys)

    # MRREL에서 관계 추출
    symptom_to_diseases = defaultdict(set)  # symptom_cui → {disease_cui}
    disease_to_symptoms = defaultdict(set)  # disease_cui → {symptom_cui}

    with open(UMLS_DIR / "MRREL.RRF", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            cui1, cui2 = parts[0], parts[4]
            rela = parts[7].lower() if len(parts) > 7 else ""

            relevant_relas = {
                "has_finding", "finding_site_of", "associated_with",
                "manifestation_of", "has_manifestation", "clinically_associated_with",
                "may_be_finding_of", "disease_has_finding",
            }
            if rela not in relevant_relas:
                continue

            if is_symptom(cui1) and is_disease(cui2):
                symptom_to_diseases[cui1].add(cui2)
                disease_to_symptoms[cui2].add(cui1)
            elif is_symptom(cui2) and is_disease(cui1):
                symptom_to_diseases[cui2].add(cui1)
                disease_to_symptoms[cui1].add(cui2)

    print(f"    증상 CUI with relations: {len(symptom_to_diseases)}")
    print(f"    질병 CUI with relations: {len(disease_to_symptoms)}")
    return symptom_to_diseases, disease_to_symptoms


def score_evidence_ratio(
    confirmed_cuis: Set[str],
    denied_cuis: Set[str],
    disease_cui: str,
    disease_to_symptoms: Dict[str, Set[str]],
) -> float:
    """Evidence Ratio scoring: c/(c+d+1) * c."""
    disease_symptoms = disease_to_symptoms.get(disease_cui, set())
    if not disease_symptoms:
        return 0.0
    c = len(confirmed_cuis & disease_symptoms)
    d = len(denied_cuis & disease_symptoms)
    if c == 0:
        return 0.0
    return (c / (c + d + 1)) * c


def evaluate(
    cases: List[dict],
    symptom_to_diseases: Dict[str, Set[str]],
    disease_to_symptoms: Dict[str, Set[str]],
    name_to_cui: Dict,
    preferred: Dict,
) -> dict:
    """Complete profile 평가."""
    print(f"\n  Evaluating {len(cases)} cases...")

    # 모든 알려진 disease CUI 수집
    all_disease_cuis = set(disease_to_symptoms.keys())
    print(f"  Known disease CUIs in KG: {len(all_disease_cuis)}")

    results = {
        "total": 0, "hit1": 0, "hit3": 0, "hit5": 0, "hit10": 0,
        "no_confirmed": 0, "no_disease_cui": 0, "no_candidates": 0,
    }
    source_results = defaultdict(lambda: {"total": 0, "hit1": 0, "hit3": 0, "hit5": 0})
    dept_results = defaultdict(lambda: {"total": 0, "hit1": 0, "hit3": 0})

    for case in tqdm(cases, desc="Scoring"):
        confirmed = case["confirmed_cuis"]
        denied = case["denied_cuis"]
        gt_cui = case["disease_cui"]

        if not confirmed:
            results["no_confirmed"] += 1
            continue
        if not gt_cui:
            results["no_disease_cui"] += 1
            continue

        # 후보 질병: confirmed 증상과 연결된 모든 질병
        candidate_diseases = set()
        for s_cui in confirmed:
            candidate_diseases.update(symptom_to_diseases.get(s_cui, set()))

        # candidate_diseases에 후보 질병 CUI도 추가
        for cd_name, cd_cui in case.get("candidate_cuis", {}).items():
            candidate_diseases.add(cd_cui)

        if not candidate_diseases:
            results["no_candidates"] += 1
            results["total"] += 1
            source_results[case["source"]]["total"] += 1
            continue

        # 각 후보에 대해 evidence ratio 계산
        scores = []
        for d_cui in candidate_diseases:
            score = score_evidence_ratio(confirmed, denied, d_cui, disease_to_symptoms)
            scores.append((d_cui, score))

        scores.sort(key=lambda x: -x[1])
        top_k_cuis = [cui for cui, _ in scores[:10]]

        results["total"] += 1
        source = case["source"]
        source_results[source]["total"] += 1
        dept = case.get("department", "")
        dept_results[dept]["total"] += 1

        if gt_cui in top_k_cuis[:1]:
            results["hit1"] += 1
            source_results[source]["hit1"] += 1
            dept_results[dept]["hit1"] += 1
        if gt_cui in top_k_cuis[:3]:
            results["hit3"] += 1
            source_results[source]["hit3"] += 1
            dept_results[dept]["hit3"] += 1
        if gt_cui in top_k_cuis[:5]:
            results["hit5"] += 1
            source_results[source]["hit5"] += 1
        if gt_cui in top_k_cuis[:10]:
            results["hit10"] += 1

    return results, dict(source_results), dict(dept_results)


def main():
    print("=" * 60)
    print("DxBench Complete Profile Evaluation")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] DxBench 로드...")
    cases = load_dxbench_data()

    # 2. Build UMLS index
    print("\n[2/4] UMLS 인덱스 구축...")
    name_to_cui, preferred = build_umls_name_index()

    # 3. Map symptoms/diseases to CUIs
    print("\n[3/4] 증상/질병 CUI 매핑...")
    mapped_cases, map_stats = map_symptoms_to_cuis(cases, name_to_cui)
    print(f"  증상 매핑: {map_stats['mapped_symptoms']}/{map_stats['total_symptoms']} "
          f"({map_stats['mapped_symptoms']/max(map_stats['total_symptoms'],1)*100:.1f}%)")
    print(f"  질병 매핑: {map_stats['mapped_diseases']}/{map_stats['total_diseases']} "
          f"({map_stats['mapped_diseases']/max(map_stats['total_diseases'],1)*100:.1f}%)")

    # Build symptom-disease KG index
    print("\n  증상-질병 관계 인덱스 구축...")
    symptom_to_diseases, disease_to_symptoms = build_symptom_disease_index(name_to_cui)

    # 4. Evaluate
    print("\n[4/4] 평가...")
    start = time.time()
    results, source_results, dept_results = evaluate(
        mapped_cases, symptom_to_diseases, disease_to_symptoms, name_to_cui, preferred
    )
    elapsed = time.time() - start

    # Report
    t = results["total"]
    print(f"\n{'='*60}")
    print(f"DXBENCH COMPLETE PROFILE RESULTS")
    print(f"{'='*60}")
    if t > 0:
        print(f"Evaluated: {t} cases")
        print(f"Hit@1:  {results['hit1']}/{t} ({results['hit1']/t*100:.1f}%)")
        print(f"Hit@3:  {results['hit3']}/{t} ({results['hit3']/t*100:.1f}%)")
        print(f"Hit@5:  {results['hit5']}/{t} ({results['hit5']/t*100:.1f}%)")
        print(f"Hit@10: {results['hit10']}/{t} ({results['hit10']/t*100:.1f}%)")
        print(f"No confirmed syms: {results['no_confirmed']}")
        print(f"No disease CUI: {results['no_disease_cui']}")
        print(f"No candidates: {results['no_candidates']}")

        print(f"\nSource별:")
        for src, sr in sorted(source_results.items()):
            st = sr["total"]
            if st > 0:
                print(f"  {src}: Hit@1={sr['hit1']/st*100:.1f}%, "
                      f"Hit@3={sr['hit3']/st*100:.1f}%, "
                      f"Hit@5={sr.get('hit5',0)/st*100:.1f}% (n={st})")

        print(f"\nDepartment별 (Hit@1):")
        for dept, dr in sorted(dept_results.items(), key=lambda x: -x[1]["total"]):
            dt = dr["total"]
            if dt >= 5:
                print(f"  {dept}: {dr['hit1']/dt*100:.1f}% (n={dt})")

    print(f"\nTime: {elapsed:.1f}s")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    output = {
        "dataset": "dxbench",
        "method": "complete_profile_evidence_ratio",
        "scoring": "evidence_ratio",
        "mapping_stats": map_stats,
        "results": results,
        "source_results": source_results,
        "dept_results": dept_results,
        "elapsed_seconds": elapsed,
    }
    with open(OUTPUT_DIR / "dxbench_benchmark.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {OUTPUT_DIR / 'dxbench_benchmark.json'}")


if __name__ == "__main__":
    main()
