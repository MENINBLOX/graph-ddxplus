#!/usr/bin/env python3
"""RareBench Complete Profile 평가.

RareBench 환자의 전체 phenotype 목록을 주고,
GraphTrace의 scoring function으로 진단 정확도를 측정한다.

Interactive inquiry는 불가 (denied symptom 정보 없음).
→ Complete profile setting으로만 평가.

Usage:
    uv run python scripts/benchmark_rarebench.py
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

RAREBENCH_DIR = Path("data/rarebench")
RAREBENCH_MAPPING_DIR = Path("external/RareBench/mapping")
MEDDX_DATA_DIR = Path("external/meddxagent/ddxdriver/benchmarks/data/rarebench")


def load_rarebench_data() -> Tuple[dict, dict, dict, dict]:
    """RareBench 데이터 로드."""
    # UMLS 매핑
    with open(RAREBENCH_DIR / "hpo_umls_mapping.json") as f:
        hpo_mapping = json.load(f)["mapping"]
    with open(RAREBENCH_DIR / "disease_umls_mapping.json") as f:
        disease_mapping = json.load(f)["mapping"]

    # 원본 RareBench 데이터 (MEDDxAgent 형식)
    with open(MEDDX_DATA_DIR / "rarebench_phenotype_mapping.json", encoding="utf-8-sig") as f:
        rb_pheno = json.load(f)
    with open(MEDDX_DATA_DIR / "rarebench_disease_mapping.json", encoding="utf-8-sig") as f:
        rb_disease = json.load(f)

    # 진단 옵션
    with open(MEDDX_DATA_DIR / "diagnosis_options.json") as f:
        diagnosis_options = json.load(f)

    return hpo_mapping, disease_mapping, rb_pheno, rb_disease, diagnosis_options


def load_patient_data() -> List[dict]:
    """HuggingFace에서 RareBench 환자 데이터 로드 (data.zip → jsonl)."""
    from huggingface_hub import hf_hub_download
    import zipfile

    path = hf_hub_download(
        repo_id="chenxz/RareBench",
        filename="data.zip",
        repo_type="dataset",
        revision="4bb064a52a253cfaa5ee228926e86af3ec0e5731",
    )

    subsets = ["RAMEDIS", "MME", "PUMCH_ADM"]
    all_patients = []

    with zipfile.ZipFile(path) as z:
        for subset in subsets:
            jsonl_name = f"data/{subset}.jsonl"
            print(f"  Loading {jsonl_name}...", end=" ")
            try:
                with z.open(jsonl_name) as f:
                    for i, line in enumerate(f):
                        entry = json.loads(line)
                        all_patients.append({
                            "subset": subset,
                            "id": f"{subset}_{i}",
                            "phenotypes": entry.get("Phenotype", []),
                            "diseases": entry.get("RareDisease", []),
                        })
                count = sum(1 for p in all_patients if p["subset"] == subset)
                print(f"{count} cases")
            except KeyError:
                print(f"  [skip] {jsonl_name} not found")

    return all_patients


def build_hpo_disease_index(
    hpo_mapping: dict,
    disease_mapping: dict,
    rb_pheno: dict,
    rb_disease: dict,
    diagnosis_options: dict,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
    """HPO→CUI, Disease→CUI 매핑 + 진단 옵션 CUI 인덱스."""
    # HPO code → UMLS CUI
    hpo_to_cui = {}
    for hp_code, info in hpo_mapping.items():
        if info.get("umls_cui"):
            hpo_to_cui[hp_code] = info["umls_cui"]

    # Disease code → UMLS CUI
    disease_to_cui = {}
    for code, info in disease_mapping.items():
        if info.get("umls_cui"):
            disease_to_cui[code] = info["umls_cui"]

    # 진단 옵션: subset별 disease name → CUI 매핑
    # MEDDxAgent의 disease_mapping은 disease_codes → display_name
    # 우리는 display_name → CUI가 필요
    option_name_to_cuis = defaultdict(set)
    for code, info in disease_mapping.items():
        if info.get("umls_cui"):
            d_name = info.get("disease_name", "")
            # disease_name에서 첫 번째 이름 추출 (슬래시로 구분)
            for part in d_name.split("/"):
                option_name_to_cuis[part.strip().lower()].add(info["umls_cui"])

    return hpo_to_cui, disease_to_cui, option_name_to_cuis


def score_complete_profile(
    patient_hpo_cuis: Set[str],
    disease_cuis: Dict[str, str],
    all_hpo_cuis: Set[str],
) -> List[Tuple[str, str, float]]:
    """Complete profile scoring: evidence ratio.

    환자의 HPO CUI 목록과 각 질병의 HPO CUI 목록 간 유사도 계산.
    여기서는 KG 없이 직접 계산 (UMLS relationship 기반).

    간단한 Jaccard-like scoring:
    score = |patient_symptoms ∩ disease_symptoms| / |patient_symptoms ∪ disease_symptoms|
    """
    # 이 방식은 KG relationship이 필요하므로,
    # MRREL에서 HPO CUI → Disease CUI 관계를 사용해야 한다.
    # 현재는 placeholder - 실제로는 MRREL 기반 매핑이 필요
    pass


def evaluate_with_mrrel(
    patients: List[dict],
    hpo_to_cui: Dict[str, str],
    disease_to_cui: Dict[str, str],
    rb_disease: dict,
) -> dict:
    """MRREL 기반 complete profile 평가.

    1. 환자 HPO codes → CUIs
    2. 각 CUI가 MRREL에서 연결된 Disease CUI 집계
    3. 가장 많이 연결된 Disease = 예측
    """
    print(f"\n[MRREL 기반 평가]")

    # MRREL에서 HPO CUI → Disease CUI 관계 로드
    mrrel_path = Path("data/umls_extracted/MRREL.RRF")
    if not mrrel_path.exists():
        print("  MRREL.RRF 없음 - 간단 매칭으로 대체")
        return evaluate_simple_matching(patients, hpo_to_cui, disease_to_cui, rb_disease)

    print("  MRREL.RRF에서 HPO-Disease 관계 추출...")
    hpo_cuis = set(hpo_to_cui.values())
    disease_cuis_set = set(disease_to_cui.values())

    # CUI → CUI 관계 (HPO ↔ any)
    hpo_related = defaultdict(set)  # hpo_cui → {related_cui}
    with open(mrrel_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            cui1, cui2 = parts[0], parts[4]
            if cui1 in hpo_cuis:
                hpo_related[cui1].add(cui2)
            if cui2 in hpo_cuis:
                hpo_related[cui2].add(cui1)

    print(f"  HPO CUIs with relations: {len(hpo_related)}")

    # Semantic type 로드 (Disease CUI 식별용)
    print("  MRSTY.RRF에서 Disease semantic types 로드...")
    disease_stys = {"T047", "T019", "T191", "T046"}
    cui_is_disease = set()
    with open(Path("data/umls_extracted/MRSTY.RRF"), encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if parts[1] in disease_stys:
                cui_is_disease.add(parts[0])

    print(f"  Disease CUIs (by STY): {len(cui_is_disease)}")

    # 환자별 평가
    results = {"total": 0, "hit1": 0, "hit3": 0, "hit10": 0, "no_match": 0}
    subset_results = defaultdict(lambda: {"total": 0, "hit1": 0, "hit3": 0, "hit10": 0})

    for patient in tqdm(patients, desc="Evaluating"):
        # 환자 HPO → CUI
        patient_cuis = set()
        for hp in patient["phenotypes"]:
            if hp in hpo_to_cui:
                patient_cuis.add(hpo_to_cui[hp])

        if not patient_cuis:
            continue

        # 환자 HPO CUI → 관련 Disease CUI 집계
        disease_scores = defaultdict(int)
        for hpo_cui in patient_cuis:
            for related_cui in hpo_related.get(hpo_cui, set()):
                if related_cui in cui_is_disease or related_cui in disease_cuis_set:
                    disease_scores[related_cui] += 1

        if not disease_scores:
            results["no_match"] += 1
            results["total"] += 1
            subset_results[patient["subset"]]["total"] += 1
            continue

        # Top-K 예측
        sorted_diseases = sorted(disease_scores.items(), key=lambda x: -x[1])
        top_k_cuis = [cui for cui, _ in sorted_diseases[:10]]

        # Ground truth CUI(s)
        gt_cuis = set()
        for d_code in patient["diseases"]:
            if d_code in disease_to_cui:
                gt_cuis.add(disease_to_cui[d_code])

        if not gt_cuis:
            continue

        results["total"] += 1
        subset_results[patient["subset"]]["total"] += 1

        if gt_cuis & set(top_k_cuis[:1]):
            results["hit1"] += 1
            subset_results[patient["subset"]]["hit1"] += 1
        if gt_cuis & set(top_k_cuis[:3]):
            results["hit3"] += 1
            subset_results[patient["subset"]]["hit3"] += 1
        if gt_cuis & set(top_k_cuis[:10]):
            results["hit10"] += 1
            subset_results[patient["subset"]]["hit10"] += 1

    return results, subset_results


def evaluate_simple_matching(patients, hpo_to_cui, disease_to_cui, rb_disease):
    """MRREL 없이 간단한 매칭 평가 (fallback)."""
    # TODO: implement
    return {"total": 0, "hit1": 0, "hit3": 0, "hit10": 0}, {}


def main():
    print("=" * 60)
    print("RareBench Complete Profile Evaluation")
    print("=" * 60)

    # Load data
    print("\n[1/3] 데이터 로드...")
    hpo_mapping, disease_mapping, rb_pheno, rb_disease, diagnosis_options = load_rarebench_data()

    print(f"  HPO→CUI: {sum(1 for v in hpo_mapping.values() if v.get('umls_cui'))}")
    print(f"  Disease→CUI: {sum(1 for v in disease_mapping.values() if v.get('umls_cui'))}")

    hpo_to_cui, disease_to_cui, option_name_to_cuis = build_hpo_disease_index(
        hpo_mapping, disease_mapping, rb_pheno, rb_disease, diagnosis_options
    )

    # Load patients
    print("\n[2/3] 환자 데이터 로드...")
    patients = load_patient_data()
    print(f"  총 환자: {len(patients)}")

    # Evaluate
    print("\n[3/3] Complete Profile 평가...")
    start = time.time()
    results, subset_results = evaluate_with_mrrel(patients, hpo_to_cui, disease_to_cui, rb_disease)
    elapsed = time.time() - start

    # Report
    print(f"\n{'='*60}")
    print(f"RAREBENCH COMPLETE PROFILE RESULTS")
    print(f"{'='*60}")
    t = results["total"]
    if t > 0:
        print(f"Total evaluated:  {t}")
        print(f"Hit@1:  {results['hit1']}/{t} ({results['hit1']/t*100:.1f}%)")
        print(f"Hit@3:  {results['hit3']}/{t} ({results['hit3']/t*100:.1f}%)")
        print(f"Hit@10: {results['hit10']}/{t} ({results['hit10']/t*100:.1f}%)")
        print(f"No match: {results.get('no_match', 0)}")

        print(f"\nSubset별:")
        for subset, sr in sorted(subset_results.items()):
            st = sr["total"]
            if st > 0:
                print(f"  {subset}: Hit@1={sr['hit1']/st*100:.1f}%, Hit@3={sr['hit3']/st*100:.1f}%, Hit@10={sr['hit10']/st*100:.1f}% (n={st})")
    else:
        print("평가된 환자 없음")

    print(f"\nTime: {elapsed:.1f}s")
    print(f"{'='*60}")

    # Save
    output = {
        "dataset": "rarebench",
        "method": "complete_profile_mrrel",
        "results": results,
        "subset_results": dict(subset_results),
        "elapsed_seconds": elapsed,
    }
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "rarebench_benchmark.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: results/rarebench_benchmark.json")


if __name__ == "__main__":
    main()
