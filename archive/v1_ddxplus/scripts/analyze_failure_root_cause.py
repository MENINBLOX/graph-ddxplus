#!/usr/bin/env python3
"""실패 케이스 근본 원인 분석.

confirmed=1인 케이스에서:
1. 환자의 증상 CUI 목록
2. KG가 질문한 증상 CUI 목록
3. 왜 일치하지 않는지
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DDXPlusLoader
from src.umls_kg import UMLSKG


def analyze_single_case(patient_idx: int, loader: DDXPlusLoader, kg: UMLSKG) -> dict:
    """단일 케이스 상세 분석."""
    patients = loader.load_patients(split="test", n_samples=None, severity=None)
    patient = patients[patient_idx]

    # GT 질환
    gt_disease_eng = loader.fr_to_eng.get(patient.pathology, patient.pathology)
    gt_cui = loader.get_disease_cui(gt_disease_eng)

    # 환자 증상 CUI
    patient_symptoms = {}
    for ev_str in patient.evidences:
        code = ev_str.split("_@_")[0] if "_@_" in ev_str else ev_str
        cui = loader.get_symptom_cui(code)
        symptom_name = loader.symptom_mapping.get(code, {}).get("name_eng", code)
        patient_symptoms[code] = {
            "cui": cui,
            "name": symptom_name,
            "in_kg": cui is not None,
        }

    # 초기 증상
    initial_code = patient.initial_evidence
    initial_cui = loader.get_symptom_cui(initial_code)

    # KG 시뮬레이션
    kg.reset_state()
    if initial_cui:
        kg.state.add_confirmed(initial_cui)

    kg_asked_symptoms = []
    patient_positive_cuis = {s["cui"] for s in patient_symptoms.values() if s["cui"]}

    for i in range(10):  # 처음 10개 질문만 분석
        candidates = kg.get_candidate_symptoms(
            initial_cui=initial_cui,
            limit=10,
            confirmed_cuis=kg.state.confirmed_cuis,
            denied_cuis=kg.state.denied_cuis,
            asked_cuis=kg.state.asked_cuis,
        )

        if not candidates:
            break

        selected = candidates[0]
        matched = selected.cui in patient_positive_cuis

        kg_asked_symptoms.append({
            "cui": selected.cui,
            "name": selected.name,
            "matched": matched,
            "disease_coverage": selected.disease_coverage,
        })

        if matched:
            kg.state.add_confirmed(selected.cui)
        else:
            kg.state.add_denied(selected.cui)

    # GT 질환이 KG에서 어떤 증상과 연결되어 있는지
    gt_disease_symptoms = []
    if gt_cui:
        query = """
        MATCH (d:Disease {cui: $disease_cui})<-[:INDICATES]-(s:Symptom)
        RETURN s.cui AS cui, s.name AS name
        """
        with kg.driver.session() as session:
            result = session.run(query, disease_cui=gt_cui)
            for record in result:
                gt_disease_symptoms.append({
                    "cui": record["cui"],
                    "name": record["name"],
                })

    # 환자 증상 중 GT 질환과 연결된 것
    patient_cuis = {s["cui"] for s in patient_symptoms.values() if s["cui"]}
    gt_symptom_cuis = {s["cui"] for s in gt_disease_symptoms}
    overlap = patient_cuis & gt_symptom_cuis

    return {
        "patient_idx": patient_idx,
        "gt_disease": gt_disease_eng,
        "gt_cui": gt_cui,
        "patient_symptoms": patient_symptoms,
        "patient_symptom_cuis": list(patient_cuis),
        "initial_symptom": {
            "code": initial_code,
            "cui": initial_cui,
        },
        "kg_asked_symptoms": kg_asked_symptoms,
        "gt_disease_symptoms_in_kg": gt_disease_symptoms[:20],  # 최대 20개
        "gt_disease_symptom_count": len(gt_disease_symptoms),
        "overlap_with_patient": list(overlap),
        "overlap_count": len(overlap),
    }


def main():
    # 실패 케이스 로드
    with open("results/failure_analysis_gtpa10.json") as f:
        failure_data = json.load(f)

    failures = failure_data["failure_cases"]
    print(f"Total failures: {len(failures)}")

    loader = DDXPlusLoader()
    kg = UMLSKG()

    # 상위 실패 질환별 샘플 분석
    disease_samples = {}
    for f in failures:
        disease = f["gt_disease_eng"]
        if disease not in disease_samples:
            disease_samples[disease] = []
        if len(disease_samples[disease]) < 2:  # 질환당 2개 샘플
            disease_samples[disease].append(f["patient_idx"])

    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)

    all_analyses = []

    for disease in ["Atrial fibrillation", "Viral pharyngitis", "Scombroid food poisoning"]:
        if disease not in disease_samples:
            continue

        print(f"\n{'='*80}")
        print(f"DISEASE: {disease}")
        print("=" * 80)

        for idx in disease_samples[disease][:1]:  # 질환당 1개만 상세 분석
            analysis = analyze_single_case(idx, loader, kg)
            all_analyses.append(analysis)

            print(f"\n[Case idx={idx}]")
            print(f"  GT Disease: {analysis['gt_disease']} (CUI: {analysis['gt_cui']})")

            print(f"\n  Patient Symptoms ({len(analysis['patient_symptoms'])}):")
            for code, info in list(analysis["patient_symptoms"].items())[:10]:
                status = "✓" if info["in_kg"] else "✗"
                print(f"    {status} {code}: {info['name']} (CUI: {info['cui']})")

            print(f"\n  Initial Symptom: {analysis['initial_symptom']}")

            print(f"\n  KG Asked Symptoms (first 10):")
            for i, s in enumerate(analysis["kg_asked_symptoms"]):
                status = "✓ MATCH" if s["matched"] else "✗ MISS"
                print(f"    {i+1}. {s['name']} (coverage={s['disease_coverage']}) [{status}]")

            print(f"\n  GT Disease has {analysis['gt_disease_symptom_count']} symptoms in KG")
            print(f"  Overlap with patient symptoms: {analysis['overlap_count']}")
            if analysis["overlap_with_patient"]:
                overlap_names = []
                for cui in analysis["overlap_with_patient"][:5]:
                    for s in analysis["gt_disease_symptoms_in_kg"]:
                        if s["cui"] == cui:
                            overlap_names.append(s["name"])
                            break
                print(f"    Overlapping: {overlap_names}")

    # 통계
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # 환자 증상이 KG에 있는 비율
    all_patient_symptoms_in_kg = []
    all_overlap_ratios = []

    for analysis in all_analyses:
        symptoms = analysis["patient_symptoms"]
        in_kg = sum(1 for s in symptoms.values() if s["in_kg"])
        total = len(symptoms)
        if total > 0:
            all_patient_symptoms_in_kg.append(in_kg / total)

        patient_cuis = len(analysis["patient_symptom_cuis"])
        overlap = analysis["overlap_count"]
        if patient_cuis > 0:
            all_overlap_ratios.append(overlap / patient_cuis)

    if all_patient_symptoms_in_kg:
        print(f"\n  Patient symptoms in KG: {sum(all_patient_symptoms_in_kg)/len(all_patient_symptoms_in_kg):.1%}")
    if all_overlap_ratios:
        print(f"  Patient symptoms overlap with GT disease: {sum(all_overlap_ratios)/len(all_overlap_ratios):.1%}")

    # 결과 저장
    output = {
        "analyses": all_analyses,
    }

    with open("results/failure_root_cause.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nResults saved to: results/failure_root_cause.json")

    kg.close()


if __name__ == "__main__":
    main()
