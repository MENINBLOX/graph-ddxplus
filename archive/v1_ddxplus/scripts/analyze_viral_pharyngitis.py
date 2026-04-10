#!/usr/bin/env python3
"""Viral pharyngitis 실패 케이스 근본 원인 분석."""

import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

NEO4J_PORTS = [7687, 7688, 7689, 7690, 7691, 7692, 7693, 7694]


def analyze_failure(args: tuple) -> dict | None:
    """실패 케이스 분석."""
    patient_idx, patient_data, loader_data, neo4j_port = args

    from src.data_loader import DDXPlusLoader, Patient
    from src.umls_kg import UMLSKG

    loader = DDXPlusLoader()
    loader._symptom_mapping = loader_data["symptom_mapping"]
    loader._disease_mapping = loader_data["disease_mapping"]
    loader._fr_to_eng = loader_data["fr_to_eng"]
    loader._conditions = loader_data["conditions"]

    try:
        uri = f"bolt://localhost:{neo4j_port}"
        kg = UMLSKG(uri=uri)
    except Exception as e:
        return {"error": str(e), "patient_idx": patient_idx}

    try:
        patient = Patient(
            age=patient_data["age"],
            sex=patient_data["sex"],
            initial_evidence=patient_data["initial_evidence"],
            evidences=patient_data["evidences"],
            pathology=patient_data["pathology"],
            differential_diagnosis=patient_data["differential_diagnosis"],
        )

        gt_disease_eng = loader.fr_to_eng.get(patient.pathology, patient.pathology)
        if gt_disease_eng != "Viral pharyngitis":
            kg.close()
            return None

        gt_cui = loader.get_disease_cui(gt_disease_eng)

        # 환자 증상 분석
        patient_symptoms = {}
        patient_positive_cuis = set()
        for ev_str in patient.evidences:
            code = ev_str.split("_@_")[0] if "_@_" in ev_str else ev_str
            cui = loader.get_symptom_cui(code)
            patient_symptoms[code] = {
                "cui": cui,
                "in_kg": False  # 나중에 채움
            }
            if cui:
                patient_positive_cuis.add(cui)

        initial_cui = loader.get_symptom_cui(patient.initial_evidence)
        if not initial_cui:
            kg.close()
            return None

        kg.reset_state()
        kg.state.add_confirmed(initial_cui)

        # 진단 수행
        asked_symptoms = []
        il = 0
        confirmed_count = 1

        for _ in range(50):
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

            if selected.cui in patient_positive_cuis:
                kg.state.add_confirmed(selected.cui)
                confirmed_count += 1
                asked_symptoms.append({
                    "cui": selected.cui,
                    "name": selected.name,
                    "matched": True,
                    "coverage": selected.disease_coverage
                })
            else:
                kg.state.add_denied(selected.cui)
                asked_symptoms.append({
                    "cui": selected.cui,
                    "name": selected.name,
                    "matched": False,
                    "coverage": selected.disease_coverage
                })

            il += 1

            should_stop, _ = kg.should_stop(max_il=50, min_il=10)
            if should_stop:
                break

        diagnosis = kg.get_diagnosis_candidates(top_k=10)
        predicted_cuis = [d.cui for d in diagnosis]
        correct_at_10 = gt_cui in predicted_cuis if gt_cui else False

        # 실패 케이스만 반환
        if correct_at_10:
            kg.close()
            return None

        # GT 질환의 KG 증상 조회
        gt_symptoms_query = """
        MATCH (d:Disease {cui: $disease_cui})<-[:INDICATES]-(s:Symptom)
        RETURN s.cui AS cui, s.name AS name
        """
        with kg.driver.session() as session:
            result = session.run(gt_symptoms_query, disease_cui=gt_cui)
            gt_disease_symptoms = [{"cui": r["cui"], "name": r["name"]} for r in result]

        gt_symptom_cuis = {s["cui"] for s in gt_disease_symptoms}

        # 환자 증상 중 KG에 있는 것 체크
        for code, info in patient_symptoms.items():
            if info["cui"] and info["cui"] in gt_symptom_cuis:
                info["in_kg"] = True

        kg.close()

        return {
            "patient_idx": patient_idx,
            "gt_disease": gt_disease_eng,
            "gt_cui": gt_cui,
            "initial_evidence": patient.initial_evidence,
            "initial_cui": initial_cui,
            "patient_symptoms": patient_symptoms,
            "patient_symptom_cuis": list(patient_positive_cuis),
            "asked_symptoms": asked_symptoms[:10],  # 처음 10개만
            "gt_disease_symptoms": gt_disease_symptoms,
            "overlap_count": len(patient_positive_cuis & gt_symptom_cuis),
            "il": il,
            "confirmed_count": confirmed_count,
            "top_diagnosis": [{"cui": d.cui, "name": d.name, "score": d.score} for d in diagnosis[:5]]
        }

    except Exception as e:
        kg.close()
        return {"error": str(e), "patient_idx": patient_idx}


def main():
    from src.data_loader import DDXPlusLoader

    loader = DDXPlusLoader()
    all_patients = loader.load_patients(split="test", n_samples=None, severity=None)

    loader_data = {
        "symptom_mapping": loader.symptom_mapping,
        "disease_mapping": loader.disease_mapping,
        "fr_to_eng": loader.fr_to_eng,
        "conditions": {
            k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
            for k, v in loader.conditions.items()
        },
    }

    patients_data = [
        {
            "age": p.age,
            "sex": p.sex,
            "initial_evidence": p.initial_evidence,
            "evidences": p.evidences,
            "pathology": p.pathology,
            "differential_diagnosis": p.differential_diagnosis,
        }
        for p in all_patients
    ]

    # Viral pharyngitis 환자만 필터
    vp_indices = []
    for idx, p in enumerate(all_patients):
        disease_eng = loader.fr_to_eng.get(p.pathology, p.pathology)
        if disease_eng == "Viral pharyngitis":
            vp_indices.append(idx)

    print(f"Total Viral pharyngitis patients: {len(vp_indices)}")

    tasks = [
        (idx, patients_data[idx], loader_data, NEO4J_PORTS[i % 8])
        for i, idx in enumerate(vp_indices)
    ]

    failures = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_failure, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Analyzing"):
            r = future.result()
            if r and "error" not in r:
                failures.append(r)

    print(f"\nTotal failures found: {len(failures)}")

    # 분석
    print("\n" + "=" * 70)
    print("Viral pharyngitis 실패 케이스 분석")
    print("=" * 70)

    # 초기 증상 분포
    initial_symptoms = Counter(f["initial_evidence"] for f in failures)
    print("\n초기 증상 분포:")
    for symptom, count in initial_symptoms.most_common(10):
        print(f"  {symptom}: {count}")

    # 환자 증상 중 매칭되지 않는 것
    unmatched_symptoms = Counter()
    for f in failures:
        for code, info in f["patient_symptoms"].items():
            if not info["in_kg"]:
                unmatched_symptoms[code] += 1

    print("\n환자 증상 중 KG에 없는 것 (상위 10개):")
    for symptom, count in unmatched_symptoms.most_common(10):
        print(f"  {symptom}: {count}")

    # KG가 질문한 증상 중 환자가 없다고 한 것
    denied_symptoms = Counter()
    for f in failures:
        for asked in f["asked_symptoms"]:
            if not asked["matched"]:
                denied_symptoms[asked["name"]] += 1

    print("\nKG가 질문했지만 환자가 거부한 증상 (상위 10개):")
    for symptom, count in denied_symptoms.most_common(10):
        print(f"  {symptom}: {count}")

    # 확인된 증상 수
    confirmed_counts = Counter(f["confirmed_count"] for f in failures)
    print("\n확인된 증상 수 분포:")
    for count in sorted(confirmed_counts.keys()):
        print(f"  {count}개: {confirmed_counts[count]}건")

    # 상위 진단
    top_diagnoses = Counter()
    for f in failures:
        if f["top_diagnosis"]:
            top_diagnoses[f["top_diagnosis"][0]["name"]] += 1

    print("\n잘못된 Top-1 진단 분포:")
    for diagnosis, count in top_diagnoses.most_common(10):
        print(f"  {diagnosis}: {count}")

    # 상세 케이스 3개 출력
    print("\n" + "=" * 70)
    print("상세 케이스 분석 (3개)")
    print("=" * 70)

    for i, f in enumerate(failures[:3]):
        print(f"\n--- Case {i+1} (patient_idx={f['patient_idx']}) ---")
        print(f"초기 증상: {f['initial_evidence']} ({f['initial_cui']})")
        print(f"환자 증상 ({len(f['patient_symptoms'])}개):")
        for code, info in list(f["patient_symptoms"].items())[:5]:
            in_kg = "✓" if info["in_kg"] else "✗"
            print(f"  {in_kg} {code}: {info['cui']}")
        print(f"KG 질문 증상:")
        for asked in f["asked_symptoms"][:5]:
            match = "✓" if asked["matched"] else "✗"
            print(f"  {match} {asked['name']} (coverage={asked['coverage']})")
        print(f"Top 진단:")
        for d in f["top_diagnosis"][:3]:
            print(f"  - {d['name']} (score={d['score']:.3f})")

    # 결과 저장
    with open("results/viral_pharyngitis_failures.json", "w") as f:
        json.dump({
            "total_failures": len(failures),
            "initial_symptom_distribution": dict(initial_symptoms),
            "unmatched_symptoms": dict(unmatched_symptoms),
            "denied_symptoms": dict(denied_symptoms),
            "confirmed_count_distribution": dict(confirmed_counts),
            "top_misdiagnoses": dict(top_diagnoses),
            "sample_cases": failures[:5]
        }, f, indent=2, ensure_ascii=False)

    print("\nResults saved to: results/viral_pharyngitis_failures.json")


if __name__ == "__main__":
    main()
