#!/usr/bin/env python3
"""Explainability Demo.

현재 KG 구조에서 제공 가능한 설명 기능 데모:
1. 일치/불일치 증상 목록
2. Coverage 기반 점수
3. Synthesized 스타일 설명 텍스트
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.umls_kg import UMLSKG


def demo_pneumonia_case() -> None:
    """폐렴 의심 케이스 데모."""
    print("=" * 70)
    print("Demo: 폐렴 의심 케이스")
    print("=" * 70)

    kg = UMLSKG()

    # 시나리오: Fever + Cough + Dyspnea + Pleuritic pain + Hemoptysis
    confirmed = {
        "C0015967",  # Fever
        "C0010200",  # Cough
        "C0013404",  # Dyspnea
        "C0008033",  # Pleuritic pain (ww_respi)
        "C0019079",  # Hemoptysis (crach_sg)
    }
    denied = {
        "C0027497",  # Nausea
        "C0011991",  # Diarrhea
        "C0085593",  # Chills
    }

    print("\n[환자 정보]")
    print("  확인된 증상: Fever, Cough, Dyspnea, Pleuritic Pain, Hemoptysis")
    print("  부정된 증상: Nausea, Diarrhea, Chills")

    # 설명 가능한 진단 결과
    results = kg.get_explained_diagnosis_candidates(
        top_k=5,
        confirmed_cuis=confirmed,
        denied_cuis=denied,
    )

    print("\n[감별 진단 결과]")
    print("-" * 70)
    for r in results:
        print(r.explanation)
        if r.unasked_symptoms:
            print(f"   → 추가 확인 필요: {', '.join(r.unasked_symptoms[:3])}")
        print()

    kg.close()


def demo_cardiac_case() -> None:
    """심장 질환 의심 케이스 데모."""
    print("=" * 70)
    print("Demo: 심장 질환 의심 케이스")
    print("=" * 70)

    kg = UMLSKG()

    # 시나리오: Pain + Palpitations + Dyspnea + Diaphoresis
    confirmed = {
        "C0030193",  # Pain (douleurxx)
        "C0030252",  # Palpitations
        "C0013404",  # Dyspnea
        "C0700590",  # Diaphoresis (diaph)
    }
    denied = {
        "C0015967",  # Fever
        "C0010200",  # Cough
    }

    print("\n[환자 정보]")
    print("  확인된 증상: Pain, Palpitations, Dyspnea, Diaphoresis")
    print("  부정된 증상: Fever, Cough")

    results = kg.get_explained_diagnosis_candidates(
        top_k=5,
        confirmed_cuis=confirmed,
        denied_cuis=denied,
    )

    print("\n[감별 진단 결과]")
    print("-" * 70)
    for r in results:
        print(r.explanation)
        print()

    kg.close()


def demo_raw_data() -> None:
    """Raw 데이터 출력 데모."""
    print("=" * 70)
    print("Demo: 원시 데이터 구조")
    print("=" * 70)

    kg = UMLSKG()

    confirmed = {"C0015967", "C0010200"}  # Fever, Cough
    denied = {"C0027497"}  # Nausea

    results = kg.get_explained_diagnosis_candidates(
        top_k=3,
        confirmed_cuis=confirmed,
        denied_cuis=denied,
    )

    print("\n[ExplainedDiagnosis 구조]")
    if results:
        r = results[0]
        print(f"  cui: {r.cui}")
        print(f"  name: {r.name}")
        print(f"  score: {r.score:.3f}")
        print(f"  rank: {r.rank}")
        print(f"  matched_symptoms: {r.matched_symptoms}")
        print(f"  denied_symptoms: {r.denied_symptoms}")
        print(f"  unasked_symptoms: {r.unasked_symptoms}")
        print(f"  matched_count: {r.matched_count}")
        print(f"  denied_count: {r.denied_count}")
        print(f"  total_symptoms: {r.total_symptoms}")
        print(f"  coverage: {r.coverage:.1%}")

    kg.close()


if __name__ == "__main__":
    demo_pneumonia_case()
    print("\n")
    demo_cardiac_case()
    print("\n")
    demo_raw_data()
