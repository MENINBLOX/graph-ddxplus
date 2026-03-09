#!/usr/bin/env python3
"""Stopping condition 파라미터 튜닝.

Grid search로 최적의 stopping threshold 찾기.
"""

import itertools
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DDXPlusLoader, Patient
from src.umls_kg import UMLSKG


@dataclass
class PatientState:
    """환자별 상태."""
    patient: Patient
    idx: int
    initial_cui: str | None = None
    confirmed_cuis: set = field(default_factory=set)
    denied_cuis: set = field(default_factory=set)
    asked_cuis: set = field(default_factory=set)
    il: int = 0
    done: bool = False


def run_benchmark(
    kg: UMLSKG,
    loader: DDXPlusLoader,
    patients: list[Patient],
    cui_to_codes: dict,
    cui_to_disease: dict,
    scoring: str,
    confidence_threshold: float,
    gap_threshold: float,
    relative_gap_threshold: float,
    min_il: int,
    max_il: int = 50,
) -> dict:
    """단일 파라미터 조합으로 벤치마크."""

    # 환자별 상태 초기화
    states = []
    for idx, patient in enumerate(patients):
        state = PatientState(patient=patient, idx=idx)
        # 초기 증상 추가
        init_cui = None
        for cui, codes in cui_to_codes.items():
            if patient.initial_evidence in codes:
                init_cui = cui
                break
        if init_cui:
            state.initial_cui = init_cui
            state.confirmed_cuis.add(init_cui)
            state.asked_cuis.add(init_cui)
        state.il = 1
        states.append(state)

    # 라운드별 처리
    for round_num in range(1, max_il + 1):
        active_states = [s for s in states if not s.done]

        if not active_states:
            break

        for state in active_states:
            # KG 상태 동기화
            kg.state.confirmed_cuis = state.confirmed_cuis.copy()
            kg.state.denied_cuis = state.denied_cuis.copy()
            kg.state.asked_cuis = state.asked_cuis.copy()

            # 중단 조건 확인
            should_stop, _ = kg.should_stop(
                max_il=max_il,
                min_il=min_il,
                confidence_threshold=confidence_threshold,
                gap_threshold=gap_threshold,
                relative_gap_threshold=relative_gap_threshold,
            )

            if should_stop or not state.initial_cui:
                state.done = True
                continue

            # KG에서 Top-1 증상 가져오기
            candidates = kg.get_candidate_symptoms(
                state.initial_cui,
                limit=10,
                confirmed_cuis=state.confirmed_cuis.copy(),
                denied_cuis=state.denied_cuis.copy(),
                asked_cuis=state.asked_cuis.copy(),
            )

            if not candidates:
                state.done = True
                continue

            # Top-1 선택
            selected = candidates[0]
            selected_cui = selected.cui

            # 환자 응답 시뮬레이션
            codes = cui_to_codes.get(selected_cui, set())
            evidences = set(state.patient.evidences)
            has_symptom = False
            for code in codes:
                if code in evidences:
                    has_symptom = True
                    break
                for ev in evidences:
                    if ev.startswith(f"{code}_@_"):
                        has_symptom = True
                        break

            state.asked_cuis.add(selected_cui)
            if has_symptom:
                state.confirmed_cuis.add(selected_cui)
            else:
                state.denied_cuis.add(selected_cui)

            state.il += 1

    # 미완료 환자 강제 완료
    for state in states:
        state.done = True

    # 진단 및 메트릭 계산
    correct = 0
    total_il = 0

    for state in states:
        # 진단
        candidates = kg.get_diagnosis_candidates(
            top_k=100,
            scoring=scoring,
            confirmed_cuis=state.confirmed_cuis.copy(),
            denied_cuis=state.denied_cuis.copy(),
        )

        predicted = None
        for c in candidates:
            if c.cui in cui_to_disease:
                predicted = cui_to_disease[c.cui]
                break

        if not predicted:
            predicted = list(cui_to_disease.values())[0]

        if predicted == state.patient.pathology:
            correct += 1
        total_il += state.il

    n = len(states)
    return {
        "gtpa_at_1": correct / n if n > 0 else 0,
        "avg_il": total_il / n if n > 0 else 0,
        "n_samples": n,
    }


def main():
    # 파라미터 그리드 (축소)
    param_grid = {
        "confidence_threshold": [0.18, 0.22, 0.25],
        "gap_threshold": [0.04, 0.06],
        "relative_gap_threshold": [1.5, 2.0],
        "min_il": [1, 3],
    }

    scoring = "v23_mild_denied"
    n_samples = 500  # 빠른 탐색
    severity = 2

    # 데이터 로드
    print("Loading data...", flush=True)
    loader = DDXPlusLoader()
    patients = loader.load_patients(split="test", n_samples=n_samples, severity=severity)
    print(f"Loaded {len(patients):,} patients", flush=True)

    # KG 연결
    print("Connecting to Neo4j...", flush=True)
    kg = UMLSKG()
    print("Connected!", flush=True)

    # CUI 매핑
    cui_to_codes = loader.build_cui_to_codes()
    cui_to_disease = {}
    for name_eng, info in loader.disease_mapping.items():
        cui = info.get("umls_cui")
        if cui:
            cond = loader.conditions.get(name_eng)
            if cond:
                cui_to_disease[cui] = cond.name_fr

    # 그리드 서치
    all_combinations = list(itertools.product(
        param_grid["confidence_threshold"],
        param_grid["gap_threshold"],
        param_grid["relative_gap_threshold"],
        param_grid["min_il"],
    ))

    print(f"\nGrid search: {len(all_combinations)} combinations", flush=True)
    print(f"Scoring: {scoring}", flush=True)
    print("=" * 70, flush=True)

    results = []
    best_result = None
    best_gtpa = 0

    for conf, gap, ratio, min_il in tqdm(all_combinations, desc="Tuning"):
        result = run_benchmark(
            kg=kg,
            loader=loader,
            patients=patients,
            cui_to_codes=cui_to_codes,
            cui_to_disease=cui_to_disease,
            scoring=scoring,
            confidence_threshold=conf,
            gap_threshold=gap,
            relative_gap_threshold=ratio,
            min_il=min_il,
        )

        result.update({
            "confidence_threshold": conf,
            "gap_threshold": gap,
            "relative_gap_threshold": ratio,
            "min_il": min_il,
            "scoring": scoring,
        })
        results.append(result)

        if result["gtpa_at_1"] > best_gtpa:
            best_gtpa = result["gtpa_at_1"]
            best_result = result

        # 진행 상황 출력 (상위 결과)
        if len(results) % 20 == 0:
            tqdm.write(f"  Best so far: GTPA@1={best_gtpa:.2%}, IL={best_result['avg_il']:.1f}")

    kg.close()

    # 결과 정렬 및 출력
    results.sort(key=lambda x: (-x["gtpa_at_1"], x["avg_il"]))

    print(f"\n{'='*70}")
    print("TOP 10 RESULTS")
    print("=" * 70)
    print(f"{'Conf':>6} {'Gap':>6} {'Ratio':>6} {'MinIL':>6} | {'GTPA@1':>8} {'Avg IL':>8}")
    print("-" * 70)

    for r in results[:10]:
        print(f"{r['confidence_threshold']:>6.2f} {r['gap_threshold']:>6.2f} "
              f"{r['relative_gap_threshold']:>6.1f} {r['min_il']:>6} | "
              f"{r['gtpa_at_1']:>7.2%} {r['avg_il']:>8.1f}")

    print("-" * 70)
    print(f"\nBest: GTPA@1={best_result['gtpa_at_1']:.2%}, Avg IL={best_result['avg_il']:.1f}")
    print(f"  confidence_threshold={best_result['confidence_threshold']}")
    print(f"  gap_threshold={best_result['gap_threshold']}")
    print(f"  relative_gap_threshold={best_result['relative_gap_threshold']}")
    print(f"  min_il={best_result['min_il']}")

    # 결과 저장
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "stopping_tuning.json"

    with open(output_file, "w") as f:
        json.dump({
            "best": best_result,
            "all_results": results,
            "param_grid": param_grid,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
