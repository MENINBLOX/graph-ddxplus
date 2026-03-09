#!/usr/bin/env python3
"""Top-N 선택 실험 (KG-only).

Top-2부터 Top-10까지 테스트하여 최적의 N 값 탐색.

Usage:
    uv run python scripts/experiment_topn.py -n 1000 --severity 2
"""

import argparse
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
    confirmed_symptoms: list = field(default_factory=list)
    denied_symptoms: list = field(default_factory=list)
    il: int = 0
    predicted: str | None = None
    done: bool = False


class TopNExperiment:
    """Top-N 실험."""

    MAX_IL = 50

    def __init__(
        self,
        top_n: int = 3,
        scoring: str = "v18_coverage",
        severity: int | None = None,
    ):
        self.top_n = top_n
        self.scoring = scoring
        self.severity = severity
        self.loader = DDXPlusLoader()

        # UMLS KG
        self.kg = UMLSKG()

        # CUI 매핑
        self._cui_to_codes = self.loader.build_cui_to_codes()
        self._cui_to_disease: dict[str, str] = {}
        self._disease_to_cui: dict[str, str] = {}
        self._build_disease_cui_mapping()

    def _build_disease_cui_mapping(self):
        """질환 CUI ↔ 이름 매핑."""
        for name_eng, info in self.loader.disease_mapping.items():
            cui = info.get("umls_cui")
            if cui:
                cond = self.loader.conditions.get(name_eng)
                if cond:
                    self._cui_to_disease[cui] = cond.name_fr
                    self._disease_to_cui[cond.name_fr] = cui

    def run(self, n_samples: int, patients: list[Patient] | None = None) -> dict:
        """실험 실행."""
        if patients is None:
            patients = self.loader.load_patients(
                split="test",
                n_samples=n_samples,
                severity=self.severity,
            )

        start_time = time.time()

        # 환자별 상태 초기화
        states = []
        for idx, patient in enumerate(patients):
            state = PatientState(patient=patient, idx=idx)
            init_cui = self._get_cui_for_code(patient.initial_evidence)
            if init_cui:
                state.initial_cui = init_cui
                state.confirmed_cuis.add(init_cui)
                state.asked_cuis.add(init_cui)
                state.confirmed_symptoms.append(
                    self.loader.symptom_mapping.get(patient.initial_evidence, {}).get("name", patient.initial_evidence)
                )
            state.il = 1
            states.append(state)

        # 라운드별 처리
        for round_num in range(1, self.MAX_IL + 1):
            active_states = [s for s in states if not s.done]

            if not active_states:
                break

            for state in active_states:
                self._process_patient_round(state)

        # 미완료 환자 강제 완료
        for state in states:
            if not state.done:
                self._make_diagnosis(state)

        elapsed = time.time() - start_time

        # 결과 계산
        total = len(states)
        correct = sum(1 for s in states if s.predicted == s.patient.pathology)
        total_il = sum(s.il for s in states)

        gtpa_at_1 = correct / total if total > 0 else 0
        avg_il = total_il / total if total > 0 else 0

        return {
            "top_n": self.top_n,
            "samples": total,
            "gtpa_at_1": gtpa_at_1,
            "avg_il": avg_il,
            "elapsed_seconds": elapsed,
        }

    def _process_patient_round(self, state: PatientState):
        """환자 1라운드 처리."""
        # KG 상태 동기화
        self._sync_kg_state(state)

        # 중단 조건 확인
        should_stop, _ = self.kg.should_stop(
            max_il=self.MAX_IL,
            min_il=3,
            confidence_threshold=0.25,
            gap_threshold=0.06,
            relative_gap_threshold=2.0,
        )
        if should_stop:
            self._make_diagnosis(state)
            return

        # KG에서 후보 증상 가져오기
        if not state.initial_cui:
            self._make_diagnosis(state)
            return

        candidates = self.kg.get_candidate_symptoms(
            state.initial_cui,
            limit=self.top_n,  # Top-N 제한
            confirmed_cuis=state.confirmed_cuis.copy(),
            denied_cuis=state.denied_cuis.copy(),
            asked_cuis=state.asked_cuis.copy(),
        )

        if not candidates:
            self._make_diagnosis(state)
            return

        # Top-1 선택 (KG-only: 항상 Top-1)
        selected = candidates[0]
        selected_cui = selected.cui
        selected_name = selected.name

        # 환자 응답 시뮬레이션
        codes = self._cui_to_codes.get(selected_cui, set())
        has_symptom = self._check_evidence(state.patient, codes)

        state.asked_cuis.add(selected_cui)
        if has_symptom:
            state.confirmed_cuis.add(selected_cui)
            state.confirmed_symptoms.append(selected_name)
        else:
            state.denied_cuis.add(selected_cui)
            state.denied_symptoms.append(selected_name)

        state.il += 1

    def _sync_kg_state(self, state: PatientState):
        """KG 내부 상태 동기화."""
        self.kg.state.confirmed_cuis = state.confirmed_cuis.copy()
        self.kg.state.denied_cuis = state.denied_cuis.copy()
        self.kg.state.asked_cuis = state.asked_cuis.copy()

    def _make_diagnosis(self, state: PatientState):
        """최종 진단 (Top-N 중 Top-1 선택)."""
        candidates = self.kg.get_diagnosis_candidates(
            top_k=self.top_n,  # Top-N 제한
            scoring=self.scoring,
            confirmed_cuis=state.confirmed_cuis.copy(),
            denied_cuis=state.denied_cuis.copy(),
        )

        # DDXPlus 질환만 필터링 후 Top-1
        for c in candidates:
            if c.cui in self._cui_to_disease:
                state.predicted = self._cui_to_disease[c.cui]
                break

        if not state.predicted:
            state.predicted = list(self._cui_to_disease.values())[0]

        state.done = True

    def _get_cui_for_code(self, code: str) -> str | None:
        """DDXPlus 코드 → CUI."""
        for cui, codes in self._cui_to_codes.items():
            if code in codes:
                return cui
        return None

    def _check_evidence(self, patient: Patient, codes: set) -> bool:
        """환자 증상 확인."""
        evidences = set(patient.evidences)
        for code in codes:
            if code in evidences:
                return True
            for ev in evidences:
                if ev.startswith(f"{code}_@_"):
                    return True
        return False

    def close(self):
        """리소스 정리."""
        if self.kg:
            self.kg.close()


def main():
    parser = argparse.ArgumentParser(description="Top-N Experiment")
    parser.add_argument("-n", "--n-samples", type=int, default=1000)
    parser.add_argument("--severity", type=int, default=2, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--scoring", type=str, default="v18_coverage")

    args = parser.parse_args()

    print(f"Top-N Experiment: {args.n_samples:,} samples, severity={args.severity}")
    print("=" * 70)

    # 환자 데이터 미리 로드 (재사용)
    loader = DDXPlusLoader()
    patients = loader.load_patients(
        split="test",
        n_samples=args.n_samples,
        severity=args.severity,
    )
    print(f"Loaded {len(patients):,} patients\n")

    results = []

    # Top-2 ~ Top-10 테스트
    for top_n in tqdm(range(2, 11), desc="Top-N Experiments"):
        experiment = TopNExperiment(
            top_n=top_n,
            scoring=args.scoring,
            severity=args.severity,
        )

        try:
            result = experiment.run(args.n_samples, patients=patients)
            results.append(result)

            tqdm.write(
                f"Top-{top_n:2d}: GTPA@1={result['gtpa_at_1']:.2%}, "
                f"IL={result['avg_il']:.1f}, Time={result['elapsed_seconds']:.1f}s"
            )
        finally:
            experiment.close()

    # 결과 요약
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Top-N':>6} {'GTPA@1':>10} {'Avg IL':>10} {'Time':>10}")
    print("-" * 70)

    best_result = max(results, key=lambda x: x["gtpa_at_1"])

    for r in results:
        marker = " *" if r == best_result else ""
        print(
            f"{r['top_n']:>6} {r['gtpa_at_1']:>9.2%} "
            f"{r['avg_il']:>10.2f} {r['elapsed_seconds']:>9.1f}s{marker}"
        )

    print("-" * 70)
    print(f"Best: Top-{best_result['top_n']} with GTPA@1={best_result['gtpa_at_1']:.2%}")

    # 결과 저장
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "topn_experiment.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
