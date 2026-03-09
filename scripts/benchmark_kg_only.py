#!/usr/bin/env python3
"""KG-Only 벤치마크 (LLM 없이 KG만 사용).

증상 선택: KG Top-1
진단 선택: KG Top-1
CPU만 사용 (Neo4j 쿼리)

Usage:
    uv run python scripts/benchmark_kg_only.py -n 27389 --severity 2
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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


class KGOnlyBenchmark:
    """KG-Only 벤치마크."""

    MAX_IL = 50

    def __init__(
        self,
        scoring: str = "v18_coverage",
        severity: int | None = None,
    ):
        self.scoring = scoring
        self.severity = severity
        self.loader = DDXPlusLoader()

        # UMLS KG
        print("Connecting to Neo4j KG...", flush=True)
        self.kg = UMLSKG()
        print("KG connected!", flush=True)

        # CUI 매핑
        self._cui_to_codes = self.loader.build_cui_to_codes()
        self._cui_to_disease: dict[str, str] = {}
        self._disease_to_cui: dict[str, str] = {}
        self._build_disease_cui_mapping()

        print(f"Disease CUI mappings: {len(self._cui_to_disease)} / 49 diseases", flush=True)

    def _build_disease_cui_mapping(self):
        """질환 CUI ↔ 이름 매핑."""
        for name_eng, info in self.loader.disease_mapping.items():
            cui = info.get("umls_cui")
            if cui:
                cond = self.loader.conditions.get(name_eng)
                if cond:
                    self._cui_to_disease[cui] = cond.name_fr
                    self._disease_to_cui[cond.name_fr] = cui

    def run(self, n_samples: int) -> dict:
        """벤치마크 실행."""
        patients = self.loader.load_patients(
            split="test",
            n_samples=n_samples,
            severity=self.severity,
        )

        print(f"\nLoaded {len(patients):,} patients (severity={self.severity})", flush=True)

        start_time = time.time()

        # 환자별 상태 초기화
        states = []
        for idx, patient in enumerate(patients):
            state = PatientState(patient=patient, idx=idx)
            # 초기 증상 추가
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
        pbar = tqdm(total=len(states), desc="Patients", unit="patient")

        for round_num in range(1, self.MAX_IL + 1):
            active_states = [s for s in states if not s.done]

            if not active_states:
                break

            # 각 환자 처리
            for state in active_states:
                self._process_patient_round(state)

            # 완료된 환자 수 업데이트
            completed = [s for s in states if s.done]
            pbar.n = len(completed)

            # 현재 정확도 계산
            if completed:
                acc = sum(1 for s in completed if s.predicted == s.patient.pathology) / len(completed)
                avg_il = sum(s.il for s in completed) / len(completed)
                pbar.set_postfix({
                    "round": round_num,
                    "avg_IL": f"{avg_il:.1f}",
                    "GTPA@1": f"{acc:.1%}",
                })
            else:
                pbar.set_postfix({"round": round_num})

            pbar.refresh()

        # 미완료 환자 강제 완료
        for state in states:
            if not state.done:
                self._make_diagnosis(state)

        pbar.n = len(states)
        pbar.refresh()
        pbar.close()

        elapsed = time.time() - start_time

        # 결과 계산
        return self._calculate_metrics(states, elapsed)

    def _process_patient_round(self, state: PatientState):
        """환자 1라운드 처리."""
        # KG 상태 동기화
        self._sync_kg_state(state)

        # 1. 중단 조건 확인
        should_stop, _ = self.kg.should_stop(
            max_il=self.MAX_IL,
            min_il=3,
            confidence_threshold=0.25,
            gap_threshold=0.06,
            relative_gap_threshold=1.5,  # 최적화: 2.0 → 1.5
        )
        if should_stop:
            self._make_diagnosis(state)
            return

        # 2. KG에서 Top-1 증상 가져오기
        if not state.initial_cui:
            self._make_diagnosis(state)
            return

        candidates = self.kg.get_candidate_symptoms(
            state.initial_cui,
            limit=10,
            confirmed_cuis=state.confirmed_cuis.copy(),
            denied_cuis=state.denied_cuis.copy(),
            asked_cuis=state.asked_cuis.copy(),
        )

        if not candidates:
            self._make_diagnosis(state)
            return

        # Top-1 선택
        selected = candidates[0]
        selected_cui = selected.cui
        selected_name = selected.name

        # 3. 환자 응답 시뮬레이션
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
        """최종 진단."""
        candidates = self.kg.get_diagnosis_candidates(
            top_k=100,
            scoring=self.scoring,
            confirmed_cuis=state.confirmed_cuis.copy(),
            denied_cuis=state.denied_cuis.copy(),
        )

        # DDXPlus 질환만 필터링
        for c in candidates:
            if c.cui in self._cui_to_disease:
                state.predicted = self._cui_to_disease[c.cui]
                break

        if not state.predicted:
            # fallback
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

    def _calculate_metrics(self, states: list[PatientState], elapsed: float) -> dict:
        """메트릭 계산."""
        total = len(states)
        correct = sum(1 for s in states if s.predicted == s.patient.pathology)
        total_il = sum(s.il for s in states)

        gtpa_at_1 = correct / total if total > 0 else 0
        avg_il = total_il / total if total > 0 else 0

        print(f"\n{'='*70}", flush=True)
        print("KG-ONLY BENCHMARK RESULTS", flush=True)
        print("=" * 70, flush=True)
        print(f"Samples: {total:,}", flush=True)
        print(f"GTPA@1: {gtpa_at_1:.2%} (KG Top-1 정확도)", flush=True)
        print(f"Avg IL: {avg_il:.2f}", flush=True)
        print(f"Time: {elapsed/60:.1f} min", flush=True)
        print(f"Throughput: {total / (elapsed/60):.1f} samples/min", flush=True)
        print("=" * 70, flush=True)
        print(f"\nvs AARLC (DDXPlus paper):", flush=True)
        print(f"  GTPA@1: {gtpa_at_1:.2%} vs 75.39% ({(gtpa_at_1-0.7539)*100:+.2f}%)", flush=True)
        print(f"  IL:     {avg_il:.2f} vs 25.75 ({avg_il-25.75:+.2f})", flush=True)

        # 결과 저장
        result = {
            "mode": "kg_only",
            "samples": total,
            "gtpa_at_1": gtpa_at_1,
            "avg_il": avg_il,
            "elapsed_seconds": elapsed,
            "scoring": self.scoring,
            "severity": self.severity,
        }

        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "kg_only_benchmark.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_file}", flush=True)

        return result

    def close(self):
        """리소스 정리."""
        if self.kg:
            self.kg.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG-Only Benchmark")
    parser.add_argument("-n", "--n-samples", type=int, default=1000)
    parser.add_argument("--severity", type=int, default=2, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--scoring", type=str, default="v18_coverage")

    args = parser.parse_args()

    print(f"KG-Only Benchmark: {args.n_samples:,} samples, severity={args.severity}")

    benchmark = KGOnlyBenchmark(
        scoring=args.scoring,
        severity=args.severity,
    )

    try:
        result = benchmark.run(args.n_samples)
    finally:
        benchmark.close()
