#!/usr/bin/env python3
"""진단 Top-K 최적화 실험.

증상 K=3 고정, 진단 K만 변경하여 최적값 탐색.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/tune_diagnosis_topk.py -n 500
"""

import argparse
import json
import re
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
    """환자별 대화 상태."""
    patient: Patient
    idx: int
    confirmed_symptoms: list[str] = field(default_factory=list)
    denied_symptoms: list[str] = field(default_factory=list)
    asked_codes: set[str] = field(default_factory=set)
    il: int = 0
    done: bool = False
    predicted: str | None = None
    confirmed_cuis: set[str] = field(default_factory=set)
    denied_cuis: set[str] = field(default_factory=set)
    asked_cuis: set[str] = field(default_factory=set)
    initial_cui: str | None = None

    def __post_init__(self):
        self.asked_codes.add(self.patient.initial_evidence)


class DiagnosisTopKTuner:
    """진단 Top-K 최적화 실험."""

    MAX_QUESTIONS = 50
    MIN_QUESTIONS = 3
    SYMPTOM_K = 3  # 증상 K는 3으로 고정

    def __init__(self, model: str, diagnosis_k: int):
        self.model = model
        self.diagnosis_k = diagnosis_k
        self.loader = DDXPlusLoader()

        # KG 초기화
        print(f"Connecting to Neo4j KG...", flush=True)
        self.kg = UMLSKG()
        print("KG connected!", flush=True)

        # CUI 매핑
        self._cui_to_codes = self.loader.build_cui_to_codes()
        self._cui_to_disease: dict[str, str] = {}
        self._disease_to_cui: dict[str, str] = {}
        self._build_disease_cui_mapping()

        # vLLM 엔진 초기화
        self._init_vllm_engine()

    def _build_disease_cui_mapping(self):
        """질환 CUI 매핑."""
        for name_eng, info in self.loader.disease_mapping.items():
            cui = info.get("umls_cui")
            if cui:
                cond = self.loader.conditions.get(name_eng)
                if cond:
                    self._cui_to_disease[cui] = cond.name_fr
                    self._disease_to_cui[cond.name_fr] = cui

    def _init_vllm_engine(self):
        """vLLM 엔진 초기화."""
        from vllm import LLM, SamplingParams

        print(f"Loading model: {self.model}", flush=True)

        self.llm = LLM(
            model=self.model,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
        )

        self.sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=128,
            stop=["\n", "</s>", "<|endoftext|>"],
        )

        print("Model loaded!", flush=True)

    def run(self, n_samples: int, severity: int = 2) -> dict:
        """벤치마크 실행."""
        patients = self.loader.load_patients(
            split="test",
            n_samples=n_samples,
            severity=severity,
        )

        print(f"\nSymptom K={self.SYMPTOM_K} (fixed), Diagnosis K={self.diagnosis_k}, Samples={len(patients):,}", flush=True)

        # 환자 상태 초기화
        states = [PatientState(patient=p, idx=i) for i, p in enumerate(patients)]

        # 초기 증상 설정
        for state in states:
            init_name = self.loader.symptom_mapping.get(
                state.patient.initial_evidence, {}
            ).get("name", state.patient.initial_evidence)
            state.confirmed_symptoms.append(init_name)

            # CUI 초기화
            init_cui = self.loader.get_symptom_cui(state.patient.initial_evidence)
            if init_cui:
                state.initial_cui = init_cui
                state.confirmed_cuis.add(init_cui)
                state.asked_cuis.add(init_cui)

        start_time = time.time()
        pbar = tqdm(total=len(states), desc=f"DxK={self.diagnosis_k}", unit="p")

        # 라운드별 배치 처리
        round_num = 0
        while True:
            active_states = [s for s in states if not s.done]
            if not active_states:
                break

            round_num += 1
            self._process_round(active_states)

            # 진행 상황 업데이트
            done_count = sum(1 for s in states if s.done)
            newly_done = done_count - pbar.n
            if newly_done > 0:
                pbar.update(newly_done)

            # 실시간 메트릭
            completed = [s for s in states if s.done]
            if completed:
                acc = sum(1 for s in completed if s.predicted == s.patient.pathology) / len(completed)
                avg_il = sum(s.il for s in states) / len(states)
                pbar.set_postfix({"GTPA@1": f"{acc:.1%}", "IL": f"{avg_il:.1f}"})

        pbar.close()

        # 결과 계산
        elapsed = time.time() - start_time
        return self._compute_results(states, elapsed)

    def _process_round(self, active_states: list[PatientState]):
        """라운드 처리."""
        question_states = []
        diagnosis_states = []

        # 종료 조건 분류
        for state in active_states:
            if state.il >= self.MAX_QUESTIONS:
                diagnosis_states.append(state)
            elif self._should_stop(state):
                diagnosis_states.append(state)
            else:
                question_states.append(state)

        # 배치 1: 증상 질문 (K=3 고정)
        if question_states:
            prompts = []
            candidates_list = []
            valid_states = []

            for state in question_states:
                candidates = self._get_symptom_candidates(state)
                if candidates:
                    candidates_list.append(candidates)
                    prompt = self._build_symptom_prompt(state, candidates)
                    prompts.append(prompt)
                    valid_states.append(state)
                else:
                    diagnosis_states.append(state)

            if prompts:
                responses = self._batch_generate(prompts)
                for state, response, candidates in zip(valid_states, responses, candidates_list):
                    self._handle_symptom_response(state, response, candidates)

        # 배치 2: 진단 (K=diagnosis_k)
        if diagnosis_states:
            prompts = []
            candidates_list = []

            for state in diagnosis_states:
                dx_candidates = self._get_diagnosis_candidates(state)
                candidates_list.append(dx_candidates)
                prompt = self._build_diagnosis_prompt(state, dx_candidates)
                prompts.append(prompt)

            responses = self._batch_generate(prompts)
            for state, response, dx_candidates in zip(diagnosis_states, responses, candidates_list):
                self._handle_diagnosis_response(state, response, dx_candidates)

    def _should_stop(self, state: PatientState) -> bool:
        """종료 조건 확인."""
        if state.il < self.MIN_QUESTIONS:
            return False

        self.kg.state.confirmed_cuis = state.confirmed_cuis.copy()
        self.kg.state.denied_cuis = state.denied_cuis.copy()
        self.kg.state.asked_cuis = state.asked_cuis.copy()

        should_stop, _ = self.kg.should_stop(max_il=self.MAX_QUESTIONS)
        return should_stop

    def _get_symptom_candidates(self, state: PatientState) -> list[tuple[str, str, int]]:
        """증상 후보 가져오기."""
        if not state.initial_cui:
            return []

        self.kg.state.confirmed_cuis = state.confirmed_cuis.copy()
        self.kg.state.denied_cuis = state.denied_cuis.copy()
        self.kg.state.asked_cuis = state.asked_cuis.copy()

        candidates = self.kg.get_candidate_symptoms(state.initial_cui, limit=10)
        return [(c.cui, c.name, c.disease_coverage) for c in candidates]

    def _get_diagnosis_candidates(self, state: PatientState) -> list[tuple[str, str, float]]:
        """진단 후보 가져오기."""
        self.kg.state.confirmed_cuis = state.confirmed_cuis.copy()
        self.kg.state.denied_cuis = state.denied_cuis.copy()
        self.kg.state.asked_cuis = state.asked_cuis.copy()

        candidates = self.kg.get_diagnosis_candidates(top_k=100, scoring="v18_coverage")

        # DDXPlus 질환만 필터링
        filtered = []
        for c in candidates:
            if c.cui in self._cui_to_disease:
                filtered.append((c.cui, c.name, c.score))

        return filtered

    def _build_symptom_prompt(self, state: PatientState, candidates: list[tuple[str, str, int]]) -> str:
        """증상 프롬프트 (K=3 고정)."""
        patient = state.patient
        init_name = self.loader.symptom_mapping.get(
            patient.initial_evidence, {}
        ).get("name", patient.initial_evidence)

        # 증상은 K=3 고정
        top_candidates = candidates[:self.SYMPTOM_K]
        total_coverage = sum(cov for _, _, cov in top_candidates) or 1
        symptom_list = "\n".join(
            [f"{i+1}. {name} ({cov/total_coverage:.0%})" for i, (_, name, cov) in enumerate(top_candidates)]
        )

        prompt = f"""Select the best symptom to ask. Options ranked by diagnostic value (1=highest):

{symptom_list}

Patient: {patient.sex}, {patient.age}, chief complaint: {init_name}

Option 1 has the highest diagnostic value. Select 1 unless there is a specific clinical reason for another option.
Answer (1-{self.SYMPTOM_K}):"""
        return prompt

    def _build_diagnosis_prompt(self, state: PatientState, dx_candidates: list[tuple[str, str, float]]) -> str:
        """진단 프롬프트 (K=diagnosis_k)."""
        patient = state.patient
        init_name = self.loader.symptom_mapping.get(
            patient.initial_evidence, {}
        ).get("name", patient.initial_evidence)

        # 진단은 diagnosis_k 적용
        display_candidates = dx_candidates[:self.diagnosis_k] if dx_candidates else []

        if display_candidates:
            total_score = sum(score for _, _, score in display_candidates) or 1
            dx_list = "\n".join(
                [f"{i+1}. {name} ({score/total_score:.0%})" for i, (_, name, score) in enumerate(display_candidates)]
            )
        else:
            dx_list = "1. Unknown"

        prompt = f"""Select the most likely diagnosis. Options ranked by probability (1=highest):

{dx_list}

Patient: {patient.sex}, {patient.age}, chief complaint: {init_name}
Confirmed: {', '.join(state.confirmed_symptoms[:5]) if state.confirmed_symptoms else 'None'}

Option 1 has the highest probability. Select 1 unless there is a specific clinical reason for another option.
Answer (1-{min(self.diagnosis_k, len(display_candidates)) if display_candidates else 1}):"""
        return prompt

    def _handle_symptom_response(self, state: PatientState, response: str, candidates: list[tuple[str, str, int]]):
        """증상 응답 처리."""
        top_candidates = candidates[:self.SYMPTOM_K]
        numbers = re.findall(r"\d+", response)
        selected_cui = None
        selected_name = None

        if numbers:
            idx = int(numbers[0]) - 1
            if 0 <= idx < len(top_candidates):
                selected_cui, selected_name, _ = top_candidates[idx]

        if not selected_cui and top_candidates:
            selected_cui, selected_name, _ = top_candidates[0]

        if not selected_cui:
            state.done = True
            return

        state.asked_cuis.add(selected_cui)
        state.il += 1

        # 환자 응답 확인
        ddxplus_codes = self._cui_to_codes.get(selected_cui, [])
        has_symptom = False

        for code in ddxplus_codes:
            if self._check_evidence(state.patient, code):
                has_symptom = True
                state.asked_codes.add(code)
                break

        if has_symptom:
            state.confirmed_cuis.add(selected_cui)
            state.confirmed_symptoms.append(selected_name)
        else:
            state.denied_cuis.add(selected_cui)
            state.denied_symptoms.append(selected_name)

    def _handle_diagnosis_response(self, state: PatientState, response: str, dx_candidates: list[tuple[str, str, float]]):
        """진단 응답 처리."""
        numbers = re.findall(r"\d+", response)
        display_candidates = dx_candidates[:self.diagnosis_k] if dx_candidates else []

        if dx_candidates:
            if numbers:
                idx = int(numbers[0]) - 1
                if 0 <= idx < len(display_candidates):
                    cui, _, _ = display_candidates[idx]
                    state.predicted = self._cui_to_disease.get(cui)

            if not state.predicted and display_candidates:
                cui, _, _ = display_candidates[0]
                state.predicted = self._cui_to_disease.get(cui)

        if not state.predicted:
            state.predicted = list(self.loader.conditions.values())[0].name_fr

        state.done = True

    def _batch_generate(self, prompts: list[str]) -> list[str]:
        """vLLM 배치 생성."""
        if not prompts:
            return []

        outputs = self.llm.generate(prompts, self.sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]

    def _check_evidence(self, patient: Patient, code: str) -> bool:
        """환자 증상 확인."""
        evidences = set(patient.evidences)

        if code in evidences:
            return True

        for ev in evidences:
            if ev.startswith(f"{code}_@_"):
                return True

        return False

    def _compute_results(self, states: list[PatientState], elapsed: float) -> dict:
        """결과 계산."""
        total = len(states)
        correct_count = sum(1 for s in states if s.predicted == s.patient.pathology)
        total_il = sum(s.il for s in states)

        gtpa_at_1 = correct_count / total
        avg_il = total_il / total

        return {
            "symptom_k": self.SYMPTOM_K,
            "diagnosis_k": self.diagnosis_k,
            "n_samples": total,
            "gtpa_at_1": gtpa_at_1,
            "avg_il": avg_il,
            "elapsed_minutes": elapsed / 60,
        }

    def close(self):
        """리소스 정리."""
        if self.kg:
            self.kg.close()


def run_diagnosis_topk_experiment(model: str, n_samples: int, k_values: list[int]) -> list[dict]:
    """진단 Top-K 실험 실행."""
    results = []

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Testing Diagnosis K = {k} (Symptom K = 3 fixed)")
        print("=" * 60)

        tuner = DiagnosisTopKTuner(model=model, diagnosis_k=k)
        result = tuner.run(n_samples)
        results.append(result)
        tuner.close()

        # GPU 메모리 정리
        import gc
        import torch
        del tuner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def print_results(results: list[dict]):
    """결과 출력."""
    print(f"\n{'='*70}")
    print("Diagnosis Top-K Optimization Results (Symptom K=3 fixed)")
    print("=" * 70)
    print(f"{'Dx K':>6} {'GTPA@1':>10} {'Avg IL':>10} {'Time':>10}")
    print("-" * 70)

    best_k = None
    best_score = 0

    for r in results:
        k = r["diagnosis_k"]
        gtpa = r["gtpa_at_1"]
        il = r["avg_il"]
        time_min = r["elapsed_minutes"]

        # Best: GTPA@1 최대화
        if gtpa > best_score:
            best_score = gtpa
            best_k = k

        marker = " *" if k == best_k else ""
        print(f"{k:>6} {gtpa:>9.1%} {il:>10.1f} {time_min:>9.1f}m{marker}")

    print("-" * 70)
    print(f"* Best Diagnosis K = {best_k} (highest GTPA@1)")

    # JSON 저장
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "diagnosis_topk_optimization.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnosis Top-K Optimization Experiment")
    parser.add_argument("-n", "--n-samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model to use")
    parser.add_argument("--k-values", type=str, default="1,2,3,4,5,6,8,10", help="Diagnosis K values to test")

    args = parser.parse_args()

    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    print(f"Symptom K: 3 (fixed)")
    print(f"Diagnosis K values to test: {k_values}")
    print(f"Model: {args.model}")
    print(f"Samples: {args.n_samples}")

    results = run_diagnosis_topk_experiment(args.model, args.n_samples, k_values)
    print_results(results)
