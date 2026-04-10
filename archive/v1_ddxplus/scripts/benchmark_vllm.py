#!/usr/bin/env python3
"""DDXPlus 벤치마크 (vLLM 배치 처리).

2개 실험군:
- Category 1: 소형 LLM only (vLLM) - baseline
- Category 2: 소형 LLM + KG (vLLM + Neo4j) - 본 연구의 제안 방법

Usage:
    # 기본: 모든 모델 + Category 1, 2 실행 (가변 IL)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_vllm.py --category 1,2 -n 10000

    # Category 2만 (Small LLM + KG)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_vllm.py --category 2 -n 10000

    # MEDDxAgent 비교: 고정 IL (5, 10, 15)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_vllm.py --category 2 --max-il 5 -n 10000
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_vllm.py --category 2 --max-il 10 -n 10000
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_vllm.py --category 2 --max-il 15 -n 10000

    # nohup 백그라운드 실행
    CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts/benchmark_vllm.py -n 10000 > benchmark.log 2>&1 &
    tail -f benchmark.log
"""

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from dotenv import load_dotenv
from tqdm import tqdm

# .env 파일 로드 (프로젝트 루트)
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DDXPlusLoader, Patient
from src.patient_simulator import PatientSimulator, ResponseType
from src.umls_kg import UMLSKG, KGState

from scripts.models_config import ALL_MODELS

# 실행별 출력 디렉토리 (main에서 설정)
RUN_OUTPUT_DIR = None  # type: Path | None


def get_output_dir() -> Path:
    """현재 실행의 출력 디렉토리 반환."""
    if RUN_OUTPUT_DIR is None:
        # fallback: 기본 results 디렉토리
        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(exist_ok=True)
        return output_dir
    return RUN_OUTPUT_DIR


@dataclass
class InteractionLog:
    """LLM + KG 상호작용 로그 (논문 분석용)."""

    round: int

    # KG Input/Output
    kg_input_confirmed: list[str] = field(default_factory=list)  # confirmed CUIs
    kg_input_denied: list[str] = field(default_factory=list)  # denied CUIs
    kg_candidate_symptoms: list[dict] = field(default_factory=list)  # [{symptom, cui, score}, ...]
    kg_diagnosis_scores: list[dict] = field(default_factory=list)  # [{disease, cui, score}, ...]
    kg_stop_decision: bool = False
    kg_confidence: float = 0.0

    # LLM Input/Output
    llm_prompt: str = ""
    llm_response: str = ""
    llm_selected_symptom: str = ""
    llm_selected_cui: str = ""

    # Patient Response
    patient_answer: bool | None = None  # True=Yes, False=No, None=진단 라운드

    # Timing
    kg_time_ms: float = 0.0
    llm_time_ms: float = 0.0


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
    predicted_dd: list[tuple[str, float]] = field(default_factory=list)  # LLM 선택 진단 [(disease, score), ...]
    kg_predicted_dd: list[tuple[str, float]] = field(default_factory=list)  # KG가 생성한 감별진단 리스트 (DDF1 평가용)

    # LLM+KG 모드용 추가 필드
    confirmed_cuis: set[str] = field(default_factory=set)
    denied_cuis: set[str] = field(default_factory=set)
    asked_cuis: set[str] = field(default_factory=set)
    initial_cui: str | None = None

    # 상호작용 로그 (논문 분석용)
    interaction_logs: list[InteractionLog] = field(default_factory=list)

    def __post_init__(self):
        self.asked_codes.add(self.patient.initial_evidence)


class VLLMBenchmark:
    """vLLM 기반 벤치마크 실행기.

    Category 1 (small_llm): LLM만 사용
        - 증상 선택: 전체 증상 목록(200+)에서 LLM이 선택
        - 진단 타이밍: MAX_QUESTIONS 도달 시 강제 진단
        - 기대 성능: 낮음 (baseline)

    Category 2 (small_llm_kg): LLM + KG 결합 (본 연구 제안 방법)
        - KG 역할 1: 후보 증상 제안 (Top-10, 관련성 높은 증상만)
        - KG 역할 2: 진단 타이밍 결정 (confidence 기반 조기 종료)
        - LLM 역할: KG 후보 중 선택, 최종 진단 결정
        - 기대 성능: 높음 (AARLC 능가)
    """

    # 고정 설정값
    MAX_QUESTIONS = 50   # 안전 상한 (KG가 조기 종료 결정)
    MIN_QUESTIONS = 3    # 최소 질문 수 (조기 종료 방지)
    TENSOR_PARALLEL_SIZE = 1
    GPU_MEMORY_UTILIZATION = 0.9  # 단독 GPU 사용 시

    def __init__(
        self,
        model: str,
        mode: Literal["small_llm", "small_llm_kg"] = "small_llm",
        max_il: int | None = None,  # MEDDxAgent 비교용: 5, 10, 15 고정 또는 None(가변)
        scoring: str = "v23_mild_denied",  # 진단 스코어링: "v23_mild_denied" (89.2%), "v18_coverage" (88.4%), "v15_ratio" (85%)
        severity: int | None = None,  # 질환 심각도 필터 (1-5, None이면 전체), 2=moderate (시뮬레이션과 동일)
        split: str = "test",  # 데이터 split: "test" (시뮬레이션과 동일) 또는 "validate"
        kg_only_diagnosis: bool = False,  # KG Top-1 직접 사용 (LLM 진단 선택 바이패스)
        top_n: int = 3,  # LLM에 제시할 후보 개수 (증상/진단 선택 시)
        reason_tokens: int = 1024,  # Stage 1 추론 max_tokens
        shuffle_candidates: bool = False,  # 후보 순서 랜덤 셔플 (position bias 테스트용)
    ):
        self.model = model
        self.mode = mode
        self.scoring = scoring
        self.severity = severity
        self.split = split
        self.kg_only_diagnosis = kg_only_diagnosis
        self.top_n = top_n
        self.reason_tokens = reason_tokens
        self.shuffle_candidates = shuffle_candidates
        # max_il이 지정되면 해당 값 사용, 아니면 MAX_QUESTIONS (가변 모드)
        self.max_questions = max_il if max_il is not None else self.MAX_QUESTIONS
        self.max_il_mode = "fixed" if max_il is not None else "adaptive"

        self.loader = DDXPlusLoader()
        self._symptom_list = self._build_symptom_list()
        self._disease_list = self._build_disease_list()

        # UMLS KG (small_llm_kg 모드용)
        self.kg: UMLSKG | None = None
        if mode == "small_llm_kg":
            self._init_kg()

        # CUI ↔ DDXPlus 역매핑
        self._cui_to_codes = self.loader.build_cui_to_codes()
        self._cui_to_disease: dict[str, str] = {}
        self._disease_to_cui: dict[str, str] = {}
        self._build_disease_cui_mapping()

        print(f"Disease CUI mappings: {len(self._cui_to_disease)} / 49 diseases", flush=True)

        # vLLM 엔진 초기화
        self._init_vllm_engine()

    def _build_symptom_list(self) -> list[tuple[str, str]]:
        """증상 목록 (code, name)."""
        symptoms = []
        for code, info in self.loader.symptom_mapping.items():
            name = info.get("name", code)
            symptoms.append((code, name))
        return sorted(symptoms, key=lambda x: x[1])

    def _build_disease_list(self) -> list[tuple[str, str]]:
        """질환 목록 (name_fr, name_eng)."""
        diseases = []
        for name, cond in self.loader.conditions.items():
            diseases.append((cond.name_fr, cond.name_eng))
        return sorted(diseases, key=lambda x: x[1])

    def _build_disease_cui_mapping(self):
        """질환 CUI ↔ 이름 매핑."""
        for name_eng, info in self.loader.disease_mapping.items():
            cui = info.get("umls_cui")
            if cui:
                cond = self.loader.conditions.get(name_eng)
                if cond:
                    self._cui_to_disease[cui] = cond.name_fr
                    self._disease_to_cui[cond.name_fr] = cui

    def _init_kg(self):
        """UMLS KG 초기화."""
        print("Connecting to Neo4j KG...", flush=True)
        try:
            self.kg = UMLSKG()
            print("KG connected!", flush=True)
        except Exception as e:
            print(f"KG connection failed: {e}", flush=True)
            print("Falling back to small_llm mode", flush=True)
            self.mode = "small_llm"
            self.kg = None

    def _init_vllm_engine(self):
        """vLLM 엔진 초기화."""
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        print(f"Loading model: {self.model}", flush=True)

        # max_model_len을 reason_tokens에 맞게 동적 설정
        # prompt (~500 tokens) + reason_tokens + buffer
        max_model_len = max(4096, self.reason_tokens + 2048)

        self.llm = LLM(
            model=self.model,
            trust_remote_code=True,
            tensor_parallel_size=self.TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=self.GPU_MEMORY_UTILIZATION,
            max_model_len=max_model_len,
        )

        # Two-Stage Prompting: 이유 생성 + 숫자 선택
        # 모든 모델에 동일하게 적용
        # Stage 1: 이유 생성 (자유 형식, 프롬프트로 간결함 요청)
        # stop tokens 없이 max_tokens만으로 제한 (모든 모델 동일)
        self.sampling_params_reason = SamplingParams(
            temperature=0.0,
            max_tokens=self.reason_tokens,  # Stage 1 추론 토큰 수
        )

        # Stage 2: 숫자 선택 (regex로 강제)
        if self.top_n <= 9:
            regex_pattern = f"[1-{self.top_n}]"
        else:
            regex_pattern = "(" + "|".join(str(i) for i in range(1, self.top_n + 1)) + ")"
        structured_params = StructuredOutputsParams(regex=regex_pattern)
        max_tokens = 2 if self.top_n >= 10 else 1

        self.sampling_params_select = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            structured_outputs=structured_params,
        )
        print(f"Two-Stage enabled: reason({self.reason_tokens} tokens) + select(regex='{regex_pattern}')", flush=True)

        print("Model loaded!", flush=True)

    def run(self, n_samples: int) -> dict:
        """벤치마크 실행."""
        patients = self.loader.load_patients(
            split=self.split,
            n_samples=n_samples,
            severity=self.severity,
        )

        category = "1 (Small LLM)" if self.mode == "small_llm" else "2 (Small LLM + KG)"

        print(f"\n{'='*70}", flush=True)
        print(f"Category: {category}", flush=True)
        print(f"Model: {self.model}", flush=True)
        print(f"Samples: {len(patients):,}", flush=True)
        print(f"Max questions: {self.max_questions} ({self.max_il_mode})", flush=True)
        print("=" * 70, flush=True)

        # 환자 상태 초기화
        states = [PatientState(patient=p, idx=i) for i, p in enumerate(patients)]

        # 초기 증상 설정
        for state in states:
            init_name = self.loader.symptom_mapping.get(
                state.patient.initial_evidence, {}
            ).get("name", state.patient.initial_evidence)
            state.confirmed_symptoms.append(init_name)

            # small_llm_kg 모드: CUI 초기화
            if self.mode == "small_llm_kg":
                init_cui = self.loader.get_symptom_cui(state.patient.initial_evidence)
                if init_cui:
                    state.initial_cui = init_cui
                    state.confirmed_cuis.add(init_cui)
                    state.asked_cuis.add(init_cui)

        start_time = time.time()

        # tqdm 진행바 설정 (완료 환자 수 기준)
        pbar = tqdm(
            total=len(states),
            desc="Patients",
            unit="p",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )

        # 라운드별 배치 처리
        round_num = 0
        while True:
            active_states = [s for s in states if not s.done]

            if not active_states:
                break

            round_num += 1

            if self.mode == "small_llm":
                self._process_round_small_llm(active_states)
            else:
                self._process_round_small_llm_kg(active_states)

            # 진행 상황 업데이트 (완료 환자 수 기준)
            current_total_il = sum(s.il for s in states)
            done_count = sum(1 for s in states if s.done)
            newly_done = done_count - pbar.n
            if newly_done > 0:
                pbar.update(newly_done)

            # 실시간 메트릭 표시
            avg_il = current_total_il / len(states) if states else 0
            completed = [s for s in states if s.done]
            if completed:
                acc = sum(1 for s in completed if s.predicted == s.patient.pathology) / len(completed)
                pbar.set_postfix({
                    "round": round_num,
                    "avg_IL": f"{avg_il:.1f}",
                    "GTPA@1": f"{acc:.1%}",
                })
            else:
                pbar.set_postfix({"round": round_num, "avg_IL": f"{avg_il:.1f}"})

        pbar.close()

        # 최종 결과 계산
        elapsed = time.time() - start_time
        results = self._compute_results(states, elapsed)

        return results

    def _process_round_small_llm(self, active_states: list[PatientState]):
        """Category 1: 소형 LLM only 모드 라운드 처리.

        KG 없이 LLM만으로 진단:
        - 증상 선택: 전체 목록(200+)에서 LLM이 직접 선택 (비효율적)
        - 진단 타이밍: MAX_QUESTIONS 도달 시 강제 진단 (스스로 판단 불가)
        - 예상 결과: 낮은 정확도 (baseline으로 KG 효과 입증용)
        """
        # MAX_QUESTIONS 미만: 계속 질문
        # MAX_QUESTIONS 이상: 진단 단계로 전환
        question_states = [s for s in active_states if s.il < self.max_questions]
        diagnosis_states = [s for s in active_states if s.il >= self.max_questions]

        # 배치 1: 증상 질문 (전체 증상 목록에서 선택)
        if question_states:
            prompts = [self._build_symptom_prompt_small_llm(s) for s in question_states]
            responses = self._batch_generate(prompts)

            for state, response in zip(question_states, responses):
                self._handle_symptom_response_small_llm(state, response)

        # 배치 2: 최종 진단
        if diagnosis_states:
            prompts = [self._build_diagnosis_prompt_small_llm(s) for s in diagnosis_states]
            responses = self._batch_generate(prompts)

            for state, response in zip(diagnosis_states, responses):
                self._handle_diagnosis_response_small_llm(state, response)

    def _process_round_small_llm_kg(self, active_states: list[PatientState]):
        """Category 2: 소형 LLM + KG 모드 라운드 처리.

        KG의 두 가지 핵심 역할:
        1. 후보 증상 제안: 2-hop 탐색으로 관련성 높은 증상 Top-10 제공
        2. 진단 타이밍 결정: confidence 기반 조기 종료 판단

        LLM의 역할:
        - KG가 제안한 증상 중 최적 선택
        - KG가 제안한 진단 후보 중 최종 결정
        """
        question_states = []
        diagnosis_states = []

        # [KG 역할 2] 진단 타이밍 결정
        for state in active_states:
            if state.il >= self.max_questions:
                # 안전장치: 최대 질문 수 도달
                diagnosis_states.append(state)
            elif self._should_stop_kg(state):
                # KG 판단: confidence 충분, 진단 가능
                diagnosis_states.append(state)
            else:
                question_states.append(state)

        # 배치 1: 증상 질문 (KG가 후보 제안)
        if question_states:
            prompts = []
            candidates_list = []
            valid_question_states = []
            interaction_logs = []

            # [KG 역할 1] 후보 증상 병렬 조회 (ThreadPoolExecutor)
            with ThreadPoolExecutor(max_workers=8) as executor:
                all_candidates = list(executor.map(
                    self._get_kg_symptom_candidates, question_states
                ))

            for state, candidates in zip(question_states, all_candidates):
                if candidates:
                    # 후보 순서 셔플 (position bias 테스트용)
                    if self.shuffle_candidates:
                        candidates = candidates[:self.top_n]  # Top-N만 셔플
                        candidates = random.sample(candidates, len(candidates))
                    candidates_list.append(candidates)
                    prompt = self._build_symptom_prompt_small_llm_kg(state, candidates)
                    prompts.append(prompt)
                    valid_question_states.append(state)

                    # 상호작용 로그 생성 (진단 후보는 나중에 병렬로 조회)
                    log = InteractionLog(
                        round=state.il + 1,
                        kg_input_confirmed=list(state.confirmed_cuis),
                        kg_input_denied=list(state.denied_cuis),
                        kg_candidate_symptoms=[
                            {"symptom": name, "cui": cui, "coverage": cov}
                            for cui, name, cov in candidates
                        ],
                        kg_diagnosis_scores=[],  # 나중에 채움
                        kg_stop_decision=False,
                    )
                    interaction_logs.append(log)
                else:
                    # 더 이상 질문할 증상 없음 → 진단으로 전환
                    diagnosis_states.append(state)

            # 진단 후보 병렬 조회 (로그용)
            if valid_question_states:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    dx_candidates_list = list(executor.map(
                        self._get_kg_diagnosis_candidates, valid_question_states
                    ))
                for log, dx_candidates in zip(interaction_logs, dx_candidates_list):
                    log.kg_diagnosis_scores = [
                        {"disease": name, "cui": cui, "score": score}
                        for cui, name, score in dx_candidates[:5]
                    ]

            if prompts:
                # LLM이 KG 후보 중에서 선택
                responses = self._batch_generate(prompts)
                for state, response, candidates, prompt, log in zip(
                    valid_question_states, responses, candidates_list, prompts, interaction_logs
                ):
                    self._handle_symptom_response_small_llm_kg(
                        state, response, candidates, prompt=prompt, interaction_log=log
                    )

        # 배치 2: 최종 진단 (KG가 후보 제안, LLM이 최종 결정)
        if diagnosis_states:
            prompts = []
            interaction_logs = []

            # KG가 진단 후보 병렬 조회 (점수 기반 Top-10)
            with ThreadPoolExecutor(max_workers=8) as executor:
                candidates_list = list(executor.map(
                    self._get_kg_diagnosis_candidates, diagnosis_states
                ))

            # 후보 순서 셔플 (position bias 테스트용)
            if self.shuffle_candidates:
                shuffled_candidates_list = []
                for dx_candidates in candidates_list:
                    dx_top_n = dx_candidates[:self.top_n]
                    shuffled = random.sample(dx_top_n, len(dx_top_n))
                    shuffled_candidates_list.append(shuffled)
                candidates_list = shuffled_candidates_list

            for state, dx_candidates in zip(diagnosis_states, candidates_list):
                prompt = self._build_diagnosis_prompt_small_llm_kg(state, dx_candidates)
                prompts.append(prompt)

                # 진단 상호작용 로그 생성
                log = InteractionLog(
                    round=state.il + 1,
                    kg_input_confirmed=list(state.confirmed_cuis),
                    kg_input_denied=list(state.denied_cuis),
                    kg_candidate_symptoms=[],  # 진단 라운드에서는 증상 후보 없음
                    kg_diagnosis_scores=[
                        {"disease": name, "cui": cui, "score": score}
                        for cui, name, score in dx_candidates[:10]
                    ],
                    kg_stop_decision=True,
                    kg_confidence=dx_candidates[0][2] if dx_candidates else 0.0,
                )
                interaction_logs.append(log)

            # LLM이 최종 진단 결정
            responses = self._batch_generate(prompts)
            for state, response, dx_candidates, prompt, log in zip(
                diagnosis_states, responses, candidates_list, prompts, interaction_logs
            ):
                self._handle_diagnosis_response_small_llm_kg(
                    state, response, dx_candidates, prompt=prompt, interaction_log=log
                )

    def _should_stop_kg(self, state: PatientState) -> bool:
        """[KG 역할 2] 진단 타이밍 결정.

        종료 조건 (OR):
        1. Top-1 confidence ≥ 0.8
        2. Top-1과 Top-2 격차 ≥ 0.3
        3. 단일 질환만 남음
        4. 더 이상 질문할 증상 없음

        NOTE: should_stop은 내부적으로 get_diagnosis_candidates를 호출하며,
        self.state를 수정해야 함. 병렬화는 _process_round_small_llm_kg에서 처리.
        """
        if not self.kg or state.il < self.MIN_QUESTIONS:
            return False

        # should_stop은 self.state를 사용하므로 임시로 설정
        # (이 메서드는 순차적으로 호출되므로 문제없음)
        self.kg.state.confirmed_cuis = state.confirmed_cuis.copy()
        self.kg.state.denied_cuis = state.denied_cuis.copy()
        self.kg.state.asked_cuis = state.asked_cuis.copy()

        should_stop, _ = self.kg.should_stop(max_il=self.max_questions)
        return should_stop

    def _get_kg_symptom_candidates(self, state: PatientState) -> list[tuple[str, str, int]]:
        """[KG 역할 1] 후보 증상 제안.

        2-hop 탐색: 주호소 → 관련 질환 → 감별 증상
        결과: (CUI, 이름, disease_coverage) Top-10

        NOTE: 스레드 안전 - self.kg.state를 수정하지 않고 파라미터로 전달
        """
        if not self.kg or not state.initial_cui:
            return []

        # 스레드 안전: 상태를 파라미터로 전달
        candidates = self.kg.get_candidate_symptoms(
            state.initial_cui,
            limit=10,
            confirmed_cuis=state.confirmed_cuis.copy(),
            denied_cuis=state.denied_cuis.copy(),
            asked_cuis=state.asked_cuis.copy(),
        )
        return [(c.cui, c.name, c.disease_coverage) for c in candidates]

    def _get_kg_diagnosis_candidates(self, state: PatientState) -> list[tuple[str, str, float]]:
        """KG에서 진단 후보 가져오기 (DDXPlus 질환만 필터링 + 누적확률 커트오프).

        학술적 근거:
        - Cumulative Probability Cutoff: 누적 확률이 95%에 도달할 때까지만 포함
        - 이는 95% Confidence Interval 개념과 유사
        - 확률이 매우 낮은 "noise" 후보들을 자연스럽게 제거

        NOTE: 스레드 안전 - self.kg.state를 수정하지 않고 파라미터로 전달
        """
        if not self.kg:
            return []

        # DDXPlus 49개 질환 전체를 커버하도록 충분히 가져옴
        # 스레드 안전: 상태를 파라미터로 전달
        candidates = self.kg.get_diagnosis_candidates(
            top_k=100,
            scoring=self.scoring,
            confirmed_cuis=state.confirmed_cuis.copy(),
            denied_cuis=state.denied_cuis.copy(),
        )

        # DDXPlus 질환만 필터링
        filtered = []
        for c in candidates:
            if c.cui in self._cui_to_disease:
                filtered.append((c.cui, c.name, c.score))

        # Adaptive Probability Cutoff for Differential Diagnosis
        # 실험 결과 (tune_cutoff.py, 500 cases):
        # - min_prob=0.02가 DDF1 최대화 (48.7%)
        # - DDR=71.9%, DDP=36.8%, 평균 21개 후보
        if filtered:
            total_prob = sum(score for _, _, score in filtered)
            if total_prob > 0:
                # Step 1: 최소 확률 임계값 (2% 미만 제외)
                min_prob_threshold = 0.02
                filtered = [(c, n, s) for c, n, s in filtered
                           if s / total_prob >= min_prob_threshold]

                # Step 2: 최소 1개 보장
                if not filtered and candidates:
                    # 필터 후 비어있으면 top-1만 포함
                    for c in candidates:
                        if c.cui in self._cui_to_disease:
                            filtered = [(c.cui, c.name, c.score)]
                            break

        return filtered

    # =========================================================================
    # 소형 LLM only 모드 (카테고리 2)
    # =========================================================================

    def _build_symptom_prompt_small_llm(self, state: PatientState) -> str:
        """소형 LLM 증상 선택 프롬프트."""
        patient = state.patient
        init_name = self.loader.symptom_mapping.get(
            patient.initial_evidence, {}
        ).get("name", patient.initial_evidence)

        available = [
            (code, name)
            for code, name in self._symptom_list
            if code not in state.asked_codes
        ][:30]

        symptom_list = "\n".join(
            [f"{i+1}. {name}" for i, (_, name) in enumerate(available)]
        )

        # H-DDx 스타일 프롬프트 (ACL 2025)
        prompt = f"""You are a medical diagnostic assistant.
Based on the patient's sex, age, and clinical evidence, select the next symptom to ask.

PATIENT:
- Sex: {patient.sex}
- Age: {patient.age}
- Chief complaint: {init_name}
- Confirmed symptoms: {', '.join(state.confirmed_symptoms[:5]) if state.confirmed_symptoms else 'None'}
- Denied symptoms: {', '.join(state.denied_symptoms[:5]) if state.denied_symptoms else 'None'}

CANDIDATE SYMPTOMS:
{symptom_list}

Based on clinical reasoning, select the most diagnostically valuable symptom.
Respond with ONLY the number (1-{len(available)}):"""
        return prompt

    def _build_diagnosis_prompt_small_llm(self, state: PatientState) -> str:
        """소형 LLM 진단 프롬프트."""
        patient = state.patient
        init_name = self.loader.symptom_mapping.get(
            patient.initial_evidence, {}
        ).get("name", patient.initial_evidence)

        disease_list = "\n".join(
            [f"{i+1}. {name}" for i, (_, name) in enumerate(self._disease_list)]
        )

        # H-DDx 스타일 프롬프트 (ACL 2025)
        prompt = f"""You are a medical diagnostic assistant.
Based on the patient's sex, age, and clinical evidence, select the most likely diagnosis.

PATIENT:
- Sex: {patient.sex}
- Age: {patient.age}
- Chief complaint: {init_name}
- Confirmed symptoms: {', '.join(state.confirmed_symptoms)}
- Denied symptoms: {', '.join(state.denied_symptoms) if state.denied_symptoms else 'None'}

DIAGNOSIS CANDIDATES:
{disease_list}

Respond with ONLY the number (1-{len(self._disease_list)}):"""
        return prompt

    def _handle_symptom_response_small_llm(self, state: PatientState, response: str):
        """소형 LLM 증상 응답 처리."""
        available = [
            (code, name)
            for code, name in self._symptom_list
            if code not in state.asked_codes
        ][:30]

        numbers = re.findall(r"\d+", response)
        selected_code = None

        if numbers:
            idx = int(numbers[0]) - 1
            if 0 <= idx < len(available):
                selected_code = available[idx][0]

        if not selected_code:
            if available:
                selected_code = available[0][0]
            else:
                state.done = True
                state.predicted = self._disease_list[0][0] if self._disease_list else None
                return

        state.asked_codes.add(selected_code)
        state.il += 1

        symptom_name = self.loader.symptom_mapping.get(selected_code, {}).get(
            "name", selected_code
        )

        has_symptom = self._check_evidence(state.patient, selected_code)

        if has_symptom:
            state.confirmed_symptoms.append(symptom_name)
        else:
            state.denied_symptoms.append(symptom_name)

    def _handle_diagnosis_response_small_llm(self, state: PatientState, response: str):
        """소형 LLM 진단 응답 처리."""
        numbers = re.findall(r"\d+", response)

        if numbers:
            idx = int(numbers[0]) - 1
            if 0 <= idx < len(self._disease_list):
                state.predicted = self._disease_list[idx][0]

        if not state.predicted:
            state.predicted = self._disease_list[0][0]

        # 감별진단 목록: LLM only 모드에서는 top-1만 저장
        if state.predicted:
            state.predicted_dd = [(state.predicted, 1.0)]

        state.done = True

    # =========================================================================
    # 소형 LLM + KG 모드 (카테고리 3)
    # =========================================================================

    def _build_symptom_prompt_small_llm_kg(
        self, state: PatientState, candidates: list[tuple[str, str, int]]
    ) -> str:
        """소형 LLM + KG 증상 선택 프롬프트."""
        patient = state.patient
        init_name = self.loader.symptom_mapping.get(
            patient.initial_evidence, {}
        ).get("name", patient.initial_evidence)

        dx_candidates = self._get_kg_diagnosis_candidates(state)
        dx_info = ""
        if dx_candidates:
            dx_list = [f"- {name} ({score:.0%})" for _, name, score in dx_candidates[:self.top_n]]
            dx_info = "\n".join(dx_list)

        # Top-N만 제시 (선택지 제한)
        top_candidates = candidates[:self.top_n]
        total_coverage = sum(cov for _, _, cov in top_candidates) or 1
        symptom_list = "\n".join(
            [f"{i+1}. {name} ({cov/total_coverage:.0%})" for i, (_, name, cov) in enumerate(top_candidates)]
        )

        # Two-Stage 프롬프트 (Stage 1: 이유 생성)
        prompt = f"""Select the most informative symptom to ask next for differential diagnosis.

Patient: {patient.sex}, {patient.age}
Chief complaint: {init_name}
Confirmed: {', '.join(state.confirmed_symptoms[:5]) if state.confirmed_symptoms else 'None'}
Denied: {', '.join(state.denied_symptoms[:5]) if state.denied_symptoms else 'None'}

Candidate symptoms:
{symptom_list}

Brief reason for your selection:"""
        return prompt

    def _build_diagnosis_prompt_small_llm_kg(
        self, state: PatientState, dx_candidates: list[tuple[str, str, float]]
    ) -> str:
        """소형 LLM + KG 진단 프롬프트."""
        patient = state.patient
        init_name = self.loader.symptom_mapping.get(
            patient.initial_evidence, {}
        ).get("name", patient.initial_evidence)

        # Top-N만 LLM에 표시 (선택지 제한, 증상 프롬프트와 일치)
        display_candidates = dx_candidates[:self.top_n] if dx_candidates else []

        if display_candidates:
            # 확률(percentage)로 표시 (증상 프롬프트와 동일한 방식)
            total_score = sum(score for _, _, score in display_candidates) or 1
            dx_list = "\n".join(
                [f"{i+1}. {name} ({score/total_score:.0%})" for i, (_, name, score) in enumerate(display_candidates)]
            )
        else:
            dx_list = "\n".join(
                [f"{i+1}. {name}" for i, (_, name) in enumerate(self._disease_list[:self.top_n])]
            )

        # Two-Stage 프롬프트 (Stage 1: 이유 생성)
        prompt = f"""Select the most likely diagnosis based on the clinical evidence.

Patient: {patient.sex}, {patient.age}
Chief complaint: {init_name}
Confirmed: {', '.join(state.confirmed_symptoms[:5]) if state.confirmed_symptoms else 'None'}
Denied: {', '.join(state.denied_symptoms[:5]) if state.denied_symptoms else 'None'}

Candidate diagnoses:
{dx_list}

Brief reason for your selection:"""
        return prompt

    def _extract_selection(
        self, response: str, max_n: int, candidates: list[tuple] | None = None
    ) -> int | None:
        """LLM 응답에서 선택 번호 추출.

        Two-Stage: regex로 강제된 숫자 (1-N)
        """
        cleaned = response.strip()

        # Two-Stage: regex로 강제되어 숫자만 있음
        try:
            idx = int(cleaned) - 1
            if 0 <= idx < max_n:
                return idx
        except ValueError:
            pass

        # Fallback: 숫자 추출 (1~max_n 범위만)
        numbers = re.findall(r'\b(\d+)\b', cleaned)
        valid_numbers = [int(n) for n in numbers if 1 <= int(n) <= max_n]

        if valid_numbers:
            return valid_numbers[-1] - 1  # 0-indexed

        return None

    def _handle_symptom_response_small_llm_kg(
        self, state: PatientState, response: str, candidates: list[tuple[str, str, int]],
        prompt: str = "", interaction_log: InteractionLog | None = None
    ):
        """소형 LLM + KG 증상 응답 처리."""
        # Top-N만 사용 (프롬프트와 일치)
        top_candidates = candidates[:self.top_n]

        # 선택 번호 추출 (Thinking 모델 대응: 이름 매칭 포함)
        idx = self._extract_selection(response, len(top_candidates), top_candidates)
        selected_cui = None
        selected_name = None

        if idx is not None and 0 <= idx < len(top_candidates):
            selected_cui, selected_name, _ = top_candidates[idx]

        if not selected_cui and top_candidates:
            selected_cui, selected_name, _ = top_candidates[0]

        if not selected_cui:
            state.done = True
            return

        state.asked_cuis.add(selected_cui)
        state.il += 1

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

        # 상호작용 로깅
        if interaction_log:
            interaction_log.llm_prompt = prompt
            interaction_log.llm_response = response
            interaction_log.llm_selected_symptom = selected_name or ""
            interaction_log.llm_selected_cui = selected_cui or ""
            interaction_log.patient_answer = has_symptom
            state.interaction_logs.append(interaction_log)

    def _handle_diagnosis_response_small_llm_kg(
        self, state: PatientState, response: str, dx_candidates: list[tuple[str, str, float]],
        prompt: str = "", interaction_log: InteractionLog | None = None
    ):
        """소형 LLM + KG 진단 응답 처리."""
        # LLM에 표시된 후보 (top N, 프롬프트와 일치)
        display_candidates = dx_candidates[:self.top_n] if dx_candidates else []

        if dx_candidates:
            # KG가 생성한 전체 감별진단 리스트 저장 (DDF1 평가용)
            for cui, _, score in dx_candidates:
                disease_fr = self._cui_to_disease.get(cui)
                if disease_fr:
                    state.kg_predicted_dd.append((disease_fr, score))

            # KG Top-1 직접 사용 모드 (LLM 진단 선택 바이패스)
            if self.kg_only_diagnosis:
                cui, _, _ = display_candidates[0]
                state.predicted = self._cui_to_disease.get(cui)
            else:
                # 선택 번호 추출 (Thinking 모델 대응: 이름 매칭 포함)
                idx = self._extract_selection(response, len(display_candidates), display_candidates)
                if idx is not None and 0 <= idx < len(display_candidates):
                    cui, _, _ = display_candidates[idx]
                    state.predicted = self._cui_to_disease.get(cui)

                if not state.predicted and display_candidates:
                    cui, _, _ = display_candidates[0]
                    state.predicted = self._cui_to_disease.get(cui)
        else:
            # KG 후보가 없는 경우: 전체 질환 목록에서 선택
            idx = self._extract_selection(response, len(self._disease_list[:self.top_n]))
            if idx is not None and 0 <= idx < len(self._disease_list):
                state.predicted = self._disease_list[idx][0]

        if not state.predicted:
            state.predicted = self._disease_list[0][0]

        # LLM이 선택한 진단을 predicted_dd에 저장 (GTPA 계산용)
        if state.predicted:
            # kg_predicted_dd에서 해당 질환의 점수 찾기
            score = 1.0
            for d, s in state.kg_predicted_dd:
                if d == state.predicted:
                    score = s
                    break
            state.predicted_dd = [(state.predicted, score)]

        state.done = True

        # 진단 상호작용 로깅
        if interaction_log:
            interaction_log.llm_prompt = prompt
            interaction_log.llm_response = response
            interaction_log.llm_selected_symptom = state.predicted or ""
            interaction_log.patient_answer = None  # 진단 라운드는 환자 응답 없음
            state.interaction_logs.append(interaction_log)

    # =========================================================================
    # 공통 유틸리티
    # =========================================================================

    def _format_prompt_for_model(self, prompt: str, stage: int = 1) -> str:
        """모델별 프롬프트 형식 적용."""
        # EXAONE 모델은 chat template 형식 필요
        if "EXAONE" in self.model:
            # Stage 1: 구체적인 이유를 요청
            if stage == 1:
                # "Brief reason" 부분을 더 구체적으로 변경
                prompt = prompt.replace(
                    "Brief reason for your selection:",
                    "Which symptom number (1-5) would you select and why? Explain in one sentence:"
                )
            return f"[|user|]\n{prompt}[|endofturn|]\n[|assistant|]\n"
        return prompt

    def _batch_generate(self, prompts: list[str]) -> list[str]:
        """Two-Stage vLLM 배치 생성.

        Stage 1: 이유 생성 (자유 형식, 짧게)
        Stage 2: 숫자 선택 (regex 강제)

        Returns: Stage 2 응답 리스트 (숫자만 포함)
        """
        if not prompts:
            return []

        # 모델별 프롬프트 형식 적용 (Stage 1)
        formatted_prompts = [self._format_prompt_for_model(p, stage=1) for p in prompts]

        # Stage 1: 이유 생성
        stage1_outputs = self.llm.generate(formatted_prompts, self.sampling_params_reason)
        reasons = [out.outputs[0].text.strip() for out in stage1_outputs]

        # 이유 로깅 (디버그용, 처음 10개 배치)
        if hasattr(self, "_reason_log_count"):
            self._reason_log_count += 1
        else:
            self._reason_log_count = 1

        if self._reason_log_count <= 10:
            print(f"\n[Two-Stage Debug #{self._reason_log_count}]", flush=True)
            # 빈 이유 개수 체크
            empty_reasons = sum(1 for r in reasons if not r or len(r) < 5)
            print(f"  Empty/short reasons: {empty_reasons}/{len(reasons)}", flush=True)
            for i, (prompt, reason) in enumerate(zip(prompts[:3], reasons[:3])):
                # 프롬프트에서 후보 목록 추출
                lines = prompt.split('\n')
                candidates_start = False
                candidates = []
                for line in lines:
                    if 'Candidate' in line:
                        candidates_start = True
                        continue
                    if candidates_start and line.strip():
                        if line.startswith('Brief') or line.startswith('Respond'):
                            break
                        candidates.append(line.strip())
                print(f"  [{i+1}] Candidates: {candidates[:5]}", flush=True)
                print(f"      Reason (len={len(reason)}): '{reason[:500]}...'", flush=True)

        # Stage 2: 숫자 선택 (이유를 컨텍스트로 추가)
        stage2_prompts = []
        for prompt, reason in zip(prompts, reasons):
            # 기존 프롬프트에서 JSON 형식 요청 부분 제거하고 이유 추가
            # 간단히: 원래 프롬프트 + 이유 + 숫자 요청
            stage2_content = f"""{prompt}

Your reasoning: {reason}

Based on your reasoning, respond with just the number (1-{self.top_n}):"""
            stage2_prompt = self._format_prompt_for_model(stage2_content, stage=2)
            stage2_prompts.append(stage2_prompt)

        stage2_outputs = self.llm.generate(stage2_prompts, self.sampling_params_select)
        selections = [out.outputs[0].text.strip() for out in stage2_outputs]

        # Stage 2 결과 로깅 및 불일치 분석
        if self._reason_log_count <= 10:
            for i, (prompt, reason, sel) in enumerate(zip(prompts[:3], reasons[:3], selections[:3])):
                # Stage 1 추론에서 언급된 번호 추출
                import re
                # 추론에서 "option 3", "select 3", "choice 3", "answer is 3" 등 패턴 찾기
                mentioned_numbers = re.findall(
                    r'(?:option|select|choice|answer|number|choose|pick|recommend)\s*(?:is\s*)?(\d)',
                    reason.lower()
                )
                # 마지막으로 언급된 번호가 최종 결론일 가능성 높음
                inferred_choice = mentioned_numbers[-1] if mentioned_numbers else None

                # 불일치 여부 확인
                match_status = "✅" if inferred_choice == sel else "❌ MISMATCH" if inferred_choice else "?"
                print(f"      -> Selection: {sel} | Inferred from reason: {inferred_choice} {match_status}", flush=True)

                # 불일치 케이스 상세 로깅
                if inferred_choice and inferred_choice != sel:
                    # 프롬프트에서 후보 목록 추출
                    lines = prompt.split('\n')
                    candidates = [l.strip() for l in lines if l.strip().startswith(('1.', '2.', '3.', '4.', '5.'))]
                    print(f"         ⚠️ Stage 1 추론: '{inferred_choice}번' 추천", flush=True)
                    print(f"         ⚠️ Stage 2 선택: '{sel}번' 선택", flush=True)
                    print(f"         ⚠️ 후보 목록: {candidates[:5]}", flush=True)
                    # 추론 마지막 200자 (결론 부분)
                    print(f"         ⚠️ 추론 결론부: '...{reason[-300:]}'", flush=True)

        # 전체 배치에서 불일치 통계 (샘플링)
        if self._reason_log_count <= 5:
            import re
            mismatch_count = 0
            total_with_inference = 0
            for reason, sel in zip(reasons[:100], selections[:100]):  # 첫 100개만 샘플
                mentioned = re.findall(
                    r'(?:option|select|choice|answer|number|choose|pick|recommend)\s*(?:is\s*)?(\d)',
                    reason.lower()
                )
                if mentioned:
                    total_with_inference += 1
                    if mentioned[-1] != sel:
                        mismatch_count += 1
            if total_with_inference > 0:
                print(f"  [Mismatch Stats] {mismatch_count}/{total_with_inference} mismatches ({100*mismatch_count/total_with_inference:.1f}%)", flush=True)

        return selections

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
        """최종 결과 계산."""
        total = len(states)
        correct_count = 0
        total_il = 0

        # DDR/DDF1 계산용
        total_recall_num = 0
        total_recall_den = 0
        total_precision_num = 0
        total_precision_den = 0

        # GTPA (확률 기반) 계산용
        total_gtpa_prob = 0.0

        for s in states:
            # GTPA@1
            if s.predicted == s.patient.pathology:
                correct_count += 1

            # IL
            total_il += s.il

            # GTPA (확률 기반): KG가 정답 진단에 할당한 확률
            gt_pathology = s.patient.pathology
            gt_prob = 0.0
            # LLM+KG 모드에서는 kg_predicted_dd 사용, LLM only에서는 predicted_dd 사용
            dd_for_gtpa = s.kg_predicted_dd if s.kg_predicted_dd else s.predicted_dd
            if dd_for_gtpa:
                pred_dd_dict = {d: score for d, score in dd_for_gtpa}
                gt_prob = pred_dd_dict.get(gt_pathology, 0.0)
                # 점수를 확률로 정규화
                total_score = sum(score for _, score in dd_for_gtpa)
                if total_score > 0:
                    gt_prob = gt_prob / total_score
            total_gtpa_prob += gt_prob

            # DDR/DDF1 계산 (KG가 생성한 감별진단 리스트 기준)
            # Ground truth DD: patient.differential_diagnosis에서 질환명 추출
            gt_dd = {d for d, _ in s.patient.differential_diagnosis}

            # KG Predicted DD: KG가 생성한 감별진단 리스트에서 질환명 추출
            # LLM+KG 모드에서는 kg_predicted_dd 사용, LLM only에서는 predicted_dd 사용
            if s.kg_predicted_dd:
                pred_dd = {d for d, _ in s.kg_predicted_dd}
            else:
                pred_dd = {d for d, _ in s.predicted_dd}

            intersection = len(gt_dd & pred_dd)

            total_recall_num += intersection
            total_recall_den += len(gt_dd) if gt_dd else 1
            total_precision_num += intersection
            total_precision_den += len(pred_dd) if pred_dd else 1

        # 메트릭 계산
        gtpa = total_gtpa_prob / total  # GTPA (확률 기반)
        gtpa_at_1 = correct_count / total
        avg_il = total_il / total
        ddr = total_recall_num / total_recall_den if total_recall_den > 0 else 0.0
        ddp = total_precision_num / total_precision_den if total_precision_den > 0 else 0.0
        ddf1 = 2 * ddr * ddp / (ddr + ddp) if (ddr + ddp) > 0 else 0.0

        # 디버그: 세부 정보
        print(f"\n[DEBUG] Metrics breakdown:", flush=True)
        print(f"  GTPA (prob): {gtpa:.2%} (KG가 정답에 부여한 확률)", flush=True)
        print(f"  GTPA@1: {gtpa_at_1:.2%} (LLM이 선택한 Top-1 정확도)", flush=True)
        print(f"  Total GT diseases: {total_recall_den}", flush=True)
        print(f"  Total KG predicted diseases: {total_precision_den}", flush=True)
        print(f"  Intersection (matches): {total_recall_num}", flush=True)
        print(f"  DDR = {total_recall_num}/{total_recall_den} = {ddr:.2%} (KG 기준)", flush=True)
        print(f"  DDP = {total_precision_num}/{total_precision_den} = {ddp:.2%} (KG 기준)", flush=True)

        category = "1" if self.mode == "small_llm" else "2"

        print(f"\n{'='*70}", flush=True)
        print(f"FINAL RESULTS - Category {category}", flush=True)
        print(f"Model: {self.model}", flush=True)
        print("=" * 70, flush=True)
        print(f"Samples: {total:,}", flush=True)
        print(f"[LLM 성능]", flush=True)
        print(f"  GTPA@1: {gtpa_at_1:.2%} (LLM이 선택한 Top-1 정확도)", flush=True)
        print(f"[KG 성능]", flush=True)
        print(f"  GTPA:   {gtpa:.2%} (KG가 정답에 부여한 확률)", flush=True)
        print(f"  DDR:    {ddr:.2%} (KG 감별진단 recall)", flush=True)
        print(f"  DDP:    {ddp:.2%} (KG 감별진단 precision)", flush=True)
        print(f"  DDF1:   {ddf1:.2%} (KG 감별진단 F1)", flush=True)
        print(f"[기타]", flush=True)
        print(f"  Avg IL: {avg_il:.2f}", flush=True)
        print(f"  Time:   {elapsed/60:.1f} min", flush=True)
        print(f"Throughput: {total / (elapsed/60):.1f} samples/min", flush=True)
        print("=" * 70, flush=True)
        print(f"\nvs AARLC (DDXPlus paper):", flush=True)
        print(f"  GTPA:   {gtpa:.2%} vs 98.82% ({(gtpa-0.9882)*100:+.2f}%)", flush=True)
        print(f"  GTPA@1: {gtpa_at_1:.2%} vs 75.39% ({(gtpa_at_1-0.7539)*100:+.2f}%)", flush=True)
        print(f"  DDR:    {ddr:.2%} vs 97.73% ({(ddr-0.9773)*100:+.2f}%)", flush=True)
        print(f"  DDF1:   {ddf1:.2%} vs 78.24% ({(ddf1-0.7824)*100:+.2f}%)", flush=True)
        print(f"  IL:     {avg_il:.2f} vs 25.75 ({avg_il-25.75:+.2f})", flush=True)

        self._save_results(total, gtpa, gtpa_at_1, ddr, ddp, ddf1, avg_il, elapsed, category)
        self._save_interaction_logs(states, category)

        return {
            "category": int(category),
            "model": self.model,
            "mode": self.mode,
            "n_samples": total,
            "gtpa": gtpa,
            "gtpa_at_1": gtpa_at_1,
            "ddr": ddr,
            "ddp": ddp,
            "ddf1": ddf1,
            "avg_il": avg_il,
            "elapsed_minutes": elapsed / 60,
            "throughput": total / (elapsed / 60),
        }

    def _save_results(
        self, total: int, gtpa: float, gtpa_at_1: float, ddr: float, ddp: float, ddf1: float, avg_il: float, elapsed: float, category: str
    ):
        """결과 저장."""
        model_short = self.model.split("/")[-1].replace("-", "_").replace(".", "_")
        output_dir = get_output_dir()

        output_file = output_dir / f"cat{category}_{model_short}_n{total}.txt"

        with open(output_file, "w") as f:
            f.write(f"Category: {category}\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Mode: {self.mode}\n")
            f.write(f"Samples: {total:,}\n")
            f.write(f"GTPA:   {gtpa:.2%} (probability-based)\n")
            f.write(f"GTPA@1: {gtpa_at_1:.2%}\n")
            f.write(f"DDR:    {ddr:.2%}\n")
            f.write(f"DDP:    {ddp:.2%}\n")
            f.write(f"DDF1:   {ddf1:.2%}\n")
            f.write(f"Avg IL: {avg_il:.2f}\n")
            f.write(f"Time: {elapsed/60:.1f} min\n")
            f.write(f"Throughput: {total / (elapsed/60):.1f} samples/min\n\n")
            f.write("vs AARLC (DDXPlus paper):\n")
            f.write(f"  GTPA:   {gtpa:.2%} vs 98.82% ({(gtpa-0.9882)*100:+.2f}%)\n")
            f.write(f"  GTPA@1: {gtpa_at_1:.2%} vs 75.39% ({(gtpa_at_1-0.7539)*100:+.2f}%)\n")
            f.write(f"  DDR:    {ddr:.2%} vs 97.73% ({(ddr-0.9773)*100:+.2f}%)\n")
            f.write(f"  DDF1:   {ddf1:.2%} vs 78.24% ({(ddf1-0.7824)*100:+.2f}%)\n")
            f.write(f"  IL:     {avg_il:.2f} vs 25.75 ({avg_il-25.75:+.2f})\n")

        print(f"\nResults saved to: {output_file}", flush=True)

    def _save_interaction_logs(
        self, states: list[PatientState], category: str
    ):
        """상호작용 로그 저장 (논문 분석용).

        JSONL 형식으로 저장하여 각 환자의 전체 진단 과정을 분석 가능하게 함.
        """
        if self.mode != "small_llm_kg":
            return  # KG 모드만 로깅

        model_short = self.model.split("/")[-1].replace("-", "_").replace(".", "_")
        output_dir = get_output_dir()
        log_file = output_dir / f"interaction_logs_cat{category}_{model_short}.jsonl"

        import dataclasses

        with open(log_file, "w") as f:
            for state in states:
                if not state.interaction_logs:
                    continue

                patient_log = {
                    "patient_idx": state.idx,
                    "pathology_gt": state.patient.pathology,
                    "pathology_pred": state.predicted,
                    "correct": state.predicted == state.patient.pathology,
                    "total_rounds": state.il,
                    "initial_symptom": state.patient.initial_evidence,
                    "interactions": [
                        dataclasses.asdict(log) for log in state.interaction_logs
                    ],
                }
                f.write(json.dumps(patient_log, ensure_ascii=False) + "\n")

        print(f"Interaction logs saved to: {log_file}", flush=True)

    def close(self):
        """리소스 정리."""
        if self.kg:
            self.kg.close()


def run_category(
    category: int,
    models: list[str],
    n_samples: int = 10000,
    max_il: int | None = None,
    scoring: str = "v23_mild_denied",
    severity: int | None = None,
    kg_only_diagnosis: bool = False,
    top_n: int = 3,
    reason_tokens: int = 1024,
    shuffle_candidates: bool = False,
) -> list[dict]:
    """카테고리별 벤치마크 실행.

    Args:
        category: 1=Small LLM, 2=Small LLM + KG
        models: 벤치마크할 모델 목록 (models_config.py에서 관리)
        n_samples: 샘플 수
        max_il: 고정 IL (MEDDxAgent 비교용), None이면 가변
        scoring: 진단 스코어링 전략 ("v23_mild_denied", "v18_coverage", "v15_ratio", "v7_additive")
        severity: 질환 심각도 필터 (1-5)
        kg_only_diagnosis: KG Top-1 직접 사용 (LLM 진단 선택 바이패스)
        top_n: LLM에 제시할 후보 개수
        reason_tokens: Stage 1 추론 max_tokens
        shuffle_candidates: 후보 순서 랜덤 셔플 (position bias 테스트)
    """
    il_info = f"IL={max_il}" if max_il else "IL=adaptive"
    shuffle_info = ", shuffle" if shuffle_candidates else ""

    if category == 1:
        # 소형 LLM only
        print("\n" + "=" * 70)
        print(f"Category 1: Small LLM (vLLM, {il_info})")
        print("=" * 70)

        return _run_vllm_models(models, "small_llm", n_samples, max_il, scoring, severity, kg_only_diagnosis, top_n, reason_tokens, shuffle_candidates)

    elif category == 2:
        # 소형 LLM + KG
        kg_mode = "KG-only-dx" if kg_only_diagnosis else "LLM-select"
        print("\n" + "=" * 70)
        print(f"Category 2: Small LLM + KG (vLLM + Neo4j, {il_info}, {scoring}, {kg_mode}, Top-{top_n}, reason={reason_tokens}{shuffle_info})")
        print("=" * 70)

        return _run_vllm_models(models, "small_llm_kg", n_samples, max_il, scoring, severity, kg_only_diagnosis, top_n, reason_tokens, shuffle_candidates)

    else:
        raise ValueError(f"Invalid category: {category}. Use 1 or 2.")


def _run_vllm_models(
    models: list[str],
    mode: str,
    n_samples: int,
    max_il: int | None = None,
    scoring: str = "v23_mild_denied",
    severity: int | None = None,
    kg_only_diagnosis: bool = False,
    top_n: int = 3,
    reason_tokens: int = 1024,
    shuffle_candidates: bool = False,
) -> list[dict]:
    """vLLM 모델 순차 실행."""
    all_results = []
    mode_name = "Small LLM" if mode == "small_llm" else "Small LLM + KG"
    il_info = f"IL={max_il}" if max_il else "IL=adaptive"
    kg_info = ", KG-only-dx" if kg_only_diagnosis else ""
    top_n_info = f", Top-{top_n}" if top_n != 3 else ""
    shuffle_info = ", shuffle" if shuffle_candidates else ""

    # 모델별 진행 상황 (tqdm)
    model_pbar = tqdm(
        models,
        desc=f"Models ({mode_name}, vLLM, {il_info}{kg_info}{top_n_info}{shuffle_info})",
        unit="model",
        position=0,
    )

    for model in model_pbar:
        model_short = model.split("/")[-1] if "/" in model else model
        model_pbar.set_description(f"Model: {model_short[:30]}")

        try:
            benchmark = VLLMBenchmark(
                model=model,
                mode=mode,
                max_il=max_il,
                scoring=scoring,
                severity=severity,
                kg_only_diagnosis=kg_only_diagnosis,
                top_n=top_n,
                reason_tokens=reason_tokens,
                shuffle_candidates=shuffle_candidates,
            )
            result = benchmark.run(n_samples)
            # max_il 정보 추가
            result["max_il"] = max_il
            result["max_il_mode"] = "fixed" if max_il else "adaptive"
            all_results.append(result)

            benchmark.close()
            del benchmark
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            tqdm.write(f"❌ Error running {model}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"model": model, "error": str(e)})

    model_pbar.close()

    # 요약 출력
    _print_summary(all_results, mode)

    return all_results


def _print_summary(results: list[dict], mode: str):
    """결과 요약 출력."""
    print(f"\n{'='*90}")
    print(f"SUMMARY - {mode}")
    print("=" * 90)
    print(f"{'Model':<35} {'GTPA@1':>8} {'DDR':>8} {'DDF1':>8} {'IL':>6} {'Time':>8}")
    print("-" * 90)

    for r in results:
        if "error" in r:
            print(f"{r['model']:<35} {'ERROR':>8}")
        else:
            model_short = r["model"].split("/")[-1][:32]
            ddr = r.get("ddr", 0)
            ddf1 = r.get("ddf1", 0)
            print(
                f"{model_short:<35} {r['gtpa_at_1']:>7.1%} "
                f"{ddr:>7.1%} {ddf1:>7.1%} "
                f"{r['avg_il']:>6.1f} {r['elapsed_minutes']:>7.1f}m"
            )

    # AARLC 기준선 출력
    print("-" * 90)
    print(f"{'AARLC (DDXPlus paper)':<35} {'75.39%':>8} {'97.73%':>8} {'78.24%':>8} {'25.75':>6}")

    # JSON 저장
    output_dir = get_output_dir()
    summary_file = output_dir / f"summary_{mode}.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


def _print_all_categories_summary(results: list[dict]):
    """Category 1, 2 비교 요약 출력."""
    print(f"\n{'='*100}")
    print("COMPARISON: Category 1 (Small LLM) vs Category 2 (Small LLM + KG)")
    print("=" * 100)

    # 모델별로 그룹화
    models = {}
    for r in results:
        if "error" in r:
            continue
        model = r["model"]
        cat = r.get("category", 0)
        if model not in models:
            models[model] = {}
        models[model][cat] = r

    print(f"{'Model':<30} {'Cat':>4} {'GTPA@1':>8} {'DDR':>8} {'DDF1':>8} {'IL':>6} {'KG Δ GTPA':>10}")
    print("-" * 100)

    for model, cats in models.items():
        model_short = model.split("/")[-1][:27]

        for cat in [1, 2]:
            if cat not in cats:
                continue
            r = cats[cat]
            ddr = r.get("ddr", 0)
            ddf1 = r.get("ddf1", 0)

            # KG 효과 계산 (Cat2 - Cat1)
            kg_delta = ""
            if cat == 2 and 1 in cats:
                delta = r["gtpa_at_1"] - cats[1]["gtpa_at_1"]
                kg_delta = f"{delta*100:+.1f}%"

            print(
                f"{model_short:<30} {cat:>4} {r['gtpa_at_1']:>7.1%} "
                f"{ddr:>7.1%} {ddf1:>7.1%} "
                f"{r['avg_il']:>6.1f} {kg_delta:>10}"
            )

    # AARLC 기준선
    print("-" * 100)
    print(f"{'AARLC (DDXPlus paper)':<30} {'-':>4} {'75.39%':>8} {'97.73%':>8} {'78.24%':>8} {'25.75':>6}")

    # JSON 저장
    output_dir = get_output_dir()
    summary_file = output_dir / f"summary_all_categories.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull summary saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DDXPlus Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본: 모든 모델 + Category 1, 2 실행 (가변 IL)
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_vllm.py -n 10000

  # Category 2만 실행 (Small LLM + KG)
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_vllm.py --category 2 -n 10000

  # MEDDxAgent 비교: 고정 IL (5, 10, 15)
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_vllm.py --category 2 --max-il 5 -n 10000
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_vllm.py --category 2 --max-il 10 -n 10000
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_vllm.py --category 2 --max-il 15 -n 10000

Categories:
  1 = Small LLM only (vLLM)
  2 = Small LLM + KG (vLLM + Neo4j)

Models: scripts/models_config.py (SMALL_LLM_MODELS)
        """
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Category to run: 1, 2, or comma-separated like '1,2' (default: 1,2)",
    )
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        default=10000,
        help="Number of samples (default: 10000)",
    )
    parser.add_argument(
        "--max-il",
        type=int,
        default=None,
        choices=[5, 10, 15, 20, 25, 30, 50],
        help="Fixed max interaction length for MEDDxAgent comparison (default: adaptive)",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="v23_mild_denied",
        choices=["v23_mild_denied", "v18_coverage", "v15_ratio", "v7_additive"],
        help="Diagnosis scoring strategy: v23_mild_denied (default), v18_coverage, v15_ratio, v7_additive",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5],
        help="Disease severity filter: 1=mild, 2=moderate (default), 3=severe, 4=emergency, 5=critical",
    )
    parser.add_argument(
        "--kg-only-diagnosis",
        action="store_true",
        help="Use KG Top-1 directly for diagnosis (bypass LLM selection). Improves accuracy by ~8%%.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        choices=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Number of candidates to show LLM for selection (default: 3)",
    )
    parser.add_argument(
        "--reason-tokens",
        type=int,
        default=1024,
        help="Max tokens for Stage 1 reasoning (default: 1024)",
    )
    parser.add_argument(
        "--shuffle-candidates",
        action="store_true",
        help="Shuffle candidate order randomly (test position bias)",
    )

    args = parser.parse_args()

    # 실행별 출력 디렉토리 생성 (YYYYMMDD_HHMMSS 형식)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_OUTPUT_DIR = Path(__file__).parent.parent / "results" / timestamp
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Results will be saved to: {RUN_OUTPUT_DIR}")

    # 카테고리 파싱 (먼저 해서 run_info에 포함)
    if args.category:
        try:
            categories = [int(c.strip()) for c in args.category.split(",")]
            for cat in categories:
                if cat not in [1, 2]:
                    parser.error(f"Invalid category: {cat}. Use 1 or 2.")
        except ValueError:
            parser.error(f"Invalid category format: {args.category}. Use integers like '1', '2', or '1,2'.")
    else:
        categories = [1, 2]  # 기본값

    # 실행 정보 저장
    run_info = {
        "timestamp": timestamp,
        "categories": categories,
        "n_samples": args.n_samples,
        "models": ALL_MODELS,
        "max_il": args.max_il,
        "max_il_mode": "fixed" if args.max_il else "adaptive",
        "scoring": args.scoring,
        "severity": args.severity,
        "kg_only_diagnosis": args.kg_only_diagnosis,
        "top_n": args.top_n,
        "reason_tokens": args.reason_tokens,
        "shuffle_candidates": args.shuffle_candidates,
    }
    with open(RUN_OUTPUT_DIR / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    il_info = f"IL={args.max_il}" if args.max_il else "IL=adaptive"
    severity_info = f"severity={args.severity}" if args.severity else "all"
    kg_dx_info = ", kg-only-diagnosis" if args.kg_only_diagnosis else ""
    top_n_info = f", Top-{args.top_n}" if args.top_n != 3 else ""
    shuffle_info = ", shuffle" if args.shuffle_candidates else ""
    print(f"Categories: {categories}, Samples: {args.n_samples:,}, {il_info}, scoring={args.scoring}, {severity_info}{kg_dx_info}{top_n_info}{shuffle_info}")

    # 벤치마크 실행
    all_results = []

    for cat in categories:
        results = run_category(
            category=cat,
            models=ALL_MODELS,
            n_samples=args.n_samples,
            max_il=args.max_il,
            scoring=args.scoring,
            severity=args.severity,
            kg_only_diagnosis=args.kg_only_diagnosis,
            top_n=args.top_n,
            reason_tokens=args.reason_tokens,
            shuffle_candidates=args.shuffle_candidates,
        )
        all_results.extend(results)

    # 여러 카테고리 실행 시 비교 요약 출력
    if len(categories) > 1:
        _print_all_categories_summary(all_results)

    if all_results:
        print(f"\n✅ Completed {len(all_results)} benchmark(s)")
        print(f"📁 Results saved to: {RUN_OUTPUT_DIR}")
