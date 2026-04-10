"""UMLS Knowledge Graph 인터페이스.

Neo4j 기반 UMLS KG 쿼리 및 진단 로직.

Interpretability (해석 가능성) 특징:
=================================

1. Intrinsic Interpretability (내재적 해석가능성)
   - 진단 과정 자체가 해석 가능한 구조
   - Black-box 모델의 Post-hoc 설명과 달리 과정이 투명

2. Evidence-based Reasoning (근거 기반 추론)
   - 각 진단에 matched_symptoms, denied_symptoms 명시
   - Coverage 기반 점수 계산 (matched_count / total_symptoms)

3. Traceable Decision Path (추적 가능한 결정 경로)
   - DiagnosticTrace: 전체 진단 과정 기록
   - DiagnosticStep: 각 질문-응답의 영향 추적
   - SymptomSelectionReason: 증상 선택 이유 명시

4. Reproducibility (재현성)
   - 동일 입력 → 동일 출력 (결정적 알고리즘)
   - KG 구조로 결과 검증 가능

주요 클래스:
- ExplainedDiagnosis: 설명이 포함된 진단 결과
- DiagnosticTrace: 진단 과정 전체 추적
- DiagnosticStep: 단일 질문-응답 단계
- SymptomSelectionReason: 증상 선택 이유
"""

from dataclasses import dataclass, field

from neo4j import GraphDatabase


@dataclass
class DiagnosisCandidate:
    """진단 후보."""

    cui: str
    name: str
    score: float  # 확신도 (0-1)
    confirmed_count: int  # 일치하는 confirmed 증상 수
    total_symptoms: int  # 질환의 총 증상 수


@dataclass
class RelatedDisease:
    """증상과 연관된 질환 정보."""

    cui: str
    name: str
    score: float  # 정규화 점수 (0-1)
    matched_symptoms: int  # 일치하는 confirmed 증상 수
    total_symptoms: int  # 질환의 총 증상 수


@dataclass
class SymptomCandidate:
    """다음 질문 후보 증상."""

    cui: str
    name: str
    disease_coverage: int  # 연결된 질환 수
    information_gain: float = 0.0
    related_diseases: list[RelatedDisease] = field(default_factory=list)  # Top 연관 질환


@dataclass
class ExplainedDiagnosis:
    """설명 가능한 진단 결과.

    현재 KG 구조(INDICATES 관계만 존재)에서 제공 가능한 설명:
    - 일치/불일치 증상 목록
    - Coverage 기반 점수
    - Synthesized 스타일 설명 텍스트
    """

    cui: str
    name: str
    score: float  # 정규화된 확신도 (0-1)
    rank: int  # 순위 (1부터 시작)

    # 증상 목록 (실제 이름)
    matched_symptoms: list[str]  # 확인된 증상 중 이 질환과 연관된 것
    denied_symptoms: list[str]  # 부정된 증상 중 이 질환과 연관된 것
    unasked_symptoms: list[str]  # 아직 질문하지 않은 연관 증상

    # 통계
    matched_count: int
    denied_count: int
    total_symptoms: int
    coverage: float  # matched_count / total_symptoms

    # 설명 텍스트
    explanation: str


@dataclass
class SymptomSelectionReason:
    """증상 선택 이유 (해석 가능성).

    각 증상이 왜 선택되었는지 명시적으로 기록.
    """

    symptom_cui: str
    symptom_name: str
    selection_reason: str  # "initial", "high_coverage", "information_gain"
    disease_coverage: int  # 연관 질환 수
    top_diseases: list[str]  # 연관 Top 질환 이름
    information_gain: float = 0.0


@dataclass
class DiagnosticStep:
    """진단 과정의 단일 단계 (해석 가능성).

    각 질문-응답이 진단에 미친 영향을 추적.
    """

    step: int  # 단계 번호 (1부터)
    symptom_asked: str  # 질문한 증상 이름
    symptom_cui: str
    patient_response: str  # "yes" or "no"
    selection_reason: str  # 왜 이 증상을 선택했는지

    # 이 단계 이후의 진단 상태
    top_diagnosis: str  # Top-1 진단명
    top_score: float  # Top-1 점수
    candidates_remaining: int  # 남은 후보 질환 수

    # 진단 변화 추적
    diagnosis_changed: bool = False  # Top-1이 바뀌었는지
    previous_top: str | None = None  # 이전 Top-1


@dataclass
class DiagnosticTrace:
    """전체 진단 과정 추적 (해석 가능성).

    진단 과정 전체를 기록하여 재현성과 해석가능성 제공.
    Black-box 모델과 달리 결정 과정을 완전히 추적 가능.
    """

    # 입력
    initial_symptom: str
    initial_symptom_cui: str

    # 진단 과정
    steps: list[DiagnosticStep] = field(default_factory=list)

    # 최종 결과
    final_diagnosis: str | None = None
    final_score: float = 0.0
    stop_reason: str = ""  # 종료 이유

    # 요약 통계
    total_questions: int = 0
    confirmed_count: int = 0
    denied_count: int = 0

    def add_step(self, step: DiagnosticStep) -> None:
        """단계 추가."""
        self.steps.append(step)
        self.total_questions = len(self.steps)

    def to_explanation(self) -> str:
        """사람이 읽을 수 있는 설명 생성."""
        lines = [
            "=== Diagnostic Reasoning Trace ===",
            f"Initial symptom: {self.initial_symptom}",
            "",
            "--- Questioning Process ---",
        ]

        for step in self.steps:
            response = "✓ Yes" if step.patient_response == "yes" else "✗ No"
            lines.append(
                f"Q{step.step}: {step.symptom_asked}? → {response}"
            )
            lines.append(f"      Reason: {step.selection_reason}")
            lines.append(f"      Top diagnosis: {step.top_diagnosis} ({step.top_score:.1%})")
            if step.diagnosis_changed:
                lines.append(f"      ⚠ Changed from: {step.previous_top}")
            lines.append("")

        lines.extend([
            "--- Final Diagnosis ---",
            f"Diagnosis: {self.final_diagnosis}",
            f"Confidence: {self.final_score:.1%}",
            f"Stop reason: {self.stop_reason}",
            f"Total questions: {self.total_questions}",
            f"Confirmed symptoms: {self.confirmed_count}",
            f"Denied symptoms: {self.denied_count}",
        ])

        return "\n".join(lines)


@dataclass
class KGState:
    """KG 상태."""

    confirmed_cuis: set[str] = field(default_factory=set)
    denied_cuis: set[str] = field(default_factory=set)
    asked_cuis: set[str] = field(default_factory=set)

    def add_confirmed(self, cui: str) -> None:
        """VALID_YES 증상 추가."""
        self.confirmed_cuis.add(cui)
        self.asked_cuis.add(cui)

    def add_denied(self, cui: str) -> None:
        """VALID_NO 또는 INVALID 증상 추가."""
        self.denied_cuis.add(cui)
        self.asked_cuis.add(cui)


class UMLSKG:
    """UMLS Knowledge Graph.

    Neo4j 기반 2-hop 탐색 및 진단 로직 구현.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password123",
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.state = KGState()

    def close(self) -> None:
        """드라이버 종료."""
        self.driver.close()

    def reset_state(self) -> None:
        """상태 초기화."""
        self.state = KGState()

    def get_related_diseases(self, symptom_cui: str) -> list[DiagnosisCandidate]:
        """증상 CUI와 연결된 질환 목록 조회."""
        query = """
        MATCH (s:Symptom {cui: $cui})-[:INDICATES]->(d:Disease)
        OPTIONAL MATCH (d)<-[:INDICATES]-(other:Symptom)
        WITH d, count(DISTINCT other) AS total_symptoms
        RETURN d.cui AS cui, d.name AS name, total_symptoms
        ORDER BY total_symptoms DESC
        """
        with self.driver.session() as session:
            result = session.run(query, cui=symptom_cui)
            return [
                DiagnosisCandidate(
                    cui=r["cui"],
                    name=r["name"],
                    score=0.0,
                    confirmed_count=1,
                    total_symptoms=r["total_symptoms"],
                )
                for r in result
            ]

    def get_candidate_symptoms(
        self,
        initial_cui: str,
        limit: int = 10,
        *,
        confirmed_cuis: set[str] | None = None,
        denied_cuis: set[str] | None = None,
        asked_cuis: set[str] | None = None,
    ) -> list[SymptomCandidate]:
        """초기 증상으로부터 2-hop 탐색하여 후보 증상 반환.

        Args:
            initial_cui: 주호소 CUI
            limit: 반환할 최대 증상 수
            confirmed_cuis: confirmed 증상 CUI 집합 (None이면 self.state 사용)
            denied_cuis: denied 증상 CUI 집합 (None이면 self.state 사용)
            asked_cuis: 이미 질문한 증상 CUI 집합 (None이면 self.state 사용)

        Returns:
            SymptomCandidate 리스트 (disease_coverage 내림차순)
        """
        # 스레드 안전: 인자로 받은 상태 사용, 없으면 self.state 사용
        _confirmed = confirmed_cuis if confirmed_cuis is not None else self.state.confirmed_cuis
        _denied = denied_cuis if denied_cuis is not None else self.state.denied_cuis
        _asked = asked_cuis if asked_cuis is not None else self.state.asked_cuis

        if not _confirmed and not _denied:
            # 초기 상태: 주호소만으로 탐색
            return self._get_initial_candidates(initial_cui, limit, asked_cuis=_asked)
        else:
            # 누적 상태: confirmed/denied 조건 적용
            return self._get_accumulated_candidates(
                limit,
                confirmed_cuis=_confirmed,
                denied_cuis=_denied,
                asked_cuis=_asked,
            )

    def _get_initial_candidates(
        self,
        initial_cui: str,
        limit: int,
        *,
        asked_cuis: set[str] | None = None,
    ) -> list[SymptomCandidate]:
        """초기 2-hop 탐색 (실제 증상 우선)."""
        _asked = asked_cuis if asked_cuis is not None else self.state.asked_cuis
        query = """
        MATCH (s:Symptom {cui: $initial_cui})-[:INDICATES]->(d:Disease)
        MATCH (d)<-[:INDICATES]-(related:Symptom)
        WHERE related.cui <> $initial_cui
          AND NOT related.cui IN $asked_cuis
        WITH related, count(DISTINCT d) AS disease_coverage
        // 실제 증상을 우선 정렬 (is_antecedent=false first)
        RETURN related.cui AS cui,
               related.name AS name,
               disease_coverage,
               CASE WHEN related.is_antecedent = false THEN 0 ELSE 1 END AS priority
        ORDER BY priority ASC, disease_coverage DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(
                query,
                initial_cui=initial_cui,
                asked_cuis=list(_asked),
                limit=limit,
            )
            return [
                SymptomCandidate(
                    cui=r["cui"],
                    name=r["name"],
                    disease_coverage=r["disease_coverage"],
                )
                for r in result
            ]

    def _get_accumulated_candidates(
        self,
        limit: int,
        *,
        confirmed_cuis: set[str] | None = None,
        denied_cuis: set[str] | None = None,
        asked_cuis: set[str] | None = None,
    ) -> list[SymptomCandidate]:
        """confirmed/denied 조건을 적용한 후보 증상 탐색 (Co-occurrence 기반).

        Co-occurrence 전략: confirmed 증상과 동시 출현하는 증상 우선 선택.
        - GTPA@1: +0.49pp (88.14% → 88.63%)
        - Avg IL: -2.3 (26.4 → 24.1)
        - Confirm Rate: +1.9pp (13.8% → 15.7%)
        - GTPA@10: -0.12pp (99.83% → 99.71%)
        """
        _confirmed = confirmed_cuis if confirmed_cuis is not None else self.state.confirmed_cuis
        _denied = denied_cuis if denied_cuis is not None else self.state.denied_cuis
        _asked = asked_cuis if asked_cuis is not None else self.state.asked_cuis

        query = """
        // Co-occurrence 기반 증상 선택
        // confirmed 증상과 함께 자주 나타나는 증상을 우선 선택
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH DISTINCT d

        // denied가 과도하게 누적된 질환은 조기 제외
        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        WITH d, count(DISTINCT denied) AS denied_count
        WHERE denied_count < 5

        WITH collect(DISTINCT d) AS valid_diseases
        WHERE size(valid_diseases) > 0

        UNWIND valid_diseases AS d
        MATCH (d)<-[:INDICATES]-(next:Symptom)
        WHERE NOT next.cui IN $confirmed_cuis
          AND NOT next.cui IN $denied_cuis
          AND NOT next.cui IN $asked_cuis

        // Co-occurrence score: 얼마나 많은 confirmed 증상과 동일 질환을 공유하는지
        WITH next, d
        MATCH (d)<-[:INDICATES]-(conf:Symptom)
        WHERE conf.cui IN $confirmed_cuis
        WITH next, count(DISTINCT d) AS coverage, count(DISTINCT conf) AS cooccur_count,
             CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority

        // cooccur_count * coverage = co-occurrence score
        // 높을수록 confirmed와 함께 나타날 확률이 높음
        RETURN next.cui AS cui,
               next.name AS name,
               coverage AS disease_coverage,
               priority
        ORDER BY priority ASC, toFloat(cooccur_count) * coverage DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(
                query,
                confirmed_cuis=list(_confirmed),
                denied_cuis=list(_denied),
                asked_cuis=list(_asked),
                limit=limit,
            )
            return [
                SymptomCandidate(
                    cui=r["cui"],
                    name=r["name"],
                    disease_coverage=r["disease_coverage"],
                )
                for r in result
            ]

    def get_related_diseases_for_symptom(
        self,
        symptom_cui: str,
        top_k: int = 3,
        *,
        confirmed_cuis: set[str] | None = None,
        denied_cuis: set[str] | None = None,
    ) -> list[RelatedDisease]:
        """증상과 연관된 상위 질환 목록 반환 (설명용).

        Args:
            symptom_cui: 증상 CUI
            top_k: 반환할 최대 질환 수
            confirmed_cuis: confirmed 증상 CUI 집합
            denied_cuis: denied 증상 CUI 집합

        Returns:
            RelatedDisease 리스트 (score 내림차순, 정규화됨)
        """
        _confirmed = confirmed_cuis if confirmed_cuis is not None else self.state.confirmed_cuis
        _denied = denied_cuis if denied_cuis is not None else self.state.denied_cuis

        # 증상과 연결된 질환들의 점수 계산
        query = """
        MATCH (s:Symptom {cui: $symptom_cui})-[:INDICATES]->(d:Disease)
        OPTIONAL MATCH (d)<-[:INDICATES]-(all_symptom:Symptom)
        WITH d, collect(DISTINCT all_symptom.cui) AS disease_symptom_cuis,
             count(DISTINCT all_symptom) AS total_symptoms
        WITH d, disease_symptom_cuis, total_symptoms,
             [c IN $confirmed_cuis WHERE c IN disease_symptom_cuis] AS matched_confirmed,
             [c IN $denied_cuis WHERE c IN disease_symptom_cuis] AS matched_denied
        WITH d, total_symptoms,
             size(matched_confirmed) + 1 AS confirmed_count,  // +1 for the symptom itself
             size(matched_denied) AS denied_count
        WITH d, confirmed_count, denied_count, total_symptoms,
             (toFloat(confirmed_count) / (toFloat(total_symptoms) + 1.0) * toFloat(confirmed_count))
             * (1.0 - 0.1 * toFloat(denied_count) / (toFloat(total_symptoms) + 1.0)) AS raw_score
        WHERE raw_score > 0
        WITH collect({
            cui: d.cui, name: d.name, raw_score: raw_score,
            confirmed_count: confirmed_count, total_symptoms: total_symptoms
        }) AS all_candidates
        WITH all_candidates,
             reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
        UNWIND all_candidates AS c
        RETURN c.cui AS cui, c.name AS name,
               CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
               c.confirmed_count AS confirmed_count,
               c.total_symptoms AS total_symptoms
        ORDER BY score DESC
        LIMIT $top_k
        """
        with self.driver.session() as session:
            result = session.run(
                query,
                symptom_cui=symptom_cui,
                confirmed_cuis=list(_confirmed),
                denied_cuis=list(_denied),
                top_k=top_k,
            )
            return [
                RelatedDisease(
                    cui=r["cui"],
                    name=r["name"],
                    score=r["score"],
                    matched_symptoms=r["confirmed_count"],
                    total_symptoms=r["total_symptoms"],
                )
                for r in result
            ]

    def enrich_candidates_with_diseases(
        self,
        candidates: list[SymptomCandidate],
        top_k_diseases: int = 3,
        *,
        confirmed_cuis: set[str] | None = None,
        denied_cuis: set[str] | None = None,
    ) -> list[SymptomCandidate]:
        """후보 증상에 연관 질환 정보 추가.

        Args:
            candidates: 후보 증상 리스트
            top_k_diseases: 각 증상당 표시할 최대 질환 수
            confirmed_cuis: confirmed 증상 CUI 집합
            denied_cuis: denied 증상 CUI 집합

        Returns:
            관련 질환 정보가 추가된 SymptomCandidate 리스트
        """
        for candidate in candidates:
            candidate.related_diseases = self.get_related_diseases_for_symptom(
                symptom_cui=candidate.cui,
                top_k=top_k_diseases,
                confirmed_cuis=confirmed_cuis,
                denied_cuis=denied_cuis,
            )
        return candidates

    def get_diagnosis_candidates(
        self,
        top_k: int = 5,
        scoring: str = "v15_ratio",
        *,
        confirmed_cuis: set[str] | None = None,
        denied_cuis: set[str] | None = None,
    ) -> list[DiagnosisCandidate]:
        """현재 상태 기반 진단 후보 반환.

        점수 계산 전략:

        v15_ratio (기본, 권장):
        - score = confirmed / (confirmed + denied + 1) × confirmed
        - denied 증상의 비율을 직접 반영하여 정확도 극대화
        - GTPA@1: 86.0%, IL: 24.4 (1000 cases, 최적화된 stopping criteria)

        v23_mild_denied:
        - score = (confirmed / (total + 1) × confirmed) × (1 - 0.1 × denied/total)
        - v18_coverage + 약한 denied 패널티 (0.1)
        - GTPA@1: 80.5%, IL: 24.4 (1000 cases)

        v18_coverage:
        - score = confirmed / (total_symptoms + 1) × confirmed
        - 질환의 총 증상 수를 고려하여 coverage 기반 점수 계산
        - GTPA@1: 78.9%, IL: 24.4 (1000 cases)

        v7_additive (이전 버전):
        - score = confirmed - 0.5 × denied
        - GTPA@1: 75%, IL: 11.0 (100 cases)

        Args:
            top_k: 반환할 최대 질환 수
            scoring: 스코어링 전략 ("v23_mild_denied", "v18_coverage", "v15_ratio", "v7_additive")
            confirmed_cuis: confirmed 증상 CUI 집합 (None이면 self.state 사용)
            denied_cuis: denied 증상 CUI 집합 (None이면 self.state 사용)

        Returns:
            DiagnosisCandidate 리스트 (score 내림차순)
        """
        # 스레드 안전: 인자로 받은 상태 사용, 없으면 self.state 사용
        _confirmed = confirmed_cuis if confirmed_cuis is not None else self.state.confirmed_cuis
        _denied = denied_cuis if denied_cuis is not None else self.state.denied_cuis

        if not _confirmed:
            return []

        if scoring == "v23_mild_denied":
            # v23: v18_coverage + 약한 denied 패널티 (0.1)
            # score = (confirmed / (total + 1) × confirmed) × (1 - 0.1 × denied/total)
            # 500 cases: 89.2% GTPA@1, IL 20.0
            query = """
            MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
            WHERE confirmed.cui IN $confirmed_cuis
            WITH DISTINCT d
            OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
            WITH d,
                 count(DISTINCT s) AS total_symptoms,
                 count(DISTINCT CASE WHEN s.cui IN $confirmed_cuis THEN s END) AS confirmed_count,
                 count(DISTINCT CASE WHEN s.cui IN $denied_cuis THEN s END) AS denied_count
            WHERE confirmed_count > 0
            WITH d, confirmed_count, denied_count, total_symptoms,
                 (toFloat(confirmed_count) / (toFloat(total_symptoms) + 1.0) * toFloat(confirmed_count))
                 * (1.0 - 0.1 * toFloat(denied_count) / (toFloat(total_symptoms) + 1.0)) AS raw_score
            WHERE raw_score > 0
            WITH collect({
                cui: d.cui, name: d.name, raw_score: raw_score,
                confirmed_count: confirmed_count, total_symptoms: total_symptoms
            }) AS all_candidates
            WITH all_candidates,
                 reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
            UNWIND all_candidates AS c
            RETURN c.cui AS cui, c.name AS name,
                   CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
                   c.confirmed_count AS confirmed_count,
                   c.total_symptoms AS total_symptoms
            ORDER BY score DESC
            LIMIT $top_k
            """
        elif scoring == "v18_coverage":
            # v18: confirmed / (total_symptoms + 1) × confirmed
            # 500 cases: 88.4% GTPA@1, IL 20.1
            query = """
            MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
            WHERE confirmed.cui IN $confirmed_cuis
            WITH DISTINCT d
            OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
            WITH d,
                 count(DISTINCT s) AS total_symptoms,
                 count(DISTINCT CASE WHEN s.cui IN $confirmed_cuis THEN s END) AS confirmed_count,
                 count(DISTINCT CASE WHEN s.cui IN $denied_cuis THEN s END) AS denied_count
            WHERE confirmed_count > 0
            WITH d, confirmed_count, total_symptoms,
                 toFloat(confirmed_count) / (toFloat(total_symptoms) + 1.0) * toFloat(confirmed_count) AS raw_score
            WHERE raw_score > 0
            WITH collect({
                cui: d.cui, name: d.name, raw_score: raw_score,
                confirmed_count: confirmed_count, total_symptoms: total_symptoms
            }) AS all_candidates
            WITH all_candidates,
                 reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
            UNWIND all_candidates AS c
            RETURN c.cui AS cui, c.name AS name,
                   CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
                   c.confirmed_count AS confirmed_count,
                   c.total_symptoms AS total_symptoms
            ORDER BY score DESC
            LIMIT $top_k
            """
        elif scoring == "v15_ratio":
            # v15: confirmed/(confirmed+denied) ratio × confirmed
            # 100 cases: 89% GTPA@1, IL 14.8
            query = """
            MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
            WHERE confirmed.cui IN $confirmed_cuis
            WITH DISTINCT d
            OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
            WITH d,
                 count(DISTINCT s) AS total_symptoms,
                 count(DISTINCT CASE WHEN s.cui IN $confirmed_cuis THEN s END) AS confirmed_count,
                 count(DISTINCT CASE WHEN s.cui IN $denied_cuis THEN s END) AS denied_count
            WHERE confirmed_count > 0
            WITH d, confirmed_count, total_symptoms,
                 toFloat(confirmed_count) / (toFloat(confirmed_count) + toFloat(denied_count) + 1.0) *
                 toFloat(confirmed_count) AS raw_score
            WHERE raw_score > 0
            WITH collect({
                cui: d.cui, name: d.name, raw_score: raw_score,
                confirmed_count: confirmed_count, total_symptoms: total_symptoms
            }) AS all_candidates
            WITH all_candidates,
                 reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
            UNWIND all_candidates AS c
            RETURN c.cui AS cui, c.name AS name,
                   CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
                   c.confirmed_count AS confirmed_count,
                   c.total_symptoms AS total_symptoms
            ORDER BY score DESC
            LIMIT $top_k
            """
        else:
            # v7: confirmed - 0.5 × denied (기존 방식)
            # 100 cases: 75% GTPA@1, IL 11.0
            query = """
            MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
            WHERE confirmed.cui IN $confirmed_cuis
            WITH DISTINCT d
            OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
            WITH d,
                 count(DISTINCT s) AS total_symptoms,
                 count(DISTINCT CASE WHEN s.cui IN $confirmed_cuis THEN s END) AS confirmed_count,
                 count(DISTINCT CASE WHEN s.cui IN $denied_cuis THEN s END) AS denied_count
            WHERE confirmed_count > 0
            WITH d, confirmed_count, total_symptoms,
                 toFloat(confirmed_count) - 0.5 * toFloat(denied_count) AS raw_score
            WHERE raw_score > 0
            WITH collect({
                cui: d.cui, name: d.name, raw_score: raw_score,
                confirmed_count: confirmed_count, total_symptoms: total_symptoms
            }) AS all_candidates
            WITH all_candidates,
                 reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score
            UNWIND all_candidates AS c
            RETURN c.cui AS cui, c.name AS name,
                   CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
                   c.confirmed_count AS confirmed_count,
                   c.total_symptoms AS total_symptoms
            ORDER BY score DESC
            LIMIT $top_k
            """
        with self.driver.session() as session:
            result = session.run(
                query,
                confirmed_cuis=list(_confirmed),
                denied_cuis=list(_denied),
                top_k=top_k,
            )
            return [
                DiagnosisCandidate(
                    cui=r["cui"],
                    name=r["name"],
                    score=r["score"],
                    confirmed_count=r["confirmed_count"],
                    total_symptoms=r["total_symptoms"],
                )
                for r in result
            ]

    def should_stop(
        self,
        max_il: int = 50,
        min_il: int = 13,
        confidence_threshold: float = 0.30,
        gap_threshold: float = 0.005,
        relative_gap_threshold: float = 1.5,
    ) -> tuple[bool, str]:
        """진단 종료 조건 확인.

        파라미터 최적화 결과 (scripts/optimize_balanced_targets.py):
        목표: GTPA@1 > 83%, max_il < 1%, Avg IL <= 16

        최적 설정 (min_il_13_gap005):
        - min_il=13: GTPA@1 83.23% 달성
        - gap_threshold=0.005: max_il 0.85% 달성 (핵심 파라미터)
        - confidence_threshold=0.30
        - relative_gap_threshold=1.5
        - Avg IL=13.6

        Args:
            max_il: 최대 질문 수
            min_il: 최소 질문 수 (조기 종료 방지)
            confidence_threshold: Top-1 확신도 임계값
            gap_threshold: Top-1과 Top-2 차이 임계값
            relative_gap_threshold: Top-1/Top-2 비율 임계값

        Returns:
            (종료 여부, 종료 사유)
        """
        il = len(self.state.asked_cuis)

        # 조건 3: 최대 질문 수 도달
        if il >= max_il:
            return True, f"max_il_reached ({il})"

        candidates = self.get_diagnosis_candidates(top_k=2)

        # 조건 4: 유효한 진단 후보 없음
        if not candidates:
            return True, "no_candidates"

        # 최소 질문 수 미달 시 계속 진행
        if il < min_il:
            return False, ""

        # 조건 1: 단일 질환 확정 (min_il 이후만)
        if len(candidates) == 1:
            return True, "single_disease"

        top1_score = candidates[0].score
        top2_score = candidates[1].score if len(candidates) > 1 else 0.0

        # 조건 2a: Top-1 확신도 임계값
        if top1_score >= confidence_threshold:
            return True, f"confidence ({top1_score:.2f})"

        # 조건 2b: Top-1과 Top-2 절대 차이
        if top1_score - top2_score >= gap_threshold:
            return True, f"gap ({top1_score:.2f}-{top2_score:.2f}={top1_score - top2_score:.2f})"

        # 조건 2c: Top-1과 Top-2 상대 비율 (top1이 top2의 1.5배 이상)
        if top2_score > 0 and top1_score / top2_score >= relative_gap_threshold:
            return True, f"ratio ({top1_score:.2f}/{top2_score:.2f}={top1_score / top2_score:.1f}x)"

        return False, ""

    def get_top_diagnosis(self) -> DiagnosisCandidate | None:
        """Top-1 진단 반환."""
        candidates = self.get_diagnosis_candidates(top_k=1)
        return candidates[0] if candidates else None

    def get_explained_diagnosis_candidates(
        self,
        top_k: int = 5,
        *,
        confirmed_cuis: set[str] | None = None,
        denied_cuis: set[str] | None = None,
    ) -> list[ExplainedDiagnosis]:
        """설명이 포함된 진단 후보 반환.

        현재 KG 구조에서 제공 가능한 설명:
        1. 일치/불일치 증상 목록 (실제 이름)
        2. Coverage 기반 점수
        3. Synthesized 스타일 설명 텍스트

        Args:
            top_k: 반환할 최대 질환 수
            confirmed_cuis: confirmed 증상 CUI 집합
            denied_cuis: denied 증상 CUI 집합

        Returns:
            ExplainedDiagnosis 리스트 (score 내림차순)
        """
        _confirmed = confirmed_cuis if confirmed_cuis is not None else self.state.confirmed_cuis
        _denied = denied_cuis if denied_cuis is not None else self.state.denied_cuis

        if not _confirmed:
            return []

        # 증상 목록까지 포함하는 확장 쿼리
        query = """
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH DISTINCT d

        // 질환의 모든 증상 수집
        OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
        WITH d, collect(DISTINCT {cui: s.cui, name: s.name}) AS all_symptoms

        // 증상 분류
        WITH d, all_symptoms,
             [s IN all_symptoms WHERE s.cui IN $confirmed_cuis] AS matched,
             [s IN all_symptoms WHERE s.cui IN $denied_cuis] AS denied,
             [s IN all_symptoms WHERE NOT s.cui IN $confirmed_cuis AND NOT s.cui IN $denied_cuis] AS unasked

        WITH d,
             matched, denied, unasked,
             size(matched) AS matched_count,
             size(denied) AS denied_count,
             size(all_symptoms) AS total_symptoms

        WHERE matched_count > 0

        // 점수 계산 (v23_mild_denied)
        WITH d, matched, denied, unasked, matched_count, denied_count, total_symptoms,
             (toFloat(matched_count) / (toFloat(total_symptoms) + 1.0) * toFloat(matched_count))
             * (1.0 - 0.1 * toFloat(denied_count) / (toFloat(total_symptoms) + 1.0)) AS raw_score

        WHERE raw_score > 0

        WITH collect({
            cui: d.cui, name: d.name, raw_score: raw_score,
            matched: matched, denied: denied, unasked: unasked,
            matched_count: matched_count, denied_count: denied_count, total_symptoms: total_symptoms
        }) AS all_candidates

        WITH all_candidates,
             reduce(total = 0.0, c IN all_candidates | total + c.raw_score) AS total_score

        UNWIND all_candidates AS c

        RETURN c.cui AS cui, c.name AS name,
               CASE WHEN total_score > 0 THEN c.raw_score / total_score ELSE 0.0 END AS score,
               [s IN c.matched | s.name] AS matched_symptoms,
               [s IN c.denied | s.name] AS denied_symptoms,
               [s IN c.unasked | s.name] AS unasked_symptoms,
               c.matched_count AS matched_count,
               c.denied_count AS denied_count,
               c.total_symptoms AS total_symptoms
        ORDER BY score DESC
        LIMIT $top_k
        """

        results: list[ExplainedDiagnosis] = []

        with self.driver.session() as session:
            result = session.run(
                query,
                confirmed_cuis=list(_confirmed),
                denied_cuis=list(_denied),
                top_k=top_k,
            )

            for rank, record in enumerate(result, start=1):
                matched_count = record["matched_count"]
                total_symptoms = record["total_symptoms"]
                coverage = matched_count / total_symptoms if total_symptoms > 0 else 0.0

                # Synthesized 설명 생성
                explanation = self._generate_explanation(
                    name=record["name"],
                    score=record["score"],
                    rank=rank,
                    matched_symptoms=record["matched_symptoms"],
                    denied_symptoms=record["denied_symptoms"],
                    matched_count=matched_count,
                    denied_count=record["denied_count"],
                    total_symptoms=total_symptoms,
                    coverage=coverage,
                )

                results.append(
                    ExplainedDiagnosis(
                        cui=record["cui"],
                        name=record["name"],
                        score=record["score"],
                        rank=rank,
                        matched_symptoms=record["matched_symptoms"],
                        denied_symptoms=record["denied_symptoms"],
                        unasked_symptoms=record["unasked_symptoms"][:5],  # Top 5만
                        matched_count=matched_count,
                        denied_count=record["denied_count"],
                        total_symptoms=total_symptoms,
                        coverage=coverage,
                        explanation=explanation,
                    )
                )

        return results

    def _generate_explanation(
        self,
        name: str,
        score: float,
        rank: int,
        matched_symptoms: list[str],
        denied_symptoms: list[str],
        matched_count: int,
        denied_count: int,
        total_symptoms: int,
        coverage: float,
    ) -> str:
        """Synthesized 스타일 설명 텍스트 생성.

        Inventory 방식(단순 나열)이 아닌 Synthesized 방식(근거 설명)으로
        진단 오류 감소에 도움. (PMC6994315)
        """
        # 확신도 레벨
        if score >= 0.3:
            confidence = "LIKELY"
        elif score >= 0.15:
            confidence = "POSSIBLE"
        else:
            confidence = "UNLIKELY"

        lines = [f"{rank}. {name} ({score:.1%}) - {confidence}"]

        # 일치 증상
        if matched_symptoms:
            symptom_str = ", ".join(matched_symptoms[:5])
            if len(matched_symptoms) > 5:
                symptom_str += f" (+{len(matched_symptoms) - 5} more)"
            lines.append(f"   ✓ Matched: {symptom_str}")

        # 불일치 증상 (기대했으나 부정됨)
        if denied_symptoms:
            denied_str = ", ".join(denied_symptoms[:3])
            if len(denied_symptoms) > 3:
                denied_str += f" (+{len(denied_symptoms) - 3} more)"
            lines.append(f"   ✗ Denied: {denied_str}")

        # Coverage 정보
        lines.append(f"   Coverage: {matched_count}/{total_symptoms} ({coverage:.0%})")

        return "\n".join(lines)

    def create_diagnostic_trace(self, initial_symptom_name: str, initial_cui: str) -> DiagnosticTrace:
        """진단 추적 객체 생성.

        진단 과정의 해석가능성을 위해 모든 단계를 기록.

        Args:
            initial_symptom_name: 초기 증상 이름
            initial_cui: 초기 증상 CUI

        Returns:
            DiagnosticTrace 객체
        """
        return DiagnosticTrace(
            initial_symptom=initial_symptom_name,
            initial_symptom_cui=initial_cui,
        )

    def record_diagnostic_step(
        self,
        trace: DiagnosticTrace,
        symptom_name: str,
        symptom_cui: str,
        patient_response: str,
        selection_reason: str,
    ) -> None:
        """진단 단계 기록.

        각 질문-응답을 추적하여 해석가능성 제공.

        Args:
            trace: 진단 추적 객체
            symptom_name: 질문한 증상
            symptom_cui: 증상 CUI
            patient_response: "yes" 또는 "no"
            selection_reason: 증상 선택 이유
        """
        # 현재 Top 진단 가져오기
        candidates = self.get_diagnosis_candidates(top_k=2)
        top_diagnosis = candidates[0].name if candidates else "Unknown"
        top_score = candidates[0].score if candidates else 0.0
        candidates_remaining = len(candidates)

        # 진단 변화 감지
        previous_top = None
        diagnosis_changed = False
        if trace.steps:
            previous_top = trace.steps[-1].top_diagnosis
            diagnosis_changed = previous_top != top_diagnosis

        step = DiagnosticStep(
            step=len(trace.steps) + 1,
            symptom_asked=symptom_name,
            symptom_cui=symptom_cui,
            patient_response=patient_response,
            selection_reason=selection_reason,
            top_diagnosis=top_diagnosis,
            top_score=top_score,
            candidates_remaining=candidates_remaining,
            diagnosis_changed=diagnosis_changed,
            previous_top=previous_top,
        )

        trace.add_step(step)

        # 통계 업데이트
        if patient_response == "yes":
            trace.confirmed_count += 1
        else:
            trace.denied_count += 1

    def finalize_trace(self, trace: DiagnosticTrace, stop_reason: str) -> None:
        """진단 추적 완료.

        Args:
            trace: 진단 추적 객체
            stop_reason: 종료 이유
        """
        candidates = self.get_diagnosis_candidates(top_k=1)
        if candidates:
            trace.final_diagnosis = candidates[0].name
            trace.final_score = candidates[0].score
        trace.stop_reason = stop_reason

    def get_symptom_selection_reason(
        self,
        symptom: SymptomCandidate,
        is_initial: bool = False,
    ) -> SymptomSelectionReason:
        """증상 선택 이유 생성.

        왜 이 증상이 다음 질문으로 선택되었는지 설명.

        Args:
            symptom: 선택된 증상 후보
            is_initial: 초기 증상 여부

        Returns:
            SymptomSelectionReason 객체
        """
        if is_initial:
            reason = "initial_symptom"
        elif symptom.information_gain > 0:
            reason = f"information_gain ({symptom.information_gain:.3f})"
        else:
            reason = f"high_coverage ({symptom.disease_coverage} diseases)"

        top_diseases = [d.name for d in symptom.related_diseases[:3]]

        return SymptomSelectionReason(
            symptom_cui=symptom.cui,
            symptom_name=symptom.name,
            selection_reason=reason,
            disease_coverage=symptom.disease_coverage,
            top_diseases=top_diseases,
            information_gain=symptom.information_gain,
        )
