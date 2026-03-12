"""UMLS Knowledge Graph 인터페이스.

Neo4j 기반 UMLS KG 쿼리 및 진단 로직.
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
        """confirmed/denied 조건을 적용한 후보 증상 탐색 (Information Gain 기반)."""
        _confirmed = confirmed_cuis if confirmed_cuis is not None else self.state.confirmed_cuis
        _denied = denied_cuis if denied_cuis is not None else self.state.denied_cuis
        _asked = asked_cuis if asked_cuis is not None else self.state.asked_cuis

        query = """
        // DDXPlus 최적화: confirmed 증상에서 시작해 후보 질환만 유지
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH DISTINCT d

        // denied가 과도하게 누적된 질환은 조기 제외
        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        WITH d, count(DISTINCT denied) AS denied_count
        WHERE denied_count < 5

        WITH collect(DISTINCT d) AS valid_diseases
        WITH valid_diseases, size(valid_diseases) AS total
        WHERE total > 0

        UNWIND valid_diseases AS d
        MATCH (d)<-[:INDICATES]-(next:Symptom)
        WHERE NOT next.cui IN $confirmed_cuis
          AND NOT next.cui IN $denied_cuis
          AND NOT next.cui IN $asked_cuis
        WITH next, total, count(DISTINCT d) AS coverage,
             CASE WHEN next.is_antecedent = false THEN 0 ELSE 1 END AS priority

        WITH next, coverage, total, priority,
             abs(toFloat(coverage) - toFloat(total) / 2.0) AS distance_from_optimal
        WITH next, coverage, priority,
             CASE WHEN total > 0
                  THEN 1.0 - (distance_from_optimal / (toFloat(total) / 2.0 + 0.1))
                  ELSE 0.0 END AS ig_score

        RETURN next.cui AS cui,
               next.name AS name,
               coverage AS disease_coverage,
               priority,
               ig_score
        ORDER BY priority ASC, ig_score DESC
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
        scoring: str = "v23_mild_denied",
        *,
        confirmed_cuis: set[str] | None = None,
        denied_cuis: set[str] | None = None,
    ) -> list[DiagnosisCandidate]:
        """현재 상태 기반 진단 후보 반환.

        점수 계산 전략:

        v23_mild_denied (기본, 권장):
        - score = (confirmed / (total + 1) × confirmed) × (1 - 0.1 × denied/total)
        - v18_coverage + 약한 denied 패널티 (0.1)
        - GTPA@1: 89.2%, IL: 20.0 (500 cases)

        v18_coverage:
        - score = confirmed / (total_symptoms + 1) × confirmed
        - 질환의 총 증상 수를 고려하여 coverage 기반 점수 계산
        - GTPA@1: 88.4%, IL: 20.1 (500 cases)

        v15_ratio:
        - score = confirmed / (confirmed + denied + 1) × confirmed
        - GTPA@1: 85%, IL: 15.2 (300 cases)

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
        min_il: int = 3,
        confidence_threshold: float = 0.25,
        gap_threshold: float = 0.06,
        relative_gap_threshold: float = 2.0,
    ) -> tuple[bool, str]:
        """진단 종료 조건 확인.

        1000개 케이스 최적화 결과 (scripts/optimize_cypher_v6.py):
        - min_il: 3 (조기 종료 방지)
        - confidence_threshold: 0.25 (86% 정확도 목표)
        - gap_threshold: 0.06 (적절한 격차 기준)
        - relative_gap_threshold: 2.0 (Top-1이 Top-2의 2배 이상)
        - denied_threshold: 5 (Cypher 쿼리)

        최적화 결과 (1000 samples):
        - GTPA@1: 86.3%
        - Avg IL: 21.0

        비교 (AARLC baseline):
        - GTPA@1: 75.39%, IL: 25.75

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
