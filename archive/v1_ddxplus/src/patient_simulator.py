"""환자 시뮬레이터.

DDXPlus EVIDENCES를 기반으로 환자 응답을 시뮬레이션.
"""

from dataclasses import dataclass
from enum import Enum

from src.data_loader import DDXPlusLoader, Patient


class ResponseType(Enum):
    """환자 응답 타입."""

    VALID_YES = "valid_yes"  # 역매핑 성공 + EVIDENCES 있음
    VALID_NO = "valid_no"  # 역매핑 성공 + EVIDENCES 없음
    INVALID = "invalid"  # 역매핑 실패


@dataclass
class PatientResponse:
    """환자 응답."""

    response_type: ResponseType
    ddxplus_codes: list[str]  # 매핑된 DDXPlus 코드들
    values: list[str]  # M/C 타입의 경우 구체적 값들
    cui: str  # 질문한 UMLS CUI


class PatientSimulator:
    """환자 시뮬레이터.

    LLM이 선택한 UMLS CUI를 DDXPlus 코드로 역매핑하고,
    EVIDENCES와 비교하여 응답을 생성.
    """

    def __init__(self, patient: Patient, loader: DDXPlusLoader) -> None:
        self.patient = patient
        self.loader = loader
        self._cui_to_codes = loader.build_cui_to_codes()
        self._evidences_set = set(patient.evidences)
        self._asked_cuis: set[str] = set()

    def ask(self, cui: str) -> PatientResponse:
        """UMLS CUI로 질문하고 응답 받기.

        Args:
            cui: 질문할 증상의 UMLS CUI

        Returns:
            PatientResponse
        """
        self._asked_cuis.add(cui)

        # Step 1: CUI → DDXPlus 코드 역매핑
        ddxplus_codes = self._cui_to_codes.get(cui, [])

        if not ddxplus_codes:
            # INVALID: DDXPlus에 없는 증상
            return PatientResponse(
                response_type=ResponseType.INVALID,
                ddxplus_codes=[],
                values=[],
                cui=cui,
            )

        # Step 2: EVIDENCES 확인
        matched_codes = []
        matched_values = []

        for code in ddxplus_codes:
            # Binary 타입: 코드 자체가 EVIDENCES에 있는지
            if code in self._evidences_set:
                matched_codes.append(code)
                continue

            # Multi/Categorical 타입: code_@_value 형태 검색
            for evidence in self.patient.evidences:
                if evidence.startswith(f"{code}_@_"):
                    matched_codes.append(code)
                    value = evidence.split("_@_", 1)[1]
                    matched_values.append(value)

        if matched_codes:
            return PatientResponse(
                response_type=ResponseType.VALID_YES,
                ddxplus_codes=list(set(matched_codes)),
                values=matched_values,
                cui=cui,
            )
        else:
            return PatientResponse(
                response_type=ResponseType.VALID_NO,
                ddxplus_codes=ddxplus_codes,
                values=[],
                cui=cui,
            )

    def get_initial_evidence_cui(self) -> str | None:
        """초기 증상(주호소)의 UMLS CUI 반환."""
        return self.loader.get_symptom_cui(self.patient.initial_evidence)

    def get_ground_truth_cui(self) -> str | None:
        """정답 질환의 UMLS CUI 반환."""
        return self.loader.get_pathology_cui(self.patient.pathology)

    @property
    def asked_cuis(self) -> set[str]:
        """지금까지 질문한 CUI 목록."""
        return self._asked_cuis.copy()

    @property
    def interaction_length(self) -> int:
        """현재까지의 IL (질문 수)."""
        return len(self._asked_cuis)
