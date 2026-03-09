"""DDXPlus 데이터 로더.

DDXPlus 데이터셋과 UMLS 매핑을 로드하는 모듈.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Patient:
    """DDXPlus 환자 데이터."""

    age: int
    sex: str
    initial_evidence: str
    evidences: list[str]
    pathology: str  # Ground truth (French)
    differential_diagnosis: list[tuple[str, float]]

    @classmethod
    def from_row(cls, row: pd.Series) -> "Patient":
        """CSV row에서 Patient 생성."""
        # Parse EVIDENCES (string representation of list)
        evidences_str = row["EVIDENCES"]
        if isinstance(evidences_str, str):
            evidences = eval(evidences_str)  # Safe: controlled data
        else:
            evidences = []

        # Parse DIFFERENTIAL_DIAGNOSIS
        dd_str = row["DIFFERENTIAL_DIAGNOSIS"]
        if isinstance(dd_str, str):
            dd = eval(dd_str)
        else:
            dd = []

        return cls(
            age=int(row["AGE"]),
            sex=row["SEX"],
            initial_evidence=row["INITIAL_EVIDENCE"],
            evidences=evidences,
            pathology=row["PATHOLOGY"],
            differential_diagnosis=dd,
        )


@dataclass
class Evidence:
    """DDXPlus Evidence 정의."""

    code: str
    question_en: str
    data_type: str  # B, M, C
    is_antecedent: bool
    possible_values: list[str] = field(default_factory=list)


@dataclass
class Condition:
    """DDXPlus 질환 정의."""

    name: str  # English name (key)
    name_fr: str  # French name (PATHOLOGY에서 사용)
    name_eng: str
    icd10: str
    symptoms: list[str]
    antecedents: list[str]
    severity: int


class DDXPlusLoader:
    """DDXPlus 데이터셋 로더."""

    def __init__(self, data_dir: str | Path = "data/ddxplus") -> None:
        self.data_dir = Path(data_dir)
        self._evidences: dict[str, Evidence] | None = None
        self._conditions: dict[str, Condition] | None = None
        self._symptom_mapping: dict[str, dict] | None = None
        self._disease_mapping: dict[str, dict] | None = None
        self._fr_to_eng: dict[str, str] | None = None

    @property
    def evidences(self) -> dict[str, Evidence]:
        """Evidence 정의 로드."""
        if self._evidences is None:
            self._load_evidences()
        return self._evidences  # type: ignore

    @property
    def conditions(self) -> dict[str, Condition]:
        """질환 정의 로드."""
        if self._conditions is None:
            self._load_conditions()
        return self._conditions  # type: ignore

    @property
    def symptom_mapping(self) -> dict[str, dict]:
        """증상 → UMLS 매핑."""
        if self._symptom_mapping is None:
            self._load_symptom_mapping()
        return self._symptom_mapping  # type: ignore

    @property
    def disease_mapping(self) -> dict[str, dict]:
        """질환 → UMLS 매핑."""
        if self._disease_mapping is None:
            self._load_disease_mapping()
        return self._disease_mapping  # type: ignore

    @property
    def fr_to_eng(self) -> dict[str, str]:
        """프랑스어 질환명 → 영어 질환명."""
        if self._fr_to_eng is None:
            self._load_conditions()
        return self._fr_to_eng  # type: ignore

    def _load_evidences(self) -> None:
        """release_evidences.json 로드."""
        path = self.data_dir / "release_evidences.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self._evidences = {}
        for code, info in data.items():
            self._evidences[code] = Evidence(
                code=code,
                question_en=info.get("question_en", ""),
                data_type=info.get("data_type", "B"),
                is_antecedent=info.get("is_antecedent", False),
                possible_values=info.get("possible-values", []),
            )

    def _load_conditions(self) -> None:
        """release_conditions.json 로드."""
        path = self.data_dir / "release_conditions.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self._conditions = {}
        self._fr_to_eng = {}

        for name, info in data.items():
            condition = Condition(
                name=name,
                name_fr=info.get("cond-name-fr", ""),
                name_eng=info.get("cond-name-eng", name),
                icd10=info.get("icd10-id", ""),
                symptoms=list(info.get("symptoms", {}).keys()),
                antecedents=list(info.get("antecedents", {}).keys()),
                severity=info.get("severity", 0),
            )
            self._conditions[name] = condition
            self._fr_to_eng[condition.name_fr] = name

    def _load_symptom_mapping(self) -> None:
        """umls_mapping.json 로드."""
        path = self.data_dir / "umls_mapping.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._symptom_mapping = data.get("mapping", {})

    def _load_disease_mapping(self) -> None:
        """disease_umls_mapping.json 로드."""
        path = self.data_dir / "disease_umls_mapping.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._disease_mapping = data.get("mapping", {})

    def load_patients(
        self,
        split: str = "test",
        n_samples: int | None = None,
        random_state: int = 42,
        severity: int | None = None,
    ) -> list[Patient]:
        """환자 데이터 로드.

        Args:
            split: "test" or "train" or "validate"
            n_samples: 샘플 수 (None이면 전체)
            random_state: 랜덤 시드
            severity: 질환 심각도 필터 (1-5, None이면 전체)
                     1: mild, 2: moderate, 3: severe, 4: emergency, 5: critical

        Returns:
            Patient 리스트
        """
        path = self.data_dir / f"release_{split}_patients.csv"
        df = pd.read_csv(path)

        # Severity 필터링
        if severity is not None:
            # 질환 조건 로드
            _ = self.conditions  # Ensure loaded

            # French 이름 → severity 매핑
            pathology_to_severity = {}
            for cond in self.conditions.values():
                pathology_to_severity[cond.name_fr] = cond.severity

            df["SEVERITY"] = df["PATHOLOGY"].map(pathology_to_severity)
            df = df[df["SEVERITY"] == severity].reset_index(drop=True)

        if n_samples is not None and n_samples < len(df):
            # 시뮬레이션과 동일하게 처음 N개 사용 (재현 가능성)
            df = df.head(n_samples)

        return [Patient.from_row(row) for _, row in df.iterrows()]

    def get_symptom_cui(self, code: str) -> str | None:
        """DDXPlus 증상 코드 → UMLS CUI."""
        info = self.symptom_mapping.get(code)
        return info.get("cui") if info else None

    def get_disease_cui(self, name_eng: str) -> str | None:
        """DDXPlus 질환명 (영어) → UMLS CUI."""
        info = self.disease_mapping.get(name_eng)
        return info.get("umls_cui") if info else None

    def get_pathology_cui(self, pathology_fr: str) -> str | None:
        """DDXPlus PATHOLOGY (프랑스어) → UMLS CUI."""
        name_eng = self.fr_to_eng.get(pathology_fr)
        if name_eng:
            return self.get_disease_cui(name_eng)
        return None

    def build_cui_to_codes(self) -> dict[str, list[str]]:
        """UMLS CUI → DDXPlus 코드 역매핑 생성."""
        cui_to_codes: dict[str, list[str]] = {}
        for code, info in self.symptom_mapping.items():
            cui = info.get("cui")
            if cui:
                cui_to_codes.setdefault(cui, []).append(code)
        return cui_to_codes
