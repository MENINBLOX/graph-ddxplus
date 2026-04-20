"""파일럿 테스트 설정."""
from pathlib import Path

# 디렉토리
PILOT_DIR = Path("pilot")
DATA_DIR = PILOT_DIR / "data"
RESULTS_DIR = PILOT_DIR / "results"
UMLS_DIR = Path("data/umls_extracted")

# 샘플 질환 5개 (DDXPlus 49개 중 다양한 유형 선정)
# 선정 기준: PubMed 문헌 풍부, 다양한 증상 프로필, 감별 진단에서 자주 혼동
SEED_DISEASES = {
    "Pneumonia": {
        "search_term": "pneumonia",
        "ddxplus_name": "Pneumonia",
        "description": "common respiratory infection, rich symptom profile",
    },
    "Pulmonary embolism": {
        "search_term": "pulmonary embolism",
        "ddxplus_name": "Pulmonary embolism",
        "description": "emergency, overlaps with pneumonia/GERD/panic",
    },
    "GERD": {
        "search_term": "gastroesophageal reflux disease",
        "ddxplus_name": "GERD",
        "description": "common, chest pain DDx with PE and panic",
    },
    "Panic attack": {
        "search_term": "panic attack",
        "ddxplus_name": "Panic attack",
        "description": "psychiatric, overlaps with PE and GERD",
    },
    "Bronchitis": {
        "search_term": "acute bronchitis",
        "ddxplus_name": "Bronchitis",
        "description": "respiratory, DDx with pneumonia",
    },
}

# 이 5개 질환은 DDXPlus에서 서로 감별 진단이 겹치는 그룹
# Pneumonia ↔ Bronchitis (호흡기)
# Pulmonary embolism ↔ Pneumonia (흉통+호흡곤란)
# GERD ↔ Pulmonary embolism ↔ Panic attack (흉통)

# PubMed 수집 설정
ABSTRACTS_PER_DISEASE = 100
MAX_PMC_FULLTEXT = 20  # 질환당 PMC full text 최대 건수

# DISO semantic types
DISO_TYPES = {
    "T047", "T184", "T033", "T034", "T191", "T046",
    "T048", "T037", "T019", "T020", "T190", "T049",
}

# API keys (from .env)
import os
PUBMED_API_KEY = os.environ.get("PUBMED_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
