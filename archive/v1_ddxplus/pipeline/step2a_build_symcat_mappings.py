#!/usr/bin/env python3
"""Step 2a: SymCat ↔ UMLS CUI 매핑 생성.

SymCat의 801 질병 + 474 증상을 UMLS CUI에 매핑.

매핑 전략:
  1. MRCONSO.RRF에서 영어 이름 → CUI 사전 구축
  2. Exact match → Lowercased exact → Partial match (순서)
  3. 매핑 결과를 JSON으로 저장

전제조건:
  - data/symcat/symcat-801-diseases.csv (step0에서 다운로드)
  - data/umls_extracted/MRCONSO.RRF (MetamorphoSys 추출)

사용법:
  uv run python pipeline/step2a_build_symcat_mappings.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

DATA_DIR = Path("data/symcat")
UMLS_DIR = Path("data/umls_extracted")
OUTPUT_DIR = Path("data/symcat")

# SymCat parse.py에서 가져온 파서
sys.path.insert(0, str(DATA_DIR))
from parse import parse_symcat_conditions, parse_symcat_symptoms


def build_umls_name_index() -> dict[str, tuple[str, str]]:
    """MRCONSO.RRF에서 영어 이름 → (CUI, preferred_name) 인덱스 구축."""
    mrconso_path = UMLS_DIR / "MRCONSO.RRF"
    if not mrconso_path.exists():
        raise FileNotFoundError(
            f"{mrconso_path}가 없습니다. MetamorphoSys로 UMLS 데이터를 추출하세요."
        )

    print("  MRCONSO.RRF 인덱싱...")
    name_to_cui: dict[str, tuple[str, str]] = {}  # lower_name → (CUI, original_name)
    preferred: dict[str, str] = {}  # CUI → preferred_name

    with open(mrconso_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            cui = parts[0]
            lang = parts[1]
            ts = parts[2]  # Term Status (P=preferred)
            name = parts[14]

            if lang != "ENG":
                continue

            lower = name.lower().strip()
            if lower not in name_to_cui:
                name_to_cui[lower] = (cui, name)

            if ts == "P" and cui not in preferred:
                preferred[cui] = name

    print(f"    인덱싱 완료: {len(name_to_cui)} unique names, {len(preferred)} CUIs")
    return name_to_cui, preferred


def match_name_to_cui(
    name: str,
    name_to_cui: dict,
) -> Optional[Tuple[str, str, str]]:
    """이름을 UMLS CUI에 매칭. Returns (cui, umls_name, method) or None."""
    lower = name.lower().strip()

    # 1. Exact match
    if lower in name_to_cui:
        cui, umls_name = name_to_cui[lower]
        return cui, umls_name, "exact"

    # 2. Normalized match (remove parentheses, extra spaces)
    normalized = re.sub(r"\s*\(.*?\)\s*", " ", lower).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized != lower and normalized in name_to_cui:
        cui, umls_name = name_to_cui[normalized]
        return cui, umls_name, "normalized"

    # 3. Token overlap match: UMLS 이름이 쿼리를 완전히 포함하거나 그 반대
    # 최소 길이 제한 + 길이 비율 제한으로 오매칭 방지
    if len(lower) > 8:
        best_match = None
        best_ratio = 0.0
        for umls_lower, (cui, umls_name) in name_to_cui.items():
            if len(umls_lower) < 5:
                continue
            # 한쪽이 다른 쪽을 완전 포함해야 하고
            if lower in umls_lower or umls_lower in lower:
                # 길이 비율이 0.5 이상이어야 함 (너무 짧은 부분매칭 방지)
                shorter = min(len(lower), len(umls_lower))
                longer = max(len(lower), len(umls_lower))
                ratio = shorter / longer
                if ratio > 0.5 and ratio > best_ratio:
                    best_match = (cui, umls_name)
                    best_ratio = ratio
        if best_match:
            return best_match[0], best_match[1], "partial"

    return None


def build_symcat_disease_mapping(
    conditions: dict,
    name_to_cui: dict[str, tuple[str, str]],
    preferred: dict[str, str],
) -> dict:
    """SymCat 질병명 → UMLS CUI 매핑."""
    mapping = {}
    matched = 0
    failed = 0

    for cond_slug, cond_data in conditions.items():
        disease_name = cond_data["condition_name"]
        result = match_name_to_cui(disease_name, name_to_cui)

        if result:
            cui, umls_name, method = result
            pref_name = preferred.get(cui, umls_name)
            mapping[disease_name] = {
                "symcat_slug": cond_slug,
                "umls_cui": cui,
                "umls_name": pref_name,
                "match_method": method,
            }
            matched += 1
        else:
            mapping[disease_name] = {
                "symcat_slug": cond_slug,
                "umls_cui": None,
                "umls_name": None,
                "match_method": "failed",
            }
            failed += 1

    return {
        "description": "SymCat disease to UMLS CUI mapping",
        "method": "Text matching against MRCONSO.RRF (exact → normalized → partial)",
        "statistics": {
            "total": len(conditions),
            "mapped": matched,
            "failed": failed,
            "coverage": f"{matched / len(conditions) * 100:.1f}%",
        },
        "mapping": mapping,
    }


def build_symcat_symptom_mapping(
    symptoms: dict,
    name_to_cui: dict[str, tuple[str, str]],
    preferred: dict[str, str],
) -> dict:
    """SymCat 증상명 → UMLS CUI 매핑."""
    mapping = {}
    matched = 0
    failed = 0

    for sym_slug, sym_data in symptoms.items():
        symptom_name = sym_data["name"]
        result = match_name_to_cui(symptom_name, name_to_cui)

        if result:
            cui, umls_name, method = result
            pref_name = preferred.get(cui, umls_name)
            mapping[symptom_name] = {
                "symcat_slug": sym_slug,
                "umls_cui": cui,
                "umls_name": pref_name,
                "match_method": method,
            }
            matched += 1
        else:
            mapping[symptom_name] = {
                "symcat_slug": sym_slug,
                "umls_cui": None,
                "umls_name": None,
                "match_method": "failed",
            }
            failed += 1

    return {
        "description": "SymCat symptom to UMLS CUI mapping",
        "method": "Text matching against MRCONSO.RRF (exact → normalized → partial)",
        "statistics": {
            "total": len(symptoms),
            "mapped": matched,
            "failed": failed,
            "coverage": f"{matched / len(symptoms) * 100:.1f}%",
        },
        "mapping": mapping,
    }


def main() -> None:
    print("=" * 60)
    print("Step 2a: SymCat ↔ UMLS CUI 매핑 생성")
    print("=" * 60)

    disease_out = OUTPUT_DIR / "disease_umls_mapping.json"
    symptom_out = OUTPUT_DIR / "symptom_umls_mapping.json"

    if disease_out.exists() and symptom_out.exists():
        print(f"\n  [skip] 매핑 파일이 이미 존재합니다.")
        print(f"    {disease_out}")
        print(f"    {symptom_out}")
        print("  재생성하려면 기존 파일을 삭제 후 재실행하세요.")
        return

    # Parse SymCat data
    print("\n[1/4] SymCat 데이터 파싱...")
    conditions = parse_symcat_conditions(str(DATA_DIR / "symcat-801-diseases.csv"))
    symptoms = parse_symcat_symptoms(str(DATA_DIR / "symcat-474-symptoms.csv"))
    print(f"  Conditions: {len(conditions)}, Symptoms: {len(symptoms)}")

    # Build UMLS index
    print("\n[2/4] UMLS 인덱스 구축...")
    name_to_cui, preferred = build_umls_name_index()

    # Map diseases
    print("\n[3/4] 질병 매핑...")
    disease_mapping = build_symcat_disease_mapping(conditions, name_to_cui, preferred)
    with open(disease_out, "w") as f:
        json.dump(disease_mapping, f, indent=2, ensure_ascii=False)
    stats = disease_mapping["statistics"]
    print(f"  결과: {stats['mapped']}/{stats['total']} ({stats['coverage']})")
    print(f"  저장: {disease_out}")

    # Map symptoms
    print("\n[4/4] 증상 매핑...")
    symptom_mapping = build_symcat_symptom_mapping(symptoms, name_to_cui, preferred)
    with open(symptom_out, "w") as f:
        json.dump(symptom_mapping, f, indent=2, ensure_ascii=False)
    stats = symptom_mapping["statistics"]
    print(f"  결과: {stats['mapped']}/{stats['total']} ({stats['coverage']})")
    print(f"  저장: {symptom_out}")

    print("\n완료!")


if __name__ == "__main__":
    main()
