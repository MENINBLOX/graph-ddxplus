#!/usr/bin/env python3
"""Step 2: DDXPlus ↔ UMLS CUI 매핑 생성.

DDXPlus의 증상 코드/질환명을 UMLS CUI에 매핑하는 JSON 파일 생성.

- umls_mapping.json: 증상 코드 → CUI (UMLS API 검색)
- disease_umls_mapping.json: 질환명 → CUI (ICD-10 → MRCONSO)

전제조건:
  - Step 0 완료 (DDXPlus 데이터)
  - Step 1에서 추출한 UMLS MRCONSO.RRF

사용법:
  uv run python pipeline/step2_build_mappings.py
"""

import json
from pathlib import Path

DATA_DIR = Path("data/ddxplus")
UMLS_DIR = Path("data/umls_extracted")


def build_symptom_mapping() -> None:
    """DDXPlus 증상 코드 → UMLS CUI 매핑 생성."""
    output_path = DATA_DIR / "umls_mapping.json"
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        n = data.get("summary", {}).get("mapped", 0)
        print(f"  [skip] {output_path} 이미 존재 ({n}개 매핑)")
        return

    print("  증상 매핑 생성 중...")
    evidences_path = DATA_DIR / "release_evidences.json"
    with open(evidences_path) as f:
        evidences = json.load(f)

    # MRCONSO에서 영어 이름 → CUI 사전 구축
    print("    MRCONSO.RRF 인덱싱...")
    name_to_cui: dict[str, str] = {}
    with open(UMLS_DIR / "MRCONSO.RRF", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            cui = parts[0]
            lang = parts[1]
            name = parts[14].lower()
            if lang == "ENG":
                if name not in name_to_cui:
                    name_to_cui[name] = cui

    # 매핑 수행
    mapping = {}
    mapped = 0
    unmapped = 0

    for code, ev in evidences.items():
        question = ev.get("question_en", "")
        # question에서 검색어 추출: "Does the patient have X?" → "X"
        search_term = question.lower()
        for prefix in [
            "does the patient have ",
            "is the patient ",
            "has the patient ",
            "did the patient ",
        ]:
            if search_term.startswith(prefix):
                search_term = search_term[len(prefix) :].rstrip("?").strip()
                break

        # MRCONSO에서 검색
        cui = name_to_cui.get(search_term)
        if not cui:
            # 부분 매칭 시도
            for name, c in name_to_cui.items():
                if search_term in name or name in search_term:
                    cui = c
                    break

        if cui:
            # CUI 이름 가져오기
            cui_name = search_term.title()
            for name, c in name_to_cui.items():
                if c == cui:
                    cui_name = name.title()
                    break

            mapping[code] = {
                "cui": cui,
                "name": cui_name,
                "search": search_term,
            }
            mapped += 1
        else:
            unmapped += 1
            print(f"    [unmapped] {code}: {search_term}")

    output = {
        "summary": {"total": len(evidences), "mapped": mapped, "unmapped": unmapped},
        "mapping": mapping,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  저장: {output_path} ({mapped}/{len(evidences)} 매핑)")


def build_disease_mapping() -> None:
    """DDXPlus 질환 → UMLS CUI 매핑 생성 (ICD-10 경유)."""
    output_path = DATA_DIR / "disease_umls_mapping.json"
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        n = data.get("statistics", {}).get("mapped", 0)
        print(f"  [skip] {output_path} 이미 존재 ({n}개 매핑)")
        return

    print("  질환 매핑 생성 중...")
    conditions_path = DATA_DIR / "release_conditions.json"
    with open(conditions_path) as f:
        conditions = json.load(f)

    # ICD-10 → CUI 사전 구축
    print("    MRCONSO.RRF에서 ICD-10 매핑 인덱싱...")
    icd10_to_cui: dict[str, tuple[str, str]] = {}
    with open(UMLS_DIR / "MRCONSO.RRF", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            cui = parts[0]
            sab = parts[11]  # Source Abbreviation
            code = parts[13]  # Source code
            name = parts[14]

            if sab in ("ICD10CM", "ICD10"):
                normalized = code.upper().strip()
                if normalized not in icd10_to_cui:
                    icd10_to_cui[normalized] = (cui, name)

    # 매핑 수행
    mapping = {}
    mapped = 0
    failed = 0
    corrections = {"SLE": "M32"}  # 수동 보정

    for disease_name, cond in conditions.items():
        icd10 = cond.get("icd10-id", "")
        name_fr = cond.get("cond-name-fr", "")
        name_en = cond.get("cond-name-eng", disease_name)

        # ICD-10 정규화
        icd10_norm = icd10.upper().strip()
        if icd10_norm in corrections:
            icd10_norm = corrections[icd10_norm]

        # MRCONSO에서 검색
        result = icd10_to_cui.get(icd10_norm)
        if not result:
            # 상위 코드로 재시도 (J93.0 → J93)
            parent = icd10_norm.split(".")[0]
            result = icd10_to_cui.get(parent)

        if result:
            cui, umls_name = result
            mapping[disease_name] = {
                "disease_key": disease_name,
                "name_fr": name_fr,
                "name_en": name_en,
                "icd10_original": icd10,
                "icd10_normalized": icd10_norm,
                "umls_cui": cui,
                "umls_name": umls_name,
            }
            mapped += 1
        else:
            failed += 1
            print(f"    [failed] {disease_name} (ICD-10: {icd10})")

    output = {
        "description": "DDXPlus disease to UMLS CUI mapping via ICD-10",
        "method": "ICD-10 -> UMLS MRCONSO (SAB=ICD10CM/ICD10)",
        "corrections": corrections,
        "statistics": {"total": len(conditions), "mapped": mapped, "failed": failed},
        "mapping": mapping,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  저장: {output_path} ({mapped}/{len(conditions)} 매핑)")


def main() -> None:
    print("=" * 60)
    print("Step 2: DDXPlus ↔ UMLS CUI 매핑 생성")
    print("=" * 60)

    # 이미 매핑 파일이 있으면 MRCONSO 없이도 skip 가능
    sym_exists = (DATA_DIR / "umls_mapping.json").exists()
    dis_exists = (DATA_DIR / "disease_umls_mapping.json").exists()

    if sym_exists and dis_exists:
        print("\n  [skip] 매핑 파일이 이미 존재합니다.")
        print(f"    {DATA_DIR / 'umls_mapping.json'}")
        print(f"    {DATA_DIR / 'disease_umls_mapping.json'}")
        print("\n  재생성하려면 기존 파일을 삭제 후 재실행하세요.")
        print("\n완료!")
        return

    if not (UMLS_DIR / "MRCONSO.RRF").exists():
        raise FileNotFoundError(
            f"{UMLS_DIR}/MRCONSO.RRF 가 없습니다. Step 1을 먼저 실행하세요."
        )

    print("\n[1/2] 증상 매핑")
    build_symptom_mapping()

    print("\n[2/2] 질환 매핑")
    build_disease_mapping()

    print("\n완료!")


if __name__ == "__main__":
    main()
