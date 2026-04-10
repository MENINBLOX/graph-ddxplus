#!/usr/bin/env python3
"""Step 2b: RareBench HPO/OMIM/ORPHA ↔ UMLS CUI 매핑 생성.

RareBench의 HPO phenotype 코드와 OMIM/ORPHA 질병 코드를 UMLS CUI에 매핑.

매핑 전략:
  - HPO → CUI: MRCONSO.RRF에서 SAB=HPO 필터링
  - OMIM → CUI: MRCONSO.RRF에서 SAB=OMIM 필터링
  - ORPHA → CUI: MRCONSO.RRF에서 SAB=ORPHANET 필터링

전제조건:
  - external/RareBench/mapping/phenotype_mapping.json
  - external/RareBench/mapping/disease_mapping.json
  - data/umls_extracted/MRCONSO.RRF

사용법:
  uv run python pipeline/step2b_build_rarebench_mappings.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

RAREBENCH_DIR = Path("external/RareBench/mapping")
UMLS_DIR = Path("data/umls_extracted")
OUTPUT_DIR = Path("data/rarebench")


def build_source_code_index(
    sab_filter: set[str],
) -> dict[str, tuple[str, str]]:
    """MRCONSO.RRF에서 특정 SAB의 source code → (CUI, name) 인덱스."""
    mrconso_path = UMLS_DIR / "MRCONSO.RRF"
    if not mrconso_path.exists():
        raise FileNotFoundError(
            f"{mrconso_path}가 없습니다. MetamorphoSys로 UMLS 데이터를 추출하세요."
        )

    print(f"  MRCONSO.RRF 인덱싱 (SAB: {sab_filter})...")
    code_to_cui: dict[str, tuple[str, str]] = {}

    with open(mrconso_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            cui = parts[0]
            lang = parts[1]
            sab = parts[11]
            code = parts[13]
            name = parts[14]

            if sab in sab_filter and lang == "ENG":
                if code not in code_to_cui:
                    code_to_cui[code] = (cui, name)

    print(f"    {len(code_to_cui)} codes indexed")
    return code_to_cui


def build_hpo_mapping(
    phenotype_mapping: dict[str, str],
    hpo_to_cui: dict[str, tuple[str, str]],
) -> dict:
    """HPO code → UMLS CUI 매핑."""
    mapping = {}
    matched = 0
    failed = 0

    for hpo_code, hpo_name in phenotype_mapping.items():
        # MRCONSO의 HPO 코드는 "HP0000001" 형식일 수 있음
        # RareBench는 "HP:0000001" 형식
        code_variants = [
            hpo_code,
            hpo_code.replace(":", ""),
            hpo_code.replace("HP:", "HP"),
        ]

        result = None
        for variant in code_variants:
            if variant in hpo_to_cui:
                result = hpo_to_cui[variant]
                break

        if result:
            cui, umls_name = result
            mapping[hpo_code] = {
                "hpo_name": hpo_name,
                "umls_cui": cui,
                "umls_name": umls_name,
            }
            matched += 1
        else:
            mapping[hpo_code] = {
                "hpo_name": hpo_name,
                "umls_cui": None,
                "umls_name": None,
            }
            failed += 1

    return {
        "description": "RareBench HPO to UMLS CUI mapping",
        "method": "SAB=HPO in MRCONSO.RRF",
        "statistics": {
            "total": len(phenotype_mapping),
            "mapped": matched,
            "failed": failed,
            "coverage": f"{matched / len(phenotype_mapping) * 100:.1f}%",
        },
        "mapping": mapping,
    }


def build_disease_mapping(
    disease_mapping: dict[str, str],
    omim_to_cui: dict[str, tuple[str, str]],
    orpha_to_cui: dict[str, tuple[str, str]],
) -> dict:
    """OMIM/ORPHA code → UMLS CUI 매핑."""
    mapping = {}
    matched = 0
    failed = 0

    for code, disease_name in disease_mapping.items():
        result = None
        source = None

        if code.startswith("OMIM:"):
            omim_id = code.replace("OMIM:", "")
            code_variants = [omim_id, f"MTHU{omim_id}", code]
            for variant in code_variants:
                if variant in omim_to_cui:
                    result = omim_to_cui[variant]
                    source = "OMIM"
                    break

        elif code.startswith("ORPHA:"):
            orpha_id = code.replace("ORPHA:", "")
            code_variants = [orpha_id, code, f"ORPHA{orpha_id}"]
            for variant in code_variants:
                if variant in orpha_to_cui:
                    result = orpha_to_cui[variant]
                    source = "ORPHANET"
                    break

        if result:
            cui, umls_name = result
            mapping[code] = {
                "disease_name": disease_name,
                "umls_cui": cui,
                "umls_name": umls_name,
                "source": source,
            }
            matched += 1
        else:
            mapping[code] = {
                "disease_name": disease_name,
                "umls_cui": None,
                "umls_name": None,
                "source": None,
            }
            failed += 1

    return {
        "description": "RareBench disease (OMIM/ORPHA) to UMLS CUI mapping",
        "method": "SAB=OMIM/ORPHANET in MRCONSO.RRF",
        "statistics": {
            "total": len(disease_mapping),
            "mapped": matched,
            "failed": failed,
            "coverage": f"{matched / len(disease_mapping) * 100:.1f}%",
        },
        "mapping": mapping,
    }


def main() -> None:
    print("=" * 60)
    print("Step 2b: RareBench ↔ UMLS CUI 매핑 생성")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hpo_out = OUTPUT_DIR / "hpo_umls_mapping.json"
    disease_out = OUTPUT_DIR / "disease_umls_mapping.json"

    if hpo_out.exists() and disease_out.exists():
        print(f"\n  [skip] 매핑 파일이 이미 존재합니다.")
        return

    # Load RareBench data
    print("\n[1/4] RareBench 데이터 로딩...")
    with open(RAREBENCH_DIR / "phenotype_mapping.json") as f:
        phenotype_mapping = json.load(f)
    with open(RAREBENCH_DIR / "disease_mapping.json") as f:
        disease_mapping_raw = json.load(f)
    print(f"  Phenotypes: {len(phenotype_mapping)}, Diseases: {len(disease_mapping_raw)}")

    # Build UMLS indices
    print("\n[2/4] HPO → CUI 인덱스...")
    hpo_to_cui = build_source_code_index({"HPO"})

    print("\n[3/4] OMIM/ORPHANET → CUI 인덱스...")
    omim_to_cui = build_source_code_index({"OMIM"})
    orpha_to_cui = build_source_code_index({"ORPHANET"})

    # Build HPO mapping
    print("\n[4/4] 매핑 생성...")
    hpo_mapping = build_hpo_mapping(phenotype_mapping, hpo_to_cui)
    with open(hpo_out, "w") as f:
        json.dump(hpo_mapping, f, indent=2, ensure_ascii=False)
    stats = hpo_mapping["statistics"]
    print(f"  HPO: {stats['mapped']}/{stats['total']} ({stats['coverage']})")

    disease_result = build_disease_mapping(disease_mapping_raw, omim_to_cui, orpha_to_cui)
    with open(disease_out, "w") as f:
        json.dump(disease_result, f, indent=2, ensure_ascii=False)
    stats = disease_result["statistics"]
    print(f"  Disease: {stats['mapped']}/{stats['total']} ({stats['coverage']})")

    print("\n완료!")


if __name__ == "__main__":
    main()
