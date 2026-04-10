#!/usr/bin/env python3
"""Step 1a: UMLS .nlm 파일에서 MRCONSO.RRF 등 필요 파일 추출.

.nlm 파일은 ZIP 아카이브이며, 내부에 gzip 분할 파일 존재.
nested zip 구조: umls-2025AB-full.zip → .nlm (ZIP) → .RRF.aa.gz (gzip)

추출 후 SAB 필터링하여 필요한 vocabulary만 저장.

전제조건:
  - data/umls-2025AB-full.zip (NLM에서 다운로드)

사용법:
  uv run python pipeline/step1a_extract_umls.py

출력:
  - data/umls_extracted/MRCONSO.RRF (SAB 필터링됨)
  - data/umls_extracted/MRREL.RRF (SAB 필터링됨)
  - data/umls_extracted/MRSTY.RRF (전체)
"""

import gzip
import io
import zipfile
from pathlib import Path

UMLS_ZIP = Path("data/umls-2025AB-full.zip")
OUTPUT_DIR = Path("data/umls_extracted")

# 추출 대상: RRF 파일명 → (nlm 파일, gz 파트 목록)
NLM_CONTENTS = {
    "2025AB-full/2025ab-1-meta.nlm": [
        "2025AB/META/MRCONSO.RRF.aa.gz",
        "2025AB/META/MRCONSO.RRF.ab.gz",
        "2025AB/META/MRCONSO.RRF.ac.gz",
        "2025AB/META/MRSTY.RRF.gz",
    ],
    "2025AB-full/2025ab-2-meta.nlm": [
        "2025AB/META/MRREL.RRF.aa.gz",
        "2025AB/META/MRREL.RRF.ab.gz",
        "2025AB/META/MRREL.RRF.ac.gz",
        "2025AB/META/MRREL.RRF.ad.gz",
        "2025AB/META/MRREL.RRF.ae.gz",
    ],
}

# SAB 필터: GraphTrace + SymCat + RareBench에 필요한 vocabulary
REQUIRED_SABS = {
    "HPO",           # RareBench phenotypes
    "OMIM",          # RareBench diseases
    "ORPHANET",      # RareBench diseases
    "SNOMEDCT_US",   # 포괄적 임상 용어
    "ICD10CM",       # DDXPlus disease mapping
    "ICD10",         # DDXPlus disease mapping
    "MTH",           # UMLS 자체 cross-reference
    "MSH",           # MeSH
    "NCI",           # NCI Thesaurus
    "MEDLINEPLUS",   # 증상 이름 다양성
}


def extract_and_filter_mrconso(outer_zip: zipfile.ZipFile) -> None:
    """MRCONSO.RRF를 추출하면서 동시에 SAB 필터링. 메모리 효율적."""
    output_path = OUTPUT_DIR / "MRCONSO.RRF"
    nlm_name = "2025AB-full/2025ab-1-meta.nlm"
    parts = [
        "2025AB/META/MRCONSO.RRF.aa.gz",
        "2025AB/META/MRCONSO.RRF.ab.gz",
        "2025AB/META/MRCONSO.RRF.ac.gz",
    ]

    print(f"\n  MRCONSO.RRF 추출 + SAB 필터링...")
    print(f"    SAB 필터: {', '.join(sorted(REQUIRED_SABS))}")

    # nlm (inner zip)을 메모리에 로드
    print(f"    {nlm_name} 로딩 중... (2.2GB)")
    with outer_zip.open(nlm_name) as f:
        nlm_data = io.BytesIO(f.read())

    total = 0
    kept = 0
    with zipfile.ZipFile(nlm_data, "r") as inner_zip:
        with open(output_path, "w", encoding="utf-8") as out:
            for part_name in parts:
                print(f"    처리: {part_name}")
                with inner_zip.open(part_name) as gz_file:
                    with gzip.open(gz_file, "rt", encoding="utf-8") as rrf:
                        for line in rrf:
                            total += 1
                            fields = line.split("|")
                            if len(fields) > 11 and fields[11] in REQUIRED_SABS:
                                out.write(line)
                                kept += 1
                            if total % 2_000_000 == 0:
                                print(f"      {total:,} lines / {kept:,} kept")

    size_mb = output_path.stat().st_size / 1e6
    print(f"    완료: {total:,} → {kept:,} ({kept/total*100:.1f}%), {size_mb:.0f} MB")

    # 메모리 해제
    del nlm_data


def extract_and_filter_mrrel(outer_zip: zipfile.ZipFile) -> None:
    """MRREL.RRF를 추출하면서 SAB 필터링."""
    output_path = OUTPUT_DIR / "MRREL.RRF"
    nlm_name = "2025AB-full/2025ab-2-meta.nlm"
    parts = [
        "2025AB/META/MRREL.RRF.aa.gz",
        "2025AB/META/MRREL.RRF.ab.gz",
        "2025AB/META/MRREL.RRF.ac.gz",
        "2025AB/META/MRREL.RRF.ad.gz",
        "2025AB/META/MRREL.RRF.ae.gz",
    ]

    print(f"\n  MRREL.RRF 추출 + SAB 필터링...")
    print(f"    {nlm_name} 로딩 중... (1.9GB)")
    with outer_zip.open(nlm_name) as f:
        nlm_data = io.BytesIO(f.read())

    total = 0
    kept = 0
    with zipfile.ZipFile(nlm_data, "r") as inner_zip:
        with open(output_path, "w", encoding="utf-8") as out:
            for part_name in parts:
                print(f"    처리: {part_name}")
                with inner_zip.open(part_name) as gz_file:
                    with gzip.open(gz_file, "rt", encoding="utf-8") as rrf:
                        for line in rrf:
                            total += 1
                            fields = line.split("|")
                            # MRREL: SAB is column[10]
                            if len(fields) > 10 and fields[10] in REQUIRED_SABS:
                                out.write(line)
                                kept += 1
                            if total % 5_000_000 == 0:
                                print(f"      {total:,} lines / {kept:,} kept")

    size_mb = output_path.stat().st_size / 1e6
    print(f"    완료: {total:,} → {kept:,} ({kept/total*100:.1f}%), {size_mb:.0f} MB")
    del nlm_data


def extract_mrsty(outer_zip: zipfile.ZipFile) -> None:
    """MRSTY.RRF 추출 (필터링 없이 전체)."""
    output_path = OUTPUT_DIR / "MRSTY.RRF"
    nlm_name = "2025AB-full/2025ab-1-meta.nlm"
    part_name = "2025AB/META/MRSTY.RRF.gz"

    print(f"\n  MRSTY.RRF 추출 (전체)...")
    print(f"    {nlm_name} 로딩 중...")
    with outer_zip.open(nlm_name) as f:
        nlm_data = io.BytesIO(f.read())

    with zipfile.ZipFile(nlm_data, "r") as inner_zip:
        with inner_zip.open(part_name) as gz_file:
            with gzip.open(gz_file, "rt", encoding="utf-8") as rrf:
                with open(output_path, "w", encoding="utf-8") as out:
                    for line in rrf:
                        out.write(line)

    size_mb = output_path.stat().st_size / 1e6
    print(f"    완료: {size_mb:.0f} MB")
    del nlm_data


def verify_output() -> None:
    """추출 결과 검증."""
    print("\n" + "=" * 60)
    print("추출 결과")
    print("=" * 60)

    for rrf in ["MRCONSO.RRF", "MRREL.RRF", "MRSTY.RRF"]:
        path = OUTPUT_DIR / rrf
        if path.exists():
            size = path.stat().st_size
            line_count = 0
            with open(path, encoding="utf-8") as f:
                for _ in f:
                    line_count += 1
            print(f"  ✓ {rrf}: {size / 1e6:.0f} MB, {line_count:,} lines")
        else:
            print(f"  ✗ {rrf}: 없음")

    # MRCONSO SAB 분포
    mrconso = OUTPUT_DIR / "MRCONSO.RRF"
    if mrconso.exists():
        sab_counts: dict[str, int] = {}
        with open(mrconso, encoding="utf-8") as f:
            for line in f:
                parts = line.split("|")
                if len(parts) > 11:
                    sab = parts[11]
                    sab_counts[sab] = sab_counts.get(sab, 0) + 1

        print(f"\n  MRCONSO SAB 분포:")
        for sab, count in sorted(sab_counts.items(), key=lambda x: -x[1]):
            print(f"    {sab}: {count:,}")


def main() -> None:
    print("=" * 60)
    print("Step 1a: UMLS .nlm → RRF 추출 (SAB 필터링)")
    print("=" * 60)

    if (OUTPUT_DIR / "MRCONSO.RRF").exists():
        print(f"\n  [skip] {OUTPUT_DIR / 'MRCONSO.RRF'}가 이미 존재합니다.")
        verify_output()
        return

    if not UMLS_ZIP.exists():
        raise FileNotFoundError(
            f"{UMLS_ZIP}가 없습니다.\n"
            "NLM에서 UMLS 다운로드: https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n소스: {UMLS_ZIP} ({UMLS_ZIP.stat().st_size / 1e9:.1f} GB)")
    print(f"출력: {OUTPUT_DIR}/")

    with zipfile.ZipFile(UMLS_ZIP, "r") as outer_zip:
        # MRCONSO는 1-meta.nlm에서, MRREL은 2-meta.nlm에서 추출
        # 각각 별도로 처리하여 메모리 효율적으로 관리
        extract_and_filter_mrconso(outer_zip)
        extract_mrsty(outer_zip)
        extract_and_filter_mrrel(outer_zip)

    verify_output()

    print(f"\n완료!")
    print(f"\n다음 단계:")
    print(f"  uv run python pipeline/step2a_build_symcat_mappings.py  # SymCat 매핑")
    print(f"  uv run python pipeline/step2b_build_rarebench_mappings.py  # RareBench 매핑")


if __name__ == "__main__":
    main()
