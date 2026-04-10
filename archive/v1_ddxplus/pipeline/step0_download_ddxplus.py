#!/usr/bin/env python3
"""Step 0: DDXPlus 데이터셋 다운로드.

HuggingFace에서 DDXPlus 데이터를 다운로드하여 data/ddxplus/에 저장.
"""

import json
import shutil
from pathlib import Path

DATA_DIR = Path("data/ddxplus")


def download_ddxplus() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "datasets 패키지가 필요합니다: uv add datasets"
        )

    print("=== DDXPlus 데이터셋 다운로드 ===")

    # 1. Patient CSV 다운로드
    for split in ["train", "validate", "test"]:
        csv_path = DATA_DIR / f"release_{split}_patients.csv"
        if csv_path.exists():
            print(f"  [skip] {csv_path} 이미 존재")
            continue

        print(f"  다운로드: {split}...")
        ds = load_dataset(
            "airi-institute/DDXPlus", split=split, trust_remote_code=True
        )
        ds.to_csv(str(csv_path), index=False)
        print(f"  저장: {csv_path} ({len(ds):,}건)")

    # 2. Metadata JSON 다운로드
    # DDXPlus 원본 리포지토리에서 evidences, conditions JSON 필요
    evidences_path = DATA_DIR / "release_evidences.json"
    conditions_path = DATA_DIR / "release_conditions.json"

    if not evidences_path.exists() or not conditions_path.exists():
        print("\n  [주의] release_evidences.json, release_conditions.json은")
        print("  DDXPlus GitHub에서 수동 다운로드가 필요합니다:")
        print("  https://github.com/mila-iqia/ddxplus/tree/master/data/release")
        print()

        try:
            import urllib.request

            base = "https://raw.githubusercontent.com/mila-iqia/ddxplus/master/data/release"
            for fname in ["release_evidences.json", "release_conditions.json"]:
                fpath = DATA_DIR / fname
                if fpath.exists():
                    print(f"  [skip] {fpath} 이미 존재")
                    continue
                url = f"{base}/{fname}"
                print(f"  다운로드: {url}")
                urllib.request.urlretrieve(url, str(fpath))
                print(f"  저장: {fpath}")
        except Exception as e:
            print(f"  자동 다운로드 실패: {e}")
            print("  수동으로 다운로드해주세요.")

    # 3. 검증
    print("\n=== 검증 ===")
    for fname in [
        "release_train_patients.csv",
        "release_validate_patients.csv",
        "release_test_patients.csv",
        "release_evidences.json",
        "release_conditions.json",
    ]:
        fpath = DATA_DIR / fname
        exists = fpath.exists()
        size = fpath.stat().st_size if exists else 0
        status = f"OK ({size:,} bytes)" if exists else "MISSING"
        print(f"  {fname}: {status}")


if __name__ == "__main__":
    download_ddxplus()
