"""Seal the evaluation sets H1 and H2 (design §1) before any development.

H1 is the official test split minus the 9,000 rows an earlier study used (development sample seed 0,
design D confirmation seed 101, design E confirmation seed 202). H2 is a hold-out of the train split.
A manifest stores the row indices (0-based, data rows after the header) and a digest of their raw bytes,
so any later change to the CSV or to the selection is detected.

    uv run python -m ddxkb.seal --root ..
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

TEST_ROWS = 134_529
TRAIN_ROWS = 1_025_602
H2_SEED = 20261022


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def data_lines(path: Path) -> list[bytes]:
    with path.open("rb") as f:
        lines = f.read().split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return lines[1:]


def rows_digest(lines: list[bytes], indices: list[int]) -> str:
    h = hashlib.sha256()
    for i in indices:
        h.update(lines[i])
        h.update(b"\n")
    return h.hexdigest()


def v2_test_rows(root: Path) -> list[int]:
    dev = random.Random(0).sample(range(TEST_ROWS), 3000)
    d = json.loads((root / "data/confirm/MANIFEST.json").read_text())["ddxplus"]["confirm_row_indices"]
    e = json.loads((root / "data/confirm_e/MANIFEST.json").read_text())["ddxplus"]["row_indices"]
    used = set(dev) | set(d) | set(e)
    if len(used) != 9000:
        raise SystemExit(f"earlier study rows overlap: {len(used)} distinct instead of 9,000")
    return sorted(used)


def manifest(name: str, csv_path: Path, root: Path, indices: list[int], how: dict[str, object]) -> dict[str, object]:
    lines = data_lines(csv_path)
    return {
        "set": name,
        "csv": str(csv_path.relative_to(root)),
        "csv_sha256": file_sha256(csv_path),
        "csv_data_rows": len(lines),
        "selection": how,
        "n": len(indices),
        "rows_sha256": rows_digest(lines, indices),
        "row_indices": indices,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(".."), help="repository root")
    a = ap.parse_args()
    root = a.root.resolve()
    out = root / "paper1/manifests"
    test_csv = root / "data/ddxplus/release_test_patients.csv"
    train_csv = root / "data/ddxplus/release_train_patients.csv"
    used = v2_test_rows(root)
    h1 = sorted(set(range(TEST_ROWS)) - set(used))
    h2 = sorted(random.Random(H2_SEED).sample(range(TRAIN_ROWS), TEST_ROWS))
    sets: list[tuple[str, Path, list[int], int, dict[str, object]]] = [
        ("H1", test_csv, h1, TEST_ROWS, {"rule": "official test minus earlier-study rows", "excluded": used}),
        ("H2", train_csv, h2, TRAIN_ROWS, {"rule": "random.Random(seed).sample(range(n), 134529)", "seed": H2_SEED}),
    ]
    for name, csv_path, rows, n_total, how in sets:
        m = manifest(name, csv_path, root, rows, how)
        if len(data_lines(csv_path)) != n_total:
            raise SystemExit(f"{name}: {m['csv_data_rows']} data rows, expected {n_total}")
        path = out / f"{name}.json"
        if path.exists():
            raise SystemExit(f"{path} exists; a seal is never overwritten")
        path.write_text(json.dumps(m))
        print(f"wrote {path}: n={m['n']} rows_sha256={str(m['rows_sha256'])[:12]}")


if __name__ == "__main__":
    main()
