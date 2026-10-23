"""Seal the evaluation sets and the subsets reported from them, before any of their rows is evaluated.

    uv run python -m ddxkb.seal --root .. test                     # T: every row of the official test split
    uv run python -m ddxkb.seal --root .. dedup                    # T-dedup: T rows whose input is not in train
    uv run --group seal python -m ddxkb.seal --root .. streambench --release-file P [--check]   # T-streambench
    uv run python -m ddxkb.seal --root .. heldout                  # H1 and H2 of the first evaluation

A manifest stores the row indices (0-based, data rows after the header), a digest of their raw bytes and the
sha256 of the CSV, so any later change to the CSV or to the selection is detected. No subcommand overwrites a
manifest.

- T (design 보완 2) is the whole official test split; the 9,000-row exclusion of H1 does not apply to it.
- H1 is the official test split minus the 9,000 rows an earlier study used (development sample seed 0,
  design D confirmation seed 101, design E confirmation seed 202); H2 is a hold-out of the train split.
  Both belong to the first evaluation and are reported again only as subsets (H1) or not at all (H2).
- T-dedup keeps the T rows whose input (`data.input_key`: age, sex, initial evidence and the set of recorded
  evidence items) appears in no train row. It is computed from the input columns of train and T only.
- T-streambench lists the T rows of StreamBench's DDXPlus test set (Wu et al. 2024, 1,764 patients, 36 per
  disease). Each StreamBench row must equal a T row in its input, its pathology and its differential (names
  mapped from English through the KB); when any row has no such T row, the manifest records that the list is
  not available instead of a row list. The rows are read from the pinned release file itself (parquet, read by
  pandas with pyarrow, which only the `seal` dependency group installs), and the manifest records the sha256 of
  that file and of the rows as JSON (`rows_json`). `--check` rebuilds the manifest from the release file and
  compares it with the sealed one instead of writing, so anyone with the release file can reproduce it.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ddxkb.data import (
    CSV_COLUMNS,
    MANIFEST_DIR,
    TEST_CSV,
    TRAIN_CSV,
    input_key,
    iter_input_keys,
    load_kb,
    parse_differential,
    string_list,
)

TEST_ROWS = 134_529
TRAIN_ROWS = 1_025_602
H2_SEED = 20261022
STREAMBENCH_ROWS = 1_764
# The release the StreamBench rows were taken from: its DDXPlus test file and that file's LFS sha256.
STREAMBENCH_RELEASE = {
    "dataset": "appier-ai-research/StreamBench",
    "revision": "a387d5d86c4fab1b7cff724a3cdea0299a716b91",
    "file": "ddxplus/test-00000-of-00001.parquet",
    "file_sha256": "4bf0ef525749c07ce7a26e87e3d40d69a0396c100dc3294a0ca3da4eef1633c5",
}
STREAMBENCH_FIELDS = ("AGE", "SEX", "INITIAL_EVIDENCE", "EVIDENCES", "PATHOLOGY", "DIFFERENTIAL_DIAGNOSIS")
STREAMBENCH_ROWS_JSON = "the release file's rows as a JSON list of objects, columns and values unchanged"


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


def _target(root: Path, name: str) -> Path:
    path = root / MANIFEST_DIR / f"{name}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a seal is never overwritten")
    return path


def _write(path: Path, m: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m))
    shown = f"n={m['n']} rows_sha256={str(m['rows_sha256'])[:12]}" if "n" in m else "not available"
    print(f"wrote {path}: {shown}")


def _sealed_test(root: Path) -> tuple[Path, dict[str, Any]]:
    """The test CSV and the T manifest, after checking that the CSV is the file T was sealed from."""
    t_path = root / MANIFEST_DIR / "T.json"
    if not t_path.exists():
        raise SystemExit("seal T first")
    t: dict[str, Any] = json.loads(t_path.read_text())
    csv_path = root / str(t["csv"])
    if file_sha256(csv_path) != t["csv_sha256"]:
        raise SystemExit(f"{t['csv']} is not the file T was sealed from")
    return csv_path, t


def seal_heldout(root: Path) -> None:
    test_csv, train_csv = root / TEST_CSV, root / TRAIN_CSV
    used = v2_test_rows(root)
    h1 = sorted(set(range(TEST_ROWS)) - set(used))
    h2 = sorted(random.Random(H2_SEED).sample(range(TRAIN_ROWS), TEST_ROWS))
    sets: list[tuple[str, Path, list[int], int, dict[str, object]]] = [
        ("H1", test_csv, h1, TEST_ROWS, {"rule": "official test minus earlier-study rows", "excluded": used}),
        ("H2", train_csv, h2, TRAIN_ROWS, {"rule": "random.Random(seed).sample(range(n), 134529)", "seed": H2_SEED}),
    ]
    for name, csv_path, rows, n_total, how in sets:
        path = _target(root, name)
        m = manifest(name, csv_path, root, rows, how)
        if m["csv_data_rows"] != n_total:
            raise SystemExit(f"{name}: {m['csv_data_rows']} data rows, expected {n_total}")
        _write(path, m)


def seal_test(root: Path) -> None:
    path = _target(root, "T")
    n_rows = len(data_lines(root / TEST_CSV))
    if n_rows != TEST_ROWS:
        raise SystemExit(f"T: {n_rows} data rows, expected {TEST_ROWS}")
    _write(
        path, manifest("T", root / TEST_CSV, root, list(range(n_rows)), {"rule": "every data row of the official test"})
    )


def dedup_rows(train_csv: Path, test_csv: Path) -> list[int]:
    """Rows of `test_csv` whose `input_key` no row of `train_csv` has; input columns only."""
    seen = set(iter_input_keys(train_csv))
    return [r for r, key in enumerate(iter_input_keys(test_csv)) if key not in seen]


def seal_dedup(root: Path) -> None:
    path = _target(root, "T-dedup")
    csv_path, t = _sealed_test(root)
    train_csv = root / TRAIN_CSV
    rows = dedup_rows(train_csv, csv_path)
    how: dict[str, object] = {
        "rule": "T rows whose (age, sex, initial evidence, set of evidence items) is in no train row",
        "key": "ddxkb.data.input_key",
        "train_csv": TRAIN_CSV,
        "train_csv_sha256": file_sha256(train_csv),
        "excluded": int(t["n"]) - len(rows),
    }
    _write(path, manifest("T-dedup", csv_path, root, rows, how))


def _record_key(
    age: str, sex: str, initial: str, evidences: Sequence[str], pathology: str, diff: Sequence[tuple[str, float]]
) -> tuple[bytes, str, tuple[tuple[str, float], ...]]:
    return (input_key(age, sex, initial, evidences), pathology, tuple(sorted(diff)))


def match_streambench(root: Path, csv_path: Path, rows: list[dict[str, Any]]) -> tuple[list[int | None], int]:
    """The T row of each StreamBench row (None when there is none), and how many rows had more than one T row
    of identical content to choose from (the earliest unclaimed one is taken)."""
    kb = load_kb(root)
    to_fr = dict(zip(kb.disease_names_en, kb.diseases, strict=True))
    groups: dict[tuple[bytes, str, tuple[tuple[str, float], ...]], list[int]] = {}
    with pd.read_csv(csv_path, dtype=str, keep_default_na=False, chunksize=200_000) as reader:
        r = 0
        for chunk in reader:
            if tuple(chunk.columns) != CSV_COLUMNS:
                raise ValueError(f"{csv_path}: unexpected columns {list(chunk.columns)}")
            for age, diff, sex, pathology, evidences, initial in chunk.itertuples(index=False, name=None):
                key = _record_key(age, sex, initial, string_list(evidences), pathology, parse_differential(diff))
                groups.setdefault(key, []).append(r)
                r += 1
    claimed: set[int] = set()
    found: list[int | None] = []
    choices = 0
    for row in rows:
        names = [row["PATHOLOGY"], *(name for name, _ in ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"]))]
        if not all(name in to_fr for name in names):
            found.append(None)
            continue
        diff = tuple((to_fr[name], float(p)) for name, p in ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"]))
        evidences = string_list(row["EVIDENCES"])
        key = _record_key(
            str(row["AGE"]), row["SEX"], row["INITIAL_EVIDENCE"], evidences, to_fr[row["PATHOLOGY"]], diff
        )
        open_rows = [r for r in groups.get(key, []) if r not in claimed]
        if not open_rows:
            found.append(None)
            continue
        choices += len(groups[key]) > 1
        claimed.add(open_rows[0])
        found.append(open_rows[0])
    return found, choices


def streambench_rows(release_file: Path) -> list[dict[str, Any]]:
    """The rows of the StreamBench release file in file order, each a mapping of its columns to Python values."""
    frame = pd.read_parquet(release_file, engine="pyarrow")
    return [{str(k): v for k, v in row.items()} for row in frame.to_dict(orient="records")]


def rows_json(rows: list[dict[str, Any]]) -> bytes:
    """STREAMBENCH_ROWS_JSON: the rows serialized with sorted keys and non-ASCII characters kept, UTF-8."""
    return json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()


def streambench_manifest(root: Path, release_file: Path) -> dict[str, Any]:
    """The T-streambench manifest made from the release file, which must be the pinned one."""
    if file_sha256(release_file) != STREAMBENCH_RELEASE["file_sha256"]:
        raise SystemExit(f"{release_file} is not {STREAMBENCH_RELEASE['file']} of the pinned StreamBench release")
    csv_path, _ = _sealed_test(root)
    rows = streambench_rows(release_file)
    if not all(set(STREAMBENCH_FIELDS) <= set(r) for r in rows):
        raise SystemExit(f"{release_file}: every row needs the fields {STREAMBENCH_FIELDS}")
    origin = {
        **STREAMBENCH_RELEASE,
        "rows_json_sha256": hashlib.sha256(rows_json(rows)).hexdigest(),
        "rows_json": STREAMBENCH_ROWS_JSON,
    }
    found, choices = match_streambench(root, csv_path, rows)
    missing = [i for i, r in enumerate(found) if r is None]
    if len(rows) != STREAMBENCH_ROWS or missing:
        reason = f"{len(missing)} of {len(rows)} StreamBench rows equal no T row (expected {STREAMBENCH_ROWS} rows)"
        return {"set": "T-streambench", "available": False, "reason": reason, "source": origin}
    how: dict[str, object] = {
        "rule": "the T row equal to each StreamBench test row in input, pathology and differential",
        "source": origin,
        "content_identical_choices": choices,
        "source_order": found,
    }
    return manifest("T-streambench", csv_path, root, sorted(r for r in found if r is not None), how)


def seal_streambench(root: Path, release_file: Path) -> None:
    path = _target(root, "T-streambench")
    _write(path, streambench_manifest(root, release_file))


def check_streambench(root: Path, release_file: Path) -> None:
    """Refuses unless the sealed T-streambench manifest is the one the release file gives now."""
    path = root / MANIFEST_DIR / "T-streambench.json"
    if not path.exists():
        raise SystemExit(f"{path} does not exist; nothing to check")
    sealed = json.loads(path.read_text())
    rebuilt = json.loads(json.dumps(streambench_manifest(root, release_file)))
    differing = sorted(k for k in sealed.keys() | rebuilt.keys() if sealed.get(k) != rebuilt.get(k))
    if differing:
        raise SystemExit(f"{path} is not the manifest the release file gives: {differing} differ")
    print(f"{path} is reproduced from {release_file}")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ddxkb.seal", description="Seal evaluation sets and their subsets.")
    ap.add_argument("--root", type=Path, default=Path(".."), help="repository root")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("test", help="T: every row of the official test split")
    sub.add_parser("dedup", help="T-dedup: T rows whose input is in no train row")
    sb = sub.add_parser("streambench", help="T-streambench: the T rows of StreamBench's DDXPlus test set")
    sb.add_argument("--release-file", type=Path, required=True, help=f"the release's {STREAMBENCH_RELEASE['file']}")
    sb.add_argument("--check", action="store_true", help="compare the sealed manifest with the release file instead")
    sub.add_parser("heldout", help="H1 and H2 of the first evaluation")
    a = ap.parse_args(argv)
    root = a.root.resolve()
    if a.command == "test":
        seal_test(root)
    elif a.command == "dedup":
        seal_dedup(root)
    elif a.command == "streambench" and a.check:
        check_streambench(root, a.release_file)
    elif a.command == "streambench":
        seal_streambench(root, a.release_file)
    else:
        seal_heldout(root)


if __name__ == "__main__":
    main()
