from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ddxkb import seal
from ddxkb.data import KB_ZIP, MANIFEST_DIR, REPO_ROOT, TEST_CSV, TRAIN_CSV, load_kb, read_manifest, subset_rows
from ddxkb.seal import data_lines, dedup_rows, match_streambench, rows_digest
from helpers import make_repo, synth_rows, write_csv

ROOT = Path(__file__).resolve().parents[2]


def test_rows_digest_depends_on_selection_and_content(tmp_path: Path) -> None:
    csv = tmp_path / "x.csv"
    csv.write_bytes(b"h\na\nb\nc\n")
    lines = data_lines(csv)
    assert lines == [b"a", b"b", b"c"]
    assert rows_digest(lines, [0, 2]) != rows_digest(lines, [0, 1])
    assert rows_digest(lines, [0, 2]) == rows_digest([b"a", b"x", b"c"], [0, 2])


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A synthetic repository whose test split has 60 rows and no manifest of T or its subsets yet."""
    make_repo(tmp_path)
    for name in ("T", "T-dedup", "T-streambench"):
        (tmp_path / MANIFEST_DIR / f"{name}.json").unlink()
    monkeypatch.setattr(seal, "TEST_ROWS", 60)
    return tmp_path


def test_t_holds_every_test_row_and_is_never_overwritten(root: Path) -> None:
    seal.main(["--root", str(root), "test"])
    t = read_manifest("T", root)
    assert t["n"] == t["csv_data_rows"] == 60 and t["row_indices"] == list(range(60))
    assert t["rows_sha256"] == rows_digest(data_lines(root / TEST_CSV), list(range(60)))
    with pytest.raises(SystemExit, match="never overwritten"):
        seal.main(["--root", str(root), "test"])


def test_t_refuses_a_test_split_of_another_size(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seal, "TEST_ROWS", 61)
    with pytest.raises(SystemExit, match="expected 61"):
        seal.main(["--root", str(root), "test"])


def test_dedup_drops_test_rows_whose_input_is_in_train(tmp_path: Path) -> None:
    train = synth_rows(20, seed=5)
    test = synth_rows(6, seed=6)
    # Row 1 repeats a train input with another pathology and differential: only the input counts. Row 4 repeats
    # a train input with its evidences reordered, which is the same set.
    test[1] = {**train[3], "pathology": "Maladie4", "differential": [("Maladie4", 1.0)]}
    test[4] = {**train[7], "evidences": list(reversed(train[7]["evidences"]))}
    test[5] = {**train[7], "age": train[7]["age"] + 1}
    write_csv(tmp_path / "train.csv", train)
    write_csv(tmp_path / "test.csv", test)
    assert dedup_rows(tmp_path / "train.csv", tmp_path / "test.csv") == [0, 2, 3, 5]


def test_dedup_is_sealed_after_t_and_from_its_file(root: Path) -> None:
    with pytest.raises(SystemExit, match="seal T first"):
        seal.main(["--root", str(root), "dedup"])
    seal.main(["--root", str(root), "test"])
    seal.main(["--root", str(root), "dedup"])
    m = read_manifest("T-dedup", root)
    assert m["row_indices"] == dedup_rows(root / TRAIN_CSV, root / TEST_CSV)
    assert m["selection"]["excluded"] == 60 - m["n"]
    assert m["selection"]["train_csv_sha256"] == seal.file_sha256(root / TRAIN_CSV)
    assert subset_rows("T-dedup", root) == frozenset(m["row_indices"])
    with pytest.raises(SystemExit, match="never overwritten"):
        seal.main(["--root", str(root), "dedup"])


def _streambench(root: Path, rows: list[int]) -> list[dict[str, Any]]:
    """Test rows in StreamBench's format: English names, evidences sorted, differential as a string."""
    english = dict(zip(load_kb(root).diseases, load_kb(root).disease_names_en, strict=True))
    test = synth_rows(60, seed=3)
    out = []
    for r in rows:
        row = test[r]
        out.append(
            {
                "AGE": row["age"],
                "SEX": row["sex"],
                "INITIAL_EVIDENCE": row["initial"],
                "EVIDENCES": repr(sorted(row["evidences"])),
                "PATHOLOGY": english[row["pathology"]],
                "DIFFERENTIAL_DIAGNOSIS": repr([[english[d], p] for d, p in row["differential"]]),
                "PATIENT_PROFILE": "unused",
            }
        )
    return out


def test_streambench_rows_are_matched_by_input_pathology_and_differential(root: Path) -> None:
    rows = _streambench(root, [7, 3, 41])
    found, choices = match_streambench(root, root / TEST_CSV, rows)
    assert (found, choices) == ([7, 3, 41], 0)
    other = dict(rows[1])
    diff = ast.literal_eval(other["DIFFERENTIAL_DIAGNOSIS"])
    diff[0][1] += 0.01
    other["DIFFERENTIAL_DIAGNOSIS"] = repr(diff)
    assert match_streambench(root, root / TEST_CSV, [rows[0], other])[0] == [7, None]
    assert match_streambench(root, root / TEST_CSV, [{**rows[0], "PATHOLOGY": "Unknown"}])[0] == [None]


def test_identical_test_rows_are_each_claimed_once(root: Path) -> None:
    test = synth_rows(60, seed=3)
    test[50] = test[20]
    write_csv(root / TEST_CSV, test)
    rows = _streambench(root, [20, 20, 20])
    assert match_streambench(root, root / TEST_CSV, rows) == ([20, 50, None], 2)


RELEASE = b"stand-in for the release file"


@pytest.fixture
def release(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in release file pinned in place of StreamBench's, with T sealed; tests set its rows."""
    monkeypatch.setitem(seal.STREAMBENCH_RELEASE, "file_sha256", hashlib.sha256(RELEASE).hexdigest())
    seal.main(["--root", str(root), "test"])
    path = root / "release.parquet"
    path.write_bytes(RELEASE)
    return path


def _release_rows(monkeypatch: pytest.MonkeyPatch, release: Path, rows: list[dict[str, Any]]) -> None:
    def read(path: Path) -> list[dict[str, Any]]:
        assert path == release
        return [dict(r) for r in rows]

    monkeypatch.setattr(seal, "streambench_rows", read)


def _seal_streambench(root: Path, release_file: Path, *flags: str) -> None:
    seal.main(["--root", str(root), "streambench", "--release-file", str(release_file), *flags])


def test_streambench_manifest(root: Path, release: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seal, "STREAMBENCH_ROWS", 3)
    rows = _streambench(root, [7, 3, 41])
    _release_rows(monkeypatch, release, rows)
    other = root / "other.parquet"
    other.write_bytes(b"another file")
    with pytest.raises(SystemExit, match="not ddxplus/test"):
        _seal_streambench(root, other)
    _seal_streambench(root, release)
    m = read_manifest("T-streambench", root)
    assert (m["n"], m["row_indices"], m["selection"]["source_order"]) == (3, [3, 7, 41], [7, 3, 41])
    source = m["selection"]["source"]
    assert (
        source["rows_json_sha256"]
        == hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    )
    assert source["file_sha256"] == hashlib.sha256(RELEASE).hexdigest()
    assert subset_rows("T-streambench", root) == frozenset({3, 7, 41})
    with pytest.raises(SystemExit, match="never overwritten"):
        _seal_streambench(root, release)


def test_streambench_check_reproduces_the_sealed_manifest(
    root: Path, release: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(seal, "STREAMBENCH_ROWS", 3)
    with pytest.raises(SystemExit, match="nothing to check"):
        _seal_streambench(root, release, "--check")
    rows = _streambench(root, [7, 3, 41])
    _release_rows(monkeypatch, release, rows)
    _seal_streambench(root, release)
    sealed = (root / MANIFEST_DIR / "T-streambench.json").read_bytes()
    _seal_streambench(root, release, "--check")
    assert capsys.readouterr().out.strip().endswith(f"is reproduced from {release}")
    # Rows other than the sealed ones, even in content only, are refused, and the manifest is left as it was.
    changed = [dict(r) for r in rows]
    changed[0]["PATIENT_PROFILE"] = "edited"
    _release_rows(monkeypatch, release, changed)
    with pytest.raises(SystemExit, match=r"\['selection'\] differ"):
        _seal_streambench(root, release, "--check")
    _release_rows(monkeypatch, release, [rows[1], rows[0], rows[2]])
    with pytest.raises(SystemExit, match=r"\['selection'\] differ"):
        _seal_streambench(root, release, "--check")
    assert (root / MANIFEST_DIR / "T-streambench.json").read_bytes() == sealed


def test_streambench_without_a_full_match_is_recorded_as_not_available(
    root: Path, release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(seal, "STREAMBENCH_ROWS", 2)
    rows = _streambench(root, [7, 3])
    rows[1]["AGE"] += 1
    _release_rows(monkeypatch, release, rows)
    _seal_streambench(root, release)
    m = read_manifest("T-streambench", root)
    assert m["available"] is False and "1 of 2" in m["reason"] and "row_indices" not in m
    assert subset_rows("T-streambench", root) is None


def test_the_release_file_is_read_as_pyarrow_lists_its_rows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [
        {"AGE": 41, "SEX": "F", "PATHOLOGY": "Bronchite", "EVIDENCES": "['toux', 'trav1_@_N']", "NOTE": "é"},
        {"AGE": 7, "SEX": "M", "PATHOLOGY": "Croup", "EVIDENCES": "[]", "NOTE": ""},
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, tmp_path / "rows.parquet")
    read = seal.streambench_rows(tmp_path / "rows.parquet")
    assert read == pq.read_table(tmp_path / "rows.parquet").to_pylist() == rows
    assert [type(r["AGE"]) for r in read] == [int, int]
    assert seal.rows_json(read) == json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()


@pytest.mark.parametrize("name", ["H1", "H2", "T", "T-dedup", "T-streambench"])
def test_manifest_matches_csv(name: str) -> None:
    path = ROOT / MANIFEST_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{name} is not sealed")
    m = json.loads(path.read_text())
    if not m.get("available", True):
        pytest.skip(f"{name} has no row list")
    csv = ROOT / m["csv"]
    if not csv.exists():
        pytest.skip("DDXPlus CSV not present")
    assert m["n"] == len(m["row_indices"]) == len(set(m["row_indices"]))
    assert rows_digest(data_lines(csv), m["row_indices"]) == m["rows_sha256"]


def test_real_t_is_the_whole_test_split_and_holds_its_subsets() -> None:
    if not (REPO_ROOT / MANIFEST_DIR / "T.json").exists():
        pytest.skip("T is not sealed")
    t = read_manifest("T")
    assert (t["n"], t["csv"], t["row_indices"]) == (134_529, TEST_CSV, list(range(134_529)))
    if not (REPO_ROOT / TEST_CSV).exists():
        pytest.skip("DDXPlus test CSV not present")
    h1 = subset_rows("H1")
    assert h1 is not None and len(h1) == 125_529
    dedup = subset_rows("T-dedup")
    assert dedup is not None and 130_000 < len(dedup) < 134_529


@pytest.mark.skipif(not (REPO_ROOT / KB_ZIP).exists(), reason="DDXPlus release files not present")
def test_real_dedup_rows_follow_from_the_input_columns() -> None:
    if not (REPO_ROOT / MANIFEST_DIR / "T-dedup.json").exists() or not (REPO_ROOT / TRAIN_CSV).exists():
        pytest.skip("T-dedup is not sealed or the train CSV is absent")
    assert dedup_rows(REPO_ROOT / TRAIN_CSV, REPO_ROOT / TEST_CSV) == read_manifest("T-dedup")["row_indices"]
