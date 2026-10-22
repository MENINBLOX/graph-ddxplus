from __future__ import annotations

import collections
import csv
import mmap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ddxkb import data
from ddxkb.data import (
    REPO_ROOT,
    UNSEAL_ENV,
    Patient,
    SealedError,
    iter_train_minus_h2,
    load_heldout,
    load_validation,
    parse_differential,
    parse_evidences,
    read_manifest,
)
from ddxkb.kb import KB
from helpers import requires_data, write_csv, write_manifest


def test_parse_evidences_pairs() -> None:
    text = "['fievre', 'douleur_lieu_@_tete', 'douleur_lieu_@_dos', 'trav1_@_N', 'douleur_intens_@_3']"
    assert parse_evidences(text) == (
        ("fievre", None),
        ("douleur_lieu", "tete"),
        ("douleur_lieu", "dos"),
        ("trav1", "N"),
        ("douleur_intens", "3"),
    )
    assert parse_evidences("[]") == ()


def test_parse_evidences_falls_back_to_literal_syntax() -> None:
    assert parse_evidences('["it\'s", "a_@_b"]') == (("it's", None), ("a", "b"))
    with pytest.raises(ValueError):
        parse_evidences("[1, 2]")


def test_parse_differential() -> None:
    text = "[['Anémie', 0.25], ['Fibrillation auriculaire/Flutter auriculaire', 1.5e-05], ['TSVP', 0]]"
    assert parse_differential(text) == (
        ("Anémie", 0.25),
        ("Fibrillation auriculaire/Flutter auriculaire", 1.5e-05),
        ("TSVP", 0.0),
    )
    assert parse_differential("[]") == ()
    assert parse_differential('[["l\'x", 0.5]]') == (("l'x", 0.5),)


def _row(i: int) -> dict[str, Any]:
    return {
        "age": 20 + i,
        "sex": "M" if i % 2 else "F",
        "pathology": "Beta",
        "evidences": ["douleur", f"douleur_intens_@_{i % 4}", "trav1_@_N"],
        "initial": "douleur",
        "differential": [("Beta", 0.75), ("Gamma", 0.25)],
    }


@pytest.fixture
def sealed_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv(UNSEAL_ENV, raising=False)
    write_csv(tmp_path / data.VALIDATION_CSV, [_row(i) for i in range(5)])
    write_csv(tmp_path / "data/ddxplus/release_train_patients.csv", [_row(100 + i) for i in range(10)])
    write_csv(tmp_path / "data/ddxplus/release_test_patients.csv", [_row(200 + i) for i in range(6)])
    write_manifest(tmp_path, "H2", "data/ddxplus/release_train_patients.csv", 10, [7, 0, 4])
    write_manifest(tmp_path, "H1", "data/ddxplus/release_test_patients.csv", 6, [1, 2, 5])
    return tmp_path


def test_validation_loader_reads_every_row(sealed_root: Path) -> None:
    v = load_validation(sealed_root)
    assert [p.row for p in v] == [0, 1, 2, 3, 4]
    assert [p.age for p in v] == [20, 21, 22, 23, 24]
    assert v[3] == Patient(
        split="validate",
        row=3,
        age=23,
        sex="M",
        pathology="Beta",
        initial="douleur",
        evidences=(("douleur", None), ("douleur_intens", "3"), ("trav1", "N")),
        differential=(("Beta", 0.75), ("Gamma", 0.25)),
    )
    assert [p.row for p in load_validation(sealed_root, limit=2)] == [0, 1]


def test_train_iterator_never_yields_h2_rows(sealed_root: Path) -> None:
    rows = list(iter_train_minus_h2(sealed_root, chunk_rows=3))
    assert [p.row for p in rows] == [1, 2, 3, 5, 6, 8, 9]
    assert [p.age for p in rows] == [120 + r for r in (1, 2, 3, 5, 6, 8, 9)]
    assert {p.split for p in rows} == {"train"}


def test_train_iterator_detects_a_csv_that_disagrees_with_the_manifest(sealed_root: Path) -> None:
    write_csv(sealed_root / "data/ddxplus/release_train_patients.csv", [_row(100 + i) for i in range(11)])
    with pytest.raises(ValueError, match="manifest"):
        list(iter_train_minus_h2(sealed_root))


def test_heldout_is_refused_without_the_flag(sealed_root: Path) -> None:
    for name in ("H1", "H2"):
        with pytest.raises(SealedError):
            load_heldout(name, sealed_root)


def test_heldout_reads_exactly_the_manifest_rows(sealed_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The unseal switch is stubbed rather than set in the environment, so no test can open the real H1/H2.
    monkeypatch.setattr(data, "unseal_requested", lambda: True)
    h1 = load_heldout("H1", sealed_root)
    assert [(p.split, p.row, p.age) for p in h1] == [("test", 1, 221), ("test", 2, 222), ("test", 5, 225)]
    h2 = load_heldout("H2", sealed_root)
    assert [(p.split, p.row, p.age) for p in h2] == [("train", 0, 120), ("train", 4, 124), ("train", 7, 127)]


def test_manifest_set_name_is_checked(sealed_root: Path) -> None:
    (sealed_root / "paper1/manifests/H1.json").write_text('{"set": "H2", "row_indices": []}')
    with pytest.raises(ValueError, match="describes set"):
        read_manifest("H1", sealed_root)


def test_real_heldout_is_refused_without_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(UNSEAL_ENV, raising=False)
    for name in ("H1", "H2"):
        with pytest.raises(SealedError):
            load_heldout(name, REPO_ROOT)


def test_real_validation_rows(real_kb: KB, validation: list[Patient]) -> None:
    assert len(validation) == 132_448
    assert [p.row for p in validation[:3]] == [0, 1, 2]
    assert validation[-1].row == 132_447
    assert all(p.split == "validate" for p in validation)
    assert {p.pathology for p in validation} <= set(real_kb.diseases)
    sums = np.array([real_kb.disease_vector(p.differential).sum() for p in validation])
    assert np.allclose(sums, 1.0, atol=1e-6)


def _open_lines(path: Path, rows: Sequence[int], n_rows: int, sealed: frozenset[int]) -> dict[int, str]:
    """Data rows `rows` of the CSV as text, located by their newline offsets.

    Rows are found from the nearer end of the file, and a sealed row is skipped by its offsets without
    being decoded.
    """
    wanted = set(rows)
    if wanted & sealed:
        raise ValueError("a sealed row was requested")
    head = {r for r in wanted if r < n_rows // 2}
    tail = wanted - head
    out: dict[int, str] = {}
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        end = len(m) - 1 if m[-1:] == b"\n" else len(m)
        start = m.find(b"\n") + 1
        for r in range(max(head, default=-1) + 1):
            stop = m.find(b"\n", start)
            if r in head:
                out[r] = m[start:stop].decode().rstrip("\r")
            start = stop + 1
        stop = end
        for r in range(n_rows - 1, min(tail, default=n_rows) - 1, -1):
            begin = m.rfind(b"\n", 0, stop) + 1
            if r in tail:
                out[r] = m[begin:stop].decode().rstrip("\r")
            stop = begin - 1
    if out.keys() != wanted:
        raise ValueError("rows not found")
    return out


def test_open_lines_reads_rows_by_offset(sealed_root: Path) -> None:
    path = sealed_root / "data/ddxplus/release_train_patients.csv"
    lines = path.read_text().splitlines()[1:]
    got = _open_lines(path, [1, 2, 8, 9], 10, frozenset({0, 4, 7}))
    assert got == {r: lines[r] for r in (1, 2, 8, 9)}
    with pytest.raises(ValueError, match="sealed"):
        _open_lines(path, [4], 10, frozenset({4}))


@requires_data
def test_real_train_minus_h2_rows_are_the_raw_rows_they_claim() -> None:
    manifest = read_manifest("H2")
    path = REPO_ROOT / str(manifest["csv"])
    if not path.exists():
        pytest.skip("DDXPlus train CSV not present")
    n_rows = int(manifest["csv_data_rows"])
    sealed = frozenset(int(r) for r in manifest["row_indices"])
    head: list[Patient] = []
    tail: collections.deque[Patient] = collections.deque(maxlen=50)
    n = 0
    for p in iter_train_minus_h2():
        if len(head) < 50:
            head.append(p)
        tail.append(p)
        n += 1
    assert n == 1_025_602 - 134_529 == 891_073
    open_rows = [r for r in range(n_rows) if r not in sealed]
    sample = [*head, *tail]
    assert [p.row for p in sample] == open_rows[:50] + open_rows[-50:]
    raw = _open_lines(path, [p.row for p in sample], n_rows, sealed)
    for p in sample:
        [(age, diff, sex, pathology, evidences, initial)] = csv.reader([raw[p.row]])
        assert (p.age, p.sex, p.pathology, p.initial) == (int(age), sex, pathology, initial), p.row
        assert p.evidences == parse_evidences(evidences), p.row
        assert p.differential == parse_differential(diff), p.row
