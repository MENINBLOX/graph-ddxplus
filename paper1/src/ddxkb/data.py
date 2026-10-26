"""Loaders for the DDXPlus knowledge base and patients, with the seal of the evaluation sets.

Development code gets the validation split and the whole train split (design 보완 2), or the train split without
the rows of H2, which `iter_train` skips unparsed. The sealed sets are
read only by `load_heldout`, and only when the environment sets DDXKB_UNSEAL=1: T, every row of the official
test split (the evaluation set of 보완 2), and H1 and H2 of the first evaluation. Subsets of T reported from
the same run (H1, T-dedup, T-streambench) are row lists in their manifests; `subset_rows` reads the list, never
a patient.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ddxkb.kb import KB

REPO_ROOT = Path(__file__).resolve().parents[3]
UNSEAL_ENV = "DDXKB_UNSEAL"
KB_ZIP = "data/ddxplus/20043374.zip"
VALIDATION_CSV = "data/ddxplus/release_validate_patients.csv"
TRAIN_CSV = "data/ddxplus/release_train_patients.csv"
TEST_CSV = "data/ddxplus/release_test_patients.csv"
MANIFEST_DIR = "paper1/manifests"
CSV_COLUMNS = ("AGE", "DIFFERENTIAL_DIAGNOSIS", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE")
INPUT_COLUMNS = ("AGE", "SEX", "EVIDENCES", "INITIAL_EVIDENCE")
VALUE_SEP = "_@_"

Heldout = Literal["H1", "H2", "T"]
Subset = Literal["H1", "T-dedup", "T-streambench"]
SUBSETS: tuple[Subset, ...] = ("H1", "T-dedup", "T-streambench")
EvidencePair = tuple[str, str | None]


class SealedError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class Patient:
    """`row` is the 0-based data row in the split's CSV, the indexing of the seal manifests."""

    split: str
    row: int
    age: int
    sex: str
    pathology: str
    initial: str
    evidences: tuple[EvidencePair, ...]
    differential: tuple[tuple[str, float], ...]


def read_release_json(root: Path = REPO_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """release_conditions.json and release_evidences.json of the KB zip, parsed with their JSON types."""
    with zipfile.ZipFile(root / KB_ZIP) as z:
        conditions: dict[str, Any] = json.loads(z.read("release_conditions.json"))
        evidences: dict[str, Any] = json.loads(z.read("release_evidences.json"))
    return conditions, evidences


def load_kb(root: Path = REPO_ROOT) -> KB:
    return KB.from_json(*read_release_json(root))


def parse_evidences(text: str) -> tuple[EvidencePair, ...]:
    """The CSV list literal -> (code, value) pairs; a binary evidence has value None."""
    return tuple(_split_pair(item) for item in string_list(text))


def _split_pair(item: str) -> EvidencePair:
    code, sep, value = item.partition(VALUE_SEP)
    return (sys.intern(code), sys.intern(value) if sep else None)


def string_list(text: str) -> list[str]:
    """A CSV list literal of strings (EVIDENCES) as a list."""
    # Release rows are list literals of plain single-quoted strings; splitting them is ~10x faster than
    # ast.literal_eval over 1M train rows. Any other quoting goes through literal_eval.
    if text == "[]":
        return []
    if text.startswith("['") and text.endswith("']") and '"' not in text and "\\" not in text:
        return text[2:-2].split("', '")
    value: Any = ast.literal_eval(text)
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"EVIDENCES is not a list of strings: {text[:80]!r}")
    return value


_DIFF_ITEM = re.compile(r"\[\s*'([^'\\\[\]]*)',\s*([-+0-9.eE]+)\s*\]")


def parse_differential(text: str) -> tuple[tuple[str, float], ...]:
    items = _DIFF_ITEM.findall(text)
    if len(items) == text.count("[") - 1:
        return tuple((sys.intern(name), float(p)) for name, p in items)
    value: Any = ast.literal_eval(text)
    return tuple((str(name), float(p)) for name, p in value)


def load_validation(root: Path = REPO_ROOT, limit: int | None = None) -> list[Patient]:
    """Validation rows in file order; `limit` keeps the first rows only."""
    return list(_read_rows(root / VALIDATION_CSV, "validate", row_ids=None, skip=None, nrows=limit))


def iter_train(root: Path = REPO_ROOT, chunk_rows: int = 100_000, without: Heldout | None = None) -> Iterator[Patient]:
    """Every train row in file order, for frequency estimation; `without` names a sealed set of train rows (H2,
    whose complement is the first protocol's frequency split), whose rows are skipped unparsed."""
    if without is None:
        yield from _read_rows(root / TRAIN_CSV, "train", row_ids=None, skip=None, chunk_rows=chunk_rows)
        return
    manifest = read_manifest(without, root)
    if manifest["csv"] != TRAIN_CSV:
        raise ValueError(f"{without} does not list train rows")
    held = frozenset(_indices(manifest))
    kept = [r for r in range(int(manifest["csv_data_rows"])) if r not in held]
    yield from _read_rows(root / TRAIN_CSV, "train", kept, skip=held.__contains__, chunk_rows=chunk_rows)


def load_heldout(name: Heldout, root: Path = REPO_ROOT) -> list[Patient]:
    """The sealed evaluation set `name` in file order. Refused unless DDXKB_UNSEAL=1."""
    if not unseal_requested():
        raise SealedError(f"{name} is sealed; only the frozen evaluation runner sets {UNSEAL_ENV}=1")
    manifest = read_manifest(name, root)
    rows = sorted(_indices(manifest))
    keep = frozenset(rows)
    split = "train" if name == "H2" else "test"
    return list(_read_rows(root / str(manifest["csv"]), split, rows, skip=lambda r: r not in keep))


def unseal_requested() -> bool:
    return os.environ.get(UNSEAL_ENV) == "1"


def read_manifest(name: str, root: Path = REPO_ROOT) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((root / MANIFEST_DIR / f"{name}.json").read_text())
    if manifest.get("set") != name:
        raise ValueError(f"manifest {name}.json describes set {manifest.get('set')!r}")
    return manifest


def subset_rows(name: Subset, root: Path = REPO_ROOT) -> frozenset[int] | None:
    """The T rows of a reporting subset, or None when its manifest records that the list was not available."""
    manifest = read_manifest(name, root)
    if not manifest.get("available", True):
        return None
    t = read_manifest("T", root)
    if (manifest["csv"], manifest["csv_sha256"]) != (t["csv"], t["csv_sha256"]):
        raise ValueError(f"{name} does not list rows of the CSV that T seals")
    rows = frozenset(_indices(manifest))
    if not rows <= frozenset(_indices(t)):
        raise ValueError(f"{name} lists rows outside T")
    return rows


def input_key(age: str, sex: str, initial: str, evidences: Sequence[str]) -> bytes:
    """Digest of a patient's input: age, sex, initial evidence and the set of recorded evidence items
    (`code` or `code_@_value`), the duplicate definition of T-dedup. The CSV text is used as is."""
    items = "\x1f".join(sorted(set(evidences)))
    return hashlib.blake2b(f"{age}\x1e{sex}\x1e{initial}\x1e{items}".encode(), digest_size=16).digest()


def iter_input_keys(path: Path, chunk_rows: int = 200_000) -> Iterator[bytes]:
    """`input_key` of every data row of a release CSV in file order, reading the input columns only."""
    with pd.read_csv(
        path, dtype=str, keep_default_na=False, usecols=list(INPUT_COLUMNS), chunksize=chunk_rows
    ) as reader:
        for chunk in reader:
            for age, sex, evidences, initial in chunk[list(INPUT_COLUMNS)].itertuples(index=False, name=None):
                yield input_key(age, sex, initial, string_list(evidences))


def _indices(manifest: dict[str, Any]) -> list[int]:
    return [int(i) for i in manifest["row_indices"]]


def _read_rows(
    path: Path,
    split: str,
    row_ids: Sequence[int] | None,
    skip: Callable[[int], bool] | None,
    nrows: int | None = None,
    chunk_rows: int = 200_000,
) -> Iterator[Patient]:
    """Parse the rows `skip` lets through; `row_ids` names them in order (None: rows are 0, 1, ...).

    Row r (0-based, after the header) is file line r + 1, the indexing of the seal manifests. Skipped
    rows are dropped by the CSV tokenizer and never become a Patient.
    """
    skiprows = None if skip is None else (lambda line: line > 0 and skip(line - 1))
    n = 0
    with pd.read_csv(
        path, dtype=str, keep_default_na=False, skiprows=skiprows, nrows=nrows, chunksize=chunk_rows
    ) as reader:
        for chunk in reader:
            if tuple(chunk.columns) != CSV_COLUMNS:
                raise ValueError(f"{path}: unexpected columns {list(chunk.columns)}")
            for age, diff, sex, pathology, evidences, initial in chunk.itertuples(index=False, name=None):
                if row_ids is not None and n >= len(row_ids):
                    raise ValueError(f"{path}: more rows than the {len(row_ids)} the manifest implies")
                yield Patient(
                    split=split,
                    row=n if row_ids is None else row_ids[n],
                    age=int(age),
                    sex=sys.intern(sex),
                    pathology=sys.intern(pathology),
                    initial=sys.intern(initial),
                    evidences=parse_evidences(evidences),
                    differential=parse_differential(diff),
                )
                n += 1
    if row_ids is not None and n != len(row_ids):
        raise ValueError(f"{path}: read {n} rows, the manifest implies {len(row_ids)}")
