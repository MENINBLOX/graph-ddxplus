"""Loaders for the DDXPlus knowledge base and patients, with the H1/H2 seal (design §1).

Development code gets the validation split and train minus H2. H1 and H2 rows are read only by
`load_heldout`, and only when the environment sets DDXKB_UNSEAL=1. The 9,000 official-test rows
outside H1 have no loader at all.
"""

from __future__ import annotations

import ast
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
MANIFEST_DIR = "paper1/manifests"
CSV_COLUMNS = ("AGE", "DIFFERENTIAL_DIAGNOSIS", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE")
VALUE_SEP = "_@_"

Heldout = Literal["H1", "H2"]
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


def load_kb(root: Path = REPO_ROOT) -> KB:
    with zipfile.ZipFile(root / KB_ZIP) as z:
        conditions = json.loads(z.read("release_conditions.json"))
        evidences = json.loads(z.read("release_evidences.json"))
    return KB.from_json(conditions, evidences)


def parse_evidences(text: str) -> tuple[EvidencePair, ...]:
    """The CSV list literal -> (code, value) pairs; a binary evidence has value None."""
    return tuple(_split_pair(item) for item in _string_list(text))


def _split_pair(item: str) -> EvidencePair:
    code, sep, value = item.partition(VALUE_SEP)
    return (sys.intern(code), sys.intern(value) if sep else None)


def _string_list(text: str) -> list[str]:
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


def iter_train_minus_h2(root: Path = REPO_ROOT, chunk_rows: int = 100_000) -> Iterator[Patient]:
    """Train rows outside H2, in file order, for frequency estimation."""
    manifest = read_manifest("H2", root)
    sealed = frozenset(_indices(manifest))
    open_rows = [r for r in range(int(manifest["csv_data_rows"])) if r not in sealed]
    path = root / str(manifest["csv"])
    yield from _read_rows(path, "train", open_rows, skip=sealed.__contains__, chunk_rows=chunk_rows)


def load_heldout(name: Heldout, root: Path = REPO_ROOT) -> list[Patient]:
    """The sealed evaluation set `name` in file order. Refused unless DDXKB_UNSEAL=1."""
    if not unseal_requested():
        raise SealedError(f"{name} is sealed; only the frozen evaluation runner sets {UNSEAL_ENV}=1")
    manifest = read_manifest(name, root)
    rows = sorted(_indices(manifest))
    keep = frozenset(rows)
    split = "test" if name == "H1" else "train"
    return list(_read_rows(root / str(manifest["csv"]), split, rows, skip=lambda r: r not in keep))


def unseal_requested() -> bool:
    return os.environ.get(UNSEAL_ENV) == "1"


def read_manifest(name: Heldout, root: Path = REPO_ROOT) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((root / MANIFEST_DIR / f"{name}.json").read_text())
    if manifest.get("set") != name:
        raise ValueError(f"manifest {name}.json describes set {manifest.get('set')!r}")
    return manifest


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
