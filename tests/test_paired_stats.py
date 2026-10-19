"""paired_stats.macro_compare reproduces the design D T4 macro numbers that design E4 fixes as its regression check."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pilot" / "scripts"))

from paired_stats import compare, macro_compare  # noqa: E402

CONFIRM_D = ROOT / "pilot" / "cache" / "v2" / "confirmD"
SETS = ["Confirm-DDXPlus", "Confirm-Phenopacket-new", "Confirm-SymCat"]


def _load(name: str) -> dict:
    path = CONFIRM_D / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path} not present")
    return json.loads(path.read_text())


@pytest.mark.parametrize(("a", "b", "k", "diff", "ci"), [
    ("gemma_1hop", "fuse_test", 10, 2.67, [1.81, 3.58]),
    ("bm25", "fuse_test", 10, 5.10, [3.59, 6.67]),
    ("gemma_1hop", "fuse_test", 1, 0.96, [0.36, 1.54]),
])
def test_macro_reproduces_confirm_d(a: str, b: str, k: int, diff: float, ci: list[float]) -> None:
    m = macro_compare(_load(a), _load(b), k, SETS)
    assert m["sets"] == sorted(SETS)
    assert m["diff"] == diff
    assert m["ci"] == ci
    assert 0 <= m["p"] < 0.05


def _toy(sizes: dict[str, int]) -> dict:
    return {"results": [{"bench": s, "ranks_lineage": [1] * n} for s, n in sizes.items()]}


def test_macro_refuses_missing_or_unequal_sets() -> None:
    a = _toy({"X": 5, "Y": 5})
    with pytest.raises(ValueError, match="missing"):
        macro_compare(a, _toy({"X": 5}), 10, ["X", "Y"])
    with pytest.raises(ValueError, match="n differs"):
        macro_compare(a, _toy({"X": 5, "Y": 4}), 10, ["X", "Y"])


def test_compare_require_refuses_skip() -> None:
    a, b = _toy({"X": 5, "Y": 5}), _toy({"X": 5, "Y": 4})
    assert [r["bench"] for r in compare(a, b, 10, boot=20)] == ["X"]
    with pytest.raises(ValueError, match="Y"):
        compare(a, b, 10, boot=20, require=["X", "Y"])
