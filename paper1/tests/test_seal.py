import json
from pathlib import Path

import pytest

from ddxkb.seal import data_lines, rows_digest

ROOT = Path(__file__).resolve().parents[2]


def test_rows_digest_depends_on_selection_and_content(tmp_path: Path) -> None:
    csv = tmp_path / "x.csv"
    csv.write_bytes(b"h\na\nb\nc\n")
    lines = data_lines(csv)
    assert lines == [b"a", b"b", b"c"]
    assert rows_digest(lines, [0, 2]) != rows_digest(lines, [0, 1])
    assert rows_digest(lines, [0, 2]) == rows_digest([b"a", b"x", b"c"], [0, 2])


@pytest.mark.parametrize("name", ["H1", "H2"])
def test_manifest_matches_csv(name: str) -> None:
    m = json.loads((ROOT / f"paper1/manifests/{name}.json").read_text())
    csv = ROOT / m["csv"]
    if not csv.exists():
        pytest.skip("DDXPlus CSV not present")
    assert m["n"] == len(m["row_indices"]) == len(set(m["row_indices"]))
    assert rows_digest(data_lines(csv), m["row_indices"]) == m["rows_sha256"]
