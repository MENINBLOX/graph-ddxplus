from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ddxkb import freeze
from ddxkb.data import KB_ZIP, MANIFEST_DIR, TRAIN_CSV, load_kb
from ddxkb.freeze import (
    FROZEN_CONFIG,
    SELECTED,
    FreezeError,
    config_digest,
    frozen_config,
    porcelain_path,
    protocol_constants,
    write_frozen,
)
from ddxkb.kb import KB
from ddxkb.policies.bed import BED_STATS_SHA256
from ddxkb.protocol import A3_EPSILONS, BUDGETS, MIN_GAIN, P2_VARIANTS, P1Config, P2Config
from ddxkb.provenance import environment, file_sha256
from helpers import SELECTION_COMMIT, make_repo, write_selected


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    make_repo(tmp_path)
    write_selected(tmp_path, P1Config(0.7, 1e-4, 0.95))
    return tmp_path


def test_the_frozen_protocol_survives_its_yaml_round_trip(repo: Path) -> None:
    digest = write_frozen(repo)
    text = (repo / FROZEN_CONFIG).read_text()
    assert text.startswith("# The frozen protocol")
    config = yaml.safe_load(text)
    assert config["config_sha256"] == digest == config_digest(config)
    assert config == frozen_config(repo)
    assert config["P1"] == {"alpha": 0.7, "eps": 0.0001, "tau": 0.95}
    assert list(config["P2"]) == [v.name for v in P2_VARIANTS]
    assert config["selected"] == {"path": SELECTED, "sha256": file_sha256(repo / SELECTED), "commit": SELECTION_COMMIT}
    assert SELECTED == "paper1/outputs/v2/dev/selected.json"
    assert FROZEN_CONFIG == "paper1/configs/protocol_v2_frozen.yaml"
    assert config["evaluation_set"] == "T"
    assert list(config["manifests"]) == ["T", "H1", "T-dedup", "T-streambench", "H2"]
    assert config["manifests"]["H2"] == file_sha256(repo / MANIFEST_DIR / "H2.json")
    assert config["manifests"]["T"] == file_sha256(repo / MANIFEST_DIR / "T.json")
    assert config["data"] == {f: file_sha256(repo / f) for f in (KB_ZIP, TRAIN_CSV)}
    assert config["budgets"] == list(BUDGETS) == [5, 10, 15, 20, 30]
    assert config["environment"] == environment()
    kb = load_kb(repo)
    assert config["constants"] == protocol_constants(kb)
    assert config["constants"]["official_disease_order"] == sorted(kb.diseases)
    assert config["constants"]["first_protocol"] == {
        "max_il": 30,
        "frequencies_without": "H2",
        "age_bin_edges": [2, 5, 8, 10, 18, 30, 33, 45, 60, 75],
    }
    assert config["constants"]["min_gain"] == MIN_GAIN
    assert (config["constants"]["max_il"], config["constants"]["hierarchy"]) == (31, "none")
    assert config["constants"]["age_bin_edges"] == [1, 5, 15, 30, 45, 60, 75]
    assert config["constants"]["a3_epsilons"] == list(A3_EPSILONS)
    assert config["subsets"] == {
        "H1": {"available": True, "n": 57, "rule": "synthetic H1"},
        "T-dedup": {"available": True, "n": 30, "rule": "synthetic T-dedup"},
        "T-streambench": {"available": True, "n": 20, "rule": "synthetic T-streambench"},
    }


def test_the_judgment_constants_of_the_revision(toy_kb: KB) -> None:
    judgments = protocol_constants(toy_kb)["judgments"]
    assert judgments["J1"] == {"min_wilson_low": 0.95, "metrics": ["gtpa1", "gtpa1_official"]}
    assert judgments["J2"] == {"policy": "P2-binary", "max_gtpa1_diff": 0.01, "il_diff_high_below": 0.0}
    assert judgments["J3'"] == {
        "policy": "B-BED",
        "bed_gtpa1": 0.9947,
        "bed_il": 5.47,
        "gtpa1_tolerance": 0.003,
        "il_tolerance": 0.5,
        "metrics": ["gtpa1", "gtpa1_official"],
        "paired": "P2-value",
    }
    constants = protocol_constants(toy_kb)
    assert constants["bed"]["stats_sha256"] == BED_STATS_SHA256 and constants["bed"]["max_questions"] == 30
    assert (constants["sensitivity_hierarchy"], constants["a4_hierarchies"]) == (
        "implicit",
        ["implicit", "parent_first", "no_children"],
    )


def test_a_subset_without_a_row_list_is_frozen_as_unavailable(repo: Path) -> None:
    unavailable = {"set": "T-streambench", "available": False, "reason": "3 of 1764 rows equal no T row"}
    (repo / MANIFEST_DIR / "T-streambench.json").write_text(json.dumps(unavailable))
    assert frozen_config(repo)["subsets"]["T-streambench"] == {"available": False, "reason": unavailable["reason"]}


def test_refuses_without_the_sealed_subsets_or_the_data_digests(repo: Path) -> None:
    (repo / MANIFEST_DIR / "T-dedup.json").unlink()
    with pytest.raises(FreezeError, match="seal"):
        frozen_config(repo)
    make_repo(repo)
    selected = json.loads((repo / SELECTED).read_text())
    del selected["provenance"]["inputs"][TRAIN_CSV]
    (repo / SELECTED).write_text(json.dumps(selected))
    with pytest.raises(FreezeError, match="sha256"):
        frozen_config(repo)


@pytest.mark.parametrize(
    "changes",
    [
        [" M paper1/src/ddxkb/kb.py"],
        ["?? paper1/outputs/v2/dev/main.json", "?? paper1/tests/test_new.py"],
        ['R  paper1/outputs/v2/dev/a.json -> paper1/src/ddxkb/"b c".py'],
    ],
)
def test_refuses_a_selection_made_with_uncommitted_code(tmp_path: Path, changes: list[str]) -> None:
    make_repo(tmp_path)
    write_selected(tmp_path, paper1_changes=changes)
    with pytest.raises(FreezeError, match="uncommitted changes under paper1/"):
        frozen_config(tmp_path)
    write_selected(tmp_path, paper1_changes=["?? paper1/outputs/v2/dev/main.json", " M paper1/outputs/v2/dev/x"])
    assert frozen_config(tmp_path)["selected"]["commit"] == SELECTION_COMMIT


def test_refuses_a_selection_without_its_commit_or_work_tree_state(repo: Path) -> None:
    selected = json.loads((repo / SELECTED).read_text())
    for key, value in (("commit", None), ("paper1_changes", None)):
        meta = {**selected["provenance"], key: value}
        (repo / SELECTED).write_text(json.dumps({**selected, "provenance": meta}))
        with pytest.raises(FreezeError, match="commit and the state of paper1/"):
            frozen_config(repo)


def test_porcelain_paths() -> None:
    assert porcelain_path("?? paper1/outputs/v2/dev/a.json") == "paper1/outputs/v2/dev/a.json"
    assert porcelain_path(" M paper1/src/x.py") == "paper1/src/x.py"
    assert porcelain_path("R  paper1/a.py -> paper1/outputs/v2/b.py") == "paper1/outputs/v2/b.py"
    assert porcelain_path('?? "paper1/outputs/v2/\\303\\251.json"') == "paper1/outputs/v2/\\303\\251.json"


def test_a_frozen_protocol_is_never_overwritten(repo: Path) -> None:
    write_frozen(repo)
    with pytest.raises(FreezeError, match="never overwritten"):
        write_frozen(repo)


def test_refuses_a_smoke_selection(tmp_path: Path) -> None:
    make_repo(tmp_path)
    with pytest.raises(FreezeError, match="does not exist"):
        frozen_config(tmp_path)
    write_selected(tmp_path, limit=2000)
    with pytest.raises(FreezeError, match="limited to 2000"):
        write_frozen(tmp_path)
    assert not (tmp_path / FROZEN_CONFIG).exists()


def test_refuses_a_configuration_under_another_variant(tmp_path: Path) -> None:
    make_repo(tmp_path)
    p2 = {v.name: P2Config(v, 1.0, 0.99).as_dict() for v in P2_VARIANTS}
    p2["P2-binary"] = P2Config(P2_VARIANTS[2], 1.0, 0.99).as_dict()
    write_selected(tmp_path, p2=p2)
    with pytest.raises(FreezeError, match="P2-value configuration under P2-binary"):
        frozen_config(tmp_path)


def test_the_cli_prints_the_digest(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    freeze.main(["--root", str(repo)])
    digest = yaml.safe_load((repo / FROZEN_CONFIG).read_text())["config_sha256"]
    assert capsys.readouterr().out.strip().endswith(digest)
    with pytest.raises(SystemExit, match=r"freeze refused: .*never overwritten"):
        freeze.main(["--root", str(repo)])


def test_digest_ignores_key_order_and_the_recorded_digest(toy_kb: KB) -> None:
    a = {"x": 1, "y": [1e-6, 0.5], "config_sha256": "anything"}
    b = {"y": [1e-6, 0.5], "x": 1}
    assert config_digest(a) == config_digest(b)
    assert config_digest(a) != config_digest({**b, "x": 2})
    assert json.loads(json.dumps(protocol_constants(toy_kb))) == protocol_constants(toy_kb)
