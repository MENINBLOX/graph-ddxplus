from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ddxkb import freeze
from ddxkb.data import KB_ZIP, MANIFEST_DIR, TRAIN_CSV, load_kb
from ddxkb.freeze import (
    DEV_OUTPUTS,
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
from ddxkb.protocol import A3_EPSILONS, BUDGETS, MIN_GAIN, P2_VARIANTS, TAU, P1Config, p2_fixed
from ddxkb.provenance import environment, file_sha256
from helpers import SELECTION_COMMIT, make_repo, write_selected


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    make_repo(tmp_path)
    write_selected(tmp_path, P1Config(0.7, 1e-4, TAU))
    return tmp_path


def test_the_frozen_protocol_survives_its_yaml_round_trip(repo: Path) -> None:
    digest = write_frozen(repo)
    text = (repo / FROZEN_CONFIG).read_text()
    assert text.startswith("# The frozen protocol")
    config = yaml.safe_load(text)
    assert config["config_sha256"] == digest == config_digest(config)
    assert config == frozen_config(repo)
    assert config["P1"] == {"alpha": 0.7, "eps": 0.0001, "tau": 0.99}
    assert config["P2"] == {
        "P2-binary": {"mode": "binary", "age_sex": False, "lam": 1.0, "tau": 0.99},
        "P2-binary-agesex": {"mode": "binary", "age_sex": True, "lam": 1.0, "tau": 0.99},
        "P2-value": {"mode": "value", "age_sex": False, "lam": 1.0, "tau": 0.99},
        "P2-value-agesex": {"mode": "value", "age_sex": True, "lam": 1.0, "tau": 0.99},
    }
    assert config["selected"] == {"path": SELECTED, "sha256": file_sha256(repo / SELECTED), "commit": SELECTION_COMMIT}
    assert SELECTED == "paper1/outputs/v3/dev/selected.json"
    assert FROZEN_CONFIG == "paper1/configs/protocol_v3_frozen.yaml"
    assert DEV_OUTPUTS == "paper1/outputs/v3/"
    assert config["evaluation_set"] == "T"
    # The first evaluation's H1 and H2 are sealed in this repository too, and are no part of the protocol.
    assert all((repo / MANIFEST_DIR / f"{name}.json").exists() for name in ("H1", "H2"))
    assert list(config["manifests"]) == ["T", "T-dedup", "T-streambench"]
    assert config["manifests"]["T"] == file_sha256(repo / MANIFEST_DIR / "T.json")
    assert config["data"] == {f: file_sha256(repo / f) for f in (KB_ZIP, TRAIN_CSV)}
    assert config["budgets"] == list(BUDGETS) == [5, 10, 15, 20, 30]
    assert config["environment"] == environment()
    kb = load_kb(repo)
    assert config["constants"] == protocol_constants(kb)
    assert config["constants"]["official_disease_order"] == sorted(kb.diseases)
    assert config["constants"]["min_gain"] == MIN_GAIN
    assert (config["constants"]["max_il"], config["constants"]["hierarchy"]) == (31, "none")
    assert config["constants"]["age_bin_edges"] == [1, 5, 15, 30, 45, 60, 75]
    assert config["constants"]["a3_epsilons"] == list(A3_EPSILONS)
    assert config["subsets"] == {
        "T-dedup": {"available": True, "n": 30, "rule": "synthetic T-dedup"},
        "T-streambench": {"available": True, "n": 20, "rule": "synthetic T-streambench"},
    }


def test_the_constants_of_the_revision(toy_kb: KB) -> None:
    constants = protocol_constants(toy_kb)
    assert (constants["tau"], constants["sensitivity_taus"], constants["p2_lambda"]) == (
        0.99,
        [0.9, 0.95, 0.99, 0.999],
        1.0,
    )
    assert constants["ladder"] == [
        ["P2-binary", "P1"],
        ["P2-binary-agesex", "P2-binary"],
        ["P2-value", "P2-binary"],
        ["P2-value-agesex", "P2-value"],
        ["B-BED", "P2-value"],
    ]
    assert constants["bed"]["reported"] == {"gtpa1": 0.9947, "il": 5.47}
    assert constants["p1_wider"] == {"alpha": [0.1, 0.9], "eps": [0.1, 1e-05]}
    assert set(constants) == {
        "max_il",
        "hierarchy",
        "sensitivity_hierarchy",
        "tau",
        "sensitivity_taus",
        "p2_lambda",
        "min_gain",
        "eig_tie_atol",
        "tie_rtol",
        "postprocess_mass",
        "severe_below",
        "official_disease_order",
        "age_bin_edges",
        "bootstrap",
        "p1_wider",
        "ladder",
        "a2_prior_lambda",
        "a3_epsilons",
        "a4_hierarchies",
        "b2_seeds",
        "bed",
    }
    assert set(constants["bed"]) == {
        "stats",
        "stats_sha256",
        "unseen",
        "stop_utility",
        "max_questions",
        "utility_tie_atol",
        "reported",
    }
    assert constants["bed"]["stats_sha256"] == BED_STATS_SHA256 and constants["bed"]["max_questions"] == 30
    assert (constants["sensitivity_hierarchy"], constants["a4_hierarchies"]) == (
        "implicit",
        ["implicit", "parent_first", "no_children"],
    )


@pytest.mark.parametrize(
    ("policy", "key", "value", "message"),
    [
        ("P1", "tau", 0.95, r"P1 stops at tau 0\.95"),
        ("P2-value-agesex", "tau", 0.999, r"P2-value-agesex stops at tau 0\.999"),
        ("P2-binary", "lam", 0.5, r"P2-binary smooths with lambda 0\.5"),
    ],
)
def test_refuses_a_selection_whose_tau_or_lambda_is_not_the_fixed_one(
    tmp_path: Path, policy: str, key: str, value: float, message: str
) -> None:
    make_repo(tmp_path)
    p1 = P1Config(0.5, 1e-3, TAU)
    p2 = {v.name: p2_fixed(v).as_dict() for v in P2_VARIANTS}
    if policy == "P1":
        p1 = P1Config(0.5, 1e-3, value)
    else:
        p2[policy][key] = value
    write_selected(tmp_path, p1, p2)
    with pytest.raises(FreezeError, match=f"does not hold the fixed settings: {message}"):
        write_frozen(tmp_path)
    assert not (tmp_path / FROZEN_CONFIG).exists()


@pytest.mark.parametrize("p1", [P1Config(0.5, 1e-5, TAU), P1Config(0.9, 1e-4, TAU), P1Config(0.42, 0.5, TAU)])
def test_refuses_a_selected_p1_outside_the_nine_candidates(tmp_path: Path, p1: P1Config) -> None:
    make_repo(tmp_path)
    write_selected(tmp_path, p1)
    with pytest.raises(FreezeError, match=r"does not hold the fixed settings: P1 with alpha .* is not one of the nine"):
        write_frozen(tmp_path)
    assert not (tmp_path / FROZEN_CONFIG).exists()


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
        ["?? paper1/outputs/v3/dev/main.json", "?? paper1/tests/test_new.py"],
        ['R  paper1/outputs/v3/dev/a.json -> paper1/src/ddxkb/"b c".py'],
        ["?? paper1/outputs/v2/dev/main.json"],
    ],
)
def test_refuses_a_selection_made_with_uncommitted_code(tmp_path: Path, changes: list[str]) -> None:
    make_repo(tmp_path)
    write_selected(tmp_path, paper1_changes=changes)
    with pytest.raises(FreezeError, match="uncommitted changes under paper1/"):
        frozen_config(tmp_path)
    write_selected(tmp_path, paper1_changes=["?? paper1/outputs/v3/dev/main.json", " M paper1/outputs/v3/dev/x"])
    assert frozen_config(tmp_path)["selected"]["commit"] == SELECTION_COMMIT


def test_refuses_a_selection_without_its_commit_or_work_tree_state(repo: Path) -> None:
    selected = json.loads((repo / SELECTED).read_text())
    for key, value in (("commit", None), ("paper1_changes", None), ("design_changes", None)):
        meta = {**selected["provenance"], key: value}
        (repo / SELECTED).write_text(json.dumps({**selected, "provenance": meta}))
        with pytest.raises(FreezeError, match="commit and the state of paper1/ and docs/design/"):
            frozen_config(repo)


def test_refuses_a_selection_made_while_head_moved(repo: Path) -> None:
    selected = json.loads((repo / SELECTED).read_text())
    meta = {**selected["provenance"], "commit_at_end": "0" * 40}
    (repo / SELECTED).write_text(json.dumps({**selected, "provenance": meta}))
    with pytest.raises(FreezeError, match="while HEAD moved"):
        frozen_config(repo)


def test_refuses_a_selection_made_before_its_design_was_committed(tmp_path: Path) -> None:
    make_repo(tmp_path)
    write_selected(tmp_path, design_changes=[" M docs/design/ddxplus_kb_redo.md"])
    with pytest.raises(FreezeError, match="uncommitted changes under docs/design/"):
        frozen_config(tmp_path)


def test_porcelain_paths() -> None:
    assert porcelain_path("?? paper1/outputs/v3/dev/a.json") == "paper1/outputs/v3/dev/a.json"
    assert porcelain_path(" M paper1/src/x.py") == "paper1/src/x.py"
    assert porcelain_path("R  paper1/a.py -> paper1/outputs/v3/b.py") == "paper1/outputs/v3/b.py"
    assert porcelain_path('?? "paper1/outputs/v3/\\303\\251.json"') == "paper1/outputs/v3/\\303\\251.json"


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
    p2 = {v.name: p2_fixed(v).as_dict() for v in P2_VARIANTS}
    p2["P2-binary"] = p2_fixed(P2_VARIANTS[2]).as_dict()
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
