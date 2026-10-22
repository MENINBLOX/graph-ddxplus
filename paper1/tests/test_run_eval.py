from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
import yaml

from ddxkb import data, experiment, run_eval
from ddxkb.data import MANIFEST_DIR, UNSEAL_ENV, Patient
from ddxkb.freeze import FROZEN_CONFIG, SELECTED, config_digest, protocol_constants, write_frozen
from ddxkb.frequencies import FrequencyCounts, count_frequencies
from ddxkb.kb import KB
from ddxkb.protocol import P2_VARIANTS, P1Config, P2Config
from ddxkb.provenance import file_sha256
from ddxkb.run_eval import FrozenProtocol, GateError, check_gates, evaluate, freeze_commit, judgments, policies
from helpers import SELECTED_P1, make_repo, write_selected


def _freeze(root: Path, p1: P1Config = SELECTED_P1) -> str:
    (root / FROZEN_CONFIG).unlink(missing_ok=True)
    write_selected(root, p1)
    return write_frozen(root)


def _rewrite(root: Path, config: dict[str, Any]) -> str:
    """Writes `config` as the protocol with its digest recomputed, as an edit by hand would."""
    config["config_sha256"] = config_digest(config)
    (root / FROZEN_CONFIG).write_text(yaml.safe_dump(config, sort_keys=False))
    return str(config["config_sha256"])


def _config(root: Path) -> dict[str, Any]:
    config: dict[str, Any] = yaml.safe_load((root / FROZEN_CONFIG).read_text())
    return config


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A frozen synthetic repository whose git state is clean and whose seal is lifted, both stubbed."""
    make_repo(tmp_path)
    _freeze(tmp_path)
    monkeypatch.setattr(run_eval, "unseal_requested", lambda: True)
    monkeypatch.setattr(run_eval, "uncommitted", lambda root: [])
    monkeypatch.setattr(run_eval, "git_head", lambda root: "c0ffee")
    monkeypatch.setattr(run_eval, "git_commits_touching", lambda root, path: ["f00d"])
    return tmp_path


def _digest(root: Path) -> str:
    return str(_config(root)["config_sha256"])


def test_all_gates_open(repo: Path) -> None:
    protocol = check_gates(repo, _digest(repo))
    assert protocol == FrozenProtocol(
        config=protocol.config, sha256=_digest(repo), commit="c0ffee", freeze_commit="f00d"
    )
    assert protocol.config["P1"] == {"alpha": 0.5, "eps": 0.001, "tau": 0.99}


def test_refuses_without_the_unseal_flag(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()
    monkeypatch.delenv(UNSEAL_ENV, raising=False)
    with pytest.raises(GateError, match="DDXKB_UNSEAL"):
        check_gates(repo, _digest(repo))


def test_refuses_a_dirty_work_tree(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval, "uncommitted", lambda root: [" M paper1/src/ddxkb/kb.py"])
    with pytest.raises(GateError, match="uncommitted"):
        check_gates(repo, _digest(repo))


def test_refuses_without_a_frozen_protocol(repo: Path) -> None:
    digest = _digest(repo)
    (repo / FROZEN_CONFIG).unlink()
    with pytest.raises(GateError, match="not frozen"):
        check_gates(repo, digest)


def test_refuses_an_edited_protocol(repo: Path) -> None:
    digest = _digest(repo)
    config = _config(repo)
    config["P1"]["tau"] = 0.95
    (repo / FROZEN_CONFIG).write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(GateError, match="recorded config_sha256"):
        check_gates(repo, digest)


def test_refuses_a_protocol_other_than_the_frozen_one(repo: Path) -> None:
    first = _digest(repo)
    second = _freeze(repo, P1Config(0.5, 1e-3, 0.95))
    check_gates(repo, second)
    with pytest.raises(GateError, match="not the frozen"):
        check_gates(repo, first)


@pytest.mark.parametrize("commits", [[], ["f00d", "beef"]])
def test_refuses_unless_one_commit_froze_the_protocol(
    repo: Path, monkeypatch: pytest.MonkeyPatch, commits: list[str]
) -> None:
    monkeypatch.setattr(run_eval, "git_commits_touching", lambda root, path: commits)
    with pytest.raises(GateError, match="exactly one commit"):
        check_gates(repo, _digest(repo))
    monkeypatch.setattr(run_eval, "git_commits_touching", lambda root, path: None)
    with pytest.raises(GateError, match="git history"):
        check_gates(repo, _digest(repo))


def test_refuses_a_selection_changed_after_the_freeze(repo: Path) -> None:
    digest = _digest(repo)
    write_selected(repo, P1Config(0.3, 1e-2, 0.9))
    with pytest.raises(GateError, match="not the selection"):
        check_gates(repo, digest)


def test_refuses_a_protocol_whose_policies_are_not_the_selected_ones(repo: Path) -> None:
    config = _config(repo)
    config["P1"]["tau"] = 0.95
    with pytest.raises(GateError, match="P1 is not the selected one"):
        check_gates(repo, _rewrite(repo, config))
    config = _config(repo)
    config["P1"]["tau"] = 0.99
    config["P2"]["P2-value"]["lam"] = 0.5
    with pytest.raises(GateError, match="P2-value is not the selected one"):
        check_gates(repo, _rewrite(repo, config))


def test_refuses_a_smoke_selection(repo: Path) -> None:
    write_selected(repo, limit=2000)
    config = _config(repo)
    config["selected"]["sha256"] = file_sha256(repo / SELECTED)
    with pytest.raises(GateError, match="smoke run"):
        check_gates(repo, _rewrite(repo, config))


def test_refuses_other_constants_or_another_environment(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    digest = _digest(repo)
    monkeypatch.setattr(run_eval, "protocol_constants", lambda: {**protocol_constants(), "min_gain": 1e-5})
    with pytest.raises(GateError, match="constants"):
        check_gates(repo, digest)
    monkeypatch.undo()
    monkeypatch.setattr(run_eval, "unseal_requested", lambda: True)
    monkeypatch.setattr(run_eval, "uncommitted", lambda root: [])
    monkeypatch.setattr(run_eval, "git_commits_touching", lambda root, path: ["f00d"])
    monkeypatch.setattr(run_eval, "environment", lambda: {"python": "3.99.0", "numpy": "0", "pandas": "0"})
    with pytest.raises(GateError, match="environment"):
        check_gates(repo, digest)


def test_refuses_other_manifests_or_csvs(repo: Path) -> None:
    digest = _digest(repo)
    manifest_path = repo / MANIFEST_DIR / "H2.json"
    original = manifest_path.read_text()
    manifest_path.write_text(original + " ")
    with pytest.raises(GateError, match="H2 manifest"):
        check_gates(repo, digest)
    manifest_path.write_text(original)
    with (repo / "data/ddxplus/release_test_patients.csv").open("a") as f:
        f.write("\n")
    with pytest.raises(GateError, match=r"release_test_patients\.csv"):
        check_gates(repo, digest)


def test_uncommitted_merges_tracked_changes_and_untracked_paper1_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(root: Path, *paths: str, untracked: bool = True) -> list[str]:
        return ["?? paper1/new.py", " M paper1/a.py"] if paths else [" M paper1/a.py", " M pilot/x.json"]

    monkeypatch.setattr(run_eval, "git_changes", fake)
    assert run_eval.uncommitted(tmp_path) == [" M paper1/a.py", " M pilot/x.json", "?? paper1/new.py"]
    monkeypatch.setattr(run_eval, "git_changes", lambda root, *paths, untracked=True: None)
    with pytest.raises(GateError, match="not a git work tree"):
        run_eval.uncommitted(tmp_path)


def _git(root: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, env=env, check=True)
    return done.stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_freeze_commit_reads_the_history_of_the_protocol_file(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    protocol = tmp_path / FROZEN_CONFIG
    protocol.parent.mkdir(parents=True)
    (tmp_path / "README").write_text("x\n")
    _git(tmp_path, "add", "README")
    _git(tmp_path, "commit", "-q", "-m", "start")
    with pytest.raises(GateError, match="found 0"):
        freeze_commit(tmp_path)
    protocol.write_text("a: 1\n")
    _git(tmp_path, "add", FROZEN_CONFIG)
    _git(tmp_path, "commit", "-q", "-m", "freeze")
    frozen = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "README").write_text("y\n")
    _git(tmp_path, "commit", "-q", "-am", "later work")
    assert freeze_commit(tmp_path) == frozen
    protocol.write_text("a: 2\n")
    _git(tmp_path, "commit", "-q", "-am", "edit the protocol")
    with pytest.raises(GateError, match="found 2"):
        freeze_commit(tmp_path)


def test_budgets_must_lie_on_the_curve(repo: Path) -> None:
    config = _config(repo)
    assert policies(config)[2] == (5, 10, 15, 20, 29)
    for bad in ([], [5, 30], [10, 5]):
        with pytest.raises(GateError, match="budgets"):
            policies({**config, "budgets": bad})


def test_each_p2_configuration_is_the_variant_it_is_listed_under(repo: Path) -> None:
    config = _config(repo)
    config["P2"]["P2-binary"] = P2Config(P2_VARIANTS[3], 1.0, 0.99).as_dict()
    with pytest.raises(GateError, match="P2-value-agesex configuration under P2-binary"):
        policies(config)


def _main(p1_low: float, diff: float, il_high: float, j3: tuple[float, float]) -> dict[str, Any]:
    return {
        "P1": {"stop": {"gtpa1": {"low": p1_low}}},
        "P2-binary": {"against": {"P1": {"stop": {"gtpa1_diff": diff, "il_diff": {"high": il_high}}}}},
        "P2-value-agesex": {"stop": {"gtpa1": {"estimate": j3[0]}, "il": {"mean": j3[1]}}},
    }


def test_judgments_apply_the_design_thresholds() -> None:
    met = judgments({"H1": _main(0.97, 0.002, -2.0, (0.99, 6.5)), "H2": _main(0.95, -0.004, -0.1, (0.0, 0.0))})
    assert (met["J1"]["met"], met["J2"]["met"], met["J3"]["met"]) == (True, True, True)
    assert met["J1"]["wilson_low"] == {"H1": 0.97, "H2": 0.95}
    assert (met["J3"]["gtpa1"], met["J3"]["il_mean"]) == (0.99, 6.5)
    missed = judgments({"H1": _main(0.97, 0.01, -2.0, (0.9846, 5.47)), "H2": _main(0.9499, 0.0, 0.01, (1.0, 1.0))})
    assert (missed["J1"]["met"], missed["J2"]["met"], missed["J3"]["met"]) == (False, False, False)
    assert not judgments({"H1": _main(0.97, 0.0, -1.0, (0.9947, 7.0)), "H2": _main(0.97, 0.0, -1.0, (0, 0))})["J3"][
        "met"
    ]


def test_the_cli_reports_a_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(UNSEAL_ENV, raising=False)
    with pytest.raises(SystemExit, match="run_eval refused: DDXKB_UNSEAL=1 is not set"):
        run_eval.main(["--config-sha256", "0" * 64, "--root", str(tmp_path)])


def test_evaluation_on_a_synthetic_repository(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "unseal_requested", lambda: True)
    counted: list[int] = []

    def counting(kb: KB, patients: Iterable[Patient]) -> FrequencyCounts:
        rows = list(patients)
        counted.extend(p.row for p in rows)
        return count_frequencies(kb, rows)

    monkeypatch.setattr(experiment, "count_frequencies", counting)
    protocol = check_gates(repo, _digest(repo))
    out = tmp_path / "eval"
    evaluate(protocol, repo, out, workers=1)
    h2_rows = set(json.loads((repo / MANIFEST_DIR / "H2.json").read_text())["row_indices"])
    # Counted once for both sets, from the train rows outside H2 only.
    assert counted == [r for r in range(150) if r not in h2_rows]
    for name, n in (("H1", 57), ("H2", 25)):
        main = json.loads((out / name / "main.json").read_text())
        assert main["set"] == name and main["P1"]["n"] == n
        assert main["P1"]["config"] == protocol.config["P1"]
        assert set(main) >= {"P1", *(v.name for v in P2_VARIANTS)}
        meta = main["provenance"]
        assert (meta["protocol_sha256"], meta["protocol_commit"], meta["split"]) == (protocol.sha256, "c0ffee", name)
        assert meta["protocol_freeze_commit"] == "f00d"
        assert meta["train_minus_h2_rows"] == 150 - len(h2_rows) == 125
        train = "data/ddxplus/release_train_patients.csv"
        assert set(meta["inputs"]) >= {f"{MANIFEST_DIR}/{name}.json", f"{MANIFEST_DIR}/H2.json", train}
        baselines = json.loads((out / name / "baselines.json").read_text())
        assert set(baselines) >= {"B0", "B1", "B2", "B6"}
        properties = json.loads((out / name / "properties.json").read_text())
        assert (properties["set"], properties["n"], properties["outside_support"]) == (name, n, 0)
        assert 0 <= properties["unique"]["share"] <= 1 and properties["provenance"]["protocol_sha256"]
        lines = gzip.decompress((out / name / "patients" / "P1.jsonl.gz").read_bytes()).decode().splitlines()
        assert len(lines) == n
    h1_rows = json.loads((repo / MANIFEST_DIR / "H1.json").read_text())["row_indices"]
    first = json.loads(gzip.decompress((out / "H1/patients/P1.jsonl.gz").read_bytes()).decode().splitlines()[0])
    assert first["row"] == h1_rows[0]
    verdicts = json.loads((out / "judgments.json").read_text())
    mains = {name: json.loads((out / name / "main.json").read_text()) for name in ("H1", "H2")}
    assert {k: v for k, v in verdicts.items() if k != "provenance"} == judgments(mains)
    assert verdicts["provenance"]["protocol_freeze_commit"] == "f00d"
    with pytest.raises(GateError, match="results exist"):
        evaluate(protocol, repo, out, workers=1)


def test_j2_bounds_the_gtpa1_difference_in_both_directions() -> None:
    lower = judgments({"H1": _main(0.97, -0.012, -2.0, (0.99, 6.5)), "H2": _main(0.97, 0.0, -2.0, (0.99, 6.5))})
    assert not lower["J2"]["met"]
