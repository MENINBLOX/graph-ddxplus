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
from ddxkb.policies import bed
from ddxkb.policies.bed import BED_STATS
from ddxkb.protocol import LADDER, P2_VARIANTS, TAU, P1Config, p2_fixed
from ddxkb.provenance import file_sha256
from ddxkb.run_eval import FrozenProtocol, GateError, check_gates, evaluate, freeze_commit, policies
from helpers import SELECTED_P1, bed_stats_sha256, make_repo, write_selected


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
    for module in (bed, run_eval):
        monkeypatch.setattr(module, "BED_STATS_SHA256", bed_stats_sha256(tmp_path))
    monkeypatch.setattr(run_eval, "unseal_requested", lambda: True)
    monkeypatch.setattr(run_eval, "uncommitted", lambda root: [])
    monkeypatch.setattr(run_eval, "git_head", lambda root: "c0ffee")
    monkeypatch.setattr(run_eval, "git_commits_touching", lambda root, path: ["f00d"])
    monkeypatch.setattr(run_eval, "git_changed_since", lambda root, commit, *paths: [])
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
    config["P1"]["eps"] = 0.01
    (repo / FROZEN_CONFIG).write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(GateError, match="recorded config_sha256"):
        check_gates(repo, digest)


def test_refuses_a_protocol_other_than_the_frozen_one(repo: Path) -> None:
    first = _digest(repo)
    second = _freeze(repo, P1Config(0.7, 1e-3, TAU))
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
    write_selected(repo, P1Config(0.3, 1e-2, TAU))
    with pytest.raises(GateError, match="not the selection"):
        check_gates(repo, digest)


def test_refuses_a_protocol_whose_policies_are_not_the_selected_ones(repo: Path) -> None:
    config = _config(repo)
    config["P1"]["alpha"] = 0.7
    with pytest.raises(GateError, match="P1 is not the selected one"):
        check_gates(repo, _rewrite(repo, config))
    config = _config(repo)
    config["P1"]["alpha"] = 0.5
    config["P2"]["P2-value"]["lam"] = 0.5
    with pytest.raises(GateError, match="P2-value is not the selected one"):
        check_gates(repo, _rewrite(repo, config))


@pytest.mark.parametrize(
    ("policy", "key", "value", "message"),
    [
        ("P1", "tau", 0.95, r"P1 stops at tau 0\.95, the protocol at 0\.99"),
        ("P2-value", "tau", 0.999, r"P2-value stops at tau 0\.999, the protocol at 0\.99"),
        ("P2-binary", "lam", 0.5, r"P2-binary smooths with lambda 0\.5, the protocol with 1\.0"),
    ],
)
def test_refuses_a_selection_whose_tau_or_lambda_is_not_the_fixed_one(
    repo: Path, policy: str, key: str, value: float, message: str
) -> None:
    """The protocol and its selection agree with each other, but not with the constants of the code."""
    p1 = SELECTED_P1
    p2 = {v.name: p2_fixed(v).as_dict() for v in P2_VARIANTS}
    if policy == "P1":
        p1 = P1Config(p1.alpha, p1.eps, value)
    else:
        p2[policy][key] = value
    write_selected(repo, p1, p2)
    config = _config(repo)
    config["selected"]["sha256"] = file_sha256(repo / SELECTED)
    config["P1"], config["P2"] = p1.as_dict(), p2
    with pytest.raises(GateError, match=f"does not hold the fixed settings of this code: {message}"):
        check_gates(repo, _rewrite(repo, config))


@pytest.mark.parametrize("p1", [P1Config(0.5, 1e-5, TAU), P1Config(0.9, 1e-4, TAU), P1Config(0.42, 0.5, TAU)])
def test_refuses_a_selected_p1_outside_the_nine_candidates(repo: Path, p1: P1Config) -> None:
    """The protocol and its selection agree with each other on a P1 that the selection could not have picked."""
    write_selected(repo, p1)
    config = _config(repo)
    config["selected"]["sha256"] = file_sha256(repo / SELECTED)
    config["P1"] = p1.as_dict()
    with pytest.raises(GateError, match=r"fixed settings of this code: P1 with alpha .* is not one of the nine"):
        check_gates(repo, _rewrite(repo, config))


def test_refuses_a_protocol_for_another_evaluation_set(repo: Path) -> None:
    config = _config(repo)
    config["evaluation_set"] = "H1"
    with pytest.raises(GateError, match="the protocol evaluates H1, this runner T"):
        check_gates(repo, _rewrite(repo, config))


def test_refuses_a_smoke_selection(repo: Path) -> None:
    write_selected(repo, limit=2000)
    config = _config(repo)
    config["selected"]["sha256"] = file_sha256(repo / SELECTED)
    with pytest.raises(GateError, match="smoke run"):
        check_gates(repo, _rewrite(repo, config))


def test_refuses_other_constants_or_another_environment(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    digest = _digest(repo)
    monkeypatch.setattr(run_eval, "protocol_constants", lambda kb: {**protocol_constants(kb), "min_gain": 1e-5})
    with pytest.raises(GateError, match="constants"):
        check_gates(repo, digest)
    monkeypatch.undo()
    monkeypatch.setattr(run_eval, "unseal_requested", lambda: True)
    monkeypatch.setattr(run_eval, "uncommitted", lambda root: [])
    monkeypatch.setattr(run_eval, "git_commits_touching", lambda root, path: ["f00d"])
    monkeypatch.setattr(run_eval, "git_changed_since", lambda root, commit, *paths: [])
    monkeypatch.setattr(run_eval, "BED_STATS_SHA256", bed_stats_sha256(repo))
    monkeypatch.setattr(run_eval, "environment", lambda: {"python": "3.99.0", "numpy": "0", "pandas": "0"})
    with pytest.raises(GateError, match="environment"):
        check_gates(repo, digest)


def test_refuses_other_manifests_or_csvs(repo: Path) -> None:
    digest = _digest(repo)
    for name in ("T", "T-dedup", "T-streambench"):
        manifest_path = repo / MANIFEST_DIR / f"{name}.json"
        original = manifest_path.read_text()
        manifest_path.write_text(original + " ")
        with pytest.raises(GateError, match=f"{name} manifest"):
            check_gates(repo, digest)
        manifest_path.write_text(original)
    # The first evaluation's sets are no part of this protocol.
    for name in ("H1", "H2"):
        (repo / MANIFEST_DIR / f"{name}.json").unlink()
    check_gates(repo, digest)
    train = repo / "data/ddxplus/release_train_patients.csv"
    original_train = train.read_bytes()
    train.write_bytes(original_train + b"\n")
    with pytest.raises(GateError, match=r"release_train_patients\.csv is not the file the development run read"):
        check_gates(repo, digest)
    train.write_bytes(original_train)
    stats = repo / BED_STATS
    original_stats = stats.read_bytes()
    stats.write_bytes(original_stats + b" ")
    with pytest.raises(GateError, match="BED's stats file"):
        check_gates(repo, digest)
    stats.write_bytes(original_stats)
    with (repo / "data/ddxplus/release_test_patients.csv").open("a") as f:
        f.write("\n")
    with pytest.raises(GateError, match=r"release_test_patients\.csv"):
        check_gates(repo, digest)


def test_uncommitted_reads_paper1_and_the_design_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[tuple[tuple[str, ...], bool]] = []

    def fake(root: Path, *paths: str, untracked: bool = True) -> list[str]:
        asked.append((paths, untracked))
        return ["?? paper1/new.py", " M docs/design/ddxplus_kb_redo.md"]

    monkeypatch.setattr(run_eval, "git_changes", fake)
    assert run_eval.uncommitted(tmp_path) == ["?? paper1/new.py", " M docs/design/ddxplus_kb_redo.md"]
    assert asked == [(("paper1", "docs/design"), True)]
    monkeypatch.setattr(run_eval, "git_changes", lambda root, *paths, untracked=True: None)
    with pytest.raises(GateError, match="not a git work tree"):
        run_eval.uncommitted(tmp_path)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_changes_outside_the_gated_paths_do_not_block(tmp_path: Path) -> None:
    for relative in ("paper1/a.py", "docs/design/d.md", "pilot/cache.json", "docs/other.md"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text("x\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "start")
    (tmp_path / "pilot/cache.json").write_text("y\n")
    (tmp_path / "docs/other.md").unlink()
    (tmp_path / "pilot/new.json").write_text("z\n")
    assert run_eval.uncommitted(tmp_path) == []
    (tmp_path / "docs/design/d.md").write_text("y\n")
    (tmp_path / "paper1/b.py").write_text("y\n")
    assert run_eval.uncommitted(tmp_path) == [" M docs/design/d.md", "?? paper1/b.py"]


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


def test_refuses_code_changed_after_the_selection_run(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[tuple[str, tuple[str, ...]]] = []

    def changed(root: Path, commit: str, *paths: str) -> list[str]:
        asked.append((commit, paths))
        return ["paper1/src/ddxkb/metrics.py"]

    monkeypatch.setattr(run_eval, "git_changed_since", changed)
    with pytest.raises(GateError, match=r"paper1/src changed after the selection run's commit"):
        check_gates(repo, _digest(repo))
    assert asked == [(_config(repo)["selected"]["commit"], ("paper1/src",))]
    monkeypatch.setattr(run_eval, "git_changed_since", lambda root, commit, *paths: None)
    with pytest.raises(GateError, match="cannot compare"):
        check_gates(repo, _digest(repo))


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_code_changes_reads_paper1_src_between_the_selection_commit_and_head(tmp_path: Path) -> None:
    for relative in ("paper1/src/a.py", "paper1/outputs/v3/dev/selected.json", "docs/design/d.md"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text("x\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "selection run")
    selected = _git(tmp_path, "rev-parse", "HEAD")
    for relative in ("paper1/outputs/v3/dev/selected.json", "docs/design/d.md"):
        (tmp_path / relative).write_text("y\n")
    _git(tmp_path, "commit", "-q", "-am", "outputs and results")
    assert run_eval.code_changes(tmp_path, selected) == []
    (tmp_path / "paper1/src/a.py").write_text("y\n")
    _git(tmp_path, "commit", "-q", "-am", "a fix")
    assert run_eval.code_changes(tmp_path, selected) == ["paper1/src/a.py"]
    with pytest.raises(GateError, match="cannot compare"):
        run_eval.code_changes(tmp_path, "0" * 40)


def test_budgets_must_lie_on_the_curve(repo: Path) -> None:
    config = _config(repo)
    assert policies(config)[2] == (5, 10, 15, 20, 30)
    for bad in ([], [5, 31], [10, 5]):
        with pytest.raises(GateError, match="budgets"):
            policies({**config, "budgets": bad})


def test_each_p2_configuration_is_the_variant_it_is_listed_under(repo: Path) -> None:
    config = _config(repo)
    config["P2"]["P2-binary"] = p2_fixed(P2_VARIANTS[3]).as_dict()
    with pytest.raises(GateError, match="P2-value-agesex configuration under P2-binary"):
        policies(config)


def test_the_cli_reports_a_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(UNSEAL_ENV, raising=False)
    with pytest.raises(SystemExit, match="run_eval refused: DDXKB_UNSEAL=1 is not set"):
        run_eval.main(["--config-sha256", "0" * 64, "--root", str(tmp_path)])


@pytest.mark.parametrize(
    "inside",
    [
        "paper1/outputs/eval",
        "paper1/outputs/dev",
        "paper1/outputs/eval/again",
        "paper1/outputs/v2",
        "paper1/outputs/v2/eval",
        "paper1/outputs/v2/eval/T",
    ],
)
def test_evaluation_never_writes_into_the_outputs_of_an_earlier_run(repo: Path, inside: str) -> None:
    protocol = check_gates(repo, _digest(repo))
    with pytest.raises(GateError, match="outputs of an earlier run"):
        evaluate(protocol, repo, repo / inside, workers=1)
    assert not (repo / inside).exists()
    assert run_eval.OUTPUT_DIR == "paper1/outputs/v3/eval"


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
    # Counted once, from every train row.
    assert counted == list(range(150))
    main = json.loads((out / "T" / "main.json").read_text())
    assert main["set"] == "T" and main["P1"]["n"] == 60
    assert main["P1"]["config"] == protocol.config["P1"]
    assert set(main) == {"set", "P1", *(v.name for v in P2_VARIANTS), "ladder", "provenance"}
    meta = main["provenance"]
    assert (meta["protocol_sha256"], meta["protocol_commit"], meta["split"]) == (protocol.sha256, "c0ffee", "T")
    assert meta["protocol_freeze_commit"] == "f00d"
    assert meta["train_rows"] == 150
    assert set(meta) == {
        "design",
        "stage",
        "split",
        "n_patients",
        "commit",
        "paper1_changes",
        "design_changes",
        "commit_at_end",
        "inputs",
        "train_rows",
        "environment",
        "workers",
        "started_utc",
        "elapsed_s",
        "timings",
        "protocol_sha256",
        "protocol_freeze_commit",
        "protocol_commit",
        "subsets",
    }
    manifests = {path for path in meta["inputs"] if path.startswith(MANIFEST_DIR)}
    assert manifests == {f"{MANIFEST_DIR}/{m}.json" for m in ("T", "T-dedup", "T-streambench")}
    assert "data/ddxplus/release_train_patients.csv" in meta["inputs"]
    assert meta["subsets"] == {
        "T-dedup": {"available": True, "n": 30},
        "T-streambench": {"available": True, "n": 20},
    }
    # Each subset is cut from the same run: its report is the report of the run's rows in it.
    p2 = main["P2-binary"]
    assert set(p2["subsets"]) == {"T-dedup", "T-streambench"}
    streambench = p2["subsets"]["T-streambench"]
    assert streambench["n"] == 20 and set(streambench["against"]) == {"P1"}
    summary = {
        "config",
        "inputs",
        "hierarchy",
        "answers",
        "n",
        "failures",
        "errors",
        "stop",
        "budgets",
        "by_disease",
        "curve",
        "subsets",
        "tau_sensitivity",
    }
    assert set(main["P1"]) == summary and set(p2) == summary | {"against"}
    lines = gzip.decompress((out / "T/patients/P2-binary.jsonl.gz").read_bytes()).decode().splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["row"] for r in records] == list(range(60))
    in_subset = [r for r in records if 10 <= r["row"] < 30]
    assert streambench["stop"]["gtpa1"]["estimate"] == pytest.approx(sum(r["gtpa1"] for r in in_subset) / 20)
    assert streambench["stop"]["il"]["mean"] == pytest.approx(sum(r["il"] for r in in_subset) / 20)
    # 보완 3 §2.1: the stop at each sensitivity threshold, on T and on each subset; at 0.99 it is the policy's own.
    for name in ("P1", *(v.name for v in P2_VARIANTS)):
        assert main[name]["config"]["tau"] == TAU
        for report in (main[name], *main[name]["subsets"].values()):
            assert list(report["tau_sensitivity"]) == ["0.9", "0.95", "0.99", "0.999"]
            assert report["tau_sensitivity"]["0.99"] == report["stop"]
            assert report["tau_sensitivity"]["0.9"]["il"]["mean"] <= report["tau_sensitivity"]["0.999"]["il"]["mean"]
    assert all(main[v.name]["config"]["lam"] == 1.0 for v in P2_VARIANTS)
    # 보완 3 §2.5: the five comparisons, those between Bayes policies with the thresholds and the subsets.
    assert list(main["ladder"]) == [f"{a} - {b}" for a, b in LADDER]
    assert all(isinstance(compared, dict) for compared in main["ladder"].values())
    rung = main["ladder"]["P2-value - P2-binary"]
    assert set(rung) == {"this", "other", "factor", "stop", "5", "10", "15", "20", "30", "tau_sensitivity", "subsets"}
    assert rung["stop"]["gtpa1_diff"] == pytest.approx(
        main["P2-value"]["stop"]["gtpa1"]["estimate"] - p2["stop"]["gtpa1"]["estimate"], abs=1e-12
    )
    assert {"low", "high"} <= set(rung["stop"]["gtpa1_diff_interval"]) and {"low", "high"} <= set(
        rung["30"]["gtpa1_diff_interval"]
    )
    subset_rung = rung["subsets"]["T-streambench"]
    assert set(subset_rung) == {"stop", "5", "10", "15", "20", "30", "tau_sensitivity"}
    assert subset_rung["stop"]["il_diff"]["estimate"] == pytest.approx(
        main["P2-value"]["subsets"]["T-streambench"]["stop"]["il"]["mean"] - streambench["stop"]["il"]["mean"],
        abs=1e-12,
    )
    assert subset_rung["tau_sensitivity"]["0.99"] == subset_rung["stop"]
    baselines = json.loads((out / "T" / "baselines.json").read_text())
    assert set(baselines) >= {"B0", "B1", "B2", "B6", "B-BED"}
    assert baselines["B1"]["set_inclusion"]["subsets"]["T-streambench"]["n"] == 20
    assert baselines["B1"]["binary_majority"]["n"] == 60
    assert baselines["B1"]["binary_majority"]["shares_from"].startswith("every patient of T")
    assert (baselines["B1"]["naive_bayes/P2-value-agesex"]["inputs"], baselines["B0"]["most_frequent"]["inputs"]) == (
        ["age", "initial_evidence", "sex"],
        [],
    )
    assert set(baselines["B2"]["seeds"]["0"]["subsets"]) == {"T-dedup", "T-streambench"}
    bed_t = baselines["B-BED"]["stats_file"]
    assert bed_t["n"] == 60 and set(bed_t["against"]) == {"P1", "P2-value"}
    assert set(bed_t["subsets"]["T-dedup"]["against"]) == {"P1", "P2-value"}
    assert {"low", "high"} <= set(bed_t["against"]["P2-value"]["stop"]["il_diff"])
    assert {"low", "high"} <= set(bed_t["against"]["P2-value"]["stop"]["gtpa1_diff_interval"])
    # 보완 3 §2.5, §2.6, §2.8 for the main B-BED run, over the whole evaluated set.
    differences = baselines["B-BED"]["answer_differences"]
    assert differences["patients"] == 60 and "subsets" not in differences
    assert differences["oracle_negative_bed_positive"] + differences["both_negative"] == differences["oracle_negative"]
    assert 0 <= differences["oracle_negative_bed_filled"] <= differences["oracle_negative_bed_positive"]
    assert baselines["B-BED"]["reported"]["this_minus_bed"]["il"] == bed_t["stop"]["il"]["mean"] - 5.47
    bed_rung = baselines["B-BED"]["ladder"]["B-BED - P2-value"]
    assert {k: bed_rung[k] for k in ("stop", "5", "10", "15", "20", "30")} == bed_t["against"]["P2-value"]
    assert bed_rung["subsets"]["T-dedup"] == bed_t["subsets"]["T-dedup"]["against"]["P2-value"]
    # 보완 3 §6: main.json holds B-BED's rung too, the one of baselines.json.
    assert main["ladder"]["B-BED - P2-value"] == bed_rung
    assert set(bed_rung) == {"this", "other", "factor", "stop", "5", "10", "15", "20", "30", "subsets"}
    written = {p.name for p in (out / "T/patients").iterdir()}
    assert {"P1.jsonl.gz", "B-BED.jsonl.gz", "B-BED-counted.jsonl.gz"} <= written
    assert BED_STATS in baselines["provenance"]["inputs"]
    hierarchy = json.loads((out / "T" / "hierarchy.json").read_text())
    assert hierarchy["hierarchy"] == "implicit"
    for name in ("P1", *(v.name for v in P2_VARIANTS)):
        assert hierarchy[name]["n"] == 60 and set(hierarchy[name]["against"]) == {"P1", name}
        assert hierarchy[name]["subsets"]["T-dedup"]["n"] == 30
    properties = json.loads((out / "T" / "properties.json").read_text())
    assert (properties["set"], properties["n"], properties["outside_support"]) == ("T", 60, 0)
    # No judgment is written: the run leaves nothing but the set's directory.
    assert [path.name for path in out.iterdir()] == ["T"]
    assert {path.name for path in (out / "T").iterdir()} == {
        "main.json",
        "baselines.json",
        "hierarchy.json",
        "properties.json",
        "patients",
    }
    with pytest.raises(GateError, match="results exist"):
        evaluate(protocol, repo, out, workers=1)


def test_a_subset_without_a_row_list_is_described_and_skipped(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unavailable = {"set": "T-streambench", "available": False, "reason": "3 of 1764 rows equal no T row"}
    (repo / MANIFEST_DIR / "T-streambench.json").write_text(json.dumps(unavailable))
    rows, described = run_eval.reporting_subsets(repo)
    assert set(rows) == {"T-dedup"} and rows["T-dedup"] == frozenset(range(0, 60, 2))
    assert described["T-streambench"] == {"available": False, "reason": unavailable["reason"]}
