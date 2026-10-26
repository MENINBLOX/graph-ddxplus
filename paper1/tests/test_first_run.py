from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from ddxkb import first_run
from ddxkb.data import REPO_ROOT
from ddxkb.evaluation import write_json
from ddxkb.first_run import (
    FIRST_DEV,
    FIRST_EVAL,
    FIRST_PROTOCOL,
    FIRST_PROTOCOL_SHA256,
    FIRST_SETS,
    FIRST_SUMMARIES,
    TAIL_BLOCK,
    FirstRun,
    VerifyError,
    first_run_outputs,
    trailing_provenance,
    verify_first_run,
)
from ddxkb.freeze import config_digest
from ddxkb.provenance import file_sha256

FREEZE = "a06db1e" + "0" * 33
OUTPUTS = "6910b60" + "0" * 33


def _write(root: Path, relative: str, obj: Any) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return file_sha256(path)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A first run's protocol, selection, manifests and evaluation summaries, with the git state stubbed."""
    selected = "paper1/outputs/dev/selected.json"
    config: dict[str, Any] = {
        "design": "docs/design/ddxplus_kb_redo.md",
        "selected": {"path": selected, "sha256": _write(tmp_path, selected, {"limit": None})},
        "manifests": {name: _write(tmp_path, f"paper1/manifests/{name}.json", {"set": name}) for name in FIRST_SETS},
        "budgets": [5, 10, 15, 20, 29],
    }
    config["config_sha256"] = digest = config_digest(config)
    (tmp_path / FIRST_PROTOCOL).parent.mkdir(parents=True)
    (tmp_path / FIRST_PROTOCOL).write_text(yaml.safe_dump(config, sort_keys=False))
    stamp = {"protocol_sha256": digest, "protocol_freeze_commit": FREEZE, "protocol_commit": FREEZE}
    for s in FIRST_SETS:
        for f in FIRST_SUMMARIES:
            _write(tmp_path, f"{FIRST_EVAL}/{s}/{f}", {"set": s, "provenance": stamp})
    _write(tmp_path, f"{FIRST_EVAL}/judgments.json", {"J1": {}, "provenance": {"design": "§6", **stamp}})
    monkeypatch.setattr(first_run, "FIRST_PROTOCOL_SHA256", digest)
    monkeypatch.setattr(first_run, "git_changes", lambda root, *paths, untracked=True: [])
    monkeypatch.setattr(
        first_run, "git_commits_touching", lambda root, path: [FREEZE] if path == FIRST_PROTOCOL else [OUTPUTS]
    )
    return tmp_path


def test_a_kept_first_run_is_verified(repo: Path) -> None:
    assert verify_first_run(repo) == FirstRun(first_run.FIRST_PROTOCOL_SHA256, FREEZE, OUTPUTS)


def test_refuses_an_edited_protocol(repo: Path) -> None:
    config = yaml.safe_load((repo / FIRST_PROTOCOL).read_text())
    config["budgets"] = [5, 10, 15, 20, 30]
    (repo / FIRST_PROTOCOL).write_text(yaml.safe_dump(config))
    with pytest.raises(VerifyError, match="not the first frozen protocol"):
        verify_first_run(repo)
    config["config_sha256"] = config_digest(config)
    (repo / FIRST_PROTOCOL).write_text(yaml.safe_dump(config))
    with pytest.raises(VerifyError, match="not the first frozen protocol"):
        verify_first_run(repo)


@pytest.mark.parametrize(
    ("commits", "message"),
    [([OUTPUTS], "frozen by"), ([FREEZE, OUTPUTS], "exactly one commit; found 2"), (None, "git history")],
)
def test_refuses_another_freeze_history(
    repo: Path, monkeypatch: pytest.MonkeyPatch, commits: list[str] | None, message: str
) -> None:
    monkeypatch.setattr(first_run, "git_commits_touching", lambda root, path: commits)
    with pytest.raises(VerifyError, match=message):
        verify_first_run(repo)


def test_refuses_a_changed_selection_or_manifest(repo: Path) -> None:
    for relative in ("paper1/outputs/dev/selected.json", "paper1/manifests/H2.json"):
        path = repo / relative
        original = path.read_text()
        path.write_text(original + " ")
        with pytest.raises(VerifyError, match=f"{relative} is not the file"):
            verify_first_run(repo)
        path.write_text(original)


def test_refuses_changed_files_or_a_second_output_commit(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(first_run, "git_changes", lambda root, *paths, untracked=True: [" M " + FIRST_EVAL + "/x"])
    with pytest.raises(VerifyError, match="have changed"):
        verify_first_run(repo)
    monkeypatch.setattr(first_run, "git_changes", lambda root, *paths, untracked=True: [])
    monkeypatch.setattr(
        first_run, "git_commits_touching", lambda root, path: [FREEZE] if path == FIRST_PROTOCOL else [OUTPUTS, FREEZE]
    )
    with pytest.raises(VerifyError, match=f"{FIRST_EVAL} must have been changed by exactly one commit"):
        verify_first_run(repo)


def test_refuses_a_missing_or_foreign_summary(repo: Path) -> None:
    summary = repo / FIRST_EVAL / "H2" / "baselines.json"
    original = summary.read_text()
    summary.unlink()
    with pytest.raises(VerifyError, match="missing"):
        verify_first_run(repo)
    other = json.loads(original)
    other["provenance"]["protocol_sha256"] = "0" * 64
    summary.write_text(json.dumps(other))
    with pytest.raises(VerifyError, match="does not record the first protocol"):
        verify_first_run(repo)


def test_the_cli_reports_the_outcome(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first_run.main(["--root", str(repo)])
    assert capsys.readouterr().out.startswith(f"first run verified: protocol {first_run.FIRST_PROTOCOL_SHA256}")
    (repo / FIRST_EVAL / "judgments.json").unlink()
    with pytest.raises(SystemExit, match=r"first run not verified: .*judgments\.json is missing"):
        first_run.main(["--root", str(repo)])


def test_the_provenance_is_read_from_the_end_without_parsing_the_results(tmp_path: Path) -> None:
    stamp = {"protocol_sha256": "ab" * 32, "timings": [{"run": "P1", "seconds": 1.5}], "note": "provenance: é"}
    summary = tmp_path / "main.json"
    write_json(summary, {"P1": {"curve": list(range(3 * TAIL_BLOCK))}, "provenance": stamp})
    assert summary.stat().st_size > 4 * TAIL_BLOCK
    assert trailing_provenance(summary) == stamp
    # The results before the key are never parsed: the same tail after bytes that are no JSON at all.
    text = summary.read_text()
    tail = text[text.rindex('"provenance":') :]
    summary.write_text('{"P1": <results that are not read>,\n ' + tail)
    assert trailing_provenance(summary) == stamp
    summary.write_text(json.dumps({"set": "H1", "provenance": stamp}))
    assert trailing_provenance(summary) == stamp


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"P1": {"gtpa1": 0.98}}', "has no provenance"),
        ('{"provenance": {"a": 1}, "P1": {"gtpa1": 0.98}}', "is not a JSON object|not the last key"),
        ('{"P1": 1, "provenance": {"a": 1}, "x": 2}', "not the last key"),
        ('{"P1": 1, "provenance": [1, 2]}', "not the last key"),
    ],
)
def test_a_summary_must_end_with_its_provenance(tmp_path: Path, content: str, message: str) -> None:
    (tmp_path / "s.json").write_text(content)
    with pytest.raises(VerifyError, match=message):
        trailing_provenance(tmp_path / "s.json")


def test_first_run_outputs_names_the_directory_an_output_would_write_into(tmp_path: Path) -> None:
    for kept in (FIRST_DEV, FIRST_EVAL):
        (tmp_path / kept).mkdir(parents=True)
        assert first_run_outputs(tmp_path, tmp_path / kept) == kept
        assert first_run_outputs(tmp_path, tmp_path / kept / "T" / "x") == kept
    assert first_run_outputs(tmp_path, tmp_path / "paper1/outputs/v2/eval") is None
    assert first_run_outputs(tmp_path, tmp_path / "paper1/outputs") is None
    assert first_run_outputs(tmp_path, tmp_path / "paper1/outputs/evaluation") is None
    (tmp_path / "elsewhere").symlink_to(tmp_path / FIRST_EVAL)
    assert first_run_outputs(tmp_path, tmp_path / "elsewhere" / "T") == FIRST_EVAL
    assert first_run_outputs(tmp_path, tmp_path / "paper1/outputs/v2/../eval") == FIRST_EVAL


@pytest.mark.skipif(not (REPO_ROOT / FIRST_PROTOCOL).exists(), reason="first protocol not present")
def test_the_recorded_digest_is_the_first_protocol_of_this_repository() -> None:
    config = yaml.safe_load((REPO_ROOT / FIRST_PROTOCOL).read_text())
    assert config["config_sha256"] == config_digest(config) == FIRST_PROTOCOL_SHA256
