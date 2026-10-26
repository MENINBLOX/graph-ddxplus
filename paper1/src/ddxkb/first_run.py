"""Verify what the first evaluation left (design "결과", frozen at a06db1e), without running anything again.

    uv run python -m ddxkb.first_run

The first protocol (paper1/configs/protocol_frozen.yaml: H1 and H2, |T| <= 30) is kept with its development and
evaluation outputs (paper1/outputs/dev, paper1/outputs/eval) and reported beside the revised protocol (보완 2). This
tree runs the revised protocol, so the first one is never rerun from it; the check confirms that what is kept is
what was run:

- the protocol file holds the digest it records, FIRST_PROTOCOL_SHA256, and exactly one commit changed it, the freeze
  commit (FIRST_FREEZE_COMMIT);
- the selection and the H1 and H2 manifests it names are the files whose sha256 it records;
- no file under the protocol and output paths is modified, deleted or untracked, exactly one commit changed the
  evaluation outputs, and each evaluation summary (main, baselines and properties of H1 and H2, judgments.json)
  exists and records that protocol digest and freeze commit.

Of each summary it parses only the "provenance" object, the last key `evaluation.write_json` writes, reading the
file from its end back to that key (`trailing_provenance`); it prints no result. The runners refuse to write into
the first run's output directories (`first_run_outputs`).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ddxkb.data import MANIFEST_DIR, REPO_ROOT
from ddxkb.freeze import config_digest
from ddxkb.provenance import file_sha256, git_changes, git_commits_touching

FIRST_PROTOCOL = "paper1/configs/protocol_frozen.yaml"
FIRST_PROTOCOL_SHA256 = "af411239a9d9043b8b60d02a456882520b2ce45eb8d2198b143a04a9763b2de8"
FIRST_FREEZE_COMMIT = "a06db1e"
FIRST_DEV = "paper1/outputs/dev"
FIRST_EVAL = "paper1/outputs/eval"
FIRST_SETS = ("H1", "H2")
FIRST_SUMMARIES = ("main.json", "baselines.json", "properties.json")
PROVENANCE_KEY = b'"provenance":'
TAIL_BLOCK = 1024


class VerifyError(RuntimeError):
    pass


@dataclass(frozen=True)
class FirstRun:
    protocol_sha256: str
    freeze_commit: str
    outputs_commit: str


def first_run_outputs(root: Path, out: Path) -> str | None:
    """The first run's output directory (FIRST_DEV or FIRST_EVAL) that `out` is or lies in, else None."""
    target = out.resolve()
    for kept in (FIRST_DEV, FIRST_EVAL):
        directory = (root / kept).resolve()
        if target == directory or directory in target.parents:
            return kept
    return None


def trailing_provenance(path: Path) -> dict[str, Any]:
    """The "provenance" object that closes a JSON summary, read in blocks from the end of the file back to its key;
    nothing before the key is parsed. The key is the last top-level one, so its last occurrence is the right one
    and only the closing brace of the summary may follow the object."""
    tail = b""
    with path.open("rb") as f:
        position = f.seek(0, 2)
        while PROVENANCE_KEY not in tail:
            if position == 0:
                raise VerifyError(f"{path} has no provenance")
            step = min(TAIL_BLOCK, position)
            position -= step
            f.seek(position)
            tail = f.read(step) + tail
    text = tail[tail.rindex(PROVENANCE_KEY) + len(PROVENANCE_KEY) :].decode()
    body = text.lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(body)
    except json.JSONDecodeError as e:
        raise VerifyError(f"{path}: its provenance is not a JSON object") from e
    if not isinstance(value, dict) or body[end:].strip() != "}":
        raise VerifyError(f"{path}: provenance is not the last key of the summary")
    return value


def _only_commit(root: Path, path: str) -> str:
    commits = git_commits_touching(root, path)
    if commits is None:
        raise VerifyError(f"cannot read the git history of {root}")
    if len(commits) != 1:
        raise VerifyError(f"{path} must have been changed by exactly one commit; found {len(commits)}")
    return commits[0]


def _summaries(root: Path) -> list[str]:
    return [*(f"{FIRST_EVAL}/{s}/{f}" for s in FIRST_SETS for f in FIRST_SUMMARIES), f"{FIRST_EVAL}/judgments.json"]


def verify_first_run(root: Path) -> FirstRun:
    """The first run's protocol digest, freeze commit and evaluation-output commit once every check of the module
    docstring holds; VerifyError otherwise."""
    path = root / FIRST_PROTOCOL
    if not path.exists():
        raise VerifyError(f"{FIRST_PROTOCOL} does not exist")
    config: dict[str, Any] = yaml.safe_load(path.read_text())
    digest = config_digest(config)
    if config.get("config_sha256") != digest or digest != FIRST_PROTOCOL_SHA256:
        raise VerifyError(f"{FIRST_PROTOCOL} is not the first frozen protocol {FIRST_PROTOCOL_SHA256}")
    frozen_at = _only_commit(root, FIRST_PROTOCOL)
    if not frozen_at.startswith(FIRST_FREEZE_COMMIT):
        raise VerifyError(f"{FIRST_PROTOCOL} was frozen by {frozen_at}, not {FIRST_FREEZE_COMMIT}")
    named = {str(config["selected"]["path"]): config["selected"]["sha256"]}
    named.update({f"{MANIFEST_DIR}/{name}.json": sha for name, sha in config["manifests"].items()})
    for relative, sha in named.items():
        if not (root / relative).exists() or file_sha256(root / relative) != sha:
            raise VerifyError(f"{relative} is not the file the first protocol names")
    changes = git_changes(root, FIRST_PROTOCOL, FIRST_DEV, FIRST_EVAL)
    if changes is None:
        raise VerifyError(f"{root} is not a git work tree")
    if changes:
        raise VerifyError(f"the first run's files have changed: {changes[:10]}")
    outputs_at = _only_commit(root, FIRST_EVAL)
    for relative in _summaries(root):
        if not (root / relative).exists():
            raise VerifyError(f"{relative} is missing")
        stamp = trailing_provenance(root / relative)
        if (stamp.get("protocol_sha256"), stamp.get("protocol_freeze_commit")) != (digest, frozen_at):
            raise VerifyError(f"{relative} does not record the first protocol and its freeze commit")
    return FirstRun(protocol_sha256=digest, freeze_commit=frozen_at, outputs_commit=outputs_at)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ddxkb.first_run", description="Verify the first evaluation's files.")
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    args = ap.parse_args(argv)
    try:
        first = verify_first_run(args.root.resolve())
    except VerifyError as e:
        raise SystemExit(f"first run not verified: {e}") from e
    print(
        f"first run verified: protocol {first.protocol_sha256}, frozen at {first.freeze_commit},"
        f" evaluation outputs from {first.outputs_commit}"
    )


if __name__ == "__main__":
    main()
