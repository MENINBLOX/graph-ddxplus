"""The single evaluation run on H1 and H2 (design §5), from the frozen protocol only.

    DDXKB_UNSEAL=1 uv run python -m ddxkb.run_eval --config-sha256 <digest> --workers N

It refuses to start unless all of these hold:

- DDXKB_UNSEAL=1 is set;
- the work tree has no uncommitted change: no modified tracked file anywhere and no untracked file under
  paper1/ (other directories hold untracked caches of other projects that this run never reads);
- paper1/configs/protocol_frozen.yaml (written by `ddxkb.freeze`, which describes its keys) exists, its
  recorded `config_sha256` is the digest of its content, and that digest is the one given on the command line;
- exactly one commit in HEAD's history changes the protocol file: the freeze commit;
- the protocol's P1 and P2 configurations are those of the selection file it names, whose sha256 it records,
  and each P2 configuration is the variant it is listed under;
- the code holds the protocol's constants, and Python, numpy and pandas are those of the development run;
- the H1 and H2 manifests on disk are the ones the protocol names, and each CSV is the file its manifest
  was made from.

The run writes, for each set, main.json (P1 and the P2 variants with curves and comparisons), baselines.json
(B0, B1, B2, B6), properties.json (KB violations and set-inclusion uniqueness) and patient-level files, and then
judgments.json with J1-J3 of design §6; each records the protocol digest, the freeze commit, the commit run
and the input sha256s. An existing result is never overwritten: a rerun after a fix goes to a new --out, and
both are reported.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ddxkb.audit import support_audit
from ddxkb.data import MANIFEST_DIR, REPO_ROOT, Heldout, load_heldout, load_kb, read_manifest, unseal_requested
from ddxkb.evaluation import write_json
from ddxkb.experiment import DESIGN_DOC, Context, baseline_suite, main_suite, provenance, utc_now
from ddxkb.freeze import FROZEN_CONFIG, config_digest, protocol_constants
from ddxkb.frequencies import FrequencyCounts
from ddxkb.protocol import (
    CURVE,
    J1_MIN_WILSON_LOW,
    J2_MAX_GTPA1_DIFF,
    J2_POLICY,
    J3_BED_GTPA1,
    J3_BED_IL,
    J3_GTPA1_TOLERANCE,
    J3_IL_TOLERANCE,
    J3_POLICY,
    P2_VARIANTS,
    P1Config,
    P2Config,
)
from ddxkb.provenance import environment, file_sha256, git_changes, git_commits_touching, git_head
from ddxkb.runner import worker_pool

OUTPUT_DIR = "paper1/outputs/eval"
SETS: tuple[Heldout, ...] = ("H1", "H2")

logger = logging.getLogger(__name__)


class GateError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenProtocol:
    config: Mapping[str, Any]
    sha256: str
    commit: str
    freeze_commit: str


def uncommitted(root: Path) -> list[str]:
    """Modified tracked files anywhere and untracked files under paper1/, as `git status --porcelain` lines."""
    tracked = git_changes(root, untracked=False)
    under_paper1 = git_changes(root, "paper1")
    if tracked is None or under_paper1 is None:
        raise GateError(f"{root} is not a git work tree")
    return list(dict.fromkeys([*tracked, *under_paper1]))


def freeze_commit(root: Path) -> str:
    """The one commit in HEAD's history that changes the protocol file."""
    commits = git_commits_touching(root, FROZEN_CONFIG)
    if commits is None:
        raise GateError(f"cannot read the git history of {root}")
    if len(commits) != 1:
        raise GateError(f"{FROZEN_CONFIG} must be changed by exactly one commit, the freeze; found {len(commits)}")
    return commits[0]


def check_gates(root: Path, expected_sha256: str) -> FrozenProtocol:
    """The frozen protocol, once every condition of the module docstring holds; GateError otherwise."""
    if not unseal_requested():
        raise GateError("DDXKB_UNSEAL=1 is not set")
    changes = uncommitted(root)
    if changes:
        raise GateError(f"the work tree has uncommitted changes: {changes[:10]}")
    path = root / FROZEN_CONFIG
    if not path.exists():
        raise GateError(f"{FROZEN_CONFIG} does not exist; the protocol is not frozen")
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise GateError(f"{FROZEN_CONFIG} is not a mapping")
    digest = config_digest(config)
    if config.get("config_sha256") != digest:
        raise GateError(f"{FROZEN_CONFIG} does not match its recorded config_sha256")
    if digest != expected_sha256:
        raise GateError(f"{FROZEN_CONFIG} has digest {digest}, not the frozen {expected_sha256}")
    frozen_at = freeze_commit(root)
    _check_selection(root, config)
    policies(config)
    if config["constants"] != protocol_constants():
        raise GateError("the code no longer holds the constants of the frozen protocol")
    if config["environment"] != environment():
        raise GateError(f"this environment {environment()} is not the development run's {config['environment']}")
    for name in SETS:
        if file_sha256(root / MANIFEST_DIR / f"{name}.json") != config["manifests"][name]:
            raise GateError(f"the {name} manifest is not the one the protocol names")
        manifest = read_manifest(name, root)
        if file_sha256(root / str(manifest["csv"])) != manifest["csv_sha256"]:
            raise GateError(f"{manifest['csv']} is not the file the {name} manifest was made from")
    commit = git_head(root)
    if commit is None:
        raise GateError("no commit checked out")
    return FrozenProtocol(config=config, sha256=digest, commit=commit, freeze_commit=frozen_at)


def _check_selection(root: Path, config: Mapping[str, Any]) -> None:
    selected_path = root / str(config["selected"]["path"])
    if not selected_path.exists() or file_sha256(selected_path) != config["selected"]["sha256"]:
        raise GateError(f"{config['selected']['path']} is not the selection the protocol was frozen from")
    selected = json.loads(selected_path.read_text())
    if selected["limit"] is not None:
        raise GateError(f"{config['selected']['path']} comes from a smoke run")
    if config["P1"] != selected["P1"]["config"]:
        raise GateError("the protocol's P1 is not the selected one")
    for v in P2_VARIANTS:
        if config["P2"][v.name] != selected["P2"][v.name]["config"]:
            raise GateError(f"the protocol's {v.name} is not the selected one")


def policies(config: Mapping[str, Any]) -> tuple[P1Config, list[P2Config], tuple[int, ...]]:
    p1 = P1Config.from_dict(config["P1"])
    p2 = [P2Config.from_dict(config["P2"][v.name]) for v in P2_VARIANTS]
    for v, cfg in zip(P2_VARIANTS, p2, strict=True):
        if cfg.name != v.name:
            raise GateError(f"the protocol lists a {cfg.name} configuration under {v.name}")
    budgets = tuple(int(b) for b in config["budgets"])
    if not budgets or not set(budgets) <= set(CURVE) or list(budgets) != sorted(set(budgets)):
        raise GateError(f"budgets {budgets} must rise strictly within 0-{CURVE[-1]}")
    return p1, p2, budgets


def judgments(mains: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """J1-J3 of design §6 from the main.json summaries of H1 and H2; every quantity is read at the stop."""
    j1 = {name: mains[name]["P1"]["stop"]["gtpa1"]["low"] for name in SETS}
    j2 = {
        name: {
            "gtpa1_diff": mains[name][J2_POLICY]["against"]["P1"]["stop"]["gtpa1_diff"],
            "il_diff_high": mains[name][J2_POLICY]["against"]["P1"]["stop"]["il_diff"]["high"],
        }
        for name in SETS
    }
    h1 = mains["H1"][J3_POLICY]["stop"]
    gtpa1, il = h1["gtpa1"]["estimate"], h1["il"]["mean"]
    return {
        "J1": {
            "rule": f"Wilson low of P1 tie-aware GTPA@1 >= {J1_MIN_WILSON_LOW} on H1 and H2",
            "wilson_low": j1,
            "met": all(low >= J1_MIN_WILSON_LOW for low in j1.values()),
        },
        "J2": {
            "rule": (
                f"|{J2_POLICY} - P1 tie-aware GTPA@1| < {J2_MAX_GTPA1_DIFF} and the upper paired-bootstrap bound of"
                f" {J2_POLICY} - P1 mean IL < 0, on H1 and H2"
            ),
            "per_set": j2,
            "met": all(abs(v["gtpa1_diff"]) < J2_MAX_GTPA1_DIFF and v["il_diff_high"] < 0 for v in j2.values()),
        },
        "J3": {
            "rule": (
                f"{J3_POLICY} on H1 within {J3_GTPA1_TOLERANCE} of BED's GTPA@1 {J3_BED_GTPA1} and within"
                f" {J3_IL_TOLERANCE} of its IL {J3_BED_IL}"
            ),
            "gtpa1": gtpa1,
            "il_mean": il,
            "met": abs(gtpa1 - J3_BED_GTPA1) <= J3_GTPA1_TOLERANCE and abs(il - J3_BED_IL) <= J3_IL_TOLERANCE,
        },
    }


def evaluate(protocol: FrozenProtocol, root: Path, out: Path, workers: int) -> None:
    """Every policy, baseline, budget curve and dataset property of the protocol on H1, then on H2, in one run."""
    existing = [path for path in (*(out / name for name in SETS), out / "judgments.json") if path.exists()]
    if existing:
        raise GateError(f"results exist at {existing}; a rerun goes to a new --out")
    p1, p2, budgets = policies(protocol.config)
    kb = load_kb(root)
    counts: FrequencyCounts | None = None
    mains: dict[str, dict[str, Any]] = {}
    stamp = {
        "protocol_sha256": protocol.sha256,
        "protocol_freeze_commit": protocol.freeze_commit,
        "protocol_commit": protocol.commit,
    }
    with worker_pool(workers) as pool:
        for name in SETS:
            started_utc, started = utc_now(), time.perf_counter()
            patients = load_heldout(name, root)
            ctx = Context(root, kb, patients, name, out / name, pool, workers, train_counts=counts)
            manifest = read_manifest(name, root)
            for relative in ("data/ddxplus/20043374.zip", f"{MANIFEST_DIR}/{name}.json", str(manifest["csv"])):
                ctx.record_input(relative)
            audit = support_audit(kb, patients)
            main, runs = main_suite(ctx, p1, p2, budgets, patients_dir=ctx.out / "patients")
            baselines = baseline_suite(ctx, p1, p2, runs, budgets)
            counts = ctx.counts()
            meta = {**provenance(ctx, f"eval {name}", started_utc, started, 0), **stamp}
            write_json(ctx.out / "main.json", {"set": name, **main, "provenance": meta})
            write_json(ctx.out / "baselines.json", {"set": name, **baselines, "provenance": meta})
            write_json(ctx.out / "properties.json", {"set": name, **audit.summary(), "provenance": meta})
            mains[name] = main
    write_json(out / "judgments.json", {**judgments(mains), "provenance": {"design": f"{DESIGN_DOC} §6", **stamp}})


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ddxkb.run_eval", description="The single evaluation run on H1 and H2.")
    ap.add_argument("--config-sha256", required=True, help="the protocol digest printed at the freeze")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default paper1/outputs/eval)")
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = args.root.resolve()
    out = args.out.resolve() if args.out is not None else root / OUTPUT_DIR
    try:
        protocol = check_gates(root, args.config_sha256)
        evaluate(protocol, root, out, args.workers)
    except GateError as e:
        raise SystemExit(f"run_eval refused: {e}") from e


if __name__ == "__main__":
    main()
