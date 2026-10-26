"""The single evaluation run on T, the whole official test split (design 보완 2), from the frozen protocol only.

    DDXKB_UNSEAL=1 uv run python -m ddxkb.run_eval --config-sha256 <digest> --workers N

It refuses to start unless all of these hold:

- DDXKB_UNSEAL=1 is set;
- the output directory is not one of the first run's (paper1/outputs/dev, paper1/outputs/eval) or inside one;
- nothing under GATED_PATHS (paper1/ and the design documents) is modified, deleted or untracked. Other directories
  hold files of other projects that this run never reads, and the data files it reads are checked by sha256;
- paper1/configs/protocol_v2_frozen.yaml (written by `ddxkb.freeze`, which describes its keys) exists, its
  recorded `config_sha256` is the digest of its content, and that digest is the one given on the command line;
- exactly one commit in HEAD's history changes the protocol file: the freeze commit;
- nothing under paper1/src differs between the selection run's commit and HEAD, so the evaluation runs the code
  that made the selection;
- the protocol's P1 and P2 configurations are those of the selection file it names, whose sha256 it records,
  and each P2 configuration is the variant it is listed under;
- the manifests on disk (T, its reporting subsets and H2) are the ones the protocol names, the test CSV is the file
  T was sealed from, the KB and train files are those the development run read, and BED's stats file is the one
  the protocol names;
- the code holds the protocol's constants, and Python, numpy (with its SIMD dispatch) and pandas are those of
  the development run.

The run writes under <out>/T/: main.json (P1 and the P2 variants with curves, each P2 against P1, and each with its
changes from the first protocol, `experiment.first_protocol_changes`), baselines.json
(B0, B1, B2, B6 and B-BED, B-BED also against P2-value), hierarchy.json (the main policies under CASANDE's implicit
answers, against their main runs), properties.json (KB violations and set-inclusion uniqueness) and patient-level
files under patients/; and <out>/judgments.json with J1, J2 and J3' of 보완 2 §8. Every summary also holds the
report of each reporting subset of T (H1, T-dedup, T-streambench when its row list was available) cut from the
same run. Each output records the protocol digest, the freeze commit, the commit run and the input sha256s. An
existing result is never overwritten: a rerun after a fix goes to a new --out, and both are reported.

The first evaluation (H1 and H2 under paper1/configs/protocol_frozen.yaml) is never rerun; `ddxkb.first_run`
verifies what it left.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from ddxkb.audit import support_audit
from ddxkb.data import (
    MANIFEST_DIR,
    REPO_ROOT,
    SUBSETS,
    load_heldout,
    load_kb,
    read_manifest,
    subset_rows,
    unseal_requested,
)
from ddxkb.evaluation import write_json
from ddxkb.experiment import (
    BED_DEFINITIONS,
    DESIGN_DOC,
    Context,
    baseline_suite,
    bed_suite,
    hierarchy_suite,
    main_suite,
    provenance,
    utc_now,
)
from ddxkb.first_run import first_run_outputs
from ddxkb.freeze import EVALUATION_SET, FROZEN_CONFIG, FROZEN_DATA, MANIFESTS, config_digest, protocol_constants
from ddxkb.policies.bed import BED_STATS, BED_STATS_SHA256
from ddxkb.protocol import (
    CURVE,
    J1_METRICS,
    J1_MIN_WILSON_LOW,
    J2_IL_DIFF_HIGH_BELOW,
    J2_MAX_GTPA1_DIFF,
    J2_POLICY,
    J3_BED_GTPA1,
    J3_BED_IL,
    J3_GTPA1_TOLERANCE,
    J3_IL_TOLERANCE,
    J3_METRICS,
    J3_PAIRED,
    J3_POLICY,
    P2_VARIANTS,
    P1Config,
    P2Config,
)
from ddxkb.provenance import (
    environment,
    file_sha256,
    git_changed_since,
    git_changes,
    git_commits_touching,
    git_head,
)
from ddxkb.runner import worker_pool

OUTPUT_DIR = "paper1/outputs/v2/eval"
GATED_PATHS = ("paper1", str(PurePosixPath(DESIGN_DOC).parent))
CODE = "paper1/src"

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
    """Modified, deleted and untracked files under GATED_PATHS, as `git status --porcelain` lines."""
    changes = git_changes(root, *GATED_PATHS)
    if changes is None:
        raise GateError(f"{root} is not a git work tree")
    return changes


def code_changes(root: Path, commit: str) -> list[str]:
    """Files under CODE that differ between `commit` and HEAD."""
    changed = git_changed_since(root, commit, CODE)
    if changed is None:
        raise GateError(f"cannot compare {CODE} at {commit} with HEAD")
    return changed


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
    changed = code_changes(root, str(config["selected"]["commit"]))
    if changed:
        raise GateError(f"{CODE} changed after the selection run's commit: {changed[:10]}")
    policies(config)
    if config["evaluation_set"] != EVALUATION_SET:
        raise GateError(f"the protocol evaluates {config['evaluation_set']}, this runner {EVALUATION_SET}")
    for name in MANIFESTS:
        if file_sha256(root / MANIFEST_DIR / f"{name}.json") != config["manifests"][name]:
            raise GateError(f"the {name} manifest is not the one the protocol names")
    manifest = read_manifest(EVALUATION_SET, root)
    if file_sha256(root / str(manifest["csv"])) != manifest["csv_sha256"]:
        raise GateError(f"{manifest['csv']} is not the file the {EVALUATION_SET} manifest was made from")
    for relative in FROZEN_DATA:
        if file_sha256(root / relative) != config["data"][relative]:
            raise GateError(f"{relative} is not the file the development run read")
    if not (root / BED_STATS).exists() or file_sha256(root / BED_STATS) != BED_STATS_SHA256:
        raise GateError(f"{BED_STATS} is not BED's stats file")
    if config["constants"] != protocol_constants(load_kb(root)):
        raise GateError("the code no longer holds the constants of the frozen protocol")
    if config["environment"] != environment():
        raise GateError(f"this environment {environment()} is not the development run's {config['environment']}")
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


def judgments(main: Mapping[str, Any], baselines: Mapping[str, Any]) -> dict[str, Any]:
    """J1, J2 and J3' of 보완 2 §8 from the main.json and baselines.json summaries of T, all read at the stop."""
    lows = {metric: main["P1"]["stop"][metric]["low"] for metric in J1_METRICS}
    j2 = main[J2_POLICY]["against"]["P1"]["stop"]
    gtpa1_diff, il_diff_high = j2["gtpa1_diff"], j2["il_diff"]["high"]
    bed = baselines[J3_POLICY]["stats_file"]
    bed_gtpa1 = {metric: bed["stop"][metric]["estimate"] for metric in J3_METRICS}
    bed_il = bed["stop"]["il"]["mean"]
    paired = bed["against"][J3_PAIRED]["stop"]
    return {
        "J1": {
            "rule": f"Wilson low of P1 GTPA@1 >= {J1_MIN_WILSON_LOW} on T for each of {list(J1_METRICS)}",
            "wilson_low": lows,
            "met": all(low >= J1_MIN_WILSON_LOW for low in lows.values()),
        },
        "J2": {
            "rule": (
                f"|{J2_POLICY} - P1 tie-aware GTPA@1| < {J2_MAX_GTPA1_DIFF} and the upper paired-bootstrap bound of"
                f" {J2_POLICY} - P1 mean IL < {J2_IL_DIFF_HIGH_BELOW}, on T"
            ),
            "gtpa1_diff": gtpa1_diff,
            "il_diff_high": il_diff_high,
            "met": abs(gtpa1_diff) < J2_MAX_GTPA1_DIFF and il_diff_high < J2_IL_DIFF_HIGH_BELOW,
        },
        "J3'": {
            "rule": (
                f"{J3_POLICY} with BED's stats file on T within {J3_GTPA1_TOLERANCE} of BED's GTPA@1 {J3_BED_GTPA1}"
                f" for each of {list(J3_METRICS)}, and its mean IL within {J3_IL_TOLERANCE} of BED's {J3_BED_IL}"
            ),
            "gtpa1": bed_gtpa1,
            "il_mean": bed_il,
            "met": all(abs(g - J3_BED_GTPA1) <= J3_GTPA1_TOLERANCE for g in bed_gtpa1.values())
            and abs(bed_il - J3_BED_IL) <= J3_IL_TOLERANCE,
            "paired": {
                "comparison": f"{J3_POLICY} - {J3_PAIRED} on the same rows, reported, not judged",
                "answers": BED_DEFINITIONS["answers"],
                **paired,
            },
        },
    }


def reporting_subsets(root: Path) -> tuple[dict[str, frozenset[int]], dict[str, Any]]:
    """The subsets with a row list, and what every subset manifest says (n, or why there is no list)."""
    rows: dict[str, frozenset[int]] = {}
    described: dict[str, Any] = {}
    for name in SUBSETS:
        found = subset_rows(name, root)
        if found is None:
            described[name] = {"available": False, "reason": read_manifest(name, root)["reason"]}
        else:
            rows[name] = found
            described[name] = {"available": True, "n": len(found)}
    return rows, described


def evaluate(protocol: FrozenProtocol, root: Path, out: Path, workers: int) -> None:
    """Every policy, baseline, budget curve and dataset property of the protocol on T in one run, with the
    reporting subsets cut from it."""
    name = EVALUATION_SET
    kept = first_run_outputs(root, out)
    if kept is not None:
        raise GateError(f"{out} is inside {kept}, the first run's outputs, which are never written")
    existing = [path for path in (out / name, out / "judgments.json") if path.exists()]
    if existing:
        raise GateError(f"results exist at {existing}; a rerun goes to a new --out")
    p1, p2, budgets = policies(protocol.config)
    kb = load_kb(root)
    subsets, described = reporting_subsets(root)
    stamp = {
        "protocol_sha256": protocol.sha256,
        "protocol_freeze_commit": protocol.freeze_commit,
        "protocol_commit": protocol.commit,
    }
    with worker_pool(workers) as pool:
        started_utc, started = utc_now(), time.perf_counter()
        patients = load_heldout(name, root)
        ctx = Context(root, kb, patients, name, out / name, pool, workers)
        manifest = read_manifest(name, root)
        for relative in ("data/ddxplus/20043374.zip", str(manifest["csv"])):
            ctx.record_input(relative)
        for subset in MANIFESTS:
            ctx.record_input(f"{MANIFEST_DIR}/{subset}.json")
        audit = support_audit(kb, patients)
        patients_dir = ctx.out / "patients"
        main, runs = main_suite(ctx, p1, p2, budgets, patients_dir=patients_dir, subsets=subsets, first_protocol=True)
        baselines = baseline_suite(ctx, p1, p2, runs, budgets, subsets=subsets)
        baselines["B-BED"] = bed_suite(ctx, runs, budgets, patients_dir=patients_dir, subsets=subsets)
        hierarchy = hierarchy_suite(ctx, p1, p2, runs, budgets, subsets=subsets)
        meta = {**provenance(ctx, f"eval {name}", started_utc, started, 0), **stamp, "subsets": described}
        write_json(ctx.out / "main.json", {"set": name, **main, "provenance": meta})
        write_json(ctx.out / "baselines.json", {"set": name, **baselines, "provenance": meta})
        write_json(ctx.out / "hierarchy.json", {"set": name, **hierarchy, "provenance": meta})
        write_json(ctx.out / "properties.json", {"set": name, **audit.summary(), "provenance": meta})
    design = f"{DESIGN_DOC} 보완 2 §8"
    write_json(out / "judgments.json", {**judgments(main, baselines), "provenance": {"design": design, **stamp}})


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ddxkb.run_eval", description="The single evaluation run on T.")
    ap.add_argument("--config-sha256", required=True, help="the protocol digest printed at the freeze")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help=f"output directory (default {OUTPUT_DIR})")
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
