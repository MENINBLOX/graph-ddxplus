"""Write the frozen protocol of design §5, as 보완 2 revises it, from the development selection.

    uv run python -m ddxkb.freeze

It reads paper1/outputs/v2/dev/selected.json, which must come from a grid run over every validation row with the code
of a commit (nothing under paper1/ uncommitted except the development outputs under paper1/outputs/v2/), writes
paper1/configs/protocol_v2_frozen.yaml (never over an existing one) and prints the protocol digest that the
evaluation runner takes as --config-sha256. The yaml goes into a commit of its own: the runner refuses to start
unless exactly one commit in HEAD's history touches it. The first protocol (paper1/configs/protocol_frozen.yaml)
is a separate file this tool never writes.

Besides `config_sha256`, the sha256 of the canonical JSON of every other key (sorted keys, no whitespace,
UTF-8), the yaml holds:

    design: docs/design/ddxplus_kb_redo.md
    selected: {path: paper1/outputs/v2/dev/selected.json, sha256: ..., commit: <the commit the selection ran>}
    evaluation_set: T
    manifests: {T: <sha256 of paper1/manifests/T.json>, H1: ..., T-dedup: ..., T-streambench: ..., H2: ...}
    subsets: {H1: {available: true, n: ..., rule: ...}, ...}   # what each reporting subset of T holds
    data: {data/ddxplus/20043374.zip: <sha256>, data/ddxplus/release_train_patients.csv: <sha256>}  # of the dev run
    budgets: [5, 10, 15, 20, 30]
    constants: {...}       # protocol_constants(kb): values the code fixes, compared again before evaluation
    environment: {python: ..., numpy: ..., pandas: ..., numpy_simd: [...]}   # of the development run
    P1: {alpha: ..., eps: ..., tau: ...}
    P2: {P2-binary: {mode: binary, age_sex: false, lam: ..., tau: ...}, ...one entry per P2 variant}

The constants hold the judgments of 보완 2 §8 (J1 on both GTPA@1 definitions, J2, and J3' with BED's figures and
tolerances), BED's rules with the sha256 of its stats file, so the B-BED reproduction is frozen too, the disease
order the official GTPA@k read (`KB.official_order`), and the first protocol's settings that the table of changes
(보완 2 §10) puts back. H2's manifest is frozen because those changes count frequencies without its rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from ddxkb.data import KB_ZIP, MANIFEST_DIR, REPO_ROOT, SUBSETS, TRAIN_CSV, Heldout, load_kb, read_manifest
from ddxkb.experiment import DESIGN_DIR, DESIGN_DOC
from ddxkb.frequencies import AGE_BIN_EDGES
from ddxkb.kb import KB
from ddxkb.metrics import POSTPROCESS_MASS, SEVERE_BELOW, TIE_RTOL
from ddxkb.policies.baselines import B2_SEEDS
from ddxkb.policies.bayes import EIG_TIE_ATOL
from ddxkb.policies.bed import (
    BED_STATS,
    BED_STATS_SHA256,
    MAX_QUESTIONS,
    STOP_UTILITY,
    UNSEEN,
    UTILITY_TIE_ATOL,
)
from ddxkb.protocol import (
    A2_PRIOR_LAMBDA,
    A3_EPSILONS,
    A4_HIERARCHIES,
    BUDGETS,
    FIRST_AGE_BIN_EDGES,
    FIRST_FREQUENCIES_WITHOUT,
    FIRST_MAX_IL,
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
    MAIN_HIERARCHY,
    MIN_GAIN,
    P1_WIDER,
    P2_VARIANTS,
    SENSITIVITY_HIERARCHY,
    P1Config,
    P2Config,
)
from ddxkb.provenance import file_sha256
from ddxkb.simulator import MAX_IL
from ddxkb.stats import BOOTSTRAP_B, BOOTSTRAP_SEED

FROZEN_CONFIG = "paper1/configs/protocol_v2_frozen.yaml"
SELECTED = "paper1/outputs/v2/dev/selected.json"
EVALUATION_SET: Heldout = "T"
MANIFESTS: tuple[str, ...] = (EVALUATION_SET, *SUBSETS, FIRST_FREQUENCIES_WITHOUT)
# The selection is made by a development run, whose outputs are the only files under paper1/ it may leave
# uncommitted.
DEV_OUTPUTS = "paper1/outputs/v2/"
FROZEN_DATA = (KB_ZIP, TRAIN_CSV)
HEADER = "# The frozen protocol of design §5 and 보완 2, written by `python -m ddxkb.freeze`. Never edit it by hand.\n"


class FreezeError(RuntimeError):
    pass


def config_digest(config: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of every key except `config_sha256`."""
    body = {k: v for k, v in config.items() if k != "config_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def protocol_constants(kb: KB) -> dict[str, Any]:
    """The values fixed in code that the evaluation must run with: the length cap and hierarchy rules, stop and
    tie rules, metric and interval settings, the disease order of the official metrics over `kb`, the ablation,
    baseline and BED settings, the first protocol's settings and the judgment thresholds."""
    return {
        "max_il": MAX_IL,
        "hierarchy": str(MAIN_HIERARCHY),
        "sensitivity_hierarchy": str(SENSITIVITY_HIERARCHY),
        "min_gain": MIN_GAIN,
        "eig_tie_atol": EIG_TIE_ATOL,
        "tie_rtol": TIE_RTOL,
        "postprocess_mass": POSTPROCESS_MASS,
        "severe_below": SEVERE_BELOW,
        "official_disease_order": [kb.diseases[i] for i in kb.official_order],
        "age_bin_edges": list(AGE_BIN_EDGES),
        "bootstrap": {"b": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED},
        "p1_wider": {name: list(values) for name, values in P1_WIDER.items()},
        "a2_prior_lambda": A2_PRIOR_LAMBDA,
        "a3_epsilons": list(A3_EPSILONS),
        "a4_hierarchies": [str(h) for h in A4_HIERARCHIES],
        "b2_seeds": list(B2_SEEDS),
        "bed": {
            "stats": BED_STATS,
            "stats_sha256": BED_STATS_SHA256,
            "unseen": UNSEEN,
            "stop_utility": STOP_UTILITY,
            "max_questions": MAX_QUESTIONS,
            "utility_tie_atol": UTILITY_TIE_ATOL,
        },
        "first_protocol": {
            "max_il": FIRST_MAX_IL,
            "frequencies_without": FIRST_FREQUENCIES_WITHOUT,
            "age_bin_edges": list(FIRST_AGE_BIN_EDGES),
        },
        "judgments": {
            "J1": {"min_wilson_low": J1_MIN_WILSON_LOW, "metrics": list(J1_METRICS)},
            "J2": {
                "policy": J2_POLICY,
                "max_gtpa1_diff": J2_MAX_GTPA1_DIFF,
                "il_diff_high_below": J2_IL_DIFF_HIGH_BELOW,
            },
            "J3'": {
                "policy": J3_POLICY,
                "bed_gtpa1": J3_BED_GTPA1,
                "bed_il": J3_BED_IL,
                "gtpa1_tolerance": J3_GTPA1_TOLERANCE,
                "il_tolerance": J3_IL_TOLERANCE,
                "metrics": list(J3_METRICS),
                "paired": J3_PAIRED,
            },
        },
    }


def subset_definitions(root: Path) -> dict[str, Any]:
    """What each reporting subset's manifest says it holds: its rule and size, or why it has no row list."""
    out: dict[str, Any] = {}
    for name in SUBSETS:
        manifest = read_manifest(name, root)
        if manifest.get("available", True):
            out[name] = {"available": True, "n": int(manifest["n"]), "rule": str(manifest["selection"]["rule"])}
        else:
            out[name] = {"available": False, "reason": str(manifest["reason"])}
    return out


def frozen_config(root: Path, selected_path: str = SELECTED) -> dict[str, Any]:
    """The protocol of the development selection at `selected_path`, with its digest."""
    path = root / selected_path
    if not path.exists():
        raise FreezeError(f"{selected_path} does not exist; run the development grid first")
    selected = json.loads(path.read_text())
    if selected.get("limit") is not None:
        raise FreezeError(f"{selected_path} comes from a smoke run limited to {selected['limit']} rows")
    p1 = P1Config.from_dict(selected["P1"]["config"])
    p2 = {v.name: P2Config.from_dict(selected["P2"][v.name]["config"]) for v in P2_VARIANTS}
    for name, cfg in p2.items():
        if cfg.name != name:
            raise FreezeError(f"{selected_path} lists a {cfg.name} configuration under {name}")
    meta = selected["provenance"]
    commit, changes, design = meta.get("commit"), meta.get("paper1_changes"), meta.get("design_changes")
    if not commit or changes is None or design is None:
        raise FreezeError(
            f"{selected_path} does not record the commit and the state of paper1/ and {DESIGN_DIR}/ it was made from"
        )
    if meta.get("commit_at_end") != commit:
        raise FreezeError(f"{selected_path} was made while HEAD moved from {commit} to {meta.get('commit_at_end')}")
    stray = [line for line in changes if not porcelain_path(line).startswith(DEV_OUTPUTS)]
    if stray:
        raise FreezeError(f"{selected_path} was made with uncommitted changes under paper1/: {stray[:10]}")
    if design:
        # Design first: what the selection ran must already be in a committed design document.
        raise FreezeError(f"{selected_path} was made with uncommitted changes under {DESIGN_DIR}/: {design[:10]}")
    inputs = meta["inputs"]
    missing = [f for f in FROZEN_DATA if f not in inputs]
    if missing:
        raise FreezeError(f"{selected_path} does not record the sha256 of {missing}")
    absent = [name for name in MANIFESTS if not (root / MANIFEST_DIR / f"{name}.json").exists()]
    if absent:
        raise FreezeError(f"seal {absent} before the freeze")
    config: dict[str, Any] = {
        "design": DESIGN_DOC,
        "selected": {"path": selected_path, "sha256": file_sha256(path), "commit": str(commit)},
        "evaluation_set": EVALUATION_SET,
        "manifests": {name: file_sha256(root / MANIFEST_DIR / f"{name}.json") for name in MANIFESTS},
        "subsets": subset_definitions(root),
        "data": {f: inputs[f] for f in FROZEN_DATA},
        "budgets": list(BUDGETS),
        "constants": protocol_constants(load_kb(root)),
        "environment": dict(meta["environment"]),
        "P1": p1.as_dict(),
        "P2": {name: cfg.as_dict() for name, cfg in p2.items()},
    }
    config["config_sha256"] = config_digest(config)
    return config


def porcelain_path(line: str) -> str:
    """The path of a `git status --porcelain=v1` line: its new path for a rename, unquoted."""
    path = line[3:].split(" -> ")[-1]
    return path[1:-1] if len(path) > 1 and path[0] == path[-1] == '"' else path


def write_frozen(root: Path, selected_path: str = SELECTED) -> str:
    """Writes the frozen protocol and returns its digest; an existing protocol is never overwritten."""
    target = root / FROZEN_CONFIG
    if target.exists():
        raise FreezeError(f"{FROZEN_CONFIG} exists; a frozen protocol is never overwritten")
    config = frozen_config(root, selected_path)
    text = HEADER + yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    if config_digest(yaml.safe_load(text)) != config["config_sha256"]:
        raise FreezeError("the protocol does not survive a YAML round trip unchanged")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return str(config["config_sha256"])


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ddxkb.freeze", description="Write the frozen protocol.")
    ap.add_argument("--selected", default=SELECTED, help=f"the development selection (default {SELECTED})")
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    args = ap.parse_args(argv)
    try:
        digest = write_frozen(args.root.resolve(), args.selected)
    except FreezeError as e:
        raise SystemExit(f"freeze refused: {e}") from e
    print(f"wrote {FROZEN_CONFIG}; config_sha256 {digest}")


if __name__ == "__main__":
    main()
