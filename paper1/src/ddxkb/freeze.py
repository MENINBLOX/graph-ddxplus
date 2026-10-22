"""Write the frozen protocol of design §5 from the development selection.

    uv run python -m ddxkb.freeze

It reads paper1/outputs/dev/selected.json, which must come from a grid run over every validation row, writes
paper1/configs/protocol_frozen.yaml (never over an existing one) and prints the protocol digest that the
evaluation runner takes as --config-sha256. The yaml goes into a commit of its own: the runner refuses to start
unless exactly one commit in HEAD's history touches it.

Besides `config_sha256`, the sha256 of the canonical JSON of every other key (sorted keys, no whitespace,
UTF-8), the yaml holds:

    design: docs/design/ddxplus_kb_redo.md
    selected: {path: paper1/outputs/dev/selected.json, sha256: ...}
    manifests: {H1: <sha256 of paper1/manifests/H1.json>, H2: <sha256 of paper1/manifests/H2.json>}
    budgets: [5, 10, 15, 20, 29]
    constants: {...}       # protocol_constants(): values the code fixes, compared again before evaluation
    environment: {python: ..., numpy: ..., pandas: ...}   # of the development run
    P1: {alpha: ..., eps: ..., tau: ...}
    P2: {P2-binary: {mode: binary, age_sex: false, lam: ..., tau: ...}, ...one entry per P2 variant}
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from ddxkb.data import MANIFEST_DIR, REPO_ROOT
from ddxkb.experiment import DESIGN_DOC
from ddxkb.frequencies import AGE_BIN_EDGES
from ddxkb.metrics import POSTPROCESS_MASS, TIE_RTOL
from ddxkb.policies.baselines import B2_SEEDS
from ddxkb.policies.bayes import EIG_TIE_ATOL
from ddxkb.protocol import (
    A2_PRIOR_LAMBDA,
    A3_EPSILONS,
    BUDGETS,
    J1_MIN_WILSON_LOW,
    J2_MAX_GTPA1_DIFF,
    J2_POLICY,
    J3_BED_GTPA1,
    J3_BED_IL,
    J3_GTPA1_TOLERANCE,
    J3_IL_TOLERANCE,
    J3_POLICY,
    MIN_GAIN,
    P2_VARIANTS,
    P1Config,
    P2Config,
)
from ddxkb.provenance import file_sha256
from ddxkb.simulator import MAX_IL
from ddxkb.stats import BOOTSTRAP_B, BOOTSTRAP_SEED

FROZEN_CONFIG = "paper1/configs/protocol_frozen.yaml"
SELECTED = "paper1/outputs/dev/selected.json"
HEADER = "# The frozen protocol of design §5, written by `python -m ddxkb.freeze`. Never edit it by hand.\n"


class FreezeError(RuntimeError):
    pass


def config_digest(config: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of every key except `config_sha256`."""
    body = {k: v for k, v in config.items() if k != "config_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def protocol_constants() -> dict[str, Any]:
    """The values fixed in code that the evaluation must run with: stop and tie rules, metric and interval
    settings, the ablation and baseline settings, and the design §6 judgment thresholds."""
    return {
        "max_il": MAX_IL,
        "min_gain": MIN_GAIN,
        "eig_tie_atol": EIG_TIE_ATOL,
        "tie_rtol": TIE_RTOL,
        "postprocess_mass": POSTPROCESS_MASS,
        "age_bin_edges": list(AGE_BIN_EDGES),
        "bootstrap": {"b": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED},
        "a2_prior_lambda": A2_PRIOR_LAMBDA,
        "a3_epsilons": list(A3_EPSILONS),
        "b2_seeds": list(B2_SEEDS),
        "judgments": {
            "J1": {"min_wilson_low": J1_MIN_WILSON_LOW},
            "J2": {"policy": J2_POLICY, "max_gtpa1_diff": J2_MAX_GTPA1_DIFF},
            "J3": {
                "policy": J3_POLICY,
                "bed_gtpa1": J3_BED_GTPA1,
                "bed_il": J3_BED_IL,
                "gtpa1_tolerance": J3_GTPA1_TOLERANCE,
                "il_tolerance": J3_IL_TOLERANCE,
            },
        },
    }


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
    config: dict[str, Any] = {
        "design": DESIGN_DOC,
        "selected": {"path": selected_path, "sha256": file_sha256(path)},
        "manifests": {name: file_sha256(root / MANIFEST_DIR / f"{name}.json") for name in ("H1", "H2")},
        "budgets": list(BUDGETS),
        "constants": protocol_constants(),
        "environment": dict(selected["provenance"]["environment"]),
        "P1": p1.as_dict(),
        "P2": {name: cfg.as_dict() for name, cfg in p2.items()},
    }
    config["config_sha256"] = config_digest(config)
    return config


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
