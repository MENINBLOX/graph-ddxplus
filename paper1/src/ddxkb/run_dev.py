"""Development runs on the validation split (design §3-§5 as 보완 2 and 보완 3 revise them). No sealed set is read.

    uv run python -m ddxkb.run_dev --stage grid|ablations|properties|all --workers N [--limit N] [--out DIR]

Stages, each writing JSON under the output directory (default paper1/outputs/v3/dev). paper1/outputs/dev and
paper1/outputs/eval hold the first protocol's runs and paper1/outputs/v2 those of 결과 2; an output directory in any
of them is refused.

- grid (보완 3 §2.1-2.3): every trajectory is run once to the cap, without a stop, and its stop at each threshold of
  SENSITIVITY_TAUS is derived from it and appended to configs_tried.jsonl as one line, where lines of earlier runs
  stay, tagged with their start time and commit. The trajectories are the nine P1 candidates (alpha x eps) and
  each P2 variant at the fixed lambda and at the sensitivity lambdas. A line's "role" says what it is for:
  "selection" for the nine P1 lines at TAU, among which the design §5 objective picks P1 into selected.json;
  "fixed" for each P2 variant at P2_LAMBDA and TAU, its one configuration, which selected.json records;
  "sensitivity" for the other thresholds and lambdas, never selected; "wider" for the P1 values one step beyond
  an end of a list where the selection sits: the best configuration at TAU of the grid they widen
  (`protocol.p1_wider_grid`) is recorded in selected.json but never selected. P1 at its selection and the P2
  variants are then run again, each with its own stop, for main.json (intervals, curves, each P2 against P1, each
  policy's stop at the sensitivity thresholds, and the factor-by-factor comparisons of 보완 3 §2.5 between these
  policies; B-BED's is in baselines.json, which main.json names in its place) and patient-level files under
  patients/.
- ablations: A1-A6 into ablations.json, B0, B1, B2, B6 and B-BED into baselines.json, and the hierarchy sensitivity
  analysis into hierarchy.json, at budgets 5/10/15/20/30 and at the stop, each against P1 (the difference in
  GTPA@1 with its paired-bootstrap interval, McNemar on fixed-order GTPA@1, paired bootstrap of the IL
  difference). A4 compares the hierarchy rules of 보완 2 §9 for P1. B-BED's entry also holds its answer
  differences from the protocol's oracle, its differences from BED's reported figures and its comparison with
  P2-value (보완 3 §2.5, §2.6, §2.8). Needs the selected.json of a grid run over the same validation rows.
- properties: KB violations and set-inclusion uniqueness into properties.json.

`--limit N` keeps the first N validation rows, for smoke runs; its default output directory is then
paper1/outputs/v3/dev_limitN, so a smoke run never mixes into the configurations tried on the full split.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ddxkb.audit import support_audit
from ddxkb.data import REPO_ROOT, load_kb, load_validation, unseal_requested
from ddxkb.evaluation import Scored, report, write_json
from ddxkb.experiment import (
    Context,
    baseline_suite,
    bed_suite,
    hierarchy_suite,
    main_suite,
    protocol_of,
    provenance,
    second_run_outputs,
    utc_now,
)
from ddxkb.first_run import first_run_outputs
from ddxkb.policies.ablations import CoverageOrder, Heard, heard_answers
from ddxkb.policies.bayes import BayesPolicy, frequency_prior, membership_table
from ddxkb.protocol import (
    A2_PRIOR_LAMBDA,
    A3_EPSILONS,
    A4_HIERARCHIES,
    BUDGETS,
    CURVE,
    MIN_GAIN,
    P2_LAMBDA,
    P2_SENSITIVITY_LAMBDAS,
    P2_VARIANTS,
    SENSITIVITY_HIERARCHY,
    SENSITIVITY_TAUS,
    TAU,
    P1Config,
    P2Config,
    P2Variant,
    check_fixed,
    p1_boundary,
    p1_grid,
    p1_wider_grid,
    p2_fixed,
    selection_key,
)
from ddxkb.runner import worker_pool
from ddxkb.simulator import AnswerMode, Hierarchy

STAGES = ("grid", "ablations", "properties")
OUTPUT_DIR = "paper1/outputs/v3/dev"
TRIED = "configs_tried.jsonl"
SELECTED = "selected.json"
OBJECTIVE = (
    f"among the nine P1 candidates (alpha x eps) at tau {TAU}: validation tie-aware GTPA@1 at the stop (IL <= 31: 30"
    " questions after the initial evidence), highest at two decimals of a percent; then the smaller mean IL; then"
    " the earlier candidate in listed order (design §5, 보완 3 §2.3)"
)
FIXED = (
    f"every policy stops at tau {TAU} and every P2 variant smooths with lambda {P2_LAMBDA}; neither is selected"
    " (보완 3 §2.1-2.2)"
)

logger = logging.getLogger(__name__)

Config = P1Config | P2Config


def stage_grid(ctx: Context, limit: int | None) -> None:
    started_utc, started, first = utc_now(), time.perf_counter(), len(ctx.timings)
    run = {"run": started_utc, "commit": ctx.git_state["commit"]}
    tried = ctx.out / TRIED
    p1_entries: list[tuple[P1Config, dict[str, Any]]] = []
    for order, cfg in enumerate(p1_grid()):
        trajectory = ctx.score(f"grid P1 alpha={cfg.alpha} eps={cfg.eps}", _no_stop(cfg.policy(ctx.kb)), CURVE)
        lines = _append(tried, _lines(cfg, trajectory, run, limit, "selection", order))
        p1_entries.append((cfg, _at_tau(lines)))
    p2: dict[str, tuple[P2Config, dict[str, Any]]] = {}
    for variant in P2_VARIANTS:
        for lam in (P2_LAMBDA, *P2_SENSITIVITY_LAMBDAS):
            p2cfg = P2Config(variant, lam, TAU)
            policy = _no_stop(p2cfg.policy(ctx.kb, ctx.counts()))
            trajectory = ctx.score(f"grid {variant.name} lambda={lam}", policy, CURVE)
            role = "fixed" if p2cfg == p2_fixed(variant) else "sensitivity"
            lines = _append(tried, _lines(p2cfg, trajectory, run, limit, role))
            if role == "fixed":
                p2[variant.name] = (p2cfg, _at_tau(lines))
    p1, p1_record = _best(p1_entries)
    boundary = _wider_p1(ctx, p1, p1_entries, run, limit)
    chosen = [cfg for cfg, _ in p2.values()]
    main, runs = main_suite(ctx, p1, chosen, patients_dir=ctx.out / "patients", sensitivity=True)
    _check_rerun("P1", p1_record, runs["P1"])
    for name, (_, record) in p2.items():
        _check_rerun(name, record, runs[name])
    selected = {
        "objective": OBJECTIVE,
        "fixed": FIXED,
        "n": len(ctx.patients),
        "limit": limit,
        "P1": {**_choice(p1, p1_record), "boundary": boundary},
        "P2": {name: _setting(cfg, record) for name, (cfg, record) in p2.items()},
    }
    meta = provenance(ctx, "grid", started_utc, started, first)
    write_json(ctx.out / SELECTED, {**selected, "provenance": meta})
    write_json(ctx.out / "main.json", {"limit": limit, **main, "provenance": meta})


def _wider_p1(
    ctx: Context,
    selected: P1Config,
    entries: Sequence[tuple[P1Config, dict[str, Any]]],
    run: dict[str, Any],
    limit: int | None,
) -> dict[str, Any]:
    """보완 3 §2.3: the grid one value wider at the ends of the alpha and eps lists where `selected` sits, its
    new configurations logged with role "wider" and their order in that grid, and the configuration the objective
    would pick there at TAU; nothing when no end is reached. The nine candidates keep their lines; only the new
    configurations are run and logged."""
    ends = p1_boundary(selected)
    if not ends:
        return {"ends": {}, "wider_grid": None}
    wider = p1_wider_grid(selected)
    known = dict(entries)
    keyed = {cfg: _keyed(known[cfg], order) for order, cfg in enumerate(wider) if cfg in known}
    for order, cfg in enumerate(wider):
        if cfg in known:
            continue
        name = f"wider grid P1 alpha={cfg.alpha} eps={cfg.eps}"
        trajectory = ctx.score(name, _no_stop(cfg.policy(ctx.kb)), CURVE)
        keyed[cfg] = _at_tau(_append(ctx.out / TRIED, _lines(cfg, trajectory, run, limit, "wider", order)))
    best_cfg, best = min(keyed.items(), key=lambda e: tuple(e[1]["selection_key"]))
    return {
        "ends": ends,
        "wider_grid": {
            "values": {
                name: sorted({getattr(cfg, name) for cfg in wider}, reverse=name == "eps") for name in ("alpha", "eps")
            },
            "n": len(wider),
            "new": len(wider) - len(known),
            "best": _choice(best_cfg, best),
            "best_is_selected": best_cfg == selected,
        },
    }


def _keyed(record: dict[str, Any], order: int) -> dict[str, Any]:
    """`record` with the order and selection key of another listing."""
    stop = record["stop"]
    key = selection_key(stop["gtpa1"]["estimate"], stop["il"]["mean"], order)
    return {**record, "order": order, "selection_key": list(key)}


def _no_stop(policy: BayesPolicy) -> BayesPolicy:
    """The grid runs every trajectory to the cap and applies each stop afterwards (`Scored.with_stop`)."""
    return dataclasses.replace(policy, tau=None, min_gain=None)


def _lines(
    cfg: Config, trajectory: Scored, run: dict[str, Any], limit: int | None, role: str, order: int | None = None
) -> list[dict[str, Any]]:
    """The configs_tried lines of one trajectory: `cfg` stopped at each of SENSITIVITY_TAUS, each stop derived from
    the trajectory.

    `role` is that of the line at TAU; the lines at the other thresholds are "sensitivity", except those of a
    "wider" configuration, which is never a candidate at any threshold. `order` is the configuration's place among
    the P1 configurations the objective compares (None for P2, which has no selection), and only the lines at TAU
    that the objective reads, "selection" and "wider", carry a selection key.
    """
    records = []
    for tau in SENSITIVITY_TAUS:
        scored = trajectory.with_stop(tau, MIN_GAIN)
        summary = report(scored, bootstrap=False)
        line_role = role if tau == TAU or role == "wider" else "sensitivity"
        key = None
        if order is not None and tau == TAU:
            key = list(selection_key(summary["stop"]["gtpa1"]["estimate"], summary["stop"]["il"]["mean"], order))
        records.append(
            {
                **run,
                "policy": cfg.name,
                "role": line_role,
                "config": dataclasses.replace(cfg, tau=tau).as_dict(),
                "order": order,
                "n": scored.n,
                "limit": limit,
                "failures": summary["failures"],
                "stop": summary["stop"],
                "curve": summary["curve"],
                "selection_key": key,
                "stop_sha256": scored.stop_sha256(),
            }
        )
    return records


def _at_tau(lines: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The line at TAU among the lines of one trajectory."""
    [line] = [r for r in lines if r["config"]["tau"] == TAU]
    return line


def _append(tried: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tried.parent.mkdir(parents=True, exist_ok=True)
    with tried.open("a") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return records


def _best[C](entries: Sequence[tuple[C, dict[str, Any]]]) -> tuple[C, dict[str, Any]]:
    return min(entries, key=lambda e: tuple(e[1]["selection_key"]))


def _setting(cfg: Config, record: dict[str, Any]) -> dict[str, Any]:
    stop = record["stop"]
    return {"config": cfg.as_dict(), "gtpa1": stop["gtpa1"]["estimate"], "il_mean": stop["il"]["mean"]}


def _choice(cfg: Config, record: dict[str, Any]) -> dict[str, Any]:
    stop = record["stop"]
    return {
        "config": cfg.as_dict(),
        "order": record["order"],
        "gtpa1": stop["gtpa1"]["estimate"],
        "il_mean": stop["il"]["mean"],
        "selection_key": record["selection_key"],
    }


def _check_rerun(name: str, record: dict[str, Any], rerun: Scored) -> None:
    """A configuration run with its own stop must reproduce the stop derived in the grid."""
    if rerun.stop_sha256() != record["stop_sha256"]:
        raise RuntimeError(f"{name}: the rerun with its own stop differs from the stop derived in the grid")


def read_selected(out: Path, n: int, limit: int | None) -> tuple[P1Config, list[P2Config]]:
    """The configurations of a grid run over the same `n` validation rows and `limit`: the selected P1, one of the
    nine candidates, and the fixed P2 variants, all holding the fixed settings of this code
    (`protocol.check_fixed`)."""
    path = out / SELECTED
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run --stage grid first")
    selected = json.loads(path.read_text())
    if (selected["n"], selected["limit"]) != (n, limit):
        raise ValueError(
            f"{path} was selected on {selected['n']} rows (limit {selected['limit']}), not on {n} (limit {limit})"
        )
    p1 = P1Config.from_dict(selected["P1"]["config"])
    p2 = [P2Config.from_dict(selected["P2"][v.name]["config"]) for v in P2_VARIANTS]
    check_fixed(p1, p2)
    return p1, p2


A4_DEFINITIONS = {
    str(Hierarchy.NONE): "main protocol: only the asked evidence is answered; a subquestion may come before its parent",
    str(Hierarchy.IMPLICIT): (
        "CASANDE's implicit answers: a negative parent answers its subquestions negative and a positive subquestion"
        " its parent positive, without a turn"
    ),
    str(Hierarchy.PARENT_FIRST): "a subquestion can be asked only once its parent is known positive",
    str(Hierarchy.NO_CHILDREN): "subquestions are never asked",
}


def stage_ablations(ctx: Context, limit: int | None) -> None:
    started_utc, started, first = utc_now(), time.perf_counter(), len(ctx.timings)
    p1cfg, p2cfgs = read_selected(ctx.out, len(ctx.patients), limit)
    references, runs = main_suite(ctx, p1cfg, p2cfgs)
    kb, counts = ctx.kb, ctx.counts()
    p1 = p1cfg.policy(kb)

    def run(name: str, policy: BayesPolicy, against: dict[str, Scored] | None = None) -> dict[str, Any]:
        scored = ctx.score(name, policy, CURVE)
        return {
            **protocol_of(policy),
            **report(scored, points=BUDGETS, against={"P1": runs["P1"], **(against or {})}),
        }

    hierarchy = hierarchy_suite(ctx, p1cfg, p2cfgs, runs)
    train_prior = frequency_prior(counts, A2_PRIOR_LAMBDA)
    a4: dict[str, Any] = {str(Hierarchy.NONE): "P1"}
    for rule in A4_HIERARCHIES:
        if rule is SENSITIVITY_HIERARCHY:
            a4[str(rule)] = "P1 in hierarchy.json"
        else:
            a4[str(rule)] = run(f"A4 {rule}", dataclasses.replace(p1, hierarchy=rule))
    a4["definitions"] = A4_DEFINITIONS
    ablations: dict[str, Any] = {
        "A1": {
            "both": "P1",
            "positive_only": run(
                "A1 positive only", dataclasses.replace(p1, table=heard_answers(p1.table, Heard.POSITIVE))
            ),
            "negative_only": run(
                "A1 negative only", dataclasses.replace(p1, table=heard_answers(p1.table, Heard.NEGATIVE))
            ),
        },
        "A2": {
            "uniform": "P1",
            "train_prior": run("A2 train prior", dataclasses.replace(p1, prior=train_prior)),
            "prior_lambda": A2_PRIOR_LAMBDA,
        },
        "A3": {
            f"{eps:g}": run(f"A3 eps={eps:g}", dataclasses.replace(p1, table=membership_table(kb, p1cfg.alpha, eps)))
            for eps in A3_EPSILONS
        },
        "A4": a4,
        "A5": {
            "eig": "P1",
            "random": "B2 in baselines.json",
            "coverage": run("A5 coverage", dataclasses.replace(p1, selector=CoverageOrder())),
        },
        "A6": {},
    }
    for cfg in p2cfgs:
        if cfg.variant.mode is AnswerMode.BINARY:
            value = P2Config(P2Variant(AnswerMode.VALUE, cfg.variant.age_sex), cfg.lam, cfg.tau)
            name = f"{cfg.name} in value mode"
            ablations["A6"][name] = {
                "config": value.as_dict(),
                **run(f"A6 {name}", value.policy(kb, counts), against={cfg.name: runs[cfg.name]}),
            }
    baselines = baseline_suite(ctx, p1cfg, p2cfgs, runs)
    baselines["B-BED"] = bed_suite(ctx, runs, patients_dir=ctx.out / "patients")
    meta = provenance(ctx, "ablations", started_utc, started, first)
    common = {"limit": limit, "budgets": list(BUDGETS)}
    write_json(ctx.out / "ablations.json", {**common, **ablations, "references": references, "provenance": meta})
    write_json(ctx.out / "baselines.json", {**common, **baselines, "provenance": meta})
    write_json(ctx.out / "hierarchy.json", {**common, **hierarchy, "provenance": meta})


def stage_properties(ctx: Context, limit: int | None) -> None:
    started_utc, started, first = utc_now(), time.perf_counter(), len(ctx.timings)
    t = time.perf_counter()
    audit = support_audit(ctx.kb, ctx.patients)
    ctx.timings.append({"run": "support audit", "seconds": round(time.perf_counter() - t, 2)})
    meta = provenance(ctx, "properties", started_utc, started, first)
    write_json(ctx.out / "properties.json", {"limit": limit, **audit.summary(), "provenance": meta})


STAGE_RUNNERS = {"grid": stage_grid, "ablations": stage_ablations, "properties": stage_properties}


def run_dev(stages: Sequence[str], root: Path, out: Path, workers: int, limit: int | None = None) -> None:
    if unseal_requested():
        raise RuntimeError("development runs refuse to start with DDXKB_UNSEAL=1 set")
    kept = first_run_outputs(root, out) or second_run_outputs(root, out)
    if kept is not None:
        raise ValueError(f"{out} is inside {kept}, the outputs of an earlier run, which are never written")
    unknown = [s for s in stages if s not in STAGE_RUNNERS]
    if unknown:
        raise ValueError(f"unknown stages {unknown}")
    kb = load_kb(root)
    patients = load_validation(root, limit)
    with worker_pool(workers) as pool:
        ctx = Context(root=root, kb=kb, patients=patients, split="validate", out=out, pool=pool, workers=workers)
        ctx.record_input("data/ddxplus/20043374.zip")
        ctx.record_input("data/ddxplus/release_validate_patients.csv")
        for stage in stages:
            logger.info("stage %s on %d validation patients", stage, len(patients))
            STAGE_RUNNERS[stage](ctx, limit)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ddxkb.run_dev", description=__doc__.split("\n")[0] if __doc__ else "")
    ap.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="the first N validation rows only (smoke runs)")
    ap.add_argument("--out", type=Path, default=None, help=f"output directory (default {OUTPUT_DIR})")
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = args.root.resolve()
    default_out = OUTPUT_DIR if args.limit is None else f"{OUTPUT_DIR}_limit{args.limit}"
    out = args.out.resolve() if args.out is not None else root / default_out
    run_dev(STAGES if args.stage == "all" else (args.stage,), root, out, args.workers, args.limit)


if __name__ == "__main__":
    main()
