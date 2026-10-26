"""Development runs on the validation split (design §3-§5 as 보완 2 revises them). No sealed set is read.

    uv run python -m ddxkb.run_dev --stage grid|ablations|v1|properties|all --workers N [--limit N] [--out DIR]

Stages, each writing JSON under the output directory (default paper1/outputs/v2/dev). paper1/outputs/dev and
paper1/outputs/eval hold the first protocol's runs, and an output directory in either is refused.

- grid: the 27 P1 candidates and the four P2 grids of 8 over every validation patient, IL <= 31. Every
  configuration and its metrics is appended to configs_tried.jsonl, where lines of earlier runs stay, tagged with
  their start time and commit; the design §5 objective picks one per policy among this run's lines of the original
  grids into selected.json. Where the P1 selection sits at an end of a P1 list, the grid one value wider there
  (`protocol.p1_wider_grid`) is run too, its new configurations logged as grid "wider" and its best one recorded
  in selected.json, but never selected (보완 2 §4). The selected configurations are then run again for main.json
  (intervals, curves, each P2 against P1, each policy's changes from the first protocol) and patient-level files
  under patients/.
- ablations: A1-A6 into ablations.json, B0, B1, B2, B6 and B-BED into baselines.json, and the hierarchy sensitivity
  analysis into hierarchy.json, at budgets 5/10/15/20/30 and at the stop, each against P1 (McNemar on GTPA@1,
  paired bootstrap of the IL difference). A4 compares the hierarchy rules of 보완 2 §9 for P1. Needs the
  selected.json of a grid run over the same validation rows.
- v1: the earlier study's rule (appendix) into v1_replay.json.
- properties: KB violations and set-inclusion uniqueness into properties.json.

`--limit N` keeps the first N validation rows, for smoke runs; its default output directory is then
paper1/outputs/v2/dev_limitN, so a smoke run never mixes into the configurations tried on the full split.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ddxkb.audit import support_audit
from ddxkb.data import REPO_ROOT, load_kb, load_validation, unseal_requested
from ddxkb.evaluation import Scored, report, v1_report, write_json
from ddxkb.experiment import (
    Context,
    baseline_suite,
    bed_suite,
    hierarchy_suite,
    main_suite,
    protocol_of,
    provenance,
    utc_now,
)
from ddxkb.first_run import first_run_outputs
from ddxkb.policies.ablations import CoverageOrder, Heard, heard_answers
from ddxkb.policies.bayes import BayesPolicy, frequency_prior, membership_table
from ddxkb.policies.v1_replay import (
    V1_DATA_DIR,
    V1Replay,
    code_graph,
    folded_cui_edges,
    load_cui_mappings,
    load_v1_edges,
    v1_cui_graph,
)
from ddxkb.protocol import (
    A2_PRIOR_LAMBDA,
    A3_EPSILONS,
    A4_HIERARCHIES,
    BUDGETS,
    CURVE,
    MIN_GAIN,
    P1_WIDER,
    P2_VARIANTS,
    SENSITIVITY_HIERARCHY,
    V1_TIE_SEEDS,
    P1Config,
    P2Config,
    P2Variant,
    p1_boundary,
    p1_grid,
    p1_wider_grid,
    p2_grid,
    selection_key,
)
from ddxkb.runner import worker_pool
from ddxkb.simulator import AnswerMode, Hierarchy

STAGES = ("grid", "ablations", "v1", "properties")
OUTPUT_DIR = "paper1/outputs/v2/dev"
TRIED = "configs_tried.jsonl"
SELECTED = "selected.json"
OBJECTIVE = (
    "validation tie-aware GTPA@1 (IL <= 31: 30 questions after the initial evidence), highest at two decimals of a "
    "percent; then the smaller mean IL; then the earlier candidate in listed order (design §5, 보완 2)"
)
V1_FILES = ("kg_edges.json", "umls_mapping.json", "disease_umls_mapping.json")
J4_TARGET = 81.13
J4_TOLERANCE = 0.5

logger = logging.getLogger(__name__)

Config = P1Config | P2Config


def stage_grid(ctx: Context, limit: int | None) -> None:
    started_utc, started, first = utc_now(), time.perf_counter(), len(ctx.timings)
    run = {"run": started_utc, "commit": ctx.git_state["commit"]}
    tried = ctx.out / TRIED
    p1_entries: list[tuple[P1Config, dict[str, Any]]] = []
    # Lines for the wider tau values of every original (alpha, eps) trajectory, kept until the selection shows
    # whether the wider grid needs them, so that no trajectory is run twice.
    held: dict[P1Config, dict[str, Any]] = {}
    for (alpha, eps), p1_group in itertools.groupby(enumerate(p1_grid()), key=lambda c: (c[1].alpha, c[1].eps)):
        p1_configs = list(p1_group)
        trajectory = ctx.score(f"grid P1 alpha={alpha} eps={eps}", _no_stop(p1_configs[0][1].policy(ctx.kb)), CURVE)
        records = _append(tried, _lines(p1_configs, trajectory, run, limit))
        p1_entries += [(cfg, record) for (_, cfg), record in zip(p1_configs, records, strict=True)]
        wider_taus = [(-1, P1Config(alpha, eps, tau)) for tau in P1_WIDER["tau"]]
        held.update(zip((c for _, c in wider_taus), _lines(wider_taus, trajectory, run, limit), strict=True))
    p2_entries: dict[str, list[tuple[P2Config, dict[str, Any]]]] = {}
    for variant in P2_VARIANTS:
        entries = p2_entries.setdefault(variant.name, [])
        for lam, p2_group in itertools.groupby(enumerate(p2_grid(variant)), key=lambda c: c[1].lam):
            p2_configs = list(p2_group)
            policy = _no_stop(p2_configs[0][1].policy(ctx.kb, ctx.counts()))
            trajectory = ctx.score(f"grid {variant.name} lambda={lam}", policy, CURVE)
            records = _append(tried, _lines(p2_configs, trajectory, run, limit))
            entries += [(cfg, record) for (_, cfg), record in zip(p2_configs, records, strict=True)]
    p1, p1_record = _best(p1_entries)
    boundary = _wider_p1(ctx, p1, p1_entries, held, run, limit)
    p2 = {name: _best(entries) for name, entries in p2_entries.items()}
    chosen = [cfg for cfg, _ in p2.values()]
    main, runs = main_suite(ctx, p1, chosen, patients_dir=ctx.out / "patients", first_protocol=True)
    _check_rerun("P1", p1_record, runs["P1"])
    for name, (_, record) in p2.items():
        _check_rerun(name, record, runs[name])
    selected = {
        "objective": OBJECTIVE,
        "n": len(ctx.patients),
        "limit": limit,
        "P1": {**_choice(p1, p1_record), "boundary": boundary},
        "P2": {name: _choice(cfg, record) for name, (cfg, record) in p2.items()},
    }
    meta = provenance(ctx, "grid", started_utc, started, first)
    write_json(ctx.out / SELECTED, {**selected, "provenance": meta})
    write_json(ctx.out / "main.json", {"limit": limit, **main, "provenance": meta})


def _wider_p1(
    ctx: Context,
    selected: P1Config,
    entries: Sequence[tuple[P1Config, dict[str, Any]]],
    held: dict[P1Config, dict[str, Any]],
    run: dict[str, Any],
    limit: int | None,
) -> dict[str, Any]:
    """보완 2 §4: the grid one value wider at the ends of the P1 lists where `selected` sits, logged as grid "wider"
    with its order in that grid, and the configuration the objective would pick there; nothing when no end is
    reached. Original configurations keep their lines; only the new ones are run and logged."""
    ends = p1_boundary(selected)
    if not ends:
        return {"ends": {}, "wider_grid": None}
    wider = p1_wider_grid(selected)
    order = {cfg: k for k, cfg in enumerate(wider)}
    known = {cfg: record for cfg, record in entries}
    new = [(order[cfg], cfg) for cfg in wider if cfg not in known]
    lines: list[dict[str, Any]] = []
    fresh = [(k, cfg) for k, cfg in new if cfg not in held]
    for (alpha, eps), group in itertools.groupby(fresh, key=lambda c: (c[1].alpha, c[1].eps)):
        configs = list(group)
        trajectory = ctx.score(f"wider grid P1 alpha={alpha} eps={eps}", _no_stop(configs[0][1].policy(ctx.kb)), CURVE)
        lines += _lines(configs, trajectory, run, limit, grid="wider")
    lines += [_keyed({**held[cfg], "grid": "wider"}, k) for k, cfg in new if cfg in held]
    lines.sort(key=lambda r: int(r["order"]))
    _append(ctx.out / TRIED, lines)
    keyed = {cfg: _keyed(record, order[cfg]) for cfg, record in known.items()}
    keyed.update({P1Config.from_dict(r["config"]): r for r in lines})
    best_cfg, best = min(keyed.items(), key=lambda e: tuple(e[1]["selection_key"]))
    return {
        "ends": ends,
        "wider_grid": {
            "values": {
                name: sorted({getattr(cfg, name) for cfg in wider}, reverse=name == "eps")
                for name in ("alpha", "eps", "tau")
            },
            "n": len(wider),
            "new": len(new),
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
    configs: Sequence[tuple[int, Config]],
    trajectory: Scored,
    run: dict[str, Any],
    limit: int | None,
    grid: str = "original",
) -> list[dict[str, Any]]:
    """One configs_tried line per configuration, each stop derived from the shared trajectory."""
    records = []
    for order, cfg in configs:
        scored = trajectory.with_stop(cfg.tau, MIN_GAIN)
        summary = report(scored, bootstrap=False)
        key = selection_key(summary["stop"]["gtpa1"]["estimate"], summary["stop"]["il"]["mean"], order)
        records.append(
            {
                **run,
                "policy": cfg.name,
                "grid": grid,
                "config": cfg.as_dict(),
                "order": order,
                "n": scored.n,
                "limit": limit,
                "failures": summary["failures"],
                "stop": summary["stop"],
                "curve": summary["curve"],
                "selection_key": list(key),
                "stop_sha256": scored.stop_sha256(),
            }
        )
    return records


def _append(tried: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tried.parent.mkdir(parents=True, exist_ok=True)
    with tried.open("a") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return records


def _best[C](entries: Sequence[tuple[C, dict[str, Any]]]) -> tuple[C, dict[str, Any]]:
    return min(entries, key=lambda e: tuple(e[1]["selection_key"]))


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
    """The selected configuration run with its own stop must reproduce the stop derived in the grid."""
    if rerun.stop_sha256() != record["stop_sha256"]:
        raise RuntimeError(f"{name}: the rerun with its own stop differs from the stop derived in the grid")


def read_selected(out: Path, n: int, limit: int | None) -> tuple[P1Config, list[P2Config]]:
    """The configurations selected by a grid run over the same `n` validation rows and `limit`."""
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


def stage_v1(ctx: Context, limit: int | None) -> None:
    started_utc, started, first = utc_now(), time.perf_counter(), len(ctx.timings)
    for name in V1_FILES:
        ctx.record_input(f"{V1_DATA_DIR}/{name}")
    cui = v1_cui_graph(ctx.kb, ctx.root)
    graphs = {"cui": cui, "code": code_graph(ctx.kb), "code_no_subattributes": code_graph(ctx.kb, True)}
    runs: dict[str, Any] = {}
    for (graph_name, graph), bug in itertools.product(graphs.items(), (True, False)):
        name = f"{graph_name}/trav1_bug" if bug else graph_name
        runs[name] = v1_report(ctx.score_v1(f"v1 {name}", V1Replay(graph, trav1_bug=bug)))
    seeds = {
        str(s): v1_report(ctx.score_v1(f"v1 cui/trav1_bug tie seed {s}", V1Replay(cui, trav1_bug=True, tie_seed=s)))
        for s in V1_TIE_SEEDS
    }
    observed = 100 * runs["cui/trav1_bug"]["top1"]["estimate"]
    out = {
        "limit": limit,
        "rule": "experiment_final_75.py as rebuilt in the review (§2.2-2.5): deny 6, v15 score, stable top 3, 223 max",
        "ties": "ascending node id unless a tie seed shuffles them",
        "runs": runs,
        "tie_seeds": {
            "run": "cui/trav1_bug",
            "seeds": seeds,
            "range": {
                "top1": _range([r["top1"]["estimate"] for r in seeds.values()]),
                "gtpa1": _range([r["gtpa1"]["estimate"] for r in seeds.values()]),
                "il_mean": _range([r["il"]["mean"] for r in seeds.values()]),
            },
        },
        "J4": {
            "quantity": "v1-order GTPA@1 (%) of cui/trav1_bug on validation",
            "observed": observed,
            "target": J4_TARGET,
            "tolerance": J4_TOLERANCE,
            "met": abs(observed - J4_TARGET) <= J4_TOLERANCE,
        },
    }
    write_json(ctx.out / "v1_replay.json", {**out, "provenance": provenance(ctx, "v1", started_utc, started, first)})


def _range(values: list[float]) -> list[float]:
    return [min(values), max(values)]


def stage_properties(ctx: Context, limit: int | None) -> None:
    started_utc, started, first = utc_now(), time.perf_counter(), len(ctx.timings)
    t = time.perf_counter()
    audit = support_audit(ctx.kb, ctx.patients)
    ctx.timings.append({"run": "support audit", "seconds": round(time.perf_counter() - t, 2)})
    out: dict[str, Any] = {"limit": limit, **audit.summary()}
    if all((ctx.root / V1_DATA_DIR / name).exists() for name in V1_FILES):
        code_cui, disease_cui = load_cui_mappings(ctx.root)
        folded = folded_cui_edges(ctx.kb, code_cui, disease_cui)
        v1 = set(load_v1_edges(ctx.root))
        out["kb_folded_to_cuis"] = {"edges": len(folded), "v1_graph_edges": len(v1), "shared": len(folded & v1)}
        for name in V1_FILES:
            ctx.record_input(f"{V1_DATA_DIR}/{name}")
    meta = provenance(ctx, "properties", started_utc, started, first)
    write_json(ctx.out / "properties.json", {**out, "provenance": meta})


STAGE_RUNNERS = {"grid": stage_grid, "ablations": stage_ablations, "v1": stage_v1, "properties": stage_properties}


def run_dev(stages: Sequence[str], root: Path, out: Path, workers: int, limit: int | None = None) -> None:
    if unseal_requested():
        raise RuntimeError("development runs refuse to start with DDXKB_UNSEAL=1 set")
    kept = first_run_outputs(root, out)
    if kept is not None:
        raise ValueError(f"{out} is inside {kept}, the first run's outputs, which are never written")
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
