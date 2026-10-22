"""Development runs on the validation split (design §3-§5). H1 and H2 are never read.

    uv run python -m ddxkb.run_dev --stage grid|ablations|v1|properties|all --workers N [--limit N]

Stages, each writing JSON under the output directory (default paper1/outputs/dev):

- grid: the P1 grid and the four P2 grids over every validation patient. Every configuration and its metrics
  is appended to configs_tried.jsonl, where lines of earlier runs stay, tagged with their start time and
  commit; the design §5 objective picks one per policy among this run's lines into selected.json. The
  selected configurations are then run again for main.json (intervals, curves, each P2 against P1) and
  patient-level files under patients/.
- ablations: A1-A6 into ablations.json and B0, B1, B2, B6 into baselines.json, at budgets 5/10/15/20/29 and
  at the stop, against P1. Needs the selected.json of a grid run over the same validation rows.
- v1: the earlier study's rule (appendix) into v1_replay.json.
- properties: KB violations and set-inclusion uniqueness into properties.json.

`--limit N` keeps the first N validation rows, for smoke runs; its default output directory is then
paper1/outputs/dev_limitN, so a smoke run never mixes into the configurations tried on the full split.
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
from ddxkb.experiment import Context, baseline_suite, main_suite, provenance, utc_now
from ddxkb.policies.ablations import CoverageOrder, Heard, heard_answers
from ddxkb.policies.bayes import BayesPolicy, frequency_prior, membership_table
from ddxkb.policies.v1_replay import (
    V1_DATA_DIR,
    V1Replay,
    code_graph,
    folded_cui_edges,
    load_cui_mappings,
    load_v1_edges,
    subattributes,
    v1_cui_graph,
)
from ddxkb.protocol import (
    A2_PRIOR_LAMBDA,
    A3_EPSILONS,
    BUDGETS,
    CURVE,
    MIN_GAIN,
    P2_VARIANTS,
    V1_TIE_SEEDS,
    P1Config,
    P2Config,
    P2Variant,
    p1_grid,
    p2_grid,
    selection_key,
)
from ddxkb.provenance import git_head
from ddxkb.runner import worker_pool
from ddxkb.simulator import AnswerMode
from ddxkb.units import QuestionUnits, code_units, cui_units

STAGES = ("grid", "ablations", "v1", "properties")
OUTPUT_DIR = "paper1/outputs/dev"
TRIED = "configs_tried.jsonl"
SELECTED = "selected.json"
OBJECTIVE = (
    "validation tie-aware GTPA@1 (|T| <= 30), highest at two decimals of a percent; then the smaller mean IL; "
    "then the earlier candidate in listed order (design §5)"
)
V1_FILES = ("kg_edges.json", "umls_mapping.json", "disease_umls_mapping.json")
J4_TARGET = 81.13
J4_TOLERANCE = 0.5

logger = logging.getLogger(__name__)

Config = P1Config | P2Config


def stage_grid(ctx: Context, limit: int | None) -> None:
    started_utc, started, first = utc_now(), time.perf_counter(), len(ctx.timings)
    run = {"run": started_utc, "commit": git_head(ctx.root)}
    tried = ctx.out / TRIED
    p1_entries: list[tuple[P1Config, dict[str, Any]]] = []
    for (alpha, eps), p1_group in itertools.groupby(enumerate(p1_grid()), key=lambda c: (c[1].alpha, c[1].eps)):
        p1_configs = list(p1_group)
        trajectory = ctx.score(f"grid P1 alpha={alpha} eps={eps}", _no_stop(p1_configs[0][1].policy(ctx.kb)), CURVE)
        records = _record(tried, p1_configs, trajectory, run, limit)
        p1_entries += [(cfg, record) for (_, cfg), record in zip(p1_configs, records, strict=True)]
    p2_entries: dict[str, list[tuple[P2Config, dict[str, Any]]]] = {}
    for variant in P2_VARIANTS:
        entries = p2_entries.setdefault(variant.name, [])
        for lam, p2_group in itertools.groupby(enumerate(p2_grid(variant)), key=lambda c: c[1].lam):
            p2_configs = list(p2_group)
            policy = _no_stop(p2_configs[0][1].policy(ctx.kb, ctx.counts()))
            trajectory = ctx.score(f"grid {variant.name} lambda={lam}", policy, CURVE)
            records = _record(tried, p2_configs, trajectory, run, limit)
            entries += [(cfg, record) for (_, cfg), record in zip(p2_configs, records, strict=True)]
    p1, p1_record = _best(p1_entries)
    p2 = {name: _best(entries) for name, entries in p2_entries.items()}
    main, runs = main_suite(ctx, p1, [cfg for cfg, _ in p2.values()], patients_dir=ctx.out / "patients")
    _check_rerun("P1", p1_record, runs["P1"])
    for name, (_, record) in p2.items():
        _check_rerun(name, record, runs[name])
    selected = {
        "objective": OBJECTIVE,
        "n": len(ctx.patients),
        "limit": limit,
        "P1": _choice(p1, p1_record),
        "P2": {name: _choice(cfg, record) for name, (cfg, record) in p2.items()},
    }
    meta = provenance(ctx, "grid", started_utc, started, first)
    write_json(ctx.out / SELECTED, {**selected, "provenance": meta})
    write_json(ctx.out / "main.json", {"limit": limit, **main, "provenance": meta})


def _no_stop(policy: BayesPolicy) -> BayesPolicy:
    """The grid runs every trajectory to the cap and applies each stop afterwards (`Scored.with_stop`)."""
    return dataclasses.replace(policy, tau=None, min_gain=None)


def _record(
    tried: Path, configs: Sequence[tuple[int, Config]], trajectory: Scored, run: dict[str, Any], limit: int | None
) -> list[dict[str, Any]]:
    """Appends one configs_tried line per configuration, each stop derived from the shared trajectory."""
    records = []
    tried.parent.mkdir(parents=True, exist_ok=True)
    with tried.open("a") as f:
        for order, cfg in configs:
            scored = trajectory.with_stop(cfg.tau, MIN_GAIN)
            summary = report(scored, bootstrap=False)
            key = selection_key(summary["stop"]["gtpa1"]["estimate"], summary["stop"]["il"]["mean"], order)
            record = {
                **run,
                "policy": cfg.name,
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
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            records.append(record)
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


def stage_ablations(ctx: Context, limit: int | None) -> None:
    started_utc, started, first = utc_now(), time.perf_counter(), len(ctx.timings)
    p1cfg, p2cfgs = read_selected(ctx.out, len(ctx.patients), limit)
    references, runs = main_suite(ctx, p1cfg, p2cfgs)
    kb, counts = ctx.kb, ctx.counts()
    p1 = p1cfg.policy(kb)

    def run(
        name: str, policy: BayesPolicy, units: QuestionUnits | None = None, against: dict[str, Scored] | None = None
    ) -> dict[str, Any]:
        scored = ctx.score(name, policy, CURVE, units)
        return report(scored, points=BUDGETS, against={"P1": runs["P1"], **(against or {})})

    code_cui, _ = load_cui_mappings(ctx.root)
    ctx.record_input(f"{V1_DATA_DIR}/umls_mapping.json")
    units = {
        "cui_v1": cui_units(kb, code_cui),
        "cui_merged": cui_units(kb, code_cui, keep_unmapped=True),
        "no_subattributes": code_units(kb, subattributes(kb)),
    }
    train_prior = frequency_prior(counts, A2_PRIOR_LAMBDA)
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
        "A4": {
            "code": "P1",
            **{name: run(f"A4 {name}", p1cfg.policy(u.kb), u) for name, u in units.items()},
            "n_units": {"code": kb.n_evidences, **{name: u.kb.n_evidences for name, u in units.items()}},
            "definitions": {
                "cui_v1": "v1's question space: codes sharing a CUI merged, codes without a CUI never asked",
                "cui_merged": "codes sharing a CUI merged, every code without a CUI a unit of its own",
                "no_subattributes": "one unit per code except the non-binary subattributes (14 in DDXPlus)",
            },
        },
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
    meta = provenance(ctx, "ablations", started_utc, started, first)
    common = {"limit": limit, "budgets": list(BUDGETS)}
    write_json(ctx.out / "ablations.json", {**common, **ablations, "references": references, "provenance": meta})
    write_json(ctx.out / "baselines.json", {**common, **baselines, "provenance": meta})


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
    ap.add_argument("--out", type=Path, default=None, help="output directory (default paper1/outputs/dev)")
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
