"""Development runs on the validation split (design §3-§5 with 보완 2 to 보완 4). No sealed set is read.

    uv run python -m ddxkb.run_dev --stage grid|ablations|properties|answer_rule|uniqueness|all --workers N
        [--limit N] [--out DIR]

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
- answer_rule (보완 4 §2): P2-value, P2-recorded (P2-value under the answer rule that also says whether a categorical
  or multi-choice evidence is recorded) and B-BED on the same patients into answer_rule.json, with the comparisons
  P2-recorded - P2-value and B-BED - P2-recorded, and P2-recorded's patient-level file under patients/. The P2-value
  and B-BED runs must reproduce what main.json, baselines.json and their patient-level files record. Needs the
  selected.json of a grid run over the same validation rows, as uniqueness does.
- uniqueness (보완 4 §3): P1, P2-binary, P2-value, P2-recorded and B-BED at their stop, split by whether matching
  the lists gives the gold disease alone, into uniqueness.json; the split must be the one properties.json counts,
  and each policy's stop the one main.json, answer_rule.json or baselines.json records.

A run of answer_rule or uniqueness over every validation row needs the records it is checked against and stops
without them; a limited run checks against those written over the same rows and goes on without the others.

`all` runs the five stages in this order and so writes every output again, grid appending its lines to
configs_tried.jsonl. Where the output directory holds a development run whose files must stay as they are,
answer_rule and then uniqueness are run by name: they write answer_rule.json, patients/P2-recorded.jsonl.gz and
uniqueness.json only.

`--limit N` keeps the first N validation rows, for smoke runs; its default output directory is then
paper1/outputs/v3/dev_limitN, so a smoke run never mixes into the configurations tried on the full split.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ddxkb.audit import list_match_is_unique, support_audit
from ddxkb.data import REPO_ROOT, load_kb, load_validation, unseal_requested
from ddxkb.evaluation import Scored, patient_lines, point_summary, report, write_json, write_patients
from ddxkb.experiment import (
    BED_MAIN,
    Context,
    Stops,
    baseline_suite,
    bed_suite,
    hierarchy_suite,
    ladder_rung,
    main_suite,
    protocol_of,
    provenance,
    rung_name,
    second_run_outputs,
    sensitivity_stops,
    tau_sensitivity,
    utc_now,
)
from ddxkb.first_run import first_run_outputs
from ddxkb.kb import BoolArray
from ddxkb.policies.ablations import CoverageOrder, Heard, heard_answers
from ddxkb.policies.bayes import BayesPolicy, frequency_prior, membership_table
from ddxkb.policies.bed import BED_STATS, BedPolicy, load_knowledge
from ddxkb.policies.episode import Policy
from ddxkb.protocol import (
    A2_PRIOR_LAMBDA,
    A3_EPSILONS,
    A4_HIERARCHIES,
    BUDGETS,
    CURVE,
    MIN_GAIN,
    P2_LAMBDA,
    P2_RECORDED,
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
from ddxkb.simulator import MAX_IL, AnswerMode, Hierarchy

STAGES = ("grid", "ablations", "properties", "answer_rule", "uniqueness")
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


def _recorded(path: Path, keys: Sequence[str], n: int, required: bool) -> Any | None:
    """What the JSON file at `path` holds under `keys`, when that is a run over `n` patients; else None, or with
    `required` an error: FileNotFoundError without the file, ValueError for another number of patients."""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{path} does not exist; the stage that writes it runs first")
        return None
    record = json.loads(path.read_text())
    for key in keys:
        record = record[key]
    if record["n"] != n:
        if required:
            raise ValueError(f"{path} records a run over {record['n']} patients, not over these {n}")
        return None
    return record


def _check_reproduced(
    ctx: Context, limit: int | None, name: str, scored: Scored, summary: Mapping[str, Any], file: str, *keys: str
) -> list[str]:
    """보완 4 §2.4: the run `scored` of a policy an earlier stage also runs, with its `summary`, against what that
    stage recorded in the same output directory: the stop summary of `file` under `keys`, and every line of the
    patient-level file patients/`name`.jsonl.gz. RuntimeError where they differ; returns the files found equal.

    A run over every validation row needs the records. A limited run compares with a record over the same number
    of rows and with nothing when there is none.
    """
    record = _recorded(ctx.out / file, keys, scored.n, required=limit is None)
    if record is None:
        return []
    if record["stop"] != summary["stop"]:
        raise RuntimeError(f"{name}: the stop of this run is not the one {file} records")
    patients = f"patients/{name}.jsonl.gz"
    if not (ctx.out / patients).exists():
        raise FileNotFoundError(f"{ctx.out / patients} does not exist; the stage that wrote {file} writes it too")
    if gzip.decompress((ctx.out / patients).read_bytes()) != b"".join(patient_lines(scored, ctx.kb)):
        raise RuntimeError(f"{name}: a patient of this run differs from {patients}")
    return [file, patients]


# 보완 4 §2.3: the paired comparisons of the answer-rule analysis, each the first policy minus the second on the same
# patients, mapped to what differs between the two.
ANSWER_RULE_LADDER = {
    (P2_RECORDED.name, "P2-value"): (
        "the answer rule: whether a categorical or multi-choice evidence is recorded, with the likelihood that fits it"
    ),
    ("B-BED", P2_RECORDED.name): (
        "what is left once both get that information: BED's utility, stop, smoothing and candidate restriction, and"
        " the one value BED's simulator gives of a multi-choice answer"
    ),
}
ANSWER_RULE_DEFINITIONS = {
    "answer_rule": (
        "recorded (보완 4 §2.1): the protocol's answers (a default value is negative, a categorical answer carries its"
        " value, a multi-choice answer all of its values) and, for a categorical or multi-choice evidence, whether the"
        " patient's record holds it with any value, its default included; a binary evidence is answered as the"
        " protocol answers it. It is the one thing BED's simulator gives beyond the protocol's answers, by reading a"
        " recorded default value as positive; the rule is not that simulator"
    ),
    "P2-value": "the main policy, run again here on the same patients",
    P2_RECORDED.name: (
        "P2-value under the recorded answer rule, with the likelihood that fits it, counted from the train records of"
        " each disease: a categorical evidence is one question over not recorded and each of its values, the default"
        " included, and a multi-choice evidence gets a recorded or not recorded outcome beside its options. Prior,"
        " smoothing, question choice and stop are those of P2-value"
    ),
    "B-BED": (
        "the main B-BED run of baselines.json (BED's stats file, answers from BED's own simulator), run again here on"
        " the same patients"
    ),
    "comparisons": (
        "each the first policy minus the second on the same patients, `factor` naming what differs between the two:"
        " at the stop and after 5, 10, 15, 20 and 30 questions, the difference in tie-aware GTPA@1 with its"
        " patient-bootstrap interval and the patients only one of the two gets right in fixed order with exact"
        " McNemar; at the stop, also the difference in mean IL with its interval; for the two P2 policies, the same at"
        " the stop of each sensitivity threshold"
    ),
    "reproduces": (
        "the records of the earlier stages in the same output directory that the P2-value and B-BED runs of this stage"
        " were found equal to, the stop summary and every line of the patient-level file (보완 4 §2.4); empty where a"
        " limited run found no record over the same rows"
    ),
}


def stage_answer_rule(ctx: Context, limit: int | None) -> None:
    started_utc, started, first = utc_now(), time.perf_counter(), len(ctx.timings)
    _, p2cfgs = read_selected(ctx.out, len(ctx.patients), limit)
    [value] = [cfg for cfg in p2cfgs if cfg.name == "P2-value"]
    recorded = p2_fixed(P2_RECORDED)
    kb, counts = ctx.kb, ctx.counts()
    summaries: dict[str, dict[str, Any]] = {}
    runs: dict[str, Scored] = {}
    stops: dict[str, Stops] = {}

    def bayes(cfg: P2Config) -> Scored:
        policy = cfg.policy(kb, counts)
        scored = ctx.score(f"answer rule {cfg.name}", policy, CURVE)
        stops[cfg.name] = sensitivity_stops(cfg.name, scored)
        summaries[cfg.name] = {
            "config": cfg.as_dict(),
            **protocol_of(policy),
            **report(scored, points=BUDGETS),
            "tau_sensitivity": tau_sensitivity(stops[cfg.name]),
        }
        runs[cfg.name] = scored.restricted(BUDGETS)
        return scored

    # The policies of the earlier stages run first: a difference from their records stops this stage before it
    # writes anything.
    scored = bayes(value)
    reproduces = {
        value.name: _check_reproduced(ctx, limit, value.name, scored, summaries[value.name], "main.json", value.name)
    }
    ctx.record_input(BED_STATS)
    bed = BedPolicy(load_knowledge(kb, ctx.root), budgets=CURVE)
    scored = ctx.score("answer rule B-BED", bed, CURVE)
    summaries["B-BED"] = {
        **protocol_of(bed),
        "answers": "BED simulator (definitions.B-BED)",
        **report(scored, points=BUDGETS),
    }
    runs["B-BED"] = scored.restricted(BUDGETS)
    reproduces["B-BED"] = _check_reproduced(
        ctx, limit, "B-BED", scored, summaries["B-BED"], "baselines.json", "B-BED", BED_MAIN
    )
    scored = bayes(recorded)
    comparisons = {
        rung_name(pair): ladder_rung(
            pair, runs, BUDGETS, {}, stops if all(name in stops for name in pair) else None, factor
        )
        for pair, factor in ANSWER_RULE_LADDER.items()
    }
    meta = provenance(ctx, "answer_rule", started_utc, started, first)
    write_patients(ctx.out / "patients" / f"{recorded.name}.jsonl.gz", scored, kb)
    write_json(
        ctx.out / "answer_rule.json",
        {
            "limit": limit,
            "budgets": list(BUDGETS),
            "definitions": ANSWER_RULE_DEFINITIONS,
            **{name: summaries[name] for name in (value.name, recorded.name, "B-BED")},
            "comparisons": comparisons,
            "reproduces": reproduces,
            "provenance": meta,
        },
    )


UNIQUENESS_DEFINITIONS = {
    "strata": (
        "unique: the gold disease is the only one whose list holds every positive evidence of the record (a default"
        " value is not positive), the `unique` figure of properties.json; not_unique: every other patient"
    ),
    "policies": (
        "each policy at its own stop, over every patient (`all`) and over each stratum: n, tie-aware GTPA@1 with its"
        " Wilson interval and mean IL, as the stop summaries of the other outputs compute them. P2-recorded is the"
        " policy of answer_rule.json, B-BED the main B-BED run (BED's stats file, BED's own simulator)"
    ),
    "lost_score": (
        "the sum over the patients of 1 - tie-aware GTPA@1 at the stop (`total`), the part of it lost on not_unique"
        " patients, and that part's share of the total (null when nothing is lost)"
    ),
    "question_limit": (
        "patients whose episode reached the limit of 30 questions after the initial evidence (IL = 31 whatever ended"
        " the episode: one that met its stop condition with the 30th answer and a failed one are among them) and those"
        " whose episode stopped earlier: n and tie-aware GTPA@1 with its Wilson interval (null for an empty group)"
    ),
    "reproduces": (
        "the records of the earlier stages in the same output directory this run was found equal to: under `strata`,"
        " properties.json for its `unique` count; under each policy, the file whose stop summary of that policy has"
        " the failures, the tie-aware GTPA@1 with its interval and the mean IL of `all`; empty where a limited run"
        " found no record over the same rows"
    ),
}
# Where the earlier stages record the stop of each policy the uniqueness stage runs again: file and keys.
UNIQUENESS_RECORDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "P1": ("main.json", ("P1",)),
    "P2-binary": ("main.json", ("P2-binary",)),
    "P2-value": ("main.json", ("P2-value",)),
    P2_RECORDED.name: ("answer_rule.json", (P2_RECORDED.name,)),
    "B-BED": ("baselines.json", ("B-BED", BED_MAIN)),
}


def _by_stratum(scored: Scored, rows: Sequence[int], unique: BoolArray) -> dict[str, Any]:
    """보완 4 §3.2-3.4 for one run at its stop; `unique[i]` says whether the patient of row `rows[i]` is in the
    unique stratum, and the run must hold these patients in this order.

    A group's GTPA@1 and mean IL are those of `evaluation.point_summary` over its patients, so `all` is the stop the
    other outputs record for the same run, to the last bit.
    """
    if unique.shape != (len(rows),) or scored.rows.tolist() != list(rows):
        raise ValueError("the strata cover other patients")
    gtpa1 = scored.column("gtpa1")

    def group(mask: BoolArray, il: bool = True) -> dict[str, Any]:
        k = int(mask.sum())
        out: dict[str, Any] = {"n": k, "gtpa1": None}
        if il:
            out["il_mean"] = None
        if k:
            summary = point_summary(scored.stop[mask], scored.il[mask], scored.gold[mask], len(scored.diseases))
            out["gtpa1"] = summary["gtpa1"]
            if il:
                out["il_mean"] = summary["il"]["mean"]
        return out

    lost = 1.0 - gtpa1
    total, not_unique = float(lost.sum()), float(lost[~unique].sum())
    at_limit = scored.il == MAX_IL
    return {
        "failures": int(scored.failed.sum()),
        "all": group(np.ones(scored.n, dtype=np.bool_)),
        "strata": {"unique": group(unique), "not_unique": group(~unique)},
        "lost_score": {
            "total": total,
            "not_unique": not_unique,
            "not_unique_share": not_unique / total if total > 0 else None,
        },
        "question_limit": {"reached": group(at_limit, il=False), "stopped_earlier": group(~at_limit, il=False)},
    }


def _check_stop_recorded(name: str, entry: Mapping[str, Any], record: Mapping[str, Any] | None) -> list[str]:
    """The `_by_stratum` entry of a policy the uniqueness stage ran again, against the summary `record` that the
    stage reporting the policy wrote of it (UNIQUENESS_RECORDS): the failures and, over every patient, the tie-aware
    GTPA@1 with its interval and the mean IL. RuntimeError where they differ; returns the file found equal, or
    nothing without a record."""
    if record is None:
        return []
    file, everyone, stop = UNIQUENESS_RECORDS[name][0], entry["all"], record["stop"]
    if (entry["failures"], everyone["gtpa1"], everyone["il_mean"]) != (
        record["failures"],
        stop["gtpa1"],
        stop["il"]["mean"],
    ):
        raise RuntimeError(f"{name}: the stop of this run is not the one {file} records")
    return [file]


def stage_uniqueness(ctx: Context, limit: int | None) -> None:
    started_utc, started, first = utc_now(), time.perf_counter(), len(ctx.timings)
    p1cfg, p2cfgs = read_selected(ctx.out, len(ctx.patients), limit)
    kb, n = ctx.kb, len(ctx.patients)
    t = time.perf_counter()
    rows = [p.row for p in ctx.patients]
    unique = np.fromiter((list_match_is_unique(kb, p) for p in ctx.patients), dtype=np.bool_, count=n)
    ctx.timings.append({"run": "unique list match", "seconds": round(time.perf_counter() - t, 2)})
    sizes = {"unique": int(unique.sum()), "not_unique": int((~unique).sum())}
    properties = _recorded(ctx.out / "properties.json", (), n, required=limit is None)
    if properties is not None and properties["unique"] != {"count": sizes["unique"], "share": sizes["unique"] / n}:
        raise RuntimeError(
            f"the unique stratum holds {sizes['unique']} of {n} patients, properties.json records"
            f" {properties['unique']['count']}"
        )
    # Read before any policy runs: a run over every validation row stops here without one of them.
    records = {
        name: _recorded(ctx.out / file, keys, n, required=limit is None)
        for name, (file, keys) in UNIQUENESS_RECORDS.items()
    }
    counts = ctx.counts()
    p2 = {cfg.name: cfg for cfg in (*p2cfgs, p2_fixed(P2_RECORDED))}
    configs: dict[str, Config] = {
        "P1": p1cfg,
        **{name: p2[name] for name in ("P2-binary", "P2-value", P2_RECORDED.name)},
    }
    policies: dict[str, Policy] = {
        name: cfg.policy(kb, ()) if isinstance(cfg, P1Config) else cfg.policy(kb, counts, ())
        for name, cfg in configs.items()
    }
    ctx.record_input(BED_STATS)
    policies["B-BED"] = BedPolicy(load_knowledge(kb, ctx.root))
    entries: dict[str, Any] = {}
    reproduces = {"strata": [] if properties is None else ["properties.json"]}
    for name, policy in policies.items():
        config = {"config": configs[name].as_dict()} if name in configs else {}
        entry = _by_stratum(ctx.score(f"uniqueness {name}", policy, ()), rows, unique)
        reproduces[name] = _check_stop_recorded(name, entry, records[name])
        entries[name] = {**config, **entry}
    meta = provenance(ctx, "uniqueness", started_utc, started, first)
    write_json(
        ctx.out / "uniqueness.json",
        {
            "limit": limit,
            "definitions": UNIQUENESS_DEFINITIONS,
            "n": n,
            "strata": {name: {"n": k, "share": k / n} for name, k in sizes.items()},
            "policies": entries,
            "reproduces": reproduces,
            "provenance": meta,
        },
    )


STAGE_RUNNERS = {
    "grid": stage_grid,
    "ablations": stage_ablations,
    "properties": stage_properties,
    "answer_rule": stage_answer_rule,
    "uniqueness": stage_uniqueness,
}


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
