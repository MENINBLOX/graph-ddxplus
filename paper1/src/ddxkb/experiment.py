"""What the development and evaluation runners share: the run context and the suites of 보완 2 (main policies with
their changes from the first protocol, baselines and reference points, the BED reproduction, and the hierarchy
sensitivity analysis)."""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ddxkb.data import MANIFEST_DIR, TRAIN_CSV, Patient, iter_train, read_release_json
from ddxkb.evaluation import MEANS, PROPORTIONS, RECALLS, Scored, ScoreTask, V1Scored, V1Task, report, write_patients
from ddxkb.frequencies import FrequencyCounts, count_frequencies
from ddxkb.kb import KB
from ddxkb.policies.baselines import (
    B2_SEEDS,
    FullProfile,
    MostFrequentDisease,
    ProfileMajority,
    SetInclusion,
    initial_only,
    random_questions,
    train_frequency_questions,
)
from ddxkb.policies.bayes import BayesPolicy
from ddxkb.policies.bed import BED_STATS, BedKnowledge, BedPolicy, count_knowledge, load_knowledge, text_default_codes
from ddxkb.policies.episode import Policy
from ddxkb.policies.v1_replay import V1Replay
from ddxkb.protocol import (
    BUDGETS,
    CURVE,
    FIRST_AGE_BIN_EDGES,
    FIRST_FREQUENCIES_WITHOUT,
    FIRST_MAX_IL,
    J3_PAIRED,
    MIN_GAIN,
    SENSITIVITY_HIERARCHY,
    P1Config,
    P2Config,
)
from ddxkb.provenance import environment, file_sha256, git_changes, git_head
from ddxkb.runner import chunks, map_items
from ddxkb.simulator import MAX_IL, AnswerMode, Hierarchy, StartInput

DESIGN_DOC = "docs/design/ddxplus_kb_redo.md"
CHUNKS_PER_WORKER = 4

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class Context:
    """One run over one patient set: KB, patients, worker pool and output directory.

    Every input file read is recorded with its sha256, and every policy run with its wall time.
    """

    root: Path
    kb: KB
    patients: Sequence[Patient]
    split: str
    out: Path
    pool: Pool | None = None
    workers: int = 1
    inputs: dict[str, str] = field(default_factory=dict)
    timings: list[dict[str, Any]] = field(default_factory=list)
    train_counts: FrequencyCounts | None = None
    first_train_counts: FrequencyCounts | None = None
    bed_train_knowledge: BedKnowledge | None = None

    def record_input(self, relative: str) -> str:
        if relative not in self.inputs:
            self.inputs[relative] = file_sha256(self.root / relative)
        return self.inputs[relative]

    def counts(self) -> FrequencyCounts:
        """The counts over every train row; the train CSV is recorded even when the counts were given."""
        self.record_input(TRAIN_CSV)
        if self.train_counts is None:
            started = time.perf_counter()
            self.train_counts = count_frequencies(self.kb, iter_train(self.root))
            self._timed("train counts", started)
        return self.train_counts

    def first_counts(self) -> FrequencyCounts:
        """The counts over the train rows outside FIRST_FREQUENCIES_WITHOUT, the first protocol's frequencies; the
        train CSV and that manifest are recorded."""
        self.record_input(TRAIN_CSV)
        self.record_input(f"{MANIFEST_DIR}/{FIRST_FREQUENCIES_WITHOUT}.json")
        if self.first_train_counts is None:
            started = time.perf_counter()
            rows = iter_train(self.root, without=FIRST_FREQUENCIES_WITHOUT)
            self.first_train_counts = count_frequencies(self.kb, rows)
            self._timed(f"train minus {FIRST_FREQUENCIES_WITHOUT} counts", started)
        return self.first_train_counts

    def bed_counts(self) -> BedKnowledge:
        """BED's statistics counted from every train row (`count_knowledge`); recorded like `counts`."""
        self.record_input(TRAIN_CSV)
        if self.bed_train_knowledge is None:
            started = time.perf_counter()
            text_defaults = text_default_codes(read_release_json(self.root)[1])
            self.bed_train_knowledge = count_knowledge(self.kb, iter_train(self.root), text_defaults)
            self._timed("BED train counts", started)
        return self.bed_train_knowledge

    def score(self, name: str, policy: Policy, budgets: tuple[int, ...], local: bool = False) -> Scored:
        """`policy` over every patient; `local` keeps the work in this process, for a policy too large to send to
        the workers chunk by chunk."""
        task = ScoreTask(policy, self.kb, budgets)
        started = time.perf_counter()
        scored = Scored.concat(map_items(task, self._chunks(), None if local else self.pool))
        self._timed(name, started, failures=int(scored.failed.sum()))
        return scored

    def score_v1(self, name: str, replay: V1Replay) -> V1Scored:
        started = time.perf_counter()
        scored = V1Scored.concat(map_items(V1Task(replay), self._chunks(), self.pool))
        self._timed(name, started, failures=int(scored.failed.sum()))
        return scored

    def _chunks(self) -> list[Sequence[Patient]]:
        return chunks(self.patients, 1 if self.pool is None else CHUNKS_PER_WORKER * self.workers)

    def _timed(self, name: str, started: float, **extra: Any) -> None:
        seconds = time.perf_counter() - started
        logger.info("%s: %.1f s %s", name, seconds, extra or "")
        self.timings.append({"run": name, "seconds": round(seconds, 2), **extra})


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def provenance(ctx: Context, stage: str, started_utc: str, started: float, first_timing: int) -> dict[str, Any]:
    """The code, inputs and environment of one stage's output.

    `inputs` lists every file the run has read by then, so a later stage of one run also lists the earlier
    stages' inputs, and `train_rows` the train rows the frequencies were counted from (None before any count),
    `first_protocol_train_rows` those of the first protocol's frequencies. `first_timing` is the index of the
    stage's first entry in `ctx.timings`.
    """
    first = ctx.first_train_counts
    return {
        "design": DESIGN_DOC,
        "stage": stage,
        "split": ctx.split,
        "n_patients": len(ctx.patients),
        "commit": git_head(ctx.root),
        "paper1_changes": git_changes(ctx.root, "paper1"),
        "inputs": dict(sorted(ctx.inputs.items())),
        "train_rows": None if ctx.train_counts is None else int(ctx.train_counts.rows.sum()),
        "first_protocol_train_rows": None if first is None else int(first.rows.sum()),
        "environment": environment(),
        "workers": ctx.workers,
        "started_utc": started_utc,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "timings": ctx.timings[first_timing:],
    }


Subsets = Mapping[str, Collection[int]]


def summarize(
    scored: Scored,
    points: Sequence[int],
    against: Mapping[str, Scored],
    subsets: Subsets,
    asks: bool = True,
    cap: int = MAX_IL,
) -> dict[str, Any]:
    """`report` of a run with length cap `cap`, and under "subsets" the same report over each subset's rows
    (comparisons too)."""
    out = report(scored, points=points, against=against, asks=asks, cap=cap)
    if subsets:
        out["subsets"] = {
            name: report(
                scored.select(rows),
                points=points,
                against={k: other.select(rows) for k, other in against.items()},
                asks=asks,
                cap=cap,
            )
            for name, rows in subsets.items()
        }
    return out


def main_suite(
    ctx: Context,
    p1: P1Config,
    p2: Sequence[P2Config],
    budgets: tuple[int, ...] = BUDGETS,
    patients_dir: Path | None = None,
    subsets: Subsets | None = None,
    first_protocol: bool = False,
) -> tuple[dict[str, Any], dict[str, Scored]]:
    """P1 and the P2 variants at their configurations, each P2 against P1.

    Returns the summaries and the runs, the runs reduced to `budgets`; with `patients_dir` it also writes
    one patient-level file per policy there. Each summary names the start inputs the policy reads, its
    hierarchy rule and answer unit, and holds the report over each of `subsets` (rows of the run's patients);
    with `first_protocol`, also each policy's `first_protocol_changes`.
    """
    configs: list[P1Config | P2Config] = [p1, *p2]
    summary: dict[str, Any] = {}
    runs: dict[str, Scored] = {}
    for cfg in configs:
        policy = cfg.policy(ctx.kb) if isinstance(cfg, P1Config) else cfg.policy(ctx.kb, ctx.counts())
        scored = ctx.score(cfg.name, policy, CURVE)
        against = {"P1": runs["P1"]} if runs else {}
        summary[cfg.name] = {
            "config": cfg.as_dict(),
            **protocol_of(policy),
            **summarize(scored, budgets, against, subsets or {}),
        }
        if first_protocol:
            summary[cfg.name]["first_protocol"] = first_protocol_changes(ctx, cfg, scored, subsets or {})
        if patients_dir is not None:
            write_patients(patients_dir / f"{cfg.name}.jsonl.gz", scored, ctx.kb)
        # Only the reported budgets are kept: a full curve is ~0.5 GB per policy on 130k patients.
        runs[cfg.name] = scored.restricted(budgets)
    return summary, runs


FIRST_PROTOCOL_CHANGES = {
    "cap30": (
        f"this run stopped at IL <= {FIRST_MAX_IL} with the initial evidence, the first protocol's cap; the questions"
        " up to there are the same"
    ),
    "train_minus_h2": f"P2 frequencies counted without the train rows of {FIRST_FREQUENCIES_WITHOUT}",
    "first_age_groups": f"age and sex priors over the first protocol's age groups (edges {list(FIRST_AGE_BIN_EDGES)})",
    "first_knowledge": "both of train_minus_h2 and first_age_groups",
    "first_setting": (
        f"the first protocol's knowledge (train_minus_h2, or first_knowledge when the prior reads age and sex)"
        f" stopped at IL <= {FIRST_MAX_IL}"
    ),
}


def first_protocol_changes(ctx: Context, cfg: P1Config | P2Config, scored: Scored, subsets: Subsets) -> dict[str, Any]:
    """보완 2 §10's table of changes for one main policy, from the same run: the policy at the same configuration
    with settings of the first protocol back (`FIRST_PROTOCOL_CHANGES`), each against `scored`, its run under the
    revised protocol, which must record every budget of CURVE.

    P1 reads no frequencies, so only its cap changes; for a P2 variant without age and sex the age groups change
    nothing, so it has no first_age_groups or first_knowledge entry.
    """

    def summary(run: Scored, cap: int = MAX_IL) -> dict[str, Any]:
        return summarize(run, (), {cfg.name: scored}, subsets, cap=cap)

    out: dict[str, Any] = {"cap30": summary(first_cap(scored, cfg.tau), FIRST_MAX_IL)}
    if isinstance(cfg, P2Config):
        first = ctx.first_counts()
        knowledge = ctx.score(f"{cfg.name} train_minus_h2", cfg.policy(ctx.kb, first), CURVE)
        out["train_minus_h2"] = summary(knowledge)
        if cfg.variant.age_sex:
            ages = cfg.policy(ctx.kb, ctx.counts(), age_edges=FIRST_AGE_BIN_EDGES)
            out["first_age_groups"] = summary(ctx.score(f"{cfg.name} first_age_groups", ages, CURVE))
            both = cfg.policy(ctx.kb, first, age_edges=FIRST_AGE_BIN_EDGES)
            knowledge = ctx.score(f"{cfg.name} first_knowledge", both, CURVE)
            out["first_knowledge"] = summary(knowledge)
        out["first_setting"] = summary(first_cap(knowledge, cfg.tau), FIRST_MAX_IL)
    return {"definitions": {k: FIRST_PROTOCOL_CHANGES[k] for k in out}, **out}


def first_cap(scored: Scored, tau: float) -> Scored:
    """`scored`, a run recording every budget of CURVE, stopped as the same policy stops under FIRST_MAX_IL: question
    choice never depends on the cap, so the trajectory up to there is the same."""
    return scored.restricted(tuple(range(FIRST_MAX_IL))).with_stop(tau, MIN_GAIN, cap=FIRST_MAX_IL)


def baseline_suite(
    ctx: Context,
    p1: P1Config,
    p2: Sequence[P2Config],
    references: Mapping[str, Scored],
    budgets: tuple[int, ...] = BUDGETS,
    subsets: Subsets | None = None,
) -> dict[str, Any]:
    """B0, B1, B2 and B6 of design §3, each against P1 and, when built on a P2 variant, against that variant,
    each also over `subsets`.

    `references` holds the runs of `main_suite`.
    """
    kb, counts = ctx.kb, ctx.counts()
    p1_policy = p1.policy(kb)
    base = {"P1": references["P1"]}

    def reference_point(
        name: str, policy: Referenced, against: Mapping[str, Scored], full_profile: bool, local: bool = False
    ) -> dict[str, Any]:
        scored = ctx.score(name, policy, (), local=local)
        return {**reference_of(policy, full_profile), **summarize(scored, (), against, subsets or {}, asks=False)}

    def inquiry(name: str, policy: BayesPolicy, against: Mapping[str, Scored]) -> dict[str, Any]:
        return {**protocol_of(policy), **summarize(ctx.score(name, policy, CURVE), budgets, against, subsets or {})}

    most_frequent = MostFrequentDisease.from_counts(kb, counts)
    out: dict[str, Any] = {
        "B0": {
            "most_frequent": reference_point("B0 most frequent", most_frequent, base, full_profile=False),
            "initial_only": reference_point("B0 initial only", initial_only(p1_policy), base, full_profile=False),
        }
    }
    majority = ProfileMajority.from_patients(kb, ctx.patients)
    kb_bayes = FullProfile(p1_policy.table, p1_policy.prior)
    b1 = {
        "set_inclusion": reference_point("B1 set inclusion", SetInclusion(kb), base, full_profile=True),
        "kb_bayes": reference_point("B1 KB-Bayes", kb_bayes, base, full_profile=True),
        "binary_majority": {
            "shares_from": f"every patient of {ctx.split}; a subset report keeps these shares",
            **reference_point("B1 binary-profile majority", majority, base, full_profile=True, local=True),
        },
    }
    for cfg in p2:
        policy = cfg.policy(kb, counts)
        against = {**base, cfg.name: references[cfg.name]}
        b1[f"naive_bayes/{cfg.name}"] = reference_point(
            f"B1 naive Bayes {cfg.name}", FullProfile(policy.table, policy.prior), against, full_profile=True
        )
    out["B1"] = b1
    seeds = {str(seed): inquiry(f"B2 seed {seed}", random_questions(p1_policy, seed), base) for seed in B2_SEEDS}
    out["B2"] = {"seeds": seeds, "over_seeds": over_seeds(seeds, budgets)}
    b6 = {"P1": inquiry("B6 P1", train_frequency_questions(p1_policy, counts), base)}
    for cfg in p2:
        policy = train_frequency_questions(cfg.policy(kb, counts), counts)
        b6[cfg.name] = inquiry(f"B6 {cfg.name}", policy, {**base, cfg.name: references[cfg.name]})
    out["B6"] = b6
    return out


BED_DEFINITIONS = {
    "stats_file": "BED's own train statistics, the stats file its code loads (J3')",
    "counted": "the same statistics counted from the train split, every value of a multi-choice evidence counted",
    "answers": (
        "BED's own simulator (`bed.record_findings`), not the protocol's oracle: a categorical or multi-choice"
        " question is answered positive whenever the evidence is on the true disease's list, its default value"
        " included, so the answer reveals list membership; comparisons with the main policies include this"
    ),
}


def bed_suite(
    ctx: Context,
    references: Mapping[str, Scored],
    budgets: tuple[int, ...] = BUDGETS,
    patients_dir: Path | None = None,
    subsets: Subsets | None = None,
) -> dict[str, Any]:
    """B-BED of 보완 2 §6 with each knowledge source of `BED_DEFINITIONS`, against P1 and against J3_PAIRED on the
    same patients; `references` holds the runs of `main_suite`. With `patients_dir` it also writes the
    patient-level files B-BED.jsonl.gz (stats file) and B-BED-counted.jsonl.gz there."""
    ctx.record_input(BED_STATS)
    knowledge = {"stats_file": load_knowledge(ctx.kb, ctx.root), "counted": ctx.bed_counts()}
    against = {name: references[name] for name in ("P1", J3_PAIRED)}
    summary: dict[str, Any] = {"definitions": BED_DEFINITIONS}
    for source, k in knowledge.items():
        name = "B-BED" if source == "stats_file" else f"B-BED-{source}"
        policy = BedPolicy(k, budgets=CURVE)
        scored = ctx.score(name, policy, CURVE)
        summary[source] = {**protocol_of(policy), **summarize(scored, budgets, against, subsets or {})}
        if patients_dir is not None:
            write_patients(patients_dir / f"{name}.jsonl.gz", scored, ctx.kb)
    return summary


def hierarchy_suite(
    ctx: Context,
    p1: P1Config,
    p2: Sequence[P2Config],
    references: Mapping[str, Scored],
    budgets: tuple[int, ...] = BUDGETS,
    subsets: Subsets | None = None,
) -> dict[str, Any]:
    """The sensitivity analysis of 보완 2 §3: P1 and each P2 variant at its configuration under CASANDE's implicit
    answers (`SENSITIVITY_HIERARCHY`), each against P1 and against its own run under the main protocol."""
    kb = ctx.kb
    policies: list[tuple[str, dict[str, Any], BayesPolicy]] = [("P1", p1.as_dict(), p1.policy(kb))]
    policies += [(cfg.name, cfg.as_dict(), cfg.policy(kb, ctx.counts())) for cfg in p2]
    out: dict[str, Any] = {"hierarchy": str(SENSITIVITY_HIERARCHY)}
    for name, config, policy in policies:
        varied = dataclasses.replace(policy, hierarchy=SENSITIVITY_HIERARCHY)
        against = {"P1": references["P1"], name: references[name]}
        scored = ctx.score(f"{name} {SENSITIVITY_HIERARCHY}", varied, CURVE)
        out[name] = {"config": config, **protocol_of(varied), **summarize(scored, budgets, against, subsets or {})}
    return out


class Referenced(Policy, Protocol):
    @property
    def start_inputs(self) -> frozenset[StartInput]: ...

    @property
    def answer_mode(self) -> AnswerMode | None: ...


class Declared(Referenced, Protocol):
    @property
    def hierarchy(self) -> Hierarchy: ...


def protocol_of(policy: Declared) -> dict[str, Any]:
    """The start inputs a policy reads, its hierarchy rule and whether it reads the revealed values, as
    summaries record them."""
    return {
        "inputs": sorted(str(i) for i in policy.start_inputs),
        "hierarchy": str(policy.hierarchy),
        "answers": str(policy.answer_mode),
    }


def reference_of(policy: Referenced, full_profile: bool) -> dict[str, Any]:
    """What a reference point without questions reads: its start inputs, whether it also reads every evidence of
    the record (B1) or none beyond them (B0), and whether it reads the revealed values (None: no answer at all)."""
    mode = policy.answer_mode
    return {
        "inputs": sorted(str(i) for i in policy.start_inputs),
        "profile": "every evidence of the record" if full_profile else "none",
        "answers": None if mode is None else str(mode),
    }


def over_seeds(reports: Mapping[str, Mapping[str, Any]], budgets: Sequence[int]) -> dict[str, Any]:
    """Mean, SD (n - 1 denominator), minimum and maximum over seeds of each estimate (design §4 for B2)."""

    def spread(values: list[float]) -> dict[str, float]:
        x = np.asarray(values, dtype=np.float64)
        sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
        return {"mean": float(x.mean()), "sd": sd, "min": float(x.min()), "max": float(x.max())}

    out: dict[str, Any] = {}
    for point in ("stop", *(str(b) for b in budgets)):
        sections = [r["stop"] if point == "stop" else r["budgets"][point] for r in reports.values()]
        out[point] = {m: spread([s[m]["estimate"] for s in sections]) for m in (*PROPORTIONS, *MEANS, *RECALLS)}
        out[point]["gtpa1_diff_vs_P1"] = spread([r["against"]["P1"][point]["gtpa1_diff"] for r in reports.values()])
    out["stop"]["il_mean"] = spread([r["stop"]["il"]["mean"] for r in reports.values()])
    return out
