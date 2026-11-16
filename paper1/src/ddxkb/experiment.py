"""What the development and evaluation runners share: the run context and the suites of 보완 2 and 보완 3 (main
policies with their stops at the sensitivity thresholds and the factor-by-factor comparisons, baselines and
reference points, the BED reproduction with its answer differences, and the hierarchy sensitivity analysis)."""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from multiprocessing.pool import Pool
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np

from ddxkb.data import TRAIN_CSV, Patient, iter_train, read_release_json
from ddxkb.evaluation import MEANS, PROPORTIONS, RECALLS, Scored, ScoreTask, compare, report, write_patients
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
from ddxkb.policies.bed import (
    BED_STATS,
    BedKnowledge,
    BedPolicy,
    answer_difference,
    count_knowledge,
    load_knowledge,
    text_default_codes,
)
from ddxkb.policies.episode import Policy
from ddxkb.protocol import (
    BED_REPORTED_GTPA1,
    BED_REPORTED_IL,
    BUDGETS,
    CURVE,
    LADDER,
    MIN_GAIN,
    SENSITIVITY_HIERARCHY,
    SENSITIVITY_TAUS,
    TAU,
    P1Config,
    P2Config,
)
from ddxkb.provenance import environment, file_sha256, git_changes, git_head
from ddxkb.runner import chunks, map_items
from ddxkb.simulator import AnswerMode, Hierarchy, StartInput

DESIGN_DOC = "docs/design/ddxplus_kb_redo.md"
DESIGN_DIR = str(PurePosixPath(DESIGN_DOC).parent)
CHUNKS_PER_WORKER = 4
# The outputs of the evaluation of 결과 2, kept as they are (보완 3 §2.10).
SECOND_RUN_OUTPUTS = "paper1/outputs/v2"

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class Context:
    """One run over one patient set: KB, patients, worker pool and output directory.

    Every input file read is recorded with its sha256, and every policy run with its wall time. `git_state` is
    the commit and the uncommitted files under paper1/ and the design documents when the run starts.
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
    bed_train_knowledge: BedKnowledge | None = None
    git_state: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.git_state = {
            "commit": git_head(self.root),
            "paper1_changes": git_changes(self.root, "paper1"),
            "design_changes": git_changes(self.root, DESIGN_DIR),
        }

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
    stages' inputs, and `train_rows` the train rows the frequencies were counted from (None before any count).
    `first_timing` is the index of the stage's first entry in `ctx.timings`. The commit and uncommitted files are
    those at the start of the run; `commit_at_end` is HEAD when the output is written, which differs when a commit
    landed during the run.
    """
    return {
        "design": DESIGN_DOC,
        "stage": stage,
        "split": ctx.split,
        "n_patients": len(ctx.patients),
        **ctx.git_state,
        "commit_at_end": git_head(ctx.root),
        "inputs": dict(sorted(ctx.inputs.items())),
        "train_rows": None if ctx.train_counts is None else int(ctx.train_counts.rows.sum()),
        "environment": environment(),
        "workers": ctx.workers,
        "started_utc": started_utc,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "timings": ctx.timings[first_timing:],
    }


def second_run_outputs(root: Path, out: Path) -> str | None:
    """SECOND_RUN_OUTPUTS when `out` is that directory or lies in it, else None. The runners refuse to write
    there, as into the first run's outputs (`first_run.first_run_outputs`)."""
    target, kept = out.resolve(), (root / SECOND_RUN_OUTPUTS).resolve()
    return SECOND_RUN_OUTPUTS if target == kept or kept in target.parents else None


Subsets = Mapping[str, Collection[int]]


def summarize(
    scored: Scored, points: Sequence[int], against: Mapping[str, Scored], subsets: Subsets, asks: bool = True
) -> dict[str, Any]:
    """`report` of a run, and under "subsets" the same report over each subset's rows (comparisons too)."""
    out = report(scored, points=points, against=against, asks=asks)
    if subsets:
        out["subsets"] = {
            name: report(
                scored.select(rows),
                points=points,
                against={k: other.select(rows) for k, other in against.items()},
                asks=asks,
            )
            for name, rows in subsets.items()
        }
    return out


Stops = Mapping[float, Scored]


def tau_key(tau: float) -> str:
    """A stop threshold as the summaries key it: "0.9", "0.95", "0.99", "0.999"."""
    return f"{tau:g}"


def sensitivity_stops(name: str, scored: Scored) -> dict[float, Scored]:
    """보완 3 §2.1: the stop of a Bayes run at each of SENSITIVITY_TAUS, derived from its one trajectory
    (`Scored.with_stop` with MIN_GAIN), each reduced to its stop. `scored` must record every budget of CURVE and
    be a run stopping at TAU: the stop derived at TAU must be the run's own."""
    stops = {tau: scored.with_stop(tau, MIN_GAIN).restricted(()) for tau in SENSITIVITY_TAUS}
    if stops[TAU].stop_sha256() != scored.stop_sha256():
        raise RuntimeError(f"{name}: the stop derived at tau {TAU} is not the policy's own stop")
    return stops


def tau_sensitivity(stops: Stops) -> dict[str, Any]:
    """The stop summary of `report` (with its intervals) at each threshold of `stops`, keyed by `tau_key`."""
    return {tau_key(tau): report(s)["stop"] for tau, s in stops.items()}


def rung_name(pair: tuple[str, str]) -> str:
    return f"{pair[0]} - {pair[1]}"


def ladder_rung(
    pair: tuple[str, str],
    runs: Mapping[str, Scored],
    budgets: Sequence[int],
    subsets: Subsets,
    stops: Mapping[str, Stops] | None = None,
    factor: str | None = None,
) -> dict[str, Any]:
    """보완 3 §2.5: one comparison of LADDER, the first policy of `pair` minus the second over the same patients.

    `compare` of the two runs at the stop and at `budgets`; with `stops` (both policies' `sensitivity_stops`),
    under "tau_sensitivity" the comparison of their stops at each threshold; and under "subsets" all of it again
    over each subset's rows. `factor` names what differs between the two policies of a pair outside LADDER
    (보완 4 §2.3); a pair of LADDER keeps the factor LADDER gives it and takes none.
    """
    this, other = pair
    if factor is not None and pair in LADDER:
        raise ValueError(f"LADDER names what differs between {this} and {other}")

    def compared(rows: Collection[int] | None) -> dict[str, Any]:
        def cut(s: Scored) -> Scored:
            return s if rows is None else s.select(rows)

        out = compare(cut(runs[this]), cut(runs[other]), points=budgets)
        if stops is not None:
            out["tau_sensitivity"] = {
                tau_key(tau): compare(cut(stops[this][tau]), cut(stops[other][tau]))["stop"] for tau in SENSITIVITY_TAUS
            }
        return out

    entry = {"this": this, "other": other, "factor": LADDER[pair] if factor is None else factor, **compared(None)}
    if subsets:
        entry["subsets"] = {name: compared(rows) for name, rows in subsets.items()}
    return entry


[BED_PAIR] = [pair for pair in LADDER if pair[0] == "B-BED"]
BED_RUNG = "B-BED in baselines.json"


def main_suite(
    ctx: Context,
    p1: P1Config,
    p2: Sequence[P2Config],
    budgets: tuple[int, ...] = BUDGETS,
    patients_dir: Path | None = None,
    subsets: Subsets | None = None,
    sensitivity: bool = False,
) -> tuple[dict[str, Any], dict[str, Scored]]:
    """P1 and the P2 variants at their configurations, each P2 against P1.

    Returns the summaries and the runs, the runs reduced to `budgets`; with `patients_dir` it also writes
    one patient-level file per policy there. Each summary names the start inputs the policy reads, its
    hierarchy rule and answer unit, and holds the report over each of `subsets` (rows of the run's patients).

    With `sensitivity`, the summaries also hold what 보완 3 §2.1 and §2.5 add: each policy's stop summary at each
    of SENSITIVITY_TAUS under "tau_sensitivity" (also in each subset's report), and under "ladder" the comparisons
    of LADDER between these policies (`ladder_rung`, with the thresholds), so `p2` must then hold every variant
    LADDER names; the comparison with B-BED (BED_PAIR) is in the summary of `bed_suite`, which "ladder" names in
    its place.
    """
    configs: list[P1Config | P2Config] = [p1, *p2]
    summary: dict[str, Any] = {}
    runs: dict[str, Scored] = {}
    stops: dict[str, Stops] = {}
    for cfg in configs:
        policy = cfg.policy(ctx.kb) if isinstance(cfg, P1Config) else cfg.policy(ctx.kb, ctx.counts())
        scored = ctx.score(cfg.name, policy, CURVE)
        against = {"P1": runs["P1"]} if runs else {}
        entry = {
            "config": cfg.as_dict(),
            **protocol_of(policy),
            **summarize(scored, budgets, against, subsets or {}),
        }
        if sensitivity:
            stops[cfg.name] = sensitivity_stops(cfg.name, scored)
            entry["tau_sensitivity"] = tau_sensitivity(stops[cfg.name])
            for subset, rows in (subsets or {}).items():
                cut = {tau: s.select(rows) for tau, s in stops[cfg.name].items()}
                entry["subsets"][subset]["tau_sensitivity"] = tau_sensitivity(cut)
        summary[cfg.name] = entry
        if patients_dir is not None:
            write_patients(patients_dir / f"{cfg.name}.jsonl.gz", scored, ctx.kb)
        # Only the reported budgets are kept: a full curve is ~0.5 GB per policy on 130k patients.
        runs[cfg.name] = scored.restricted(budgets)
    if sensitivity:
        summary["ladder"] = {
            rung_name(pair): BED_RUNG if pair == BED_PAIR else ladder_rung(pair, runs, budgets, subsets or {}, stops)
            for pair in LADDER
        }
    return summary, runs


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


BED_MAIN = "stats_file"
BED_DEFINITIONS = {
    "stats_file": "BED's own train statistics, the stats file its code loads; the main B-BED run",
    "counted": "the same statistics counted from the train split, every value of a multi-choice evidence counted",
    "answers": (
        "BED's own simulator (`bed.record_findings`), not the protocol's oracle: a categorical or multi-choice"
        " question is answered positive whenever the evidence is on the true disease's list, its default value"
        " included, so the answer reveals list membership; comparisons with the main policies include this"
    ),
    "answer_differences": (
        "the categorical and multi-choice questions the main run asked up to its stop, over every patient:"
        " valued_questions of them; oracle_negative of these the protocol's oracle answers negative (the evidence"
        " is not recorded, or recorded with its default value); oracle_negative_bed_positive of those BED's"
        " simulator answers positive, both_negative the rest; oracle_negative_bed_filled of the"
        " oracle_negative_bed_positive answers are for an evidence the record lacks, which BED fills in because the"
        " true disease knows it, and the others for a recorded default value; patients_with_difference patients got"
        " at least one answer of the oracle_negative_bed_positive kind"
    ),
    "reported": (
        "the main run minus BED's figures on the official test split (CASANDE Table 1; GTPA@1 as the official"
        " code computes it), recorded without a judgment"
    ),
    "ladder": "the main run against the policy LADDER pairs it with, on the same patients",
}


def bed_answer_differences(k: BedKnowledge, patients: Sequence[Patient], scored: Scored) -> dict[str, Any]:
    """보완 3 §2.6: where BED's simulator and the protocol's oracle answer differently, counted over the questions
    the B-BED run `scored` asked each of `patients` up to its stop (`bed.answer_difference`); a failed patient
    asked none."""
    if [p.row for p in patients] != scored.rows.tolist():
        raise ValueError("the run covers other patients")
    valued = negative = positive = filled = patients_with_difference = 0
    for i, p in enumerate(patients):
        d = answer_difference(k, p, scored.questions[i][: int(scored.il[i]) - 1])
        valued += d.valued
        negative += d.oracle_negative
        positive += d.bed_positive
        filled += d.bed_filled
        patients_with_difference += d.bed_positive > 0
    return {
        "run": BED_MAIN,
        "patients": len(patients),
        "valued_questions": valued,
        "oracle_negative": negative,
        "oracle_negative_bed_positive": positive,
        "oracle_negative_bed_filled": filled,
        "both_negative": negative - positive,
        "patients_with_difference": patients_with_difference,
        "patients_with_difference_share": patients_with_difference / len(patients),
    }


def bed_reported_differences(summary: Mapping[str, Any]) -> dict[str, Any]:
    """보완 3 §2.8: the stop of a B-BED summary minus BED's reported figures, for both GTPA@1 definitions and the
    mean IL."""
    stop = summary["stop"]
    return {
        "run": BED_MAIN,
        "bed": {"gtpa1": BED_REPORTED_GTPA1, "il": BED_REPORTED_IL},
        "this_minus_bed": {
            "gtpa1": stop["gtpa1"]["estimate"] - BED_REPORTED_GTPA1,
            "gtpa1_official": stop["gtpa1_official"]["estimate"] - BED_REPORTED_GTPA1,
            "il": stop["il"]["mean"] - BED_REPORTED_IL,
        },
    }


def bed_suite(
    ctx: Context,
    references: Mapping[str, Scored],
    budgets: tuple[int, ...] = BUDGETS,
    patients_dir: Path | None = None,
    subsets: Subsets | None = None,
) -> dict[str, Any]:
    """B-BED of 보완 2 §6 with each knowledge source of `BED_DEFINITIONS`, against P1 and against the policy LADDER
    pairs it with, on the same patients; `references` holds the runs of `main_suite`. For the main run (BED_MAIN)
    the summary also holds its answer differences, its differences from BED's reported figures and its rung of
    the ladder (보완 3 §2.5, §2.6, §2.8). With `patients_dir` it also writes the patient-level files
    B-BED.jsonl.gz (stats file) and B-BED-counted.jsonl.gz there."""
    ctx.record_input(BED_STATS)
    knowledge = {BED_MAIN: load_knowledge(ctx.kb, ctx.root), "counted": ctx.bed_counts()}
    paired = BED_PAIR[1]
    against = {name: references[name] for name in ("P1", paired)}
    summary: dict[str, Any] = {"definitions": BED_DEFINITIONS}
    for source, k in knowledge.items():
        name = "B-BED" if source == BED_MAIN else f"B-BED-{source}"
        policy = BedPolicy(k, budgets=CURVE)
        scored = ctx.score(name, policy, CURVE)
        protocol = {**protocol_of(policy), "answers": "BED simulator (definitions.answers)"}
        summary[source] = {**protocol, **summarize(scored, budgets, against, subsets or {})}
        if patients_dir is not None:
            write_patients(patients_dir / f"{name}.jsonl.gz", scored, ctx.kb)
        if source == BED_MAIN:
            summary["answer_differences"] = bed_answer_differences(k, ctx.patients, scored)
            summary["reported"] = bed_reported_differences(summary[source])
            runs = {name: scored.restricted(budgets), paired: references[paired]}
            summary["ladder"] = {rung_name(BED_PAIR): ladder_rung(BED_PAIR, runs, budgets, subsets or {})}
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
