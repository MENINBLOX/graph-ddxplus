"""What the development and evaluation runners share: the run context and the main and baseline suites."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any

import numpy as np

from ddxkb.data import MANIFEST_DIR, Patient, iter_train_minus_h2, read_manifest
from ddxkb.evaluation import PROPORTIONS, Scored, ScoreTask, V1Scored, V1Task, report, write_patients
from ddxkb.frequencies import FrequencyCounts, count_frequencies
from ddxkb.kb import KB
from ddxkb.policies.baselines import (
    B2_SEEDS,
    FullProfile,
    MostFrequentDisease,
    SetInclusion,
    initial_only,
    random_questions,
    train_frequency_questions,
)
from ddxkb.policies.episode import Policy
from ddxkb.policies.v1_replay import V1Replay
from ddxkb.protocol import BUDGETS, CURVE, P1Config, P2Config
from ddxkb.provenance import environment, file_sha256, git_changes, git_head
from ddxkb.runner import chunks, map_items
from ddxkb.units import QuestionUnits

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

    def record_input(self, relative: str) -> str:
        if relative not in self.inputs:
            self.inputs[relative] = file_sha256(self.root / relative)
        return self.inputs[relative]

    def counts(self) -> FrequencyCounts:
        """The train-minus-H2 counts; their inputs are verified and recorded even when the counts were given."""
        h2 = f"{MANIFEST_DIR}/H2.json"
        if h2 not in self.inputs:
            manifest = read_manifest("H2", self.root)
            csv = str(manifest["csv"])
            expected = manifest.get("csv_sha256")
            if expected is not None and self.record_input(csv) != expected:
                raise ValueError(f"{csv} is not the file the H2 manifest was made from")
            self.record_input(csv)
            self.record_input(h2)
        if self.train_counts is None:
            started = time.perf_counter()
            self.train_counts = count_frequencies(self.kb, iter_train_minus_h2(self.root))
            self._timed("train minus H2 counts", started)
        return self.train_counts

    def score(self, name: str, policy: Policy, budgets: tuple[int, ...], units: QuestionUnits | None = None) -> Scored:
        task = ScoreTask(policy, self.kb if units is None else units.kb, budgets, units)
        started = time.perf_counter()
        scored = Scored.concat(map_items(task, self._chunks(), self.pool))
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
    stages' inputs, and `train_minus_h2_rows` the train rows the frequencies were counted from (None before
    any count). `first_timing` is the index of the stage's first entry in `ctx.timings`.
    """
    return {
        "design": DESIGN_DOC,
        "stage": stage,
        "split": ctx.split,
        "n_patients": len(ctx.patients),
        "commit": git_head(ctx.root),
        "paper1_changes": git_changes(ctx.root, "paper1"),
        "inputs": dict(sorted(ctx.inputs.items())),
        "train_minus_h2_rows": None if ctx.train_counts is None else int(ctx.train_counts.rows.sum()),
        "environment": environment(),
        "workers": ctx.workers,
        "started_utc": started_utc,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "timings": ctx.timings[first_timing:],
    }


def main_suite(
    ctx: Context,
    p1: P1Config,
    p2: Sequence[P2Config],
    budgets: tuple[int, ...] = BUDGETS,
    patients_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Scored]]:
    """P1 and the P2 variants at their configurations, each P2 against P1.

    Returns the summaries and the runs, the runs reduced to `budgets`; with `patients_dir` it also writes
    one patient-level file per policy there.
    """
    policies: list[tuple[str, dict[str, Any], Policy]] = [("P1", p1.as_dict(), p1.policy(ctx.kb))]
    policies += [(cfg.name, cfg.as_dict(), cfg.policy(ctx.kb, ctx.counts())) for cfg in p2]
    summary: dict[str, Any] = {}
    runs: dict[str, Scored] = {}
    for name, config, policy in policies:
        scored = ctx.score(name, policy, CURVE)
        against = {"P1": runs["P1"]} if runs else {}
        summary[name] = {"config": config, **report(scored, points=budgets, against=against)}
        if patients_dir is not None:
            write_patients(patients_dir / f"{name}.jsonl.gz", scored, ctx.kb)
        # Only the reported budgets are kept: a full curve is ~0.3 GB per policy on 130k patients.
        runs[name] = scored.restricted(budgets)
    return summary, runs


def baseline_suite(
    ctx: Context,
    p1: P1Config,
    p2: Sequence[P2Config],
    references: Mapping[str, Scored],
    budgets: tuple[int, ...] = BUDGETS,
) -> dict[str, Any]:
    """B0, B1, B2 and B6 of design §3, each against P1 and, when built on a P2 variant, against that variant.

    `references` holds the runs of `main_suite`.
    """
    kb, counts = ctx.kb, ctx.counts()
    p1_policy = p1.policy(kb)
    base = {"P1": references["P1"]}

    def reference_point(name: str, policy: Policy, against: Mapping[str, Scored]) -> dict[str, Any]:
        return report(ctx.score(name, policy, ()), against=against, asks=False)

    def inquiry(name: str, policy: Policy, against: Mapping[str, Scored]) -> dict[str, Any]:
        return report(ctx.score(name, policy, CURVE), points=budgets, against=against)

    out: dict[str, Any] = {
        "B0": {
            "most_frequent": reference_point("B0 most frequent", MostFrequentDisease.from_counts(kb, counts), base),
            "initial_only": reference_point("B0 initial only", initial_only(p1_policy), base),
        }
    }
    b1 = {
        "set_inclusion": reference_point("B1 set inclusion", SetInclusion(kb), base),
        "kb_bayes": reference_point("B1 KB-Bayes", FullProfile(p1_policy.table, p1_policy.prior), base),
    }
    for cfg in p2:
        policy = cfg.policy(kb, counts)
        against = {**base, cfg.name: references[cfg.name]}
        b1[f"naive_bayes/{cfg.name}"] = reference_point(
            f"B1 naive Bayes {cfg.name}", FullProfile(policy.table, policy.prior), against
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


def over_seeds(reports: Mapping[str, Mapping[str, Any]], budgets: Sequence[int]) -> dict[str, Any]:
    """Mean, SD (n - 1 denominator), minimum and maximum over seeds of each estimate (design §4 for B2)."""

    def spread(values: list[float]) -> dict[str, float]:
        x = np.asarray(values, dtype=np.float64)
        sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
        return {"mean": float(x.mean()), "sd": sd, "min": float(x.min()), "max": float(x.max())}

    out: dict[str, Any] = {}
    for point in ("stop", *(str(b) for b in budgets)):
        sections = [r["stop"] if point == "stop" else r["budgets"][point] for r in reports.values()]
        out[point] = {m: spread([s[m]["estimate"] for s in sections]) for m in (*PROPORTIONS, "ddf1", "per")}
        out[point]["gtpa1_diff_vs_P1"] = spread([r["against"]["P1"][point]["gtpa1_diff"] for r in reports.values()])
    out["stop"]["il_mean"] = spread([r["stop"]["il"]["mean"] for r in reports.values()])
    return out
