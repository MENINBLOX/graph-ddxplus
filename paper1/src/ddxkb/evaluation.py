"""Per-patient scores of policy runs and their summaries (design §4, 보완 2 and 보완 3).

Every policy is scored the same way, at its stop and after each recorded budget: GTPA@1 tie-aware, in fixed
order and as the official code computes it, GTPA@3 and GTPA@5 tie-aware and as the official code computes them,
GTPA, DDR, DDP, DDF1, DSHM, and PER without and with the initial evidence. A patient whose episode raises scores 0
on all of them and counts as reaching |T| = MAX_IL: design §2 scores a failure as a wrong answer and reports the
count. The primary outcome is the tie-aware GTPA@1 at the stop with the mean IL (보완 3 §2.4); paired comparisons
take the difference of the tie-aware GTPA@1 and test the fixed-order one (`compare`).
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import logging
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ddxkb.data import Patient
from ddxkb.kb import KB, BoolArray, FloatArray
from ddxkb.metrics import (
    SEVERE_BELOW,
    PatientScores,
    class_means,
    il_summary,
    positive_evidence_recall,
    positive_evidence_recall_official,
    ranking,
    score_patient,
)
from ddxkb.policies.episode import Policy
from ddxkb.simulator import MAX_IL, positive_codes
from ddxkb.stats import Interval, bootstrap_intervals, mcnemar_exact, wilson_interval

IntArray = npt.NDArray[np.int64]

SCORED = tuple(f.name for f in dataclasses.fields(PatientScores))
RECALLS = ("per", "per_official")
METRICS = (*SCORED, *RECALLS)
PROPORTIONS = (
    "gtpa1",
    "gtpa1_fixed",
    "gtpa1_official",
    "gtpa3",
    "gtpa3_official",
    "gtpa5",
    "gtpa5_official",
    "gtpa",
)
BY_DISEASE = ("gtpa1", "gtpa1_official")
MEANS = ("ddr", "ddp", "ddf1", "dshm")
COLUMN = {m: i for i, m in enumerate(METRICS)}
TOP_K = 5
FAILED_IL = MAX_IL
ERRORS_SHOWN = 20

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class Scored:
    """One policy over many patients, patients in run order; diseases are indices into `diseases` (KB order).

    `stop[i]` holds METRICS at patient i's stop and `il[i]` its |T|. `at_budget[i, k]` holds METRICS after
    exactly budgets[k] questions, reached without stopping early, `peak[i, k]` the largest posterior there and
    `max_gain[i, k]` the largest EIG among the evidences still unasked. `top[i]` ranks the TOP_K most probable
    diseases at the stop, `top_p[i]` their posteriors; a stop derived by `with_stop` has neither. A failed
    patient has gold -1 only if its pathology is not in the KB.
    """

    diseases: tuple[str, ...]
    rows: IntArray
    gold: IntArray
    il: IntArray
    stop: FloatArray
    budgets: tuple[int, ...]
    at_budget: FloatArray
    peak: FloatArray
    max_gain: FloatArray
    questions: tuple[tuple[str, ...], ...]
    failed: BoolArray
    errors: tuple[tuple[int, str], ...]
    top: IntArray | None
    top_p: FloatArray | None

    @property
    def n(self) -> int:
        return len(self.rows)

    def at(self, budget: int) -> FloatArray:
        return np.asarray(self.at_budget[:, self.budgets.index(budget)])

    def column(self, metric: str, budget: int | None = None) -> FloatArray:
        """One metric per patient at the stop (budget None) or after `budget` questions."""
        scores = self.stop if budget is None else self.at(budget)
        return np.asarray(scores[:, COLUMN[metric]])

    def stop_sha256(self) -> str:
        """Digest of every patient's row, |T| and scores at the stop: equal digests mean equal stops."""
        h = hashlib.sha256()
        for a in (self.rows, self.il, self.stop):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def with_stop(self, tau: float, min_gain: float | None, cap: int = MAX_IL) -> Scored:
        """The stop of `BayesPolicy` with this tau and min_gain, applied afterwards to its run without early stop.

        `cap` is that policy's length cap, and the run must record every budget from 0 to cap - 1. The stop is
        the first budget whose posterior reaches tau or whose unasked evidences all have an EIG below
        min_gain, else the last: question choice never depends on the stop. A failed patient's IL is `cap`, as a
        failed run's is FAILED_IL under the main cap.
        """
        if self.budgets != tuple(range(cap)):
            raise ValueError(f"deriving a stop needs every budget from 0 to the length cap - 1 = {cap - 1}")
        reached = self.peak >= tau
        if min_gain is not None:
            reached |= self.max_gain < min_gain
        index = np.where(reached.any(axis=1), reached.argmax(axis=1), cap - 1)
        return dataclasses.replace(
            self,
            il=np.where(self.failed, cap, index + 1).astype(np.int64),
            stop=np.asarray(self.at_budget[np.arange(self.n), index]),
            top=None,
            top_p=None,
        )

    def select(self, rows: Collection[int]) -> Scored:
        """The same run over the patients whose row is in `rows`, in run order."""
        keep = np.isin(self.rows, np.fromiter(rows, dtype=np.int64, count=len(rows)))
        index = np.flatnonzero(keep)
        kept = frozenset(int(r) for r in self.rows[index])
        return dataclasses.replace(
            self,
            rows=self.rows[index],
            gold=self.gold[index],
            il=self.il[index],
            stop=self.stop[index],
            at_budget=self.at_budget[index],
            peak=self.peak[index],
            max_gain=self.max_gain[index],
            questions=tuple(self.questions[i] for i in index),
            failed=self.failed[index],
            errors=tuple(e for e in self.errors if e[0] in kept),
            top=None if self.top is None else self.top[index],
            top_p=None if self.top_p is None else self.top_p[index],
        )

    def restricted(self, budgets: Sequence[int]) -> Scored:
        """The same run keeping only the given budgets."""
        k = [self.budgets.index(b) for b in budgets]
        return dataclasses.replace(
            self,
            budgets=tuple(budgets),
            at_budget=self.at_budget[:, k],
            peak=self.peak[:, k],
            max_gain=self.max_gain[:, k],
        )

    @classmethod
    def concat(cls, parts: Sequence[Scored]) -> Scored:
        if not parts:
            raise ValueError("nothing to concatenate")
        budgets = parts[0].budgets
        if any(p.budgets != budgets for p in parts):
            raise ValueError("parts record different budgets")
        if any(p.diseases != parts[0].diseases for p in parts):
            raise ValueError("parts score different diseases")
        tops = [p.top for p in parts]
        top_ps = [p.top_p for p in parts]
        return cls(
            diseases=parts[0].diseases,
            rows=np.concatenate([p.rows for p in parts]),
            gold=np.concatenate([p.gold for p in parts]),
            il=np.concatenate([p.il for p in parts]),
            stop=np.concatenate([p.stop for p in parts]),
            budgets=budgets,
            at_budget=np.concatenate([p.at_budget for p in parts]),
            peak=np.concatenate([p.peak for p in parts]),
            max_gain=np.concatenate([p.max_gain for p in parts]),
            questions=tuple(q for p in parts for q in p.questions),
            failed=np.concatenate([p.failed for p in parts]),
            errors=tuple(e for p in parts for e in p.errors),
            top=None if any(t is None for t in tops) else np.concatenate([t for t in tops if t is not None]),
            top_p=None if any(t is None for t in top_ps) else np.concatenate([t for t in top_ps if t is not None]),
        )


def _scores(
    kb: KB,
    probs: FloatArray,
    gold: int,
    truth: FloatArray,
    severe: BoolArray,
    asked: Sequence[str],
    positives: frozenset[str],
    initial: str,
) -> list[float]:
    """METRICS in their order."""
    s = score_patient(probs, gold, truth, severe, kb.official_order)
    per = positive_evidence_recall(asked, positives, initial)
    per_official = positive_evidence_recall_official(asked, positives, initial)
    return [*(getattr(s, name) for name in SCORED), per, per_official]


@dataclass(frozen=True, eq=False)
class ScoreTask:
    """Runs `policy` over a chunk of patients and scores every episode; picklable for worker processes."""

    policy: Policy
    kb: KB
    budgets: tuple[int, ...]

    def __call__(self, patients: Sequence[Patient]) -> Scored:
        n, nb, m = len(patients), len(self.budgets), len(METRICS)
        severe = np.array(self.kb.severity) < SEVERE_BELOW
        gold = np.array([self.kb.disease_index.get(p.pathology, -1) for p in patients], dtype=np.int64)
        il = np.full(n, FAILED_IL, dtype=np.int64)
        stop = np.zeros((n, m))
        at_budget = np.zeros((n, nb, m))
        peak = np.zeros((n, nb))
        max_gain = np.zeros((n, nb))
        top = np.full((n, TOP_K), -1, dtype=np.int64)
        top_p = np.zeros((n, TOP_K))
        failed = np.zeros(n, dtype=np.bool_)
        questions: list[tuple[str, ...]] = []
        errors: list[tuple[int, str]] = []
        for i, p in enumerate(patients):
            try:
                ep = self.policy.run(p)
                if ep.budgets != self.budgets:
                    raise ValueError(f"episode records budgets {ep.budgets}, the task expects {self.budgets}")
                truth = self.kb.disease_vector(p.differential)
                positives = positive_codes(self.kb, p)
                stop[i] = _scores(self.kb, ep.posterior, ep.gold, truth, severe, ep.asked, positives, p.initial)
                for k, b in enumerate(self.budgets):
                    q = ep.budget_posteriors[k]
                    at_budget[i, k] = _scores(
                        self.kb, q, ep.gold, truth, severe, ep.questions[:b], positives, p.initial
                    )
                    peak[i, k] = q.max()
                max_gain[i] = ep.budget_gains
                order = ranking(ep.posterior)[:TOP_K]
                top[i, : len(order)] = order
                top_p[i, : len(order)] = ep.posterior[order]
                il[i] = ep.il
                questions.append(ep.questions)
            except Exception as e:
                logger.exception("episode failed for patient row %d", p.row)
                stop[i] = at_budget[i] = peak[i] = max_gain[i] = top_p[i] = 0.0
                top[i] = -1
                il[i] = FAILED_IL
                failed[i] = True
                questions.append(())
                errors.append((p.row, f"{type(e).__name__}: {e}"))
        return Scored(
            diseases=self.kb.diseases,
            rows=np.array([p.row for p in patients], dtype=np.int64),
            gold=gold,
            il=il,
            stop=stop,
            budgets=self.budgets,
            at_budget=at_budget,
            peak=peak,
            max_gain=max_gain,
            questions=tuple(questions),
            failed=failed,
            errors=tuple(errors),
            top=top,
            top_p=top_p,
        )


def _interval(i: Interval) -> dict[str, float]:
    return {"estimate": i.estimate, "low": i.low, "high": i.high}


def point_summary(
    scores: FloatArray, il: IntArray | None, gold: IntArray, n_diseases: int, asks: bool = True
) -> dict[str, Any]:
    """Means of METRICS over patients, with Wilson intervals for the proportions, the macro GTPA@1 (the mean
    over the diseases present of each disease's mean, gold index `gold`), and the summary of `il`.

    `asks` False leaves PER out: a reference that asks no question has nothing to recall.
    """
    n = scores.shape[0]
    means = scores.mean(axis=0)
    out: dict[str, Any] = {}
    for metric in PROPORTIONS:
        out[metric] = _interval(wilson_interval(min(1.0, float(means[COLUMN[metric]])), n))
    for metric in (*MEANS, *RECALLS) if asks else MEANS:
        out[metric] = {"estimate": float(means[COLUMN[metric]])}
    present = np.bincount(gold[gold >= 0], minlength=n_diseases) > 0
    out["macro"] = {
        metric: float(class_means(scores[:, COLUMN[metric]], gold, n_diseases)[0][present].mean())
        for metric in BY_DISEASE
    }
    out["macro"]["diseases"] = int(present.sum())
    if il is not None:
        out["il"] = dataclasses.asdict(il_summary(il, cap=MAX_IL))
    return out


def by_disease(s: Scored) -> dict[str, Any]:
    """Patients and BY_DISEASE means at the stop per gold disease present, in KB order."""
    n_diseases = len(s.diseases)
    columns = {metric: class_means(s.column(metric), s.gold, n_diseases) for metric in BY_DISEASE}
    counts = columns[BY_DISEASE[0]][1]
    return {
        name: {"n": int(counts[d]), **{metric: float(columns[metric][0][d]) for metric in BY_DISEASE}}
        for d, name in enumerate(s.diseases)
        if counts[d] > 0
    }


MCNEMAR_METRIC = "gtpa1_fixed"


def _paired(a: FloatArray, b: FloatArray) -> dict[str, Any]:
    g1, fixed, official = COLUMN["gtpa1"], COLUMN["gtpa1_fixed"], COLUMN["gtpa1_official"]
    tested = COLUMN[MCNEMAR_METRIC]
    test = mcnemar_exact(a[:, tested], b[:, tested])
    return {
        "gtpa1_diff": float(np.mean(a[:, g1] - b[:, g1])),
        "gtpa1_fixed_diff": float(np.mean(a[:, fixed] - b[:, fixed])),
        "gtpa1_official_diff": float(np.mean(a[:, official] - b[:, official])),
        "mcnemar": {
            "metric": MCNEMAR_METRIC,
            "only_this": test.only_a,
            "only_other": test.only_b,
            "p_value": test.p_value,
        },
    }


# Per-patient columns to bootstrap, each with the mapping that takes its bounds ("low", "high").
BootstrapColumns = list[tuple[dict[str, Any], FloatArray]]


def _compared(s: Scored, other: Scored, points: Sequence[int], asks: bool) -> dict[str, Any]:
    """`s` minus `other` at the stop and at the points both runs recorded, as point estimates."""
    c = {"stop": _paired(s.stop, other.stop)}
    for b in points:
        if b in other.budgets:
            c[str(b)] = _paired(s.at(b), other.at(b))
    if asks:
        c["stop"]["il_diff"] = {"estimate": float(np.mean(s.il - other.il))}
    return c


def _differences(c: dict[str, Any], s: Scored, other: Scored) -> BootstrapColumns:
    """The per-patient difference columns behind the comparison `c` of `s` with `other`, each with the mapping of
    `c` that takes its bootstrap bounds: the tie-aware GTPA@1 difference of every point, which gets its own key
    "gtpa1_diff_interval" beside the point estimate "gtpa1_diff", and the IL difference where `c` holds one."""
    out: BootstrapColumns = []
    for point, section in c.items():
        budget = None if point == "stop" else int(point)
        section["gtpa1_diff_interval"] = {"estimate": section["gtpa1_diff"]}
        out.append((section["gtpa1_diff_interval"], s.column("gtpa1", budget) - other.column("gtpa1", budget)))
    if "il_diff" in c["stop"]:
        out.append((c["stop"]["il_diff"], (s.il - other.il).astype(np.float64)))
    return out


def _fill_intervals(columns: BootstrapColumns) -> None:
    """Sets the bootstrap bounds of every column in its mapping; all come from one set of resampled patients."""
    intervals = bootstrap_intervals([column for _, column in columns])
    for (target, _), ci in zip(columns, intervals, strict=True):
        target["low"] = ci.low
        target["high"] = ci.high


def compare(s: Scored, other: Scored, *, points: Sequence[int] = (), asks: bool = True) -> dict[str, Any]:
    """`s` minus `other` over the same patients, as `report` compares a run with each run of `against` (보완 3 §2.5).

    At the stop and at each of `points` both runs recorded: the differences in GTPA@1 (tie-aware, fixed order,
    official), the discordant counts and exact McNemar on fixed-order GTPA@1 (under "mcnemar", which names that
    metric), and the patient-bootstrap interval of the tie-aware difference ("gtpa1_diff_interval"); with `asks`,
    also the mean IL difference at the stop with its interval.
    """
    if not np.array_equal(s.rows, other.rows):
        raise ValueError("the runs to compare cover other patients")
    c = _compared(s, other, points, asks)
    _fill_intervals(_differences(c, s, other))
    return c


def report(
    s: Scored,
    *,
    points: Sequence[int] = (),
    against: Mapping[str, Scored] | None = None,
    bootstrap: bool = True,
    asks: bool = True,
) -> dict[str, Any]:
    """The JSON summary of one run.

    `points` are budgets summarized like the stop; every recorded budget also enters the curve as point
    estimates, and "by_disease" holds the per-disease GTPA@1 at the stop. `against` names runs over the same
    patients to compare with, `s` minus each: the GTPA@1 differences and McNemar on fixed-order GTPA@1 at the stop
    and at the points both runs recorded, and with `asks`, the mean IL difference at the stop. `bootstrap` adds
    patient-bootstrap intervals (design §4) to DDF1, DSHM, both PERs and mean IL, and to the tie-aware GTPA@1
    difference ("gtpa1_diff_interval") and the IL difference of each comparison (보완 3 §2.5); without it a
    comparison holds point estimates only. `asks` False marks a reference without questions, whose IL and PER mean
    nothing.
    """
    others = dict(against or {})
    for name, other in others.items():
        if not np.array_equal(s.rows, other.rows):
            raise ValueError(f"{name} was run on other patients")
    n_diseases = len(s.diseases)
    out: dict[str, Any] = {
        "n": s.n,
        "failures": int(s.failed.sum()),
        "errors": [{"row": r, "error": e} for r, e in s.errors[:ERRORS_SHOWN]],
        "stop": point_summary(s.stop, s.il if asks else None, s.gold, n_diseases, asks),
    }
    out["budgets"] = {str(b): point_summary(s.at(b), None, s.gold, n_diseases, asks) for b in points}
    out["by_disease"] = by_disease(s)
    if s.budgets:
        curve: dict[str, Any] = {"budgets": list(s.budgets)}
        means = s.at_budget.mean(axis=0)
        for metric in PROPORTIONS:
            curve[metric] = [float(x) for x in means[:, COLUMN[metric]]]
        cis = [wilson_interval(min(1.0, float(x)), s.n) for x in means[:, COLUMN["gtpa1"]]]
        curve["gtpa1_low"] = [c.low for c in cis]
        curve["gtpa1_high"] = [c.high for c in cis]
        out["curve"] = curve
    comparisons = {name: _compared(s, other, points, asks) for name, other in others.items()}
    if comparisons:
        out["against"] = comparisons
    if bootstrap:
        _add_bootstrap(out, s, points, others, asks)
    return out


def _add_bootstrap(
    out: dict[str, Any], s: Scored, points: Sequence[int], others: Mapping[str, Scored], asks: bool
) -> None:
    """Fills in the bootstrap intervals in place; all of them come from one set of resampled patients."""
    targets: BootstrapColumns = []
    metrics = ("ddf1", "dshm", "per", "per_official") if asks else ("ddf1", "dshm")
    for metric in metrics:
        targets.append((out["stop"][metric], s.column(metric)))
        for b in points:
            targets.append((out["budgets"][str(b)][metric], s.column(metric, b)))
    if asks:
        targets.append((out["stop"]["il"], s.il.astype(np.float64)))
    for name, other in others.items():
        targets += _differences(out["against"][name], s, other)
    _fill_intervals(targets)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + "\n")


def patient_lines(s: Scored, kb: KB) -> Iterator[bytes]:
    """One JSON line per patient: the stop, its differential head, its scores and the tie-aware GTPA@1 curve."""
    if s.top is None or s.top_p is None:
        raise ValueError("a derived stop has no differential to write")
    top, top_p = s.top, s.top_p
    errors = dict(s.errors)

    def lines() -> Iterator[bytes]:
        for i in range(s.n):
            row = int(s.rows[i])
            head = zip(top[i], top_p[i], strict=True)
            record: dict[str, Any] = {
                "row": row,
                "pathology": kb.diseases[s.gold[i]] if s.gold[i] >= 0 else None,
                "il": int(s.il[i]),
                "asked": list(s.questions[i][: s.il[i] - 1]),
                "top": [[kb.diseases[j], round(float(p), 6)] for j, p in head if j >= 0],
            }
            record.update({m: round(float(s.stop[i, COLUMN[m]]), 6) for m in METRICS})
            record["curve_gtpa1"] = [round(float(x), 4) for x in s.at_budget[i, :, COLUMN["gtpa1"]]]
            if row in errors:
                record["error"] = errors[row]
            yield (json.dumps(record, ensure_ascii=False) + "\n").encode()

    return lines()


def write_patients(path: Path, s: Scored, kb: KB) -> None:
    """`patient_lines` gzipped into `path`.

    The gzip header carries no name or timestamp, so the same results give the same bytes.
    """
    lines = patient_lines(s, kb)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for line in lines:
            gz.write(line)
