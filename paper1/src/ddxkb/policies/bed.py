"""B-BED: the BED baseline of CASANDE ported rule for rule (design 보완 2 §6, J3' of §8).

Source: mila-iqia/Casande-RL at 5279013, `Baselines/BED`, run as `run_ddxplus.sh` runs it (solver BED_BATCH,
KL utility, search depth 1, 30 questions, threshold 0.01). CASANDE reports it on the official test split at
GTPA@1 99.47% and IL 5.47. File:line references below are to that commit.

Knowledge (`BedKnowledge`), from train records without age, sex or smoothing:
- P(d) is the share of train rows (`ddxplus_utils.py:96`, `data_utils.py:199`).
- A disease knows an evidence when one of its train records holds it; a categorical or multi-choice evidence on
  its KB list counts as held with its default value by every record lacking it (`ddxplus_utils.py:39-43`). In
  DDXPlus the known evidences are the KB lists less 6 binary entries no train record holds.
- A known binary evidence has the share of the disease's records holding it. A known categorical or multi-choice
  evidence has the share of each option, the options being every value some train record holds, plus the default
  of a multi-choice evidence, in sorted string order (`ddxplus_utils.py:101-136`, `data_utils.py:216-220`);
  an option the disease's records never hold, or a value outside the options, has 1e-7 (`QMR.py:217`).
- BED's counting keeps one value per record of a multi-choice evidence: the last one in the iteration order of
  a Python set of strings (`ddxplus_utils.py:40-54`), which string hash randomization changes between runs.
  `load_knowledge` reads the stats file the BED code loads at that commit; `count_knowledge` counts from train
  records and counts every value of a multi-choice evidence. The two agree on everything else (priors, binary and
  categorical shares are bit-identical on the local train file).

Answers come from BED's own simulator (`record_findings`, `bayesian_experimental_design_batch.py:148-244`), not
from `simulator.Oracle`: an evidence is answered positive when the record holds it, whatever its value, and the
evidences of the true disease that the record lacks but the disease knows are filled in as held with the default.
So a categorical or multi-choice question answers whether the evidence is on the true disease's list, which the
protocol's oracle (a default value is negative) does not reveal. A positive answer restricts the candidate
diseases to those that know the evidence (`:141-145`). A multi-choice answer carries only the last value the
record lists for it (`:196-199` stores each value in a new set).

Inference (`QMR.py:210-259`), over every disease: the product of the prior, P(o | d) of each positive answer
(0 when d does not know the evidence) and 1 - P(o | d) of each negative answer, normalized. A negative binary
answer uses P = 0 for an unknown evidence; a negative categorical or multi-choice answer uses the default
option, looked up by the JSON default value among string options (`bayesian_experimental_design_batch.py:170-172`),
so the integer defaults of DDXPlus never match and give 1 - 1e-7 to every disease.

Questions (`bayesian_experimental_design.py:70-193`): only evidences known to a candidate disease and not yet
asked; the posterior is already 0 outside the candidates, so the restriction narrows the scan, not the choice.
The utility is the expected KL divergence of the new posterior from the current one: over the two answers
of a binary evidence, over the options of a categorical one (the unrecorded outcome is left out), and the
largest such two-answer value over the options of a multi-choice one. The highest utility is asked; BED stops
when it is below 0.01, and after 30 questions. Neither age nor sex is read.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ddxkb.data import REPO_ROOT, Patient, read_release_json
from ddxkb.kb import KB, BoolArray, DataType, FloatArray, IndexArray
from ddxkb.policies.episode import Episode
from ddxkb.simulator import AnswerMode, Hierarchy, StartInput

UNSEEN = 1e-7
STOP_UTILITY = 0.01
MAX_QUESTIONS = 30
# Utilities closer than this are one tie, asked in KB evidence order. BED keeps the first strictly greater utility
# in Python set iteration order, so among equal utilities its pick follows float rounding and hash-table slots,
# which no port can follow. Rounding moves a utility by about 1e-16, while the 1e-7 floor makes real differences
# of about 1e-13 (skin lesions against their location), which BED resolves and this tolerance keeps.
UTILITY_TIE_ATOL = 1e-14
# BED's KL adds numpy's float epsilon inside both logarithms (`bayesian_experimental_design.py:33-37`).
_EPS = float(np.finfo(np.float64).eps)

BED_STATS = "paper1/reference/casande_bed/ddxplus_bed_stats_4.json"
BED_STATS_SHA256 = "84e781b350800eb8a2087c5749e0d1eb07d81a4e8a50ddffcab3e06e70df533c"

# Evidence index -> option indices of a positive answer; () for a binary evidence.
Findings = dict[int, tuple[int, ...]]


class BedDeadEnd(RuntimeError):
    """Every disease has likelihood 0. The original code divides 0 by 0 at its next step here."""


def text_default_codes(evidences: Mapping[str, Any]) -> frozenset[str]:
    """Codes of release_evidences.json whose default_value is a JSON string, the only defaults BED's option lookup
    can find."""
    return frozenset(code for code, e in evidences.items() if isinstance(e["default_value"], str))


@dataclass(frozen=True, eq=False)
class BedKnowledge:
    """BED's train statistics in KB order.

    `binary[d, j]` is P(present | d) of a known binary evidence (0 elsewhere). For a categorical or
    multi-choice evidence j, `options[j]` lists BED's options and `option_probs[j][d, o]` is P(option o | d),
    UNSEEN where d's records never hold it; rows of diseases that do not know j are not read.
    `default_option[j]` is the option a negative answer to j names, -1 when BED's lookup misses it.
    """

    kb: KB
    prior: FloatArray
    known: BoolArray
    binary: FloatArray
    options: tuple[tuple[str, ...], ...]
    option_probs: tuple[FloatArray, ...]
    default_option: IndexArray
    option_index: tuple[Mapping[str, int], ...] = field(init=False)
    outcomes: FloatArray = field(init=False)
    component_start: IndexArray = field(init=False)
    evidence_start: IndexArray = field(init=False)

    def __post_init__(self) -> None:
        kb = self.kb
        d, e = kb.n_diseases, kb.n_evidences
        if self.prior.shape != (d,) or self.known.shape != (d, e) or self.binary.shape != (d, e):
            raise ValueError("prior, known and binary must be in KB shape")
        if len(self.options) != e or len(self.option_probs) != e or self.default_option.shape != (e,):
            raise ValueError("options, option_probs and default_option need one entry per evidence")
        for j, ev in enumerate(kb.evidences):
            if ev.is_binary and (self.options[j] or self.default_option[j] != -1):
                raise ValueError(f"binary evidence {ev.code} has options")
            if self.option_probs[j].shape != (d, len(self.options[j])):
                raise ValueError(f"{ev.code}: option_probs does not match its options")
            if not -1 <= self.default_option[j] < len(self.options[j]):
                raise ValueError(f"{ev.code}: default option outside its options")
        if ((self.binary < 0) | (self.binary > 1)).any() or (self.binary[~self.known] != 0).any():
            raise ValueError("binary shares must lie in [0, 1] and be 0 for unknown evidences")
        for a in (self.prior, self.known, self.binary, self.default_option, *self.option_probs):
            a.flags.writeable = False
        object.__setattr__(self, "option_index", tuple({v: o for o, v in enumerate(opts)} for opts in self.options))
        self._build_outcomes()

    def _build_outcomes(self) -> None:
        """The answer likelihoods every question's utility sums over, grouped like `bayes.LikelihoodTable`: a
        binary question is one [yes, no] component, a categorical one a component over its options, and a
        multi-choice one a [chosen, not chosen] component per option, of which the utility takes the largest."""
        components: list[FloatArray] = []
        per_evidence: list[int] = []
        for j, ev in enumerate(self.kb.evidences):
            if ev.is_binary:
                parts = [np.stack([self.binary[:, j], 1.0 - self.binary[:, j]], axis=1)]
            elif ev.data_type is DataType.CATEGORICAL:
                parts = [self._held(j)] if self.options[j] else [np.zeros((self.kb.n_diseases, 1))]
            else:
                held, absent = self._held(j), 1.0 - self._read(j)
                parts = [np.stack([held[:, o], absent[:, o]], axis=1) for o in range(len(self.options[j]))]
                parts = parts or [np.zeros((self.kb.n_diseases, 1))]
            components += parts
            per_evidence.append(len(parts))
        sizes = [c.shape[1] for c in components]
        outcomes = np.concatenate(components, axis=1)
        component_start = np.cumsum([0, *sizes[:-1]], dtype=np.intp)
        evidence_start = np.cumsum([0, *per_evidence[:-1]], dtype=np.intp)
        for a in (outcomes, component_start, evidence_start):
            a.flags.writeable = False
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "component_start", component_start)
        object.__setattr__(self, "evidence_start", evidence_start)

    def _held(self, j: int) -> FloatArray:
        """P(option | d) of a positive answer naming each option; 0 for diseases that do not know j."""
        return np.where(self.known[:, j, None], self.option_probs[j], 0.0)

    def _read(self, j: int) -> FloatArray:
        """P(option | d) as a negative answer reads it: UNSEEN for diseases that do not know j."""
        return np.where(self.known[:, j, None], self.option_probs[j], UNSEEN)

    def positive_factor(self, j: int, option: int | None) -> FloatArray:
        """The likelihood of a positive answer to j; `option` is None for a binary evidence, -1 for a value
        outside BED's options."""
        if self.kb.evidences[j].is_binary:
            if option is not None:
                raise ValueError(f"binary evidence {self.kb.evidences[j].code} answered with an option")
            return np.asarray(self.binary[:, j])
        if option is None:
            raise ValueError(f"{self.kb.evidences[j].code} needs an option")
        if option == -1:
            return np.where(self.known[:, j], UNSEEN, 0.0)
        return np.asarray(self._held(j)[:, option])

    def negative_factor(self, j: int) -> FloatArray:
        """The likelihood of a negative answer to j."""
        if self.kb.evidences[j].is_binary:
            return 1.0 - self.binary[:, j]
        option = int(self.default_option[j])
        if option == -1:
            return np.full(self.kb.n_diseases, 1.0 - UNSEEN)
        return np.asarray(1.0 - self._read(j)[:, option])

    def utilities(self, posterior: FloatArray) -> FloatArray:
        """BED's KL utility of every evidence at this posterior, in KB order."""
        joint = posterior[:, None] * self.outcomes
        # Summed row by row rather than by BLAS, so the rounding, and with it every tie, is the same on any machine.
        p = joint.sum(axis=0)
        new = np.divide(joint, p, out=np.zeros_like(joint), where=p > 0)
        log_old = np.log(posterior + _EPS)
        kl = (new * (np.log(new + _EPS) - log_old[:, None])).sum(axis=0)
        expected = np.where(p > 0, p * kl, 0.0)
        per_component = np.add.reduceat(expected, self.component_start)
        return np.asarray(np.maximum.reduceat(per_component, self.evidence_start))

    def option_of(self, j: int, value: str) -> int:
        return self.option_index[j].get(value, -1)

    @classmethod
    def from_stats(cls, kb: KB, stats: Mapping[str, Any], text_defaults: frozenset[str]) -> BedKnowledge:
        """From a stats file in BED's format (`ddxplus_bed_stats_4.json`): disease -> {"prior", "findings"}.

        An evidence's options are the sorted keys of the last disease in the file holding it, as
        `data_utils.py:183,217` takes them; a disease holding a key outside them is refused (BED raises KeyError).
        """
        if sorted(stats) != sorted(kb.diseases):
            raise ValueError("the stats file names other diseases than the KB")
        d, e = kb.n_diseases, kb.n_evidences
        prior = np.zeros(d)
        known = np.zeros((d, e), dtype=np.bool_)
        binary = np.zeros((d, e))
        last_holder: dict[int, Mapping[str, Any]] = {}
        for name, entry in stats.items():
            i = kb.disease_index[name]
            prior[i] = float(entry["prior"])
            for code, value in entry["findings"].items():
                j = kb.evidence_index[code]
                known[i, j] = True
                if kb.evidences[j].is_binary:
                    if isinstance(value, dict | bool):
                        raise ValueError(f"{name}: binary evidence {code} has {value!r}")
                    binary[i, j] = float(value)
                elif not isinstance(value, dict):
                    raise ValueError(f"{name}: {code} has no option shares")
                else:
                    last_holder[j] = value
        options = tuple(tuple(sorted(last_holder[j])) if j in last_holder else () for j in range(e))
        option_probs = []
        for j in range(e):
            probs = np.full((d, len(options[j])), UNSEEN)
            for name, entry in stats.items():
                shares = entry["findings"].get(kb.evidences[j].code)
                if not isinstance(shares, dict):
                    continue
                if not set(shares) <= set(options[j]):
                    raise ValueError(f"{name}: {kb.evidences[j].code} holds values outside its options")
                probs[kb.disease_index[name]] = [float(shares.get(o, UNSEEN)) for o in options[j]]
            option_probs.append(probs)
        return cls(
            kb=kb,
            prior=prior,
            known=known,
            binary=binary,
            options=options,
            option_probs=tuple(option_probs),
            default_option=_default_options(kb, options, text_defaults),
        )


def _default_options(kb: KB, options: tuple[tuple[str, ...], ...], text_defaults: frozenset[str]) -> IndexArray:
    out = np.full(kb.n_evidences, -1, dtype=np.intp)
    for j, ev in enumerate(kb.evidences):
        if not ev.is_binary and ev.code in text_defaults and ev.default in options[j]:
            out[j] = options[j].index(ev.default)
    return out


def count_knowledge(kb: KB, patients: Iterable[Patient], text_defaults: frozenset[str]) -> BedKnowledge:
    """BED's statistics counted from train records; every value of a multi-choice evidence counts."""
    d, e = kb.n_diseases, kb.n_evidences
    rows = np.zeros(d, dtype=np.int64)
    held = np.zeros((d, e), dtype=np.int64)
    values: dict[tuple[int, str], dict[int, int]] = {}
    filled = [
        [int(j) for j in np.flatnonzero(kb.membership[i]) if not kb.evidences[j].is_binary]
        for i in range(kb.n_diseases)
    ]
    for p in patients:
        i = kb.disease_index[p.pathology]
        rows[i] += 1
        record: dict[int, set[str]] = {}
        for code, value in p.evidences:
            j = kb.evidence_index[code]
            if kb.evidences[j].is_binary:
                held[i, j] += 1
            else:
                record.setdefault(j, set()).add(_value(code, value))
        for j in filled[i]:
            record.setdefault(j, {kb.evidences[j].default})
        for j, vals in record.items():
            held[i, j] += 1
            for v in vals:
                cell = values.setdefault((j, v), {})
                cell[i] = cell.get(i, 0) + 1
    if (rows == 0).any():
        raise ValueError("every KB disease needs train records")
    per_evidence: dict[int, set[str]] = {}
    for j, v in values:
        per_evidence.setdefault(j, set()).add(v)
    for j, ev in enumerate(kb.evidences):
        if ev.data_type is DataType.MULTI_CHOICE:
            per_evidence.setdefault(j, set()).add(ev.default)
    options = tuple(tuple(sorted(per_evidence.get(j, set()))) for j in range(e))
    n = rows.astype(np.float64)
    option_probs = []
    for j, opts in enumerate(options):
        probs = np.full((d, len(opts)), UNSEEN)
        for o, v in enumerate(opts):
            for i, count in values.get((j, v), {}).items():
                probs[i, o] = count / n[i]
        option_probs.append(probs)
    is_binary = np.array([ev.is_binary for ev in kb.evidences])
    return BedKnowledge(
        kb=kb,
        prior=n / n.sum(),
        known=held > 0,
        binary=np.where(is_binary, held / n[:, None], 0.0),
        options=options,
        option_probs=tuple(option_probs),
        default_option=_default_options(kb, options, text_defaults),
    )


def _value(code: str, value: str | None) -> str:
    if value is None:
        raise ValueError(f"{code} has no value")
    return value


def load_knowledge(kb: KB, root: Path = REPO_ROOT, path: str = BED_STATS) -> BedKnowledge:
    """The stats file BED's code loads at Casande-RL@5279013, checked against its sha256."""
    raw = (root / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BED_STATS_SHA256:
        raise ValueError(f"{path} is not BED's stats file")
    _, evidences = read_release_json(root)
    return BedKnowledge.from_stats(kb, json.loads(raw), text_default_codes(evidences))


def record_findings(k: BedKnowledge, patient: Patient) -> Findings:
    """BED's simulator: every evidence the record holds, and the known categorical and multi-choice evidences of
    the true disease filled in with the default; a multi-choice evidence keeps the last value listed."""
    kb = k.kb
    out: Findings = {}
    for code, value in patient.evidences:
        j = kb.evidence_index[code]
        out[j] = () if kb.evidences[j].is_binary else (k.option_of(j, _value(code, value)),)
    truth = kb.disease_index[patient.pathology]
    for j in map(int, np.flatnonzero(k.known[truth])):
        if not kb.evidences[j].is_binary and j not in out:
            out[j] = (int(k.default_option[j]),)
    return out


def _posterior(likelihood: FloatArray, prior: FloatArray) -> FloatArray:
    joint = likelihood * prior
    total = float(joint.sum())
    if not total > 0:
        raise BedDeadEnd("every disease has likelihood 0")
    return np.asarray(joint / total)


@dataclass(frozen=True, eq=False)
class BedPolicy:
    """BED's inquiry over one patient.

    The episode stops at the first point where no candidate question has a utility of at least `threshold`
    (none above 0 counts as below), or after `max_questions` questions; IL counts the initial evidence. For each
    budget b the posterior after exactly b questions is recorded with the largest candidate utility there: the
    trajectory continues past the stop with BED's choice, and once no candidate has a positive utility BED has
    nothing to ask, so later budgets hold that posterior. An episode whose posterior vanishes raises BedDeadEnd.
    """

    knowledge: BedKnowledge
    budgets: tuple[int, ...] = ()
    threshold: float = STOP_UTILITY
    max_questions: int = MAX_QUESTIONS

    def __post_init__(self) -> None:
        if self.max_questions < 0:
            raise ValueError("max_questions must be non-negative")
        if list(self.budgets) != sorted(set(self.budgets)):
            raise ValueError("budgets must be strictly increasing")
        if self.budgets and not 0 <= self.budgets[0] <= self.budgets[-1] <= self.max_questions:
            raise ValueError(f"budgets count questions after the initial evidence: 0 to {self.max_questions}")

    @property
    def start_inputs(self) -> frozenset[StartInput]:
        return frozenset({StartInput.INITIAL_EVIDENCE})

    @property
    def hierarchy(self) -> Hierarchy:
        return Hierarchy.NONE

    @property
    def answer_mode(self) -> AnswerMode:
        return AnswerMode.VALUE

    def run(self, patient: Patient) -> Episode:
        k = self.knowledge
        kb = k.kb
        first = kb.evidence_index[patient.initial]
        if not kb.evidences[first].is_binary:
            raise ValueError(f"initial evidence {patient.initial} is not binary")
        findings = record_findings(k, patient)
        if first not in findings:
            raise ValueError(f"initial evidence {patient.initial} is not in the patient record")
        likelihood = k.positive_factor(first, None)
        candidates = k.known[:, first].copy()
        asked = np.zeros(kb.n_evidences, dtype=np.bool_)
        asked[first] = True
        questions: list[str] = []
        snapshots: list[FloatArray] = []
        gains: list[float] = []
        stop: tuple[int, FloatArray] | None = None
        last_budget = self.budgets[-1] if self.budgets else 0
        while True:
            q = _posterior(likelihood, k.prior)
            n = len(questions)
            utility = k.utilities(q)
            open_ = k.known[candidates].any(axis=0) & ~asked
            best = max(float(utility[open_].max()), 0.0) if open_.any() else 0.0
            if n in self.budgets:
                snapshots.append(q)
                gains.append(best)
            if stop is None and (n == self.max_questions or best < self.threshold or best <= 0.0):
                stop = (n + 1, q)
            if stop is not None and n >= last_budget:
                break
            if best <= 0.0:
                left = [b for b in self.budgets if b > n]
                snapshots += [q] * len(left)
                gains += [0.0] * len(left)
                break
            j = int(np.flatnonzero(open_ & (utility >= best - UTILITY_TIE_ATOL))[0])
            asked[j] = True
            questions.append(kb.evidences[j].code)
            answer = findings.get(j)
            if answer is None:
                likelihood = likelihood * k.negative_factor(j)
                continue
            for option in answer or (None,):
                likelihood = likelihood * k.positive_factor(j, option)
            candidates &= k.known[:, j]
        if stop is None:
            raise RuntimeError("the episode ended without a stop")
        return Episode(
            row=patient.row,
            gold=kb.disease_index[patient.pathology],
            questions=tuple(questions),
            il=stop[0],
            posterior=stop[1],
            budgets=self.budgets,
            budget_posteriors=np.stack(snapshots) if snapshots else np.empty((0, kb.n_diseases)),
            budget_gains=np.asarray(gains, dtype=np.float64),
        )
