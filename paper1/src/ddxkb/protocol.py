"""The candidate configurations, fixed settings, stop rules, budgets, selection objective and paired comparisons of
design §3-§5 as 보완 2 and 보완 3 revise them, fixed before development.

보완 3 §2.1-2.3: every Bayes policy stops at the one threshold TAU, P2 smooths with the one P2_LAMBDA, and only P1's
alpha and eps are selected, among nine candidates. The other thresholds of SENSITIVITY_TAUS and the other lambdas
of P2_SENSITIVITY_LAMBDAS are reported as sensitivity, never selected.

The frozen protocol (`ddxkb.freeze`) records every value here that the evaluation runs with, and the evaluation
runner refuses to start when the code no longer holds the recorded values.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ddxkb.frequencies import FrequencyCounts
from ddxkb.kb import KB
from ddxkb.policies.bayes import BayesPolicy, frequency_prior, frequency_table, membership_table, uniform_prior
from ddxkb.simulator import MAX_IL, AnswerMode, Hierarchy

# 보완 3 §2.1: the stop threshold of P1, of every P2 variant and of everything built from P1. It is not selected:
# tau trades accuracy for questions, which the accuracy objective cannot weigh.
TAU = 0.99
# 보완 3 §2.1: the thresholds whose stops are reported for every Bayes main policy, each derived from the one
# trajectory run to the cap (`Scored.with_stop`); TAU is one of them.
SENSITIVITY_TAUS = (0.9, 0.95, 0.99, 0.999)
P1_ALPHAS = (0.3, 0.5, 0.7)
P1_EPSILONS = (1e-2, 1e-3, 1e-4)
# 보완 3 §2.3: the value one step beyond the first and the last of each P1 list, in listed order. The development
# record holds these where the selection sits at an end of a list; the selection itself stays within the nine.
# alpha keeps its step of 0.2 and eps its factor of 10.
P1_WIDER = {"alpha": (0.1, 0.9), "eps": (1e-1, 1e-5)}
# 보완 3 §2.2: the Laplace smoothing of every P2 variant (add one), not selected. The other lambdas are run on
# validation only, as sensitivity.
P2_LAMBDA = 1.0
P2_SENSITIVITY_LAMBDAS = (0.5,)

# Questions after the initial evidence: the reported budgets, and every budget for the accuracy curve (0-30).
BUDGETS = (5, 10, 15, 20, 30)
CURVE = tuple(range(MAX_IL))
# 보완 2 §3: the main protocol propagates nothing between a question and its subquestions. CASANDE's implicit
# answers are the sensitivity analysis of every main policy; A4 (§9) adds the rules that hold subquestions back.
MAIN_HIERARCHY = Hierarchy.NONE
SENSITIVITY_HIERARCHY = Hierarchy.IMPLICIT
A4_HIERARCHIES = (Hierarchy.IMPLICIT, Hierarchy.PARENT_FIRST, Hierarchy.NO_CHILDREN)

# Design §3: P1 and P2 also stop once no unasked evidence has an EIG of at least this many nats. Lists nested
# in other lists (stable angina in unstable angina) leave posteriors near 0.8 that no question can move.
MIN_GAIN = 1e-6

A2_PRIOR_LAMBDA = 1.0
A3_EPSILONS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-9)

# 보완 3 §2.5: the paired comparisons that change one factor at a time, each the first policy minus the second on
# the same patients, mapped to the factor that differs between the two.
LADDER = {
    ("P2-binary", "P1"): "the lists replaced by train frequencies (prior and likelihood)",
    ("P2-binary-agesex", "P2-binary"): "the prior by age and sex",
    ("P2-value", "P2-binary"): "the values of the answers",
    ("P2-value-agesex", "P2-value"): "the prior by age and sex",
    ("B-BED", "P2-value"): "BED's utility, stop, smoothing and simulator (design E)",
}
# 보완 3 §2.8: BED's figures on the official test split (CASANDE Table 1), computed the official way. B-BED's
# differences from them are recorded, not judged.
BED_REPORTED_GTPA1 = 0.9947
BED_REPORTED_IL = 5.47


@dataclass(frozen=True)
class P1Config:
    alpha: float
    eps: float
    tau: float

    @property
    def name(self) -> str:
        return "P1"

    def policy(self, kb: KB, budgets: tuple[int, ...] = CURVE) -> BayesPolicy:
        table = membership_table(kb, self.alpha, self.eps)
        return BayesPolicy(table, uniform_prior(kb), self.tau, budgets, min_gain=MIN_GAIN, hierarchy=MAIN_HIERARCHY)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> P1Config:
        return cls(alpha=float(d["alpha"]), eps=float(d["eps"]), tau=float(d["tau"]))


@dataclass(frozen=True)
class P2Variant:
    mode: AnswerMode
    age_sex: bool

    @property
    def name(self) -> str:
        return f"P2-{self.mode}" + ("-agesex" if self.age_sex else "")


# Design §3 lists the variants as binary/value answers x age and sex unused/used, in this order.
P2_VARIANTS = tuple(P2Variant(m, a) for m in (AnswerMode.BINARY, AnswerMode.VALUE) for a in (False, True))


@dataclass(frozen=True)
class P2Config:
    variant: P2Variant
    lam: float
    tau: float

    @property
    def name(self) -> str:
        return self.variant.name

    def policy(self, kb: KB, counts: FrequencyCounts, budgets: tuple[int, ...] = CURVE) -> BayesPolicy:
        return BayesPolicy(
            frequency_table(kb, counts, self.lam, self.variant.mode),
            frequency_prior(counts, self.lam, self.variant.age_sex),
            self.tau,
            budgets,
            min_gain=MIN_GAIN,
            hierarchy=MAIN_HIERARCHY,
        )

    def as_dict(self) -> dict[str, Any]:
        return {"mode": str(self.variant.mode), "age_sex": self.variant.age_sex, "lam": self.lam, "tau": self.tau}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> P2Config:
        variant = P2Variant(AnswerMode(d["mode"]), bool(d["age_sex"]))
        return cls(variant=variant, lam=float(d["lam"]), tau=float(d["tau"]))


def p1_grid() -> tuple[P1Config, ...]:
    """The nine P1 candidates in listed order (보완 3 §2.3), eps varying fastest, each stopping at TAU."""
    return tuple(P1Config(a, e, TAU) for a, e in itertools.product(P1_ALPHAS, P1_EPSILONS))


def p1_boundary(cfg: P1Config) -> dict[str, str]:
    """The selected parameters of `cfg` (alpha, eps) at an end of their P1 list, mapped to "first" or "last"."""
    out = {}
    for name, values in (("alpha", P1_ALPHAS), ("eps", P1_EPSILONS)):
        value = getattr(cfg, name)
        if value not in values:
            raise ValueError(f"{name} = {value} is not a P1 candidate")
        if value == values[0]:
            out[name] = "first"
        elif value == values[-1]:
            out[name] = "last"
    return out


def p1_wider_grid(cfg: P1Config) -> tuple[P1Config, ...]:
    """The P1 grid with each list one value longer at the end where `cfg` sits (`P1_WIDER`), in listed order with
    eps varying fastest; the nine candidates are part of it. Without a parameter at an end it is the P1 grid."""
    ends = p1_boundary(cfg)
    lists = []
    for name, values in (("alpha", P1_ALPHAS), ("eps", P1_EPSILONS)):
        before, after = P1_WIDER[name]
        end = ends.get(name)
        lists.append((before, *values) if end == "first" else (*values, after) if end == "last" else values)
    return tuple(P1Config(a, e, TAU) for a, e in itertools.product(*lists))


def p2_fixed(variant: P2Variant) -> P2Config:
    """The one configuration of a P2 variant (보완 3 §2.1-2.2): P2_LAMBDA and TAU."""
    return P2Config(variant, P2_LAMBDA, TAU)


def check_fixed(p1: P1Config, p2: Sequence[P2Config]) -> None:
    """ValueError unless every configuration stops at TAU, every P2 configuration smooths with P2_LAMBDA and P1 is
    one of the nine candidates: what 보완 3 §2.1-2.3 fixes for every run that is reported."""
    configs: list[P1Config | P2Config] = [p1, *p2]
    for cfg in configs:
        if cfg.tau != TAU:
            raise ValueError(f"{cfg.name} stops at tau {cfg.tau}, the protocol at {TAU}")
    if p1 not in p1_grid():
        raise ValueError(f"P1 with alpha {p1.alpha} and eps {p1.eps} is not one of the nine candidates")
    for p2cfg in p2:
        if p2cfg.lam != P2_LAMBDA:
            raise ValueError(f"{p2cfg.name} smooths with lambda {p2cfg.lam}, the protocol with {P2_LAMBDA}")


def selection_key(gtpa1: float, il_mean: float, order: int) -> tuple[float, float, int]:
    """Design §5, applied to the P1 candidates at TAU (보완 3 §2.3): the larger tie-aware GTPA@1 as a percentage
    at two decimals, then the smaller mean IL, then the earlier candidate in listed order. Smaller keys win."""
    return (-round(100 * gtpa1, 2), il_mean, order)
