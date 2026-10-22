"""The candidate configurations, stop rules, budgets, selection objective and judgment thresholds of design
§3-§6, fixed before development.

The frozen protocol (`ddxkb.freeze`) records every value here, and the evaluation runner refuses to start
when the code no longer holds the recorded values.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass
from typing import Any

from ddxkb.frequencies import FrequencyCounts
from ddxkb.kb import KB
from ddxkb.policies.bayes import BayesPolicy, frequency_prior, frequency_table, membership_table, uniform_prior
from ddxkb.simulator import MAX_IL, AnswerMode

P1_ALPHAS = (0.3, 0.5, 0.7)
P1_EPSILONS = (1e-2, 1e-3, 1e-4)
P1_TAUS = (0.9, 0.95, 0.99)
P2_LAMBDAS = (0.5, 1.0)
P2_TAUS = (0.9, 0.95, 0.99, 0.999)

# Questions after the initial evidence: the reported budgets, and every budget for the accuracy curve.
BUDGETS = (5, 10, 15, 20, 29)
CURVE = tuple(range(MAX_IL))

# Design §3: P1 and P2 also stop once no unasked evidence has an EIG of at least this many nats. Lists nested
# in other lists (stable angina in unstable angina) leave posteriors near 0.8 that no question can move.
MIN_GAIN = 1e-6

A2_PRIOR_LAMBDA = 1.0
A3_EPSILONS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-9)
V1_TIE_SEEDS = tuple(range(10))

# Design §6: J1 on H1 and H2, J2 on both at the stop, J3 on H1 against BED's reported test figures.
J1_MIN_WILSON_LOW = 0.95
J2_POLICY = "P2-binary"
J2_MAX_GTPA1_DIFF = 0.01
J3_POLICY = "P2-value-agesex"
J3_BED_GTPA1 = 0.9947
J3_BED_IL = 5.47
J3_GTPA1_TOLERANCE = 0.01
J3_IL_TOLERANCE = 1.5


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
        return BayesPolicy(table, uniform_prior(kb), self.tau, budgets, min_gain=MIN_GAIN)

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
        )

    def as_dict(self) -> dict[str, Any]:
        return {"mode": str(self.variant.mode), "age_sex": self.variant.age_sex, "lam": self.lam, "tau": self.tau}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> P2Config:
        variant = P2Variant(AnswerMode(d["mode"]), bool(d["age_sex"]))
        return cls(variant=variant, lam=float(d["lam"]), tau=float(d["tau"]))


def p1_grid() -> tuple[P1Config, ...]:
    """The 27 P1 candidates in listed order (design §5), tau varying fastest."""
    return tuple(P1Config(a, e, t) for a, e, t in itertools.product(P1_ALPHAS, P1_EPSILONS, P1_TAUS))


def p2_grid(variant: P2Variant) -> tuple[P2Config, ...]:
    """The 8 candidates of one P2 variant in listed order, tau varying fastest."""
    return tuple(P2Config(variant, lam, tau) for lam, tau in itertools.product(P2_LAMBDAS, P2_TAUS))


def selection_key(gtpa1: float, il_mean: float, order: int) -> tuple[float, float, int]:
    """Design §5: the larger tie-aware GTPA@1 as a percentage at two decimals, then the smaller mean IL,
    then the earlier candidate in listed order. Smaller keys win."""
    return (-round(100 * gtpa1, 2), il_mean, order)
