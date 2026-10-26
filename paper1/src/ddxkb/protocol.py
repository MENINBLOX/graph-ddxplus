"""The candidate configurations, stop rules, budgets, selection objective and judgment thresholds of design
§3-§6 as 보완 2 revises them, fixed before development.

The frozen protocol (`ddxkb.freeze`) records every value here, and the evaluation runner refuses to start
when the code no longer holds the recorded values.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass
from typing import Any

from ddxkb.data import Heldout
from ddxkb.frequencies import AGE_BIN_EDGES, FrequencyCounts
from ddxkb.kb import KB
from ddxkb.policies.bayes import BayesPolicy, frequency_prior, frequency_table, membership_table, uniform_prior
from ddxkb.simulator import MAX_IL, AnswerMode, Hierarchy

P1_ALPHAS = (0.3, 0.5, 0.7)
P1_EPSILONS = (1e-2, 1e-3, 1e-4)
P1_TAUS = (0.9, 0.95, 0.99)
# 보완 2 §4: the value one step beyond the first and the last of each P1 list, in listed order. The development
# record holds these where the selection sits at an end of a list; the selection itself stays within the 27.
# alpha keeps its step of 0.2 and eps its factor of 10; tau continues as the P2 grid does above 0.99 and doubles
# 1 - tau below 0.9, as from 0.95 to 0.9.
P1_WIDER = {"alpha": (0.1, 0.9), "eps": (1e-1, 1e-5), "tau": (0.8, 0.999)}
P2_LAMBDAS = (0.5, 1.0)
P2_TAUS = (0.9, 0.95, 0.99, 0.999)

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

# 보완 2 §10 reports each change from the first protocol (paper1/configs/protocol_frozen.yaml) on T in the same run:
# each main policy at its configuration with settings of the first protocol back (`experiment.first_protocol_changes`).
# Those settings: |T| <= 30 with the initial evidence, P2 frequencies counted without the train rows of H2, and
# the age groups chosen from train - H2.
FIRST_MAX_IL = 30
FIRST_FREQUENCIES_WITHOUT: Heldout = "H2"
FIRST_AGE_BIN_EDGES = (2, 5, 8, 10, 18, 30, 33, 45, 60, 75)

A2_PRIOR_LAMBDA = 1.0
A3_EPSILONS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-9)
V1_TIE_SEEDS = tuple(range(10))

# 보완 2 §8, on T at the stop: J1 for both GTPA@1 definitions, J2 on the tie-aware one and the IL difference.
J1_MIN_WILSON_LOW = 0.95
J1_METRICS = ("gtpa1", "gtpa1_official")
J2_POLICY = "P2-binary"
J2_MAX_GTPA1_DIFF = 0.01
J2_IL_DIFF_HIGH_BELOW = 0.0
# J3': B-BED against BED's figures on the official test (CASANDE Table 1), for both GTPA@1 definitions, and the
# same-row comparison of the P2 variant closest to BED with it (reported, not judged).
J3_POLICY = "B-BED"
J3_BED_GTPA1 = 0.9947
J3_BED_IL = 5.47
J3_GTPA1_TOLERANCE = 0.003
J3_IL_TOLERANCE = 0.5
J3_METRICS = ("gtpa1", "gtpa1_official")
J3_PAIRED = "P2-value"


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

    def policy(
        self,
        kb: KB,
        counts: FrequencyCounts,
        budgets: tuple[int, ...] = CURVE,
        age_edges: tuple[int, ...] = AGE_BIN_EDGES,
    ) -> BayesPolicy:
        return BayesPolicy(
            frequency_table(kb, counts, self.lam, self.variant.mode),
            frequency_prior(counts, self.lam, self.variant.age_sex, age_edges),
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
    """The 27 P1 candidates in listed order (design §5), tau varying fastest."""
    return tuple(P1Config(a, e, t) for a, e, t in itertools.product(P1_ALPHAS, P1_EPSILONS, P1_TAUS))


def p1_boundary(cfg: P1Config) -> dict[str, str]:
    """The parameters of `cfg` at an end of their P1 list, mapped to "first" or "last"."""
    out = {}
    for name, values in (("alpha", P1_ALPHAS), ("eps", P1_EPSILONS), ("tau", P1_TAUS)):
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
    tau varying fastest; the 27 candidates are part of it. Without a parameter at an end it is the P1 grid."""
    ends = p1_boundary(cfg)
    lists = []
    for name, values in (("alpha", P1_ALPHAS), ("eps", P1_EPSILONS), ("tau", P1_TAUS)):
        before, after = P1_WIDER[name]
        end = ends.get(name)
        lists.append((before, *values) if end == "first" else (*values, after) if end == "last" else values)
    return tuple(P1Config(a, e, t) for a, e, t in itertools.product(*lists))


def p2_grid(variant: P2Variant) -> tuple[P2Config, ...]:
    """The 8 candidates of one P2 variant in listed order, tau varying fastest."""
    return tuple(P2Config(variant, lam, tau) for lam, tau in itertools.product(P2_LAMBDAS, P2_TAUS))


def selection_key(gtpa1: float, il_mean: float, order: int) -> tuple[float, float, int]:
    """Design §5: the larger tie-aware GTPA@1 as a percentage at two decimals, then the smaller mean IL,
    then the earlier candidate in listed order. Smaller keys win."""
    return (-round(100 * gtpa1, 2), il_mean, order)
