from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from ddxkb.data import Patient, iter_train, load_kb, load_validation, read_release_json
from ddxkb.evaluation import ScoreTask, compare, report
from ddxkb.experiment import (
    BED_PAIR,
    BED_RUNG,
    SECOND_RUN_OUTPUTS,
    Context,
    bed_answer_differences,
    bed_reported_differences,
    ladder_rung,
    main_suite,
    second_run_outputs,
    sensitivity_stops,
    tau_key,
    tau_sensitivity,
)
from ddxkb.kb import KB
from ddxkb.policies.bayes import BayesPolicy, membership_table, uniform_prior
from ddxkb.policies.bed import BedPolicy, answer_difference, count_knowledge, load_knowledge, text_default_codes
from ddxkb.protocol import CURVE, LADDER, MIN_GAIN, P2_VARIANTS, SENSITIVITY_TAUS, TAU, P1Config, p2_fixed
from helpers import binary_kb, make_repo, requires_data, toy_patient


def test_second_run_outputs_names_the_directory_of_the_second_evaluation(tmp_path: Path) -> None:
    assert SECOND_RUN_OUTPUTS == "paper1/outputs/v2"
    kept = tmp_path / SECOND_RUN_OUTPUTS
    kept.mkdir(parents=True)
    for inside in (kept, kept / "dev", kept / "eval" / "T", kept / "dev_limit3000"):
        assert second_run_outputs(tmp_path, inside) == SECOND_RUN_OUTPUTS
    for outside in ("paper1/outputs/v3/dev", "paper1/outputs/v3/eval", "paper1/outputs", "paper1/outputs/v20/dev"):
        assert second_run_outputs(tmp_path, tmp_path / outside) is None
    (tmp_path / "elsewhere").symlink_to(kept)
    assert second_run_outputs(tmp_path, tmp_path / "elsewhere" / "eval") == SECOND_RUN_OUTPUTS
    assert second_run_outputs(tmp_path, tmp_path / "paper1/outputs/v3/../v2/dev") == SECOND_RUN_OUTPUTS


def _wide_kb() -> KB:
    """31 binary evidences, the fewest a trajectory to the protocol's cap needs; three diseases with nested lists."""
    codes = [f"e{j:02d}" for j in range(31)]
    lists = {"A": codes[:12], "B": codes[:8], "C": codes[4:20]}
    return binary_kb(lists, codes)


def _wide_patients(n: int) -> list[Patient]:
    records = [
        (["e00", "e05", "e09", "e11"], "e00", "A"),
        (["e00", "e03", "e06"], "e00", "B"),
        (["e04", "e15", "e19"], "e04", "C"),
        (["e05", "e07"], "e05", "B"),
        (["e06", "e08", "e10"], "e06", "A"),
    ]
    return [
        toy_patient(record, initial=initial, pathology=pathology, row=i)
        for i, (record, initial, pathology) in enumerate(records[k % len(records)] for k in range(n))
    ]


def _bayes(kb: KB, tau: float, alpha: float = 0.5) -> BayesPolicy:
    return BayesPolicy(membership_table(kb, alpha, 1e-2), uniform_prior(kb), tau, CURVE, min_gain=MIN_GAIN)


def test_sensitivity_stops_are_the_policy_run_at_each_threshold() -> None:
    kb, patients = _wide_kb(), _wide_patients(20)
    own = ScoreTask(_bayes(kb, TAU), kb, CURVE)(patients)
    stops = sensitivity_stops("P1", own)
    assert tuple(stops) == SENSITIVITY_TAUS
    lengths = []
    for tau, derived in stops.items():
        alone = ScoreTask(_bayes(kb, tau), kb, CURVE)(patients)
        assert derived.stop_sha256() == alone.stop_sha256()
        assert derived.budgets == ()
        lengths.append(float(derived.il.mean()))
    assert lengths == sorted(lengths) and lengths[0] < lengths[-1]
    # At the protocol's threshold the derived stop is the run's own, and so is its summary with the intervals.
    assert stops[TAU].stop_sha256() == own.stop_sha256()
    summaries = tau_sensitivity(stops)
    assert list(summaries) == ["0.9", "0.95", "0.99", "0.999"] == [tau_key(tau) for tau in SENSITIVITY_TAUS]
    assert summaries["0.99"] == report(own)["stop"]
    assert summaries["0.9"]["il"]["mean"] == lengths[0] and {"low", "high"} <= set(summaries["0.9"]["il"])


def test_sensitivity_stops_refuse_a_run_stopping_at_another_threshold() -> None:
    kb, patients = _wide_kb(), _wide_patients(20)
    other = ScoreTask(_bayes(kb, 0.9), kb, CURVE)(patients)
    with pytest.raises(RuntimeError, match=r"the stop derived at tau 0\.99 is not the policy's own stop"):
        sensitivity_stops("P1", other)
    with pytest.raises(ValueError, match="length cap"):
        sensitivity_stops("P1", other.restricted((5, 10)))


def test_a_ladder_rung_compares_two_runs_at_the_stop_the_budgets_and_the_thresholds() -> None:
    kb, patients = _wide_kb(), _wide_patients(20)
    pair = next(iter(LADDER))
    full = {
        name: ScoreTask(_bayes(kb, TAU, alpha), kb, CURVE)(patients)
        for name, alpha in zip(pair, (0.7, 0.3), strict=True)
    }
    stops = {name: sensitivity_stops(name, s) for name, s in full.items()}
    runs = {name: s.restricted((5, 10)) for name, s in full.items()}
    rows = frozenset(range(0, 20, 3))
    rung = ladder_rung(pair, runs, (5, 10), {"part": rows}, stops)
    this, other = pair
    assert (rung["this"], rung["other"], rung["factor"]) == (this, other, LADDER[pair])
    assert {k: rung[k] for k in ("stop", "5", "10")} == compare(runs[this], runs[other], points=(5, 10))
    assert rung["stop"]["il_diff"]["estimate"] == float(np.mean(runs[this].il - runs[other].il)) != 0.0
    for tau in SENSITIVITY_TAUS:
        expected = compare(stops[this][tau], stops[other][tau])["stop"]
        assert rung["tau_sensitivity"][tau_key(tau)] == expected
    assert rung["tau_sensitivity"]["0.99"] == rung["stop"]
    part = rung["subsets"]["part"]
    assert set(part) == {"stop", "5", "10", "tau_sensitivity"}
    assert part["stop"] == compare(runs[this].select(rows), runs[other].select(rows))["stop"]
    assert (
        part["tau_sensitivity"]["0.9"] == compare(stops[this][0.9].select(rows), stops[other][0.9].select(rows))["stop"]
    )
    plain = ladder_rung(pair, runs, (5, 10), {})
    assert set(plain) == {"this", "other", "factor", "stop", "5", "10"}
    # A pair outside LADDER brings its own factor (보완 4 §2.3) and is compared the same way, minuend first.
    outside = (other, this)
    assert outside not in LADDER
    named = ladder_rung(outside, runs, (5, 10), {}, stops, factor="the other way round")
    assert (named["this"], named["other"], named["factor"]) == (other, this, "the other way round")
    assert {k: named[k] for k in ("stop", "5", "10")} == compare(runs[other], runs[this], points=(5, 10))
    assert named["stop"]["il_diff"]["estimate"] == -rung["stop"]["il_diff"]["estimate"]
    assert set(named) == set(rung) - {"subsets"}
    # A pair of LADDER keeps the factor LADDER gives it, and a pair outside it needs one.
    with pytest.raises(ValueError, match=f"LADDER names what differs between {this} and {other}"):
        ladder_rung(pair, runs, (5, 10), {}, factor="another")
    with pytest.raises(KeyError):
        ladder_rung(outside, runs, (5, 10), {})


def test_the_ladder_names_only_the_bed_comparison_in_place_of_a_rung(tmp_path: Path) -> None:
    make_repo(tmp_path, n_validation=12)
    ctx = Context(tmp_path, load_kb(tmp_path), load_validation(tmp_path), "validate", tmp_path / "out")
    p1 = P1Config(0.5, 1e-3, TAU)
    assert BED_PAIR == ("B-BED", "P2-value")
    summary, _ = main_suite(ctx, p1, [p2_fixed(v) for v in P2_VARIANTS], sensitivity=True)
    named = {name for name, rung in summary["ladder"].items() if not isinstance(rung, dict)}
    assert named == {"B-BED - P2-value"} and summary["ladder"]["B-BED - P2-value"] == BED_RUNG
    # A Bayes policy that was not run is an error, not a comparison left to another file.
    with pytest.raises(KeyError, match="P2-binary-agesex"):
        main_suite(ctx, p1, [p2_fixed(P2_VARIANTS[0])], sensitivity=True)
    assert "ladder" not in main_suite(ctx, p1, [p2_fixed(P2_VARIANTS[0])])[0]


def test_bed_answer_differences_count_the_questions_up_to_the_stop(tmp_path: Path) -> None:
    make_repo(tmp_path)
    kb = load_kb(tmp_path)
    k = count_knowledge(kb, iter_train(tmp_path), text_default_codes(read_release_json(tmp_path)[1]))
    patients = load_validation(tmp_path)
    failing = 5
    # A patient whose episode fails asked nothing.
    patients[failing] = dataclasses.replace(patients[failing], initial="douleur_lieu")
    # With this threshold the episodes stop early, and the trajectory goes on for the budgets.
    stopped = BedPolicy(k, threshold=0.3)
    scored = ScoreTask(dataclasses.replace(stopped, budgets=CURVE), kb, CURVE)(patients)
    assert [int(i) for i in np.flatnonzero(scored.failed)] == [failing]
    expected = [answer_difference(k, p, stopped.run(p).asked) for i, p in enumerate(patients) if i != failing]
    out = bed_answer_differences(k, patients, scored)
    n = len(patients)
    assert out == {
        "run": "stats_file",
        "patients": n,
        "valued_questions": sum(d.valued for d in expected),
        "oracle_negative": sum(d.oracle_negative for d in expected),
        "oracle_negative_bed_positive": sum(d.bed_positive for d in expected),
        "oracle_negative_bed_filled": sum(d.bed_filled for d in expected),
        "both_negative": sum(d.oracle_negative - d.bed_positive for d in expected),
        "patients_with_difference": sum(d.bed_positive > 0 for d in expected),
        "patients_with_difference_share": sum(d.bed_positive > 0 for d in expected) / n,
    }
    assert out["valued_questions"] > out["oracle_negative"] > out["oracle_negative_bed_positive"] > 0
    # Some differing answers are recorded defaults, the others filled in for a record that lacks the evidence.
    assert 0 < out["oracle_negative_bed_filled"] < out["oracle_negative_bed_positive"]
    assert 0 < out["patients_with_difference"] < n and out["both_negative"] > 0
    # The questions asked after the stop are not counted.
    whole = [answer_difference(k, p, scored.questions[i]) for i, p in enumerate(patients)]
    assert sum(d.valued for d in whole) > out["valued_questions"]
    with pytest.raises(ValueError, match="other patients"):
        bed_answer_differences(k, patients[:5], scored)


@requires_data
def test_bed_answer_differences_on_the_first_validation_rows(real_kb: KB, validation: list[Patient]) -> None:
    """Design E.4: B-BED asks the first 3,000 validation patients 6,930 categorical and multi-choice questions; the
    oracle answers 4,191 of them negative, and BED's simulator 1,310 of those positive, for 38.6% of the patients.
    All 1,310 are recorded default values: none is filled in."""
    patients = validation[:3000]
    k = load_knowledge(real_kb)
    out = bed_answer_differences(k, patients, ScoreTask(BedPolicy(k), real_kb, ())(patients))
    assert (out["valued_questions"], out["oracle_negative"], out["oracle_negative_bed_positive"]) == (6930, 4191, 1310)
    assert (out["both_negative"], out["patients_with_difference"]) == (2881, 1159)
    assert out["oracle_negative_bed_filled"] == 0
    assert round(100 * out["patients_with_difference_share"], 1) == 38.6


def test_bed_reported_differences_are_recorded_without_a_judgment() -> None:
    stop = {"gtpa1": {"estimate": 0.9}, "gtpa1_official": {"estimate": 0.91}, "il": {"mean": 12.0}}
    out = bed_reported_differences({"stop": stop})
    assert out == {
        "run": "stats_file",
        "bed": {"gtpa1": 0.9947, "il": 5.47},
        "this_minus_bed": {"gtpa1": 0.9 - 0.9947, "gtpa1_official": 0.91 - 0.9947, "il": 12.0 - 5.47},
    }
