from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from ddxkb.data import Patient
from ddxkb.evaluation import (
    COLUMN,
    FAILED_IL,
    METRICS,
    PROPORTIONS,
    SCORED,
    Scored,
    ScoreTask,
    V1Scored,
    V1Task,
    report,
    v1_report,
    write_patients,
)
from ddxkb.kb import KB
from ddxkb.metrics import (
    SEVERE_BELOW,
    PatientScores,
    gtpa_at_k,
    positive_evidence_recall,
    positive_evidence_recall_official,
    score_patient,
)
from ddxkb.policies.baselines import SetInclusion
from ddxkb.policies.bayes import BayesPolicy, membership_table, uniform_prior
from ddxkb.policies.v1_replay import V1Replay, code_graph
from ddxkb.simulator import MAX_IL, positive_codes
from ddxkb.stats import BOOTSTRAP_B, BOOTSTRAP_SEED, bootstrap_interval, mcnemar_exact
from helpers import make_repo, toy_patient

RECORDS = [
    (["douleur", "douleur_intens_@_2", "douleur_lieu_@_dos", "atcd", "trav1_@_N"], "douleur", "Beta"),
    (["fievre", "toux", "trav1_@_Asie"], "fievre", "Alpha"),
    (["fievre", "douleur", "douleur_intens_@_1", "douleur_lieu_@_tete", "trav1_@_N"], "fievre", "Gamma"),
    (["fievre", "trav1_@_N"], "fievre", "Gamma"),
]
CAP = 7
CURVE = tuple(range(CAP))


def _patients(n: int = 12) -> list[Patient]:
    out = []
    for i in range(n):
        record, initial, pathology = RECORDS[i % len(RECORDS)]
        out.append(toy_patient(record, initial=initial, pathology=pathology, row=100 + i))
    return out


def _p1(kb: KB, tau: float | None, min_gain: float | None = None) -> BayesPolicy:
    return BayesPolicy(membership_table(kb, 0.5, 1e-2), uniform_prior(kb), tau, CURVE, min_gain=min_gain)


def test_scores_are_the_metric_functions(toy_kb: KB) -> None:
    patients = _patients(4)
    policy = _p1(toy_kb, 0.95)
    s = ScoreTask(policy, toy_kb, CURVE)(patients)
    for i, p in enumerate(patients):
        ep = policy.run(p)
        truth = toy_kb.disease_vector(p.differential)
        severe = np.array(toy_kb.severity) < SEVERE_BELOW
        expected = score_patient(ep.posterior, ep.gold, truth, severe, toy_kb.official_order)
        for metric in SCORED:
            assert s.stop[i, COLUMN[metric]] == getattr(expected, metric)
        positives = positive_codes(toy_kb, p)
        assert s.stop[i, COLUMN["per"]] == positive_evidence_recall(ep.asked, positives, p.initial)
        assert s.stop[i, COLUMN["per_official"]] == positive_evidence_recall_official(ep.asked, positives, p.initial)
        assert s.il[i] == ep.il
        assert s.questions[i] == ep.questions
        for k in range(len(CURVE)):
            assert s.at_budget[i, k, COLUMN["gtpa3"]] == gtpa_at_k(ep.budget_posteriors[k], ep.gold, 3)
            assert s.peak[i, k] == ep.budget_posteriors[k].max()
            assert s.max_gain[i, k] == ep.budget_gains[k]
        assert s.top is not None and s.top_p is not None
        assert s.top[i, 0] == int(np.argmax(ep.posterior))
        assert s.top_p[i, 0] == ep.posterior.max()
    assert s.rows.tolist() == [100, 101, 102, 103]
    assert not s.failed.any()


@pytest.mark.parametrize("tau", [0.5, 0.8, 0.95, 0.999, 1.0])
@pytest.mark.parametrize("min_gain", [None, 1e-6, 0.05, 0.3])
def test_a_derived_stop_is_the_policy_stop(toy_kb: KB, tau: float, min_gain: float | None) -> None:
    patients = _patients()
    trajectory = ScoreTask(_p1(toy_kb, None), toy_kb, CURVE)(patients)
    derived = trajectory.with_stop(tau, min_gain, cap=CAP)
    direct = ScoreTask(_p1(toy_kb, tau, min_gain), toy_kb, CURVE)(patients)
    assert np.array_equal(derived.il, direct.il)
    assert np.array_equal(derived.stop, direct.stop)
    assert derived.stop_sha256() == direct.stop_sha256()
    assert derived.top is None


def test_the_gain_rule_changes_derived_stops(toy_kb: KB) -> None:
    trajectory = ScoreTask(_p1(toy_kb, None), toy_kb, CURVE)(_patients())
    tau_only = trajectory.with_stop(0.999, None, cap=CAP)
    assert (trajectory.with_stop(0.999, 0.05, cap=CAP).il < tau_only.il).any()


def test_a_stop_is_derived_only_from_a_run_to_the_length_cap(toy_kb: KB) -> None:
    trajectory = ScoreTask(_p1(toy_kb, None), toy_kb, CURVE)(_patients())
    for bad in (trajectory.restricted((0, 3)), trajectory.restricted(CURVE[:-1])):
        with pytest.raises(ValueError, match="length cap"):
            bad.with_stop(0.9, None, cap=CAP)
    # The default cap is the protocol's IL = 31, which this 7-evidence run never reaches.
    with pytest.raises(ValueError, match="length cap"):
        trajectory.with_stop(0.9, None)
    assert CAP < MAX_IL


def test_a_failed_patient_scores_zero_at_the_cap(toy_kb: KB) -> None:
    patients = _patients(3)
    patients[1] = dataclasses.replace(patients[1], initial="atcd")
    s = ScoreTask(_p1(toy_kb, 0.95), toy_kb, CURVE)(patients)
    assert s.failed.tolist() == [False, True, False]
    assert s.errors == ((101, "ValueError: initial evidence atcd is not positive in the patient record"),)
    assert (s.il[1], s.stop[1].sum(), s.at_budget[1].sum(), s.peak[1].sum()) == (FAILED_IL, 0, 0, 0)
    assert s.questions[1] == ()
    trajectory = ScoreTask(_p1(toy_kb, None), toy_kb, CURVE)(patients)
    assert trajectory.with_stop(0.5, 1e-6, cap=CAP).il[1] == CAP
    shorter = CAP - 2
    capped = trajectory.restricted(tuple(range(shorter))).with_stop(0.5, 1e-6, cap=shorter)
    assert capped.il[1] == shorter
    assert report(capped, cap=shorter, bootstrap=False)["failures"] == 1


def test_concat_and_restricted(toy_kb: KB) -> None:
    patients = _patients(10)
    task = ScoreTask(_p1(toy_kb, 0.9), toy_kb, CURVE)
    whole = task(patients)
    parts = Scored.concat([task(patients[:3]), task(patients[3:])])
    for field in ("rows", "gold", "il", "stop", "at_budget", "peak", "max_gain", "failed", "top", "top_p"):
        assert np.array_equal(getattr(parts, field), getattr(whole, field))
    assert parts.questions == whole.questions
    small = whole.restricted((2, 5))
    assert small.budgets == (2, 5)
    assert np.array_equal(small.at(5), whole.at(5))
    assert np.array_equal(small.max_gain, whole.max_gain[:, [2, 5]])
    short = BayesPolicy(membership_table(toy_kb, 0.5, 1e-2), uniform_prior(toy_kb), 0.9, (0, 1))
    with pytest.raises(ValueError, match="different budgets"):
        Scored.concat([task(patients[:2]), ScoreTask(short, toy_kb, (0, 1))(patients[:2])])
    with pytest.raises(ValueError, match="nothing"):
        Scored.concat([])


def test_select_keeps_the_patients_of_the_given_rows(toy_kb: KB) -> None:
    patients = _patients(10)
    patients[5] = dataclasses.replace(patients[5], initial="atcd")
    whole = ScoreTask(_p1(toy_kb, 0.9), toy_kb, CURVE)(patients)
    part = whole.select({102, 105, 107, 999})
    assert part.rows.tolist() == [102, 105, 107]
    index = [2, 5, 7]
    for field in ("gold", "il", "stop", "at_budget", "peak", "max_gain", "failed", "top", "top_p"):
        assert np.array_equal(getattr(part, field), getattr(whole, field)[index]), field
    assert part.questions == tuple(whole.questions[i] for i in index)
    assert [r for r, _ in part.errors] == [105]
    assert part.diseases == whole.diseases == toy_kb.diseases
    assert whole.select(set()).n == 0


def test_report_gives_macro_and_per_disease_gtpa1(toy_kb: KB) -> None:
    patients = _patients(12)
    s = ScoreTask(_p1(toy_kb, 0.9), toy_kb, CURVE)(patients)
    r = report(s, points=(2,), bootstrap=False)
    per = r["by_disease"]
    assert list(per) == ["Alpha", "Beta", "Gamma"]
    for name in per:
        mine = s.gold == toy_kb.disease_index[name]
        assert per[name]["n"] == int(mine.sum())
        assert per[name]["gtpa1"] == pytest.approx(s.column("gtpa1")[mine].mean())
        assert per[name]["gtpa1_official"] == pytest.approx(s.column("gtpa1_official")[mine].mean())
    macro = r["stop"]["macro"]
    assert macro["diseases"] == 3
    assert macro["gtpa1"] == pytest.approx(np.mean([v["gtpa1"] for v in per.values()]))
    beta_only = report(s.select({p.row for p in patients if p.pathology == "Beta"}), bootstrap=False)
    assert beta_only["stop"]["macro"]["diseases"] == 1 and list(beta_only["by_disease"]) == ["Beta"]
    assert r["budgets"]["2"]["macro"]["gtpa1"] == pytest.approx(
        np.mean([s.column("gtpa1", 2)[s.gold == d].mean() for d in range(3)])
    )
    assert r["stop"]["il"]["questions_mean"] == pytest.approx(s.il.mean() - 1)
    assert set(r["stop"]) >= {"gtpa1_official", "dshm", "per_official"}


def test_report_summarizes_and_compares(toy_kb: KB) -> None:
    patients = _patients(12)
    a = ScoreTask(_p1(toy_kb, 0.9), toy_kb, CURVE)(patients)
    b = ScoreTask(_p1(toy_kb, 0.999), toy_kb, CURVE)(patients)
    r = report(a, points=(2, 6), against={"other": b})
    assert r["n"] == 12 and r["failures"] == 0
    assert r["stop"]["gtpa1"]["estimate"] == pytest.approx(a.column("gtpa1").mean())
    assert r["stop"]["gtpa1"]["low"] <= r["stop"]["gtpa1"]["estimate"] <= r["stop"]["gtpa1"]["high"]
    assert r["stop"]["il"]["mean"] == a.il.mean()
    ci = bootstrap_interval(a.il)
    assert (r["stop"]["il"]["low"], r["stop"]["il"]["high"]) == (ci.low, ci.high)
    ddf1 = bootstrap_interval(a.column("ddf1", 6))
    assert (r["budgets"]["6"]["ddf1"]["low"], r["budgets"]["6"]["ddf1"]["high"]) == (ddf1.low, ddf1.high)
    assert "il" not in r["budgets"]["2"]
    assert r["curve"]["budgets"] == list(CURVE)
    assert r["curve"]["gtpa1"][2] == pytest.approx(a.column("gtpa1", 2).mean())
    c = r["against"]["other"]
    assert set(c) == {"stop", "2", "6"}
    test = mcnemar_exact(a.column("gtpa1_fixed"), b.column("gtpa1_fixed"))
    assert c["stop"]["mcnemar"] == {"only_this": test.only_a, "only_other": test.only_b, "p_value": test.p_value}
    # Paired: each replicate draws one set of patients for both runs.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    replicates = []
    for _ in range(BOOTSTRAP_B):
        idx = rng.integers(0, a.n, size=a.n)
        replicates.append(a.il[idx].mean() - b.il[idx].mean())
    low, high = np.quantile(replicates, [0.025, 0.975])
    assert c["stop"]["il_diff"]["estimate"] == float(np.mean(a.il - b.il))
    assert (c["stop"]["il_diff"]["low"], c["stop"]["il_diff"]["high"]) == pytest.approx((low, high), abs=1e-12)
    assert c["6"]["gtpa1_diff"] == pytest.approx(float(np.mean(a.column("gtpa1", 6) - b.column("gtpa1", 6))))
    plain = report(a, bootstrap=False)
    assert "low" not in plain["stop"]["il"] and "budgets" in plain and plain["budgets"] == {}


def test_a_reference_without_questions_has_no_il_or_per(toy_kb: KB) -> None:
    patients = _patients(8)
    p1 = ScoreTask(_p1(toy_kb, 0.9), toy_kb, CURVE)(patients)
    s = ScoreTask(SetInclusion(toy_kb), toy_kb, ())(patients)
    r = report(s, against={"P1": p1}, asks=False)
    assert "il" not in r["stop"] and "per" not in r["stop"] and "curve" not in r
    assert set(r["against"]["P1"]) == {"stop"} and "il_diff" not in r["against"]["P1"]["stop"]
    with pytest.raises(ValueError, match="other patients"):
        report(s, against={"P1": ScoreTask(_p1(toy_kb, 0.9), toy_kb, CURVE)(patients[:4])})


def test_patient_file_is_byte_stable(toy_kb: KB, tmp_path: Path) -> None:
    patients = _patients(5)
    patients[2] = dataclasses.replace(patients[2], initial="atcd")
    s = ScoreTask(_p1(toy_kb, 0.9), toy_kb, CURVE)(patients)
    write_patients(tmp_path / "a.jsonl.gz", s, toy_kb)
    write_patients(tmp_path / "b.jsonl.gz", s, toy_kb)
    raw = (tmp_path / "a.jsonl.gz").read_bytes()
    assert raw == (tmp_path / "b.jsonl.gz").read_bytes()
    records = [json.loads(line) for line in gzip.decompress(raw).decode().splitlines()]
    assert [r["row"] for r in records] == [100, 101, 102, 103, 104]
    assert records[0]["pathology"] == "Beta"
    assert records[0]["il"] == len(records[0]["asked"]) + 1
    assert s.top is not None
    assert records[0]["top"][0][0] == toy_kb.diseases[int(s.top[0, 0])]
    assert set(METRICS) <= set(records[0])
    assert len(records[0]["curve_gtpa1"]) == CAP
    assert records[2]["error"].startswith("ValueError") and records[2]["top"] == []
    with pytest.raises(ValueError, match="derived"):
        write_patients(
            tmp_path / "c.jsonl.gz",
            ScoreTask(_p1(toy_kb, None), toy_kb, CURVE)(patients).with_stop(0.9, None, CAP),
            toy_kb,
        )


def test_v1_task(toy_kb: KB) -> None:
    patients = _patients(8)
    replay = V1Replay(code_graph(toy_kb), deny=2)
    v = V1Task(replay)(patients)
    episodes = [replay.run(p) for p in patients]
    assert v.il.tolist() == [e.il for e in episodes]
    assert v.hits[:, 0].tolist() == [e.in_top(1) for e in episodes]
    whole = V1Scored.concat([V1Task(replay)(patients[:3]), V1Task(replay)(patients[3:])])
    assert np.array_equal(whole.hits, v.hits)
    r = v1_report(v)
    assert r["top1"]["estimate"] == v.hits[:, 0].mean()
    assert r["il"]["max"] == max(e.il for e in episodes)


def test_a_failed_v1_patient_counts_at_the_longest_interaction(toy_kb: KB) -> None:
    patients = _patients(3)
    # Subattributes have no node in this graph, so an episode starting from one raises.
    patients[1] = dataclasses.replace(patients[1], initial="douleur_lieu")
    replay = V1Replay(code_graph(toy_kb, drop_subattributes=True), deny=2, max_questions=40)
    v = V1Task(replay)(patients)
    assert v.failed.tolist() == [False, True, False]
    assert v.il[1] == 41 and v.il[0] == replay.run(patients[0]).il
    assert (v.hits[1].any(), v.tie_aware[1], v.gtpa[1]) == (False, 0.0, 0.0)


def test_metric_columns_are_the_patient_scores_then_both_recalls() -> None:
    assert (*(f.name for f in dataclasses.fields(PatientScores)), "per", "per_official") == METRICS
    assert {"gtpa1_official", "gtpa3_official", "gtpa5_official"} <= set(PROPORTIONS) <= set(SCORED)


def test_scoring_on_the_synthetic_repository(tmp_path: Path) -> None:
    from ddxkb.data import load_kb, load_validation
    from ddxkb.protocol import CURVE as FULL_CURVE

    make_repo(tmp_path, n_validation=20)
    kb = load_kb(tmp_path)
    patients = load_validation(tmp_path)
    policy = BayesPolicy(membership_table(kb, 0.5, 1e-3), uniform_prior(kb), 0.99, FULL_CURVE)
    s = ScoreTask(policy, kb, FULL_CURVE)(patients)
    assert s.n == 20 and not s.failed.any()
    assert (s.il >= 1).all() and (s.il <= MAX_IL).all()
