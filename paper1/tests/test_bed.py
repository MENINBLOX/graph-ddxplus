from __future__ import annotations

import gzip
import itertools
import json
import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ddxkb.data import REPO_ROOT, Patient, iter_train, read_release_json
from ddxkb.evaluation import Scored, ScoreTask
from ddxkb.kb import KB, DataType, FloatArray
from ddxkb.metrics import gtpa_at_k_official
from ddxkb.policies.bed import (
    BED_STATS,
    UNSEEN,
    UTILITY_TIE_ATOL,
    BedDeadEnd,
    BedKnowledge,
    BedPolicy,
    count_knowledge,
    load_knowledge,
    record_findings,
    text_default_codes,
)
from ddxkb.runner import chunks, map_items, worker_pool
from helpers import TOY_EVIDENCES, bed_stats_json, requires_data, toy_patient

TOY_TEXT_DEFAULTS = text_default_codes(TOY_EVIDENCES)

# Alpha 4 records, Beta 2, Gamma 2; Alpha's last record lacks trav1, which its list holds.
TOY_TRAIN = [
    (["fievre", "toux", "trav1_@_N"], "fievre", "Alpha"),
    (["fievre", "trav1_@_AfriqN"], "fievre", "Alpha"),
    (["fievre", "toux", "trav1_@_N"], "fievre", "Alpha"),
    (["toux"], "toux", "Alpha"),
    (["douleur", "douleur_intens_@_2", "douleur_lieu_@_tete", "douleur_lieu_@_dos", "trav1_@_N"], "douleur", "Beta"),
    (["douleur", "douleur_intens_@_3", "douleur_lieu_@_dos", "atcd", "trav1_@_N"], "douleur", "Beta"),
    (["fievre", "douleur", "douleur_intens_@_2", "douleur_lieu_@_ventre", "trav1_@_N"], "fievre", "Gamma"),
    (["fievre", "douleur_intens_@_0", "douleur_lieu_@_nulle_part", "trav1_@_Asie"], "fievre", "Gamma"),
]


def _train() -> list[Patient]:
    return [toy_patient(e, initial=i, pathology=d, row=r) for r, (e, i, d) in enumerate(TOY_TRAIN)]


@pytest.fixture
def toy(toy_kb: KB) -> BedKnowledge:
    return count_knowledge(toy_kb, _train(), TOY_TEXT_DEFAULTS)


def test_text_defaults_are_the_string_defaults_of_the_json() -> None:
    assert {"douleur_lieu", "trav1"} == TOY_TEXT_DEFAULTS


def test_counted_knowledge_follows_bed_statistics(toy_kb: KB, toy: BedKnowledge) -> None:
    ev = toy_kb.evidence_index
    assert toy.prior.tolist() == [0.5, 0.25, 0.25]
    # Binary evidences are known only once a record holds them; listed categorical and multi-choice ones always.
    known = {
        d: sorted(e.code for e, k in zip(toy_kb.evidences, toy.known[i], strict=True) if k)
        for i, d in enumerate(toy_kb.diseases)
    }
    assert known == {
        "Alpha": ["fievre", "toux", "trav1"],
        "Beta": ["atcd", "douleur", "douleur_intens", "douleur_lieu", "trav1"],
        "Gamma": ["douleur", "douleur_intens", "douleur_lieu", "fievre", "trav1"],
    }
    assert toy.binary[:, ev["fievre"]].tolist() == [0.75, 0.0, 1.0]
    assert toy.binary[:, ev["douleur"]].tolist() == [0.0, 1.0, 0.5]
    assert toy.binary[:, ev["atcd"]].tolist() == [0.0, 0.5, 0.0]
    # Options: every value some record holds (Alpha's missing trav1 counted as its default), in string order.
    assert toy.options[ev["douleur_intens"]] == ("0", "2", "3")
    assert toy.options[ev["douleur_lieu"]] == ("dos", "nulle_part", "tete", "ventre")
    assert toy.options[ev["trav1"]] == ("AfriqN", "Asie", "N")
    np.testing.assert_array_equal(
        toy.option_probs[ev["trav1"]], [[0.25, UNSEEN, 0.75], [UNSEEN, UNSEEN, 1.0], [UNSEEN, 0.5, 0.5]]
    )
    np.testing.assert_array_equal(toy.option_probs[ev["douleur_intens"]][1:], [[UNSEEN, 0.5, 0.5], [0.5, 0.5, UNSEEN]])
    # Every value of a multi-choice record counts here, so Beta's shares add up to more than 1.
    np.testing.assert_array_equal(toy.option_probs[ev["douleur_lieu"]][1], [1.0, UNSEEN, 0.5, UNSEEN])
    # The integer default of douleur_intens never matches BED's string options.
    assert toy.default_option[ev["douleur_intens"]] == -1
    assert toy.default_option[ev["douleur_lieu"]] == 1
    assert toy.default_option[ev["trav1"]] == 2
    assert toy.default_option[ev["fievre"]] == -1


def test_stats_file_round_trip(toy_kb: KB, toy: BedKnowledge) -> None:
    back = BedKnowledge.from_stats(toy_kb, json.loads(json.dumps(bed_stats_json(toy_kb, toy))), TOY_TEXT_DEFAULTS)
    for name in ("prior", "known", "binary", "default_option"):
        np.testing.assert_array_equal(getattr(back, name), getattr(toy, name))
    assert back.options == toy.options
    for j in range(toy_kb.n_evidences):
        np.testing.assert_array_equal(back.option_probs[j][toy.known[:, j]], toy.option_probs[j][toy.known[:, j]])


def test_stats_file_options_come_from_the_last_disease_holding_the_evidence(toy_kb: KB, toy: BedKnowledge) -> None:
    j = toy_kb.evidence_index["trav1"]
    stats = bed_stats_json(toy_kb, toy)
    for d in ("Alpha", "Beta"):
        del stats[d]["findings"]["trav1"]["Asie"]
    back = BedKnowledge.from_stats(toy_kb, stats, TOY_TEXT_DEFAULTS)
    assert back.options[j] == ("AfriqN", "Asie", "N")
    assert back.option_probs[j][toy_kb.disease_index["Alpha"]].tolist() == [0.25, UNSEEN, 0.75]
    del stats["Gamma"]["findings"]["trav1"]["Asie"]
    assert BedKnowledge.from_stats(toy_kb, stats, TOY_TEXT_DEFAULTS).options[j] == ("AfriqN", "N")
    # Alpha now holds a value outside Gamma's options, which BED's re-keying cannot place.
    stats["Alpha"]["findings"]["trav1"]["Asie"] = 0.1
    with pytest.raises(ValueError, match="outside its options"):
        BedKnowledge.from_stats(toy_kb, stats, TOY_TEXT_DEFAULTS)


def test_record_answers_hold_default_values_and_the_last_multi_choice_value(toy_kb: KB, toy: BedKnowledge) -> None:
    ev = toy_kb.evidence_index
    beta = toy_patient(["douleur", "douleur_lieu_@_tete", "douleur_lieu_@_dos", "trav1_@_N"], "douleur", "Beta")
    found = record_findings(toy, beta)
    assert found[ev["douleur"]] == ()
    assert found[ev["douleur_lieu"]] == (0,)  # dos, listed last; tete is lost
    assert found[ev["trav1"]] == (2,)  # N, the default value, is a positive answer
    assert found[ev["douleur_intens"]] == (-1,)  # known to Beta, missing from the record: the default, unmatched
    assert ev["atcd"] not in found and ev["fievre"] not in found
    alpha = toy_patient(["fievre"], "fievre", "Alpha")
    assert record_findings(toy, alpha) == {ev["fievre"]: (), ev["trav1"]: (2,)}


def test_answer_likelihoods(toy_kb: KB, toy: BedKnowledge) -> None:
    ev = toy_kb.evidence_index
    assert toy.positive_factor(ev["fievre"], None).tolist() == [0.75, 0.0, 1.0]
    assert toy.negative_factor(ev["fievre"]).tolist() == [0.25, 1.0, 0.0]
    j = ev["douleur_intens"]
    assert toy.positive_factor(j, 1).tolist() == [0.0, 0.5, 0.5]
    assert toy.positive_factor(j, -1).tolist() == [0.0, UNSEEN, UNSEEN]
    assert toy.negative_factor(j).tolist() == [1 - UNSEEN] * 3
    j = ev["douleur_lieu"]
    assert toy.negative_factor(j).tolist() == [1 - UNSEEN, 1 - UNSEEN, 0.5]
    assert toy.negative_factor(ev["trav1"]).tolist() == [0.25, 0.0, 0.5]
    with pytest.raises(ValueError, match="needs an option"):
        toy.positive_factor(j, None)


# A transcription of BED's inference and utility (`QMR.compute_disease_probs`, `BED.get_*_finding_utility`) over
# its own data layout, the oracle the vectorized port is checked against.
Finding = int | tuple[int, int]


def _bed_tables(k: BedKnowledge) -> tuple[list[dict[int, Any]], FloatArray]:
    d2f: list[dict[int, Any]] = []
    for i in range(k.kb.n_diseases):
        f4d: dict[int, Any] = {}
        for j in map(int, np.flatnonzero(k.known[i])):
            if k.kb.evidences[j].is_binary:
                f4d[j] = float(k.binary[i, j])
            else:
                f4d[j] = {o: float(p) for o, p in enumerate(k.option_probs[j][i])}
        d2f.append(f4d)
    return d2f, np.asarray(k.prior)


def _bed_probs(
    d2f: list[dict[int, Any]], p_d: FloatArray, pos: list[Finding], neg: list[Finding]
) -> tuple[FloatArray | int, float]:
    p_f_d = np.empty(len(d2f))
    for d, f4d in enumerate(d2f):
        prob = 1.0
        for finding in neg:
            evi, option = finding if isinstance(finding, tuple) else (finding, None)
            fp = f4d.get(evi, 0) if option is None else f4d.get(evi, {}).get(option, 1e-7)
            prob *= 1 - fp
        for finding in pos:
            evi, option = finding if isinstance(finding, tuple) else (finding, None)
            if evi not in f4d:
                prob = 0.0
                break
            fp = f4d[evi] if option is None else f4d[evi].get(option, 1e-7)
            if fp == 0.0:
                prob = 0.0
                break
            prob *= fp
        p_f_d[d] = prob
    probs = p_f_d * p_d
    joint = float(sum(probs))
    if joint == 0.0:
        return 0, 0.0
    return probs / joint, joint


def _kl(new: FloatArray | int, old: FloatArray) -> float:
    eps = np.finfo(np.float64).eps
    return float((np.log(new + eps) * new - np.log(old + eps) * new).sum())


def _bed_utility(
    k: BedKnowledge, d2f: list[dict[int, Any]], p_d: FloatArray, j: int, pos: list[Finding], neg: list[Finding]
) -> float:
    old, old_joint = _bed_probs(d2f, p_d, pos, neg)
    assert not isinstance(old, int)
    kind = k.kb.evidences[j].data_type

    def term(p: list[Finding], n: list[Finding]) -> float:
        new, joint = _bed_probs(d2f, p_d, p, n)
        return joint / old_joint * _kl(new, old)

    if kind is DataType.BINARY:
        return term([*pos, j], neg) + term(pos, [*neg, j])
    options = range(len(k.options[j]))
    if kind is DataType.CATEGORICAL:
        return sum(term([*pos, (j, o)], neg) for o in options)
    return max(term([*pos, (j, o)], neg) + term(pos, [*neg, (j, o)]) for o in options)


def _bed_episode(
    k: BedKnowledge, patient: Patient, threshold: float = 0.01, max_questions: int = 30
) -> tuple[list[str], FloatArray]:
    """BEDBatch.batch_run with eval_reset and eval_step, scanning candidates in index order."""
    kb = k.kb
    d2f, p_d = _bed_tables(k)
    findings: dict[int, Any] = {}
    for code, value in patient.evidences:
        j = kb.evidence_index[code]
        if value is None:
            findings[j] = True
        else:
            option = k.option_index[j].get(value, -1)
            findings[j] = {option} if kb.evidences[j].data_type is DataType.MULTI_CHOICE else option
    truth = kb.disease_index[patient.pathology]
    for j in map(int, np.flatnonzero(k.known[truth])):
        e = kb.evidences[j]
        if not e.is_binary and j not in findings:
            option = int(k.default_option[j])
            findings[j] = {option} if e.data_type is DataType.MULTI_CHOICE else option
    first = kb.evidence_index[patient.initial]
    pos: list[Finding] = [first]
    neg: list[Finding] = []
    candidates = [d for d in range(kb.n_diseases) if first in d2f[d]]
    asked: list[str] = []
    for _ in range(max_questions):
        open_ = set(itertools.chain.from_iterable(d2f[d] for d in candidates))
        open_ -= {f[0] if isinstance(f, tuple) else f for f in pos + neg}
        best, action = 0.0, None
        for f in sorted(open_):
            u = _bed_utility(k, d2f, p_d, f, pos, neg)
            if u > best:
                best, action = u, f
        if action is None or best < threshold:
            break
        asked.append(kb.evidences[action].code)
        answer = findings.get(action)
        if answer is None:
            neg.append(action if kb.evidences[action].is_binary else (action, int(k.default_option[action])))
            continue
        if answer is True:
            pos.append(action)
        elif isinstance(answer, set):
            pos += [(action, o) for o in answer]
        else:
            pos.append((action, answer))
        candidates = [d for d in candidates if action in d2f[d]]
    final, _ = _bed_probs(d2f, p_d, pos, neg)
    assert not isinstance(final, int)
    return asked, final


def _random_patients(kb: KB, n: int, seed: int) -> list[Patient]:
    """Records drawn from the toy lists: binary list entries with probability 0.6, a value for every listed
    categorical or multi-choice evidence (often the default), and a recorded binary initial evidence."""
    rng = random.Random(seed)
    out: list[Patient] = []
    while len(out) < n:
        d = rng.choice(kb.diseases)
        items: list[str] = []
        for code in sorted(kb.support(d), key=kb.evidence_index.__getitem__):
            e = kb.evidence(code)
            if e.is_binary:
                if rng.random() < 0.6:
                    items.append(code)
            elif e.data_type is DataType.MULTI_CHOICE:
                items += [f"{code}_@_{v}" for v in rng.sample(e.values, rng.randint(1, 2))]
            else:
                items.append(f"{code}_@_{rng.choice(e.values)}")
        binary = [c for c in items if "_@_" not in c]
        if binary:
            out.append(toy_patient(items, initial=rng.choice(binary), pathology=d, row=len(out)))
    return out


def test_utilities_match_the_transcription_of_bed(toy_kb: KB) -> None:
    k = count_knowledge(toy_kb, _random_patients(toy_kb, 200, seed=0), TOY_TEXT_DEFAULTS)
    d2f, p_d = _bed_tables(k)
    rng = random.Random(1)
    for _ in range(30):
        asked = rng.sample(range(toy_kb.n_evidences), rng.randint(0, 3))
        pos: list[Finding] = []
        neg: list[Finding] = []
        likelihood = np.ones(toy_kb.n_diseases)
        for j in asked:
            if toy_kb.evidences[j].is_binary and rng.random() < 0.5:
                pos.append(j)
                likelihood = likelihood * k.positive_factor(j, None)
            elif not toy_kb.evidences[j].is_binary and rng.random() < 0.5:
                o = rng.randrange(-1, len(k.options[j]))
                pos.append((j, o))
                likelihood = likelihood * k.positive_factor(j, o)
            else:
                neg.append(j if toy_kb.evidences[j].is_binary else (j, int(k.default_option[j])))
                likelihood = likelihood * k.negative_factor(j)
        old, _ = _bed_probs(d2f, p_d, pos, neg)
        if isinstance(old, int):
            continue
        q = likelihood * k.prior / (likelihood * k.prior).sum()
        np.testing.assert_allclose(q, old, rtol=1e-12, atol=1e-15)
        unasked = [j for j in range(toy_kb.n_evidences) if j not in asked]
        expected = [_bed_utility(k, d2f, p_d, j, pos, neg) for j in unasked]
        np.testing.assert_allclose(k.utilities(q)[unasked], expected, rtol=1e-9, atol=1e-13)


def test_episodes_match_the_transcription_of_bed(toy_kb: KB) -> None:
    k = count_knowledge(toy_kb, _random_patients(toy_kb, 200, seed=0), TOY_TEXT_DEFAULTS)
    policy = BedPolicy(k)
    for p in _random_patients(toy_kb, 60, seed=2):
        asked, final = _bed_episode(k, p)
        ep = policy.run(p)
        assert list(ep.asked) == asked, p.evidences
        assert ep.il == len(asked) + 1
        np.testing.assert_allclose(ep.posterior, final, rtol=1e-9, atol=1e-15)


def test_a_positive_default_value_restricts_the_candidates(toy_kb: KB, toy: BedKnowledge) -> None:
    # Gamma's record holds douleur_intens at its default 0: the answer is positive and rules Alpha out.
    gamma = toy_patient(["fievre", "douleur_intens_@_0", "douleur_lieu_@_nulle_part", "trav1_@_N"], "fievre", "Gamma")
    ep = BedPolicy(toy).run(gamma)
    assert ep.asked[0] in ("douleur_intens", "douleur_lieu")
    assert ep.posterior[toy_kb.disease_index["Alpha"]] == 0.0
    assert ep.posterior[toy_kb.disease_index["Gamma"]] == 1.0


def test_stop_threshold_and_question_cap(toy_kb: KB, toy: BedKnowledge) -> None:
    alpha = toy_patient(["fievre", "trav1_@_N"], "fievre", "Alpha")
    assert BedPolicy(toy, threshold=10.0).run(alpha).il == 1
    assert BedPolicy(toy, max_questions=1).run(alpha).il == 2
    ep = BedPolicy(toy).run(alpha)
    assert 1 < ep.il <= 31 and ep.questions == ep.asked


def test_budgets_continue_past_the_stop_and_hold_once_nothing_is_informative(toy_kb: KB, toy: BedKnowledge) -> None:
    alpha = toy_patient(["fievre", "trav1_@_N"], "fievre", "Alpha")
    stopped = BedPolicy(toy).run(alpha)
    ep = BedPolicy(toy, budgets=(0, 1, 30)).run(alpha)
    assert ep.il == stopped.il and ep.questions[: ep.il - 1] == stopped.asked
    np.testing.assert_array_equal(ep.posterior, stopped.posterior)
    assert ep.budget_posteriors.shape == (3, 3)
    np.testing.assert_array_equal(ep.budget_posteriors[0], BedPolicy(toy, threshold=10.0).run(alpha).posterior)
    # Seven evidences run out long before 30 questions; the curve then holds the last posterior.
    assert len(ep.questions) < toy_kb.n_evidences and ep.budget_gains[-1] == 0.0
    with pytest.raises(ValueError, match="budgets"):
        BedPolicy(toy, budgets=(31,))


def test_a_vanished_posterior_raises(toy_kb: KB) -> None:
    # No train record holds toux, so no disease knows it and an episode starting from it has no posterior.
    train = [p for p in _train() if not any(code == "toux" for code, _ in p.evidences)]
    k = count_knowledge(toy_kb, train, TOY_TEXT_DEFAULTS)
    with pytest.raises(BedDeadEnd):
        BedPolicy(k).run(toy_patient(["toux"], "toux", "Alpha"))


def test_scored_like_any_policy(toy_kb: KB, toy: BedKnowledge) -> None:
    patients = _random_patients(toy_kb, 20, seed=3)
    s = ScoreTask(BedPolicy(toy), toy_kb, ())(patients)
    assert s.n == 20 and not s.failed.any()
    assert all(1 <= il <= 31 for il in s.il)


# Real data.

ORIGINAL = Path(__file__).parent / "fixtures" / "bed_original_validate_head.json.gz"


@pytest.fixture(scope="session")
def shipped(real_kb: KB) -> BedKnowledge:
    return load_knowledge(real_kb)


@requires_data
def test_bed_stats_file_against_the_kb(real_kb: KB, shipped: BedKnowledge) -> None:
    assert (REPO_ROOT / BED_STATS).exists()
    unknown = {
        (real_kb.diseases[i], real_kb.evidences[j].code)
        for i, j in zip(*np.nonzero(real_kb.membership & ~shipped.known), strict=True)
    }
    assert unknown == {
        ("Chagas", "oedeme"),
        ("Myocardite", "irritable"),
        ("Oedème localisé ou généralisé sans atteinte pulmonaire associée", "dyspn"),
        ("Oedème localisé ou généralisé sans atteinte pulmonaire associée", "ww_dd"),
        ("Oedème localisé ou généralisé sans atteinte pulmonaire associée", "dyspn_noct"),
        ("Sarcoïdose", "lesions_peau"),
    }
    assert not (shipped.known & ~real_kb.membership).any()
    unmatched = {
        real_kb.evidences[j].code
        for j in np.flatnonzero(shipped.default_option == -1)
        if not real_kb.evidences[j].is_binary
    }
    assert unmatched == {
        "douleurxx_soudain",
        "douleurxx_intens",
        "douleurxx_precis",
        "lesions_peau_intens",
        "lesions_peau_elevee",
        "lesions_peau_prurit",
    }


@requires_data
def test_bed_stats_file_is_the_train_split_but_for_multi_choice(real_kb: KB, shipped: BedKnowledge) -> None:
    if not (REPO_ROOT / "data/ddxplus/release_train_patients.csv").exists():
        pytest.skip("DDXPlus train CSV not present")
    counted = count_knowledge(real_kb, iter_train(), text_default_codes(read_release_json()[1]))
    for name in ("prior", "known", "binary"):
        np.testing.assert_array_equal(getattr(counted, name), getattr(shipped, name))
    for j, e in enumerate(real_kb.evidences):
        if e.data_type is DataType.CATEGORICAL:
            assert counted.options[j] == shipped.options[j]
            np.testing.assert_array_equal(counted.option_probs[j], shipped.option_probs[j])
        elif e.data_type is DataType.MULTI_CHOICE:
            # BED's file keeps one value per record, so its shares of a disease sum to 1 (and 1e-7 per unseen value).
            seen = shipped.option_probs[j][shipped.known[:, j]]
            np.testing.assert_allclose(np.where(seen > UNSEEN, seen, 0.0).sum(axis=1), 1.0, rtol=0, atol=1e-12)
            # The value kept is one of the record's values, so no share exceeds the share of records holding it.
            column = [counted.options[j].index(o) for o in shipped.options[j]]
            held = counted.option_probs[j][shipped.known[:, j]][:, column]
            assert (np.where(seen > UNSEEN, seen, 0.0) <= held + 1e-12).all()


def _original() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = json.loads(gzip.decompress(ORIGINAL.read_bytes()))
    return rows


def _replay_utilities(k: BedKnowledge, patient: Patient, asked: Sequence[str]) -> FloatArray:
    """The port's utilities after `asked`, answered as BED answers them."""
    kb = k.kb
    findings = record_findings(k, patient)
    likelihood = k.positive_factor(kb.evidence_index[patient.initial], None)
    for code in asked:
        j = kb.evidence_index[code]
        answer = findings.get(j)
        if answer is None:
            likelihood = likelihood * k.negative_factor(j)
        else:
            for option in answer or (None,):
                likelihood = likelihood * k.positive_factor(j, option)
    joint = likelihood * k.prior
    return k.utilities(joint / joint.sum())


@requires_data
def test_port_reproduces_the_original_code_on_validation(
    real_kb: KB, shipped: BedKnowledge, validation: list[Patient]
) -> None:
    """The original code's output (BEDBatch.batch_run at Casande-RL@5279013 with its stats file) on the first
    validation rows: the port gives every patient the same GTPA@1, and its questions differ only after a tie
    between equally useful questions, which BED breaks by float rounding and set order."""
    policy = BedPolicy(shipped)
    original = _original()
    diverged = 0
    for rec in original:
        p = validation[rec["row"]]
        ep = policy.run(p)
        g1 = gtpa_at_k_official(ep.posterior, ep.gold, 1, real_kb.official_order) == 1.0
        assert g1 == rec["gtpa1"], rec["row"]
        if list(ep.asked) == rec["q"]:
            assert ep.il == rec["il"]
            continue
        diverged += 1
        step = next((i for i, (a, b) in enumerate(zip(ep.asked, rec["q"], strict=False)) if a != b), None)
        assert step is not None, f"row {rec['row']}: one stops where the other asks"
        u = _replay_utilities(shipped, p, ep.asked[:step])
        mine, theirs = real_kb.evidence_index[ep.asked[step]], real_kb.evidence_index[rec["q"][step]]
        assert abs(u[mine] - u[theirs]) <= UTILITY_TIE_ATOL, rec["row"]
    assert diverged <= 0.05 * len(original)


@requires_data
def test_bed_on_the_review_sample(real_kb: KB, shipped: BedKnowledge, validation: list[Patient]) -> None:
    sample = [validation[i] for i in random.Random(0).sample(range(len(validation)), 20_000)]
    processes = min(16, os.cpu_count() or 1)
    with worker_pool(processes) as pool:
        task = ScoreTask(BedPolicy(shipped), real_kb, ())
        s = Scored.concat(map_items(task, chunks(sample, 4 * processes), pool))
    assert not s.failed.any()
    g1 = round(100 * float(s.column("gtpa1_official").mean()), 2)
    assert (g1, round(float(s.il.mean()), 3)) == (99.42, 5.487)
