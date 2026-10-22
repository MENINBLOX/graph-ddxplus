from __future__ import annotations

import os
import random

import numpy as np
import pytest

from ddxkb.data import REPO_ROOT, Patient
from ddxkb.evaluation import V1_TOP, V1Scored, V1Task
from ddxkb.kb import KB
from ddxkb.policies.v1_replay import (
    V1_DATA_DIR,
    V1Replay,
    code_graph,
    cui_graph,
    folded_cui_edges,
    subattributes,
    v1_cui_graph,
)
from ddxkb.runner import chunks, map_items, worker_pool
from helpers import binary_kb, toy_patient

# s1 is on D1-D3's lists; s2 and s5 are each on two of them; D4 shares only s5.
LISTS = {"D1": ["s1", "s2", "s3", "s5"], "D2": ["s1", "s2", "s4"], "D3": ["s1", "s5"], "D4": ["s5", "s6"]}
NODES = ["s1", "s2", "s3", "s4", "s5", "s6"]


@pytest.fixture
def lists() -> KB:
    return binary_kb(LISTS, NODES)


def _d2(row: int = 0) -> Patient:
    return toy_patient(["s1", "s2", "s4"], initial="s1", pathology="D2", row=row)


def _mask(*nodes: str) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
    return np.array([n in nodes for n in NODES])


def test_hand_traced_episode(lists: KB) -> None:
    ep = V1Replay(code_graph(lists), deny=2).run(_d2())
    # Stage A: s2 and s5 are each on two lists holding s1; ascending node order asks s2 (yes).
    # Stage B: s5 scores coverage 2 x co-occurrence 2 (no); s3 and s4 tie at 1 x 2, s3 first (no); now D1
    # holds 2 denied nodes, so only D2's s4 is left (yes); D2 and D3 have nothing unasked, and the episode ends.
    assert ep.questions == ("s2", "s5", "s3", "s4")
    assert ep.il == 5
    assert ep.order == (1, 0, 2)
    scores = np.array([2 / 5 * 2, 3 / 4 * 3, 1 / 3 * 1, 0.0])
    assert ep.posterior == pytest.approx(scores / scores.sum())
    assert ep.in_top(1)
    assert (ep.gold, ep.budgets, ep.budget_posteriors.shape) == (1, (), (0, 4))


def test_deny_threshold_filters_question_generation_only(lists: KB) -> None:
    ep = V1Replay(code_graph(lists), deny=1).run(_d2())
    # After s5 (no), D1 and D3 each hold one denied node and leave the question pool: s3 is never asked.
    assert ep.questions == ("s2", "s5", "s4")
    # D1 and D3 are still ranked.
    assert ep.order == (1, 0, 2)
    assert ep.posterior == pytest.approx([12 / 43, 27 / 43, 4 / 43, 0.0])
    for deny in (0, 6):
        assert V1Replay(code_graph(lists), deny=deny).run(_d2()).questions == ("s2", "s5", "s3", "s4")


def test_ranking_scores_and_ties(lists: KB) -> None:
    pol = V1Replay(code_graph(lists))
    order, scores = pol.ranking(_mask("s1", "s2"), _mask())
    assert scores == pytest.approx([4 / 3, 4 / 3, 1 / 2, 0.0])
    assert order.tolist() == [0, 1, 2]
    order, scores = pol.ranking(_mask("s1", "s2"), _mask("s5"))
    assert scores == pytest.approx([1.0, 4 / 3, 1 / 3, 0.0])
    assert order.tolist() == [1, 0, 2]


def test_stable_top3_and_question_cap_stop_the_episode(lists: KB) -> None:
    # Top 3 after each question: (D1, D2, D3), then (D2, D1, D3) three times.
    assert V1Replay(code_graph(lists), deny=2, stable_window=2).run(_d2()).questions == ("s2", "s5", "s3")
    assert V1Replay(code_graph(lists), deny=2, stable_window=3).run(_d2()).questions == ("s2", "s5", "s3", "s4")
    capped = V1Replay(code_graph(lists), deny=2, max_questions=2).run(_d2())
    assert (capped.questions, capped.il) == (("s2", "s5"), 3)
    with pytest.raises(ValueError):
        V1Replay(code_graph(lists), stable_window=0)


def test_tie_seed_fixes_one_shuffled_node_order(lists: KB) -> None:
    firsts = {}
    for seed in range(20):
        pol = V1Replay(code_graph(lists), deny=2, tie_seed=seed)
        a, b = pol.run(_d2(row=0)), pol.run(_d2(row=5))
        assert a.questions == b.questions
        firsts[seed] = a.questions[0]
    assert set(firsts.values()) == {"s2", "s5"}
    assert V1Replay(code_graph(lists), deny=2, tie_seed=None).run(_d2()).questions[0] == "s2"


def test_trav1_bug_answers_recorded_default_values_yes(toy_kb: KB) -> None:
    g = code_graph(toy_kb)
    patient = toy_patient(
        ["douleur", "douleur_intens_@_0", "douleur_lieu_@_nulle_part", "trav1_@_N", "atcd"], initial="douleur"
    )

    def yes(bug: bool) -> set[str]:
        mask = V1Replay(g, trav1_bug=bug).answered_yes(patient)
        return {g.nodes[s] for s in np.flatnonzero(mask)}

    assert yes(False) == {"douleur", "atcd"}
    assert yes(True) == {"douleur", "douleur_intens", "douleur_lieu", "trav1", "atcd"}


def test_code_graph_can_drop_the_subattributes(toy_kb: KB) -> None:
    assert subattributes(toy_kb) == {"douleur_intens", "douleur_lieu"}
    full = code_graph(toy_kb)
    assert full.nodes == tuple(e.code for e in toy_kb.evidences)
    assert np.array_equal(full.membership, toy_kb.membership)
    assert full.disease_nodes == toy_kb.diseases
    g = code_graph(toy_kb, drop_subattributes=True)
    assert g.nodes == ("fievre", "toux", "douleur", "trav1", "atcd")
    assert np.array_equal(g.membership, toy_kb.membership[:, [0, 1, 2, 5, 6]])
    assert "douleur_lieu" not in g.code_node
    assert g.code_node["atcd"] == 4


def test_cui_graph_merges_codes_sharing_a_cui(toy_kb: KB) -> None:
    code_cui = {"fievre": "C1", "toux": "C2", "douleur": "C3", "atcd": "C3", "trav1": "C9", "absent_code": "C8"}
    disease_cui = {"Alpha": "X1", "Beta": "X2", "Gamma": "X3"}
    edges = folded_cui_edges(toy_kb, code_cui, disease_cui)
    assert edges == {
        ("C1", "X1"),
        ("C2", "X1"),
        ("C9", "X1"),
        ("C3", "X2"),
        ("C9", "X2"),
        ("C1", "X3"),
        ("C3", "X3"),
        ("C9", "X3"),
    }
    g = cui_graph(toy_kb, edges, code_cui, disease_cui)
    assert g.nodes == ("C1", "C2", "C3", "C9")
    assert g.disease_nodes == ("X1", "X2", "X3")
    assert dict(g.code_node) == {"fievre": 0, "toux": 1, "douleur": 2, "atcd": 2, "trav1": 3}
    # atcd alone answers the shared node C3.
    yes = V1Replay(g).answered_yes(toy_patient(["fievre", "atcd"], initial="fievre"))
    assert yes.tolist() == [True, False, True, False]
    with pytest.raises(ValueError, match="names no KB disease"):
        cui_graph(toy_kb, [*edges, ("C1", "X7")], code_cui, disease_cui)


def test_an_initial_evidence_without_a_node_is_a_failure(toy_kb: KB) -> None:
    code_cui = {"fievre": "C1", "douleur": "C3"}
    disease_cui = {"Alpha": "X1", "Beta": "X2", "Gamma": "X3"}
    g = cui_graph(toy_kb, folded_cui_edges(toy_kb, code_cui, disease_cui), code_cui, disease_cui)
    v = V1Task(V1Replay(g))([toy_patient(["toux", "fievre"], initial="toux", pathology="Alpha", row=3)])
    assert v.failed.tolist() == [True]
    assert v.errors[0][0] == 3 and "toux has no node" in v.errors[0][1]


def test_parallel_matches_serial(lists: KB) -> None:
    records = [(["s1", "s2", "s4"], "D2"), (["s1", "s5"], "D3"), (["s5", "s6"], "D4"), (["s1", "s2", "s3"], "D1")]
    patients = [
        toy_patient(records[i % 4][0], initial=records[i % 4][0][0], pathology=records[i % 4][1], row=i)
        for i in range(30)
    ]
    task = V1Task(V1Replay(code_graph(lists), deny=2, tie_seed=3))
    serial = task(patients)
    with worker_pool(2) as pool:
        parallel = V1Scored.concat(map_items(task, chunks(patients, 4), pool))
    for field in ("rows", "il", "hits", "tie_aware", "gtpa", "failed"):
        assert np.array_equal(getattr(serial, field), getattr(parallel, field)), field


def _require_v1_files() -> None:
    for name in ("kg_edges.json", "umls_mapping.json", "disease_umls_mapping.json"):
        if not (REPO_ROOT / V1_DATA_DIR / name).exists():
            pytest.skip(f"{name} not present")


def _replay(pol: V1Replay, patients: list[Patient]) -> tuple[float, float, float, float]:
    """v1-order GTPA@1, tie-aware GTPA@1, v1-order GTPA@10 (all %) and mean |T| including the initial evidence."""
    processes = min(16, os.cpu_count() or 1)
    with worker_pool(processes) as pool:
        v = V1Scored.concat(map_items(V1Task(pol), chunks(patients, 4 * processes), pool))
    assert len(v.rows) == len(patients) and not v.failed.any()
    return (
        round(100 * float(v.hits[:, V1_TOP.index(1)].mean()), 2),
        round(100 * float(v.tie_aware.mean()), 2),
        round(100 * float(v.hits[:, V1_TOP.index(10)].mean()), 2),
        round(float(v.il.mean()), 2),
    )


def _sample(validation: list[Patient]) -> list[Patient]:
    return [validation[i] for i in random.Random(0).sample(range(len(validation)), 20_000)]


def test_real_subattributes(real_kb: KB) -> None:
    assert len(subattributes(real_kb)) == 14
    assert not any(real_kb.evidence(c).is_binary for c in subattributes(real_kb))


def test_c2_cui_level_on_validation(real_kb: KB, validation: list[Patient]) -> None:
    # Review §2.8: 81.13% GTPA@1 (CUI-ascending ties), GTPA@10 99.73%, IL 17.15 without the initial evidence.
    _require_v1_files()
    top1, tie_aware, top10, il = _replay(V1Replay(v1_cui_graph(real_kb), trav1_bug=True), validation)
    assert (top1, top10, il) == (81.13, 99.73, 18.15)
    assert tie_aware == 80.10


def test_c2_code_level_on_the_review_sample(real_kb: KB, validation: list[Patient]) -> None:
    # Review §6.2: 31.83% tie-aware GTPA@1, IL 7.09 with the initial evidence.
    top1, tie_aware, _, il = _replay(V1Replay(code_graph(real_kb)), _sample(validation))
    assert (tie_aware, il) == (31.83, 7.09)
    assert top1 == 32.17


def test_code_level_without_subattributes_on_the_review_sample(real_kb: KB, validation: list[Patient]) -> None:
    # Review §6.2: 82.23% / 17.89, and 78.70% / 18.45 with the trav1 bug.
    g = code_graph(real_kb, drop_subattributes=True)
    sample = _sample(validation)
    assert _replay(V1Replay(g), sample)[1::2] == (82.23, 17.89)
    assert _replay(V1Replay(g, trav1_bug=True), sample)[1::2] == (78.70, 18.45)
