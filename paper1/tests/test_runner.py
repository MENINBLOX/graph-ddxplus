from __future__ import annotations

import numpy as np
import pytest

from ddxkb.data import Patient
from ddxkb.evaluation import Scored, ScoreTask
from ddxkb.kb import KB
from ddxkb.policies.bayes import BayesPolicy, membership_table, uniform_prior
from ddxkb.runner import chunks, map_items, worker_pool
from helpers import toy_patient

RECORDS = [
    (["douleur", "douleur_intens_@_2", "douleur_lieu_@_dos", "atcd", "trav1_@_N"], "douleur", "Beta"),
    (["fievre", "trav1_@_Asie"], "fievre", "Alpha"),
    (["fievre", "douleur", "douleur_intens_@_1", "douleur_lieu_@_tete", "trav1_@_N"], "fievre", "Gamma"),
]


def _patients(n: int) -> list[Patient]:
    out = []
    for i in range(n):
        record, initial, pathology = RECORDS[i % len(RECORDS)]
        # No record holds toux, so every seventh patient starts from an absent evidence and its episode raises.
        out.append(toy_patient(record, initial="toux" if i % 7 == 6 else initial, pathology=pathology, row=1000 + i))
    return out


def test_a_pooled_run_keeps_patient_order_and_matches_the_serial_run(toy_kb: KB) -> None:
    budgets = (0, 2)
    policy = BayesPolicy(membership_table(toy_kb, 0.5, 1e-3), uniform_prior(toy_kb), 0.99, budgets, min_gain=1e-6)
    task = ScoreTask(policy, toy_kb, budgets)
    patients = _patients(40)
    serial = task(patients)
    with worker_pool(2) as pool:
        pooled = Scored.concat(map_items(task, chunks(patients, 7), pool))
    assert pooled.rows.tolist() == [p.row for p in patients]
    for field in ("rows", "gold", "il", "stop", "at_budget", "peak", "max_gain", "failed", "top", "top_p"):
        assert np.array_equal(getattr(pooled, field), getattr(serial, field)), field
    assert (pooled.questions, pooled.errors) == (serial.questions, serial.errors)
    assert [i for i, f in enumerate(serial.failed) if f] == [6, 13, 20, 27, 34]


def _square(x: int) -> int:
    return x * x


def test_map_items_keeps_order_with_and_without_a_pool() -> None:
    items = list(range(23))
    with worker_pool(1) as none:
        assert none is None
        assert map_items(_square, items, none) == [x * x for x in items]
    with worker_pool(2) as pool:
        assert map_items(_square, items, pool) == [x * x for x in items]
    with pytest.raises(ValueError), worker_pool(0):
        pass


def test_chunks_cover_the_items_in_order() -> None:
    items = list(range(10))
    parts = chunks(items, 4)
    assert [list(p) for p in parts] == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert chunks(items, 1) == [items]
    assert [list(p) for p in chunks(items, 30)] == [[i] for i in items]
    assert chunks([], 3) == []
    with pytest.raises(ValueError):
        chunks(items, 0)
