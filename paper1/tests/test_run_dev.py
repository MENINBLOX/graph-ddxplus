from __future__ import annotations

import dataclasses
import gzip
import itertools
import json
import re
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ddxkb import experiment, run_dev
from ddxkb.audit import list_match_is_unique, support_audit
from ddxkb.data import TEST_CSV, TRAIN_CSV, VALIDATION_CSV, iter_train, load_kb, load_validation
from ddxkb.evaluation import METRICS, Scored, ScoreTask, report
from ddxkb.frequencies import count_frequencies
from ddxkb.kb import KB
from ddxkb.policies import bed
from ddxkb.policies.baselines import random_questions, train_frequency_questions
from ddxkb.policies.bayes import BayesPolicy
from ddxkb.policies.episode import Policy
from ddxkb.protocol import (
    LADDER,
    P2_RECORDED,
    P2_VARIANTS,
    SENSITIVITY_HIERARCHY,
    SENSITIVITY_TAUS,
    TAU,
    P1Config,
    P2Config,
    P2Variant,
    p1_grid,
    p1_wider_grid,
    p2_fixed,
    selection_key,
)
from ddxkb.run_dev import STAGES, main
from ddxkb.simulator import MAX_IL, AnswerMode, positive_codes
from helpers import (
    SYNTH_DISEASES,
    SYNTH_SYMPTOMS,
    bed_stats_json,
    bed_stats_sha256,
    make_repo,
    synth_kb_json,
    synth_rows,
    write_csv,
)

N_VALIDATION = 45
# What main.json holds for each main policy; a P2 variant also holds its comparison with P1 under "against".
MAIN_SUMMARY = {
    "config",
    "inputs",
    "hierarchy",
    "answers",
    "n",
    "failures",
    "errors",
    "stop",
    "budgets",
    "by_disease",
    "curve",
    "tau_sensitivity",
}
PROVENANCE = {
    "design",
    "stage",
    "split",
    "n_patients",
    "commit",
    "paper1_changes",
    "design_changes",
    "commit_at_end",
    "inputs",
    "train_rows",
    "environment",
    "workers",
    "started_utc",
    "elapsed_s",
    "timings",
}


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic repository with a directory in place of the test CSV, which no reader can open, whatever its
    permissions: every development run of this module must succeed without the test split."""
    root = tmp_path_factory.mktemp("repo")
    make_repo(root, n_validation=N_VALIDATION)
    (root / TEST_CSV).unlink()
    (root / TEST_CSV).mkdir()
    return root


@pytest.fixture(scope="module")
def dev_out(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("dev")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bed, "BED_STATS_SHA256", bed_stats_sha256(repo))
        run_dev.run_dev(STAGES, repo, out, workers=1)
    return out


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _tried(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / run_dev.TRIED).read_text().splitlines()]


def _config(line: dict[str, Any]) -> P1Config | P2Config:
    config = line["config"]
    return P1Config.from_dict(config) if line["policy"] == "P1" else P2Config.from_dict(config)


def test_grid_logs_every_configuration_at_every_threshold(dev_out: Path) -> None:
    tried = [r for r in _tried(dev_out) if r["role"] != "wider"]
    assert len(tried) == (9 + 4 * 2) * 4
    p1 = [r for r in tried if r["policy"] == "P1"]
    # One line per candidate and threshold, the candidates in listed order and the thresholds rising within each.
    assert [_config(r) for r in p1] == [P1Config(c.alpha, c.eps, tau) for c in p1_grid() for tau in SENSITIVITY_TAUS]
    assert [r["order"] for r in p1] == [k for k in range(9) for _ in SENSITIVITY_TAUS]
    for variant in P2_VARIANTS:
        rows = [r for r in tried if r["policy"] == variant.name]
        assert [_config(r) for r in rows] == [
            P2Config(variant, lam, tau) for lam in (1.0, 0.5) for tau in SENSITIVITY_TAUS
        ]
        assert [r["role"] for r in rows] == ["sensitivity", "sensitivity", "fixed", "sensitivity", *["sensitivity"] * 4]
        assert all(r["order"] is None and r["selection_key"] is None for r in rows)
    for r in tried:
        assert r["n"] == N_VALIDATION and r["limit"] is None
        assert len(r["curve"]["gtpa1"]) == 31


def test_only_the_nine_p1_lines_at_the_threshold_enter_the_selection(dev_out: Path) -> None:
    tried = _tried(dev_out)
    candidates = [r for r in tried if r["role"] == "selection"]
    assert [_config(r) for r in candidates] == list(p1_grid())
    for r in tried:
        stop = r["stop"]
        if r["role"] == "selection" or (r["role"] == "wider" and r["config"]["tau"] == TAU):
            key = selection_key(stop["gtpa1"]["estimate"], stop["il"]["mean"], r["order"])
            assert r["selection_key"] == list(key)
        else:
            assert r["selection_key"] is None
    selected = _json(dev_out / run_dev.SELECTED)
    best = min(candidates, key=lambda r: tuple(r["selection_key"]))
    assert selected["P1"]["config"] == best["config"] and selected["P1"]["order"] == best["order"]
    assert selected["P1"]["config"]["tau"] == TAU
    assert "0.99" in selected["objective"] and "0.99" in selected["fixed"]


def test_p2_is_fixed_and_never_selected(dev_out: Path) -> None:
    tried = _tried(dev_out)
    selected = _json(dev_out / run_dev.SELECTED)
    assert list(selected["P2"]) == [v.name for v in P2_VARIANTS]
    preferred = []
    for variant in P2_VARIANTS:
        chosen = selected["P2"][variant.name]
        assert chosen["config"] == p2_fixed(variant).as_dict() == {**chosen["config"], "lam": 1.0, "tau": 0.99}
        assert set(chosen) == {"config", "gtpa1", "il_mean"}
        rows = [r for r in tried if r["policy"] == variant.name]
        [fixed] = [r for r in rows if r["role"] == "fixed"]
        assert fixed["config"] == chosen["config"]
        assert (chosen["gtpa1"], chosen["il_mean"]) == (fixed["stop"]["gtpa1"]["estimate"], fixed["stop"]["il"]["mean"])
        keys = [selection_key(r["stop"]["gtpa1"]["estimate"], r["stop"]["il"]["mean"], k) for k, r in enumerate(rows)]
        preferred.append(rows[keys.index(min(keys))]["config"])
    # The accuracy objective would pick another lambda or threshold for some variant; none is picked.
    assert any(config != p2_fixed(v).as_dict() for v, config in zip(P2_VARIANTS, preferred, strict=True))


def _check_wider(out: Path) -> dict[str, object]:
    """The wider-grid record of a grid run against its configs_tried lines; returns the boundary record."""
    selected = _json(out / run_dev.SELECTED)["P1"]
    boundary: dict[str, object] = selected["boundary"]
    tried = [r for r in _tried(out) if r["policy"] == "P1"]
    wider_lines = [r for r in tried if r["role"] == "wider"]
    chosen = P1Config.from_dict(selected["config"])
    if not boundary["ends"]:
        assert boundary["wider_grid"] is None and not wider_lines
        return boundary
    wider = p1_wider_grid(chosen)
    new = [c for c in wider if c not in set(p1_grid())]
    assert [_config(r) for r in wider_lines] == [P1Config(c.alpha, c.eps, tau) for c in new for tau in SENSITIVITY_TAUS]
    assert [r["order"] for r in wider_lines] == [wider.index(c) for c in new for _ in SENSITIVITY_TAUS]
    record = boundary["wider_grid"]
    assert isinstance(record, dict) and (record["n"], record["new"]) == (len(wider), len(new))
    at_tau = [r for r in tried if r["config"]["tau"] == TAU]
    keys = {
        _config(r): selection_key(r["stop"]["gtpa1"]["estimate"], r["stop"]["il"]["mean"], wider.index(_config(r)))
        for r in at_tau
    }
    assert set(keys) == set(wider)
    best = min(keys, key=keys.__getitem__)
    assert record["best"]["config"] == best.as_dict() and record["best_is_selected"] == (best == chosen)
    return boundary


def test_the_wider_grid_is_recorded_only_at_a_boundary(dev_out: Path) -> None:
    _check_wider(dev_out)


def test_a_selection_at_a_boundary_records_the_wider_grid_without_choosing_from_it(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corner = P1Config(0.3, 1e-4, TAU)
    best = run_dev._best

    def forced(entries: Any) -> Any:
        return next((e for e in entries if e[0] == corner), None) or best(entries)

    monkeypatch.setattr(run_dev, "_best", forced)
    run_dev.run_dev(("grid",), repo, tmp_path, workers=1)
    boundary = _check_wider(tmp_path)
    assert boundary["ends"] == {"alpha": "first", "eps": "last"}
    record = boundary["wider_grid"]
    assert isinstance(record, dict) and record["new"] == 4 * 4 - 9
    assert record["values"] == {"alpha": [0.1, 0.3, 0.5, 0.7], "eps": [1e-2, 1e-3, 1e-4, 1e-5]}
    assert _json(tmp_path / run_dev.SELECTED)["P1"]["config"] == corner.as_dict()
    # A line derived from a trajectory without a stop is the stop of that configuration run on its own.
    kb = load_kb(repo)
    for cfg in (P1Config(0.1, 1e-5, TAU), P1Config(0.1, 1e-5, 0.999), P1Config(0.5, 1e-3, 0.9)):
        line = next(r for r in _tried(tmp_path) if r["policy"] == "P1" and r["config"] == cfg.as_dict())
        alone = ScoreTask(cfg.policy(kb, ()), kb, ())(load_validation(repo))
        assert line["stop_sha256"] == alone.stop_sha256()


def test_main_reruns_the_selected_configurations(dev_out: Path) -> None:
    selected = _json(dev_out / run_dev.SELECTED)
    main_json = _json(dev_out / "main.json")
    assert main_json["P1"]["config"] == selected["P1"]["config"]
    assert main_json["P1"]["stop"]["gtpa1"]["estimate"] == selected["P1"]["gtpa1"]
    assert main_json["P1"]["stop"]["il"]["mean"] == selected["P1"]["il_mean"]
    assert "against" not in main_json["P1"]
    assert (main_json["P1"]["inputs"], main_json["P1"]["hierarchy"]) == (["initial_evidence"], "none")
    for v in P2_VARIANTS:
        entry = main_json[v.name]
        assert entry["config"] == selected["P2"][v.name]["config"]
        assert entry["stop"]["il"]["mean"] == selected["P2"][v.name]["il_mean"]
        assert set(entry["against"]) == {"P1"}
        assert set(entry["against"]["P1"]) == {"stop", "5", "10", "15", "20", "30"}
        assert {"low", "high"} <= set(entry["against"]["P1"]["stop"]["il_diff"])
        assert set(entry["budgets"]) == {"5", "10", "15", "20", "30"}
        assert {"low", "high"} <= set(entry["stop"]["ddf1"]) and {"low", "high"} <= set(entry["stop"]["dshm"])
        inputs = ["age", "initial_evidence", "sex"] if v.age_sex else ["initial_evidence"]
        assert entry["inputs"] == inputs
    for name in ("P1", *(v.name for v in P2_VARIANTS)):
        assert main_json[name]["answers"] == ("value" if name.startswith("P2-value") else "binary")
        lines = gzip.decompress((dev_out / "patients" / f"{name}.jsonl.gz").read_bytes()).decode().splitlines()
        records = [json.loads(line) for line in lines]
        assert [r["row"] for r in records] == list(range(N_VALIDATION))
        assert sum(r["gtpa1"] for r in records) / N_VALIDATION == pytest.approx(
            main_json[name]["stop"]["gtpa1"]["estimate"]
        )
    assert set(main_json["P1"]) == MAIN_SUMMARY
    assert all(set(main_json[v.name]) == MAIN_SUMMARY | {"against"} for v in P2_VARIANTS)


def test_every_main_policy_reports_its_stop_at_each_sensitivity_threshold(repo: Path, dev_out: Path) -> None:
    main_json = _json(dev_out / "main.json")
    tried = _tried(dev_out)
    kb, patients = load_kb(repo), load_validation(repo)
    counts = count_frequencies(kb, iter_train(repo))
    for name in ("P1", *(v.name for v in P2_VARIANTS)):
        entry = main_json[name]
        sensitivity = entry["tau_sensitivity"]
        assert list(sensitivity) == ["0.9", "0.95", "0.99", "0.999"]
        # At the protocol's threshold the derived stop is the policy's own, intervals included.
        assert sensitivity["0.99"] == entry["stop"]
        lengths = [sensitivity[key]["il"]["mean"] for key in sensitivity]
        assert lengths == sorted(lengths) and lengths[0] < lengths[-1]
        for key, tau in zip(sensitivity, SENSITIVITY_TAUS, strict=True):
            stop = sensitivity[key]
            assert {"low", "high"} <= set(stop["il"]) and {"low", "high"} <= set(stop["ddf1"])
            # The same stop as the grid derived, and as the policy stopping at that threshold gives on its own.
            config = {**entry["config"], "tau": tau}
            [line] = [r for r in tried if r["policy"] == name and r["config"] == config]
            assert (stop["gtpa1"], stop["il"]["mean"]) == (line["stop"]["gtpa1"], line["stop"]["il"]["mean"])
            cfg = _config(line)
            policy = cfg.policy(kb, ()) if isinstance(cfg, P1Config) else cfg.policy(kb, counts, ())
            alone = ScoreTask(policy, kb, ())(patients)
            assert stop["gtpa1"]["estimate"] == float(alone.column("gtpa1").mean())
            assert stop["il"]["mean"] == float(alone.il.mean())


def test_the_ladder_compares_one_factor_at_a_time(dev_out: Path) -> None:
    main_json = _json(dev_out / "main.json")
    ladder = main_json["ladder"]
    assert list(ladder) == [f"{a} - {b}" for a, b in LADDER]
    assert ladder["B-BED - P2-value"] == "B-BED in baselines.json"
    for (this, other), factor in list(LADDER.items())[:4]:
        rung = ladder[f"{this} - {other}"]
        assert (rung["this"], rung["other"], rung["factor"]) == (this, other, factor)
        assert set(rung) == {"this", "other", "factor", "stop", "5", "10", "15", "20", "30", "tau_sensitivity"}
        stop = rung["stop"]
        # Minuend first: the difference is this policy's estimate minus the other's.
        gtpa1 = main_json[this]["stop"]["gtpa1"]["estimate"] - main_json[other]["stop"]["gtpa1"]["estimate"]
        il = main_json[this]["stop"]["il"]["mean"] - main_json[other]["stop"]["il"]["mean"]
        assert stop["gtpa1_diff"] == pytest.approx(gtpa1, abs=1e-12)
        assert stop["il_diff"]["estimate"] == pytest.approx(il, abs=1e-12)
        for point in ("stop", "5", "10", "15", "20", "30"):
            section = rung[point]
            interval = section["gtpa1_diff_interval"]
            assert interval["estimate"] == section["gtpa1_diff"]
            assert interval["low"] <= interval["estimate"] <= interval["high"]
            assert set(section["mcnemar"]) == {"metric", "only_this", "only_other", "p_value"}
            assert section["mcnemar"]["metric"] == "gtpa1_fixed"
            assert ("il_diff" in section) == (point == "stop")
        assert stop["il_diff"]["low"] <= stop["il_diff"]["estimate"] <= stop["il_diff"]["high"]
        assert list(rung["tau_sensitivity"]) == ["0.9", "0.95", "0.99", "0.999"]
        for key, at_tau in rung["tau_sensitivity"].items():
            assert set(at_tau) == set(stop)
            mine, theirs = main_json[this]["tau_sensitivity"][key], main_json[other]["tau_sensitivity"][key]
            assert at_tau["il_diff"]["estimate"] == pytest.approx(mine["il"]["mean"] - theirs["il"]["mean"], abs=1e-12)
        assert rung["tau_sensitivity"]["0.99"] == stop
    # The first rung is the comparison each P2 summary already holds against P1.
    first = ladder["P2-binary - P1"]
    assert {k: first[k] for k in ("stop", "5", "10", "15", "20", "30")} == main_json["P2-binary"]["against"]["P1"]


def test_ablations_and_baselines_cover_the_design(dev_out: Path) -> None:
    ablations = _json(dev_out / "ablations.json")
    assert set(ablations) >= {"A1", "A2", "A3", "A4", "A5", "A6", "references", "provenance"}
    assert set(ablations["references"]) == {"P1", *(v.name for v in P2_VARIANTS)}
    assert set(ablations["A1"]) == {"both", "positive_only", "negative_only"}
    # 보완 3 §2.7: before any question both A1 variants hold P1's posterior, the initial evidence heard.
    start = ablations["references"]["P1"]["curve"]["gtpa1"][0]
    assert all(ablations["A1"][heard]["curve"]["gtpa1"][0] == start for heard in ("positive_only", "negative_only"))
    assert start > 1 / 5
    assert set(ablations["A3"]) == {"0.1", "0.01", "0.001", "0.0001", "1e-06", "1e-09"}
    a4 = ablations["A4"]
    assert (a4["none"], a4["implicit"]) == ("P1", "P1 in hierarchy.json")
    assert set(a4["definitions"]) == {"none", "implicit", "parent_first", "no_children"}
    for rule in ("parent_first", "no_children"):
        assert a4[rule]["hierarchy"] == rule and a4[rule]["failures"] == 0
        assert set(a4[rule]["against"]["P1"]) == {"stop", "5", "10", "15", "20", "30"}
        assert {"low", "high"} <= set(a4[rule]["against"]["P1"]["stop"]["il_diff"])
    assert set(ablations["A6"]) == {"P2-binary in value mode", "P2-binary-agesex in value mode"}
    assert set(ablations["A6"]["P2-binary in value mode"]["against"]) == {"P1", "P2-binary"}
    selected_eps = _json(dev_out / run_dev.SELECTED)["P1"]["config"]["eps"]
    same = ablations["A3"][f"{selected_eps:g}"]["stop"]
    assert same["gtpa1"] == ablations["references"]["P1"]["stop"]["gtpa1"]
    assert same["il"]["mean"] == ablations["references"]["P1"]["stop"]["il"]["mean"]
    baselines = _json(dev_out / "baselines.json")
    assert set(baselines["B0"]) == {"most_frequent", "initial_only"}
    assert set(baselines["B1"]) == {
        "set_inclusion",
        "kb_bayes",
        "binary_majority",
        *(f"naive_bayes/{v.name}" for v in P2_VARIANTS),
    }
    assert baselines["B1"]["binary_majority"]["failures"] == 0
    reads = {name: (b["inputs"], b["profile"], b["answers"]) for name, b in baselines["B1"].items()}
    assert (
        reads["set_inclusion"] == reads["kb_bayes"] == (["initial_evidence"], "every evidence of the record", "binary")
    )
    assert reads["naive_bayes/P2-value-agesex"] == (
        ["age", "initial_evidence", "sex"],
        "every evidence of the record",
        "value",
    )
    assert (baselines["B0"]["most_frequent"]["inputs"], baselines["B0"]["most_frequent"]["answers"]) == ([], None)
    assert baselines["B0"]["initial_only"]["profile"] == "none"
    assert set(baselines["B1"]["binary_majority"]["against"]) == {"P1"}
    bed_runs = baselines["B-BED"]
    assert set(bed_runs) == {"definitions", "stats_file", "counted", "answer_differences", "reported", "ladder"}
    assert set(bed_runs["definitions"]) == set(bed_runs) - {"definitions"} | {"answers"}
    for source in ("stats_file", "counted"):
        entry = bed_runs[source]
        assert (entry["inputs"], entry["hierarchy"]) == (["initial_evidence"], "none")
        assert set(entry["against"]) == {"P1", "P2-value"}
        assert {"low", "high"} <= set(entry["against"]["P2-value"]["stop"]["il_diff"])
        assert "p_value" in entry["against"]["P2-value"]["stop"]["mcnemar"]
        assert set(entry["budgets"]) == {"5", "10", "15", "20", "30"} and len(entry["curve"]["gtpa1"]) == 31
    # The synthetic stats file is the train split counted as BED counts it, so both sources agree.
    assert bed_runs["stats_file"]["stop"]["gtpa1"] == bed_runs["counted"]["stop"]["gtpa1"]
    assert {f.name for f in (dev_out / "patients").iterdir()} >= {"B-BED.jsonl.gz", "B-BED-counted.jsonl.gz"}
    hierarchy = _json(dev_out / "hierarchy.json")
    assert hierarchy["hierarchy"] == "implicit"
    for name in ("P1", *(v.name for v in P2_VARIANTS)):
        assert hierarchy[name]["hierarchy"] == "implicit" and set(hierarchy[name]["against"]) == {"P1", name}
    assert set(baselines["B2"]["seeds"]) == {"0", "1", "2", "3", "4"}
    assert set(baselines["B6"]) == {"P1", *(v.name for v in P2_VARIANTS)}
    assert "il" not in baselines["B1"]["set_inclusion"]["stop"]
    spread = baselines["B2"]["over_seeds"]["stop"]["gtpa1"]
    seeds = [s["stop"]["gtpa1"]["estimate"] for s in baselines["B2"]["seeds"].values()]
    assert (spread["min"], spread["max"]) == (min(seeds), max(seeds))
    assert spread["mean"] == pytest.approx(sum(seeds) / 5)


def test_the_runs_built_on_a_main_policy_keep_its_threshold_and_smoothing(repo: Path, dev_out: Path) -> None:
    """보완 3 §2.1-2.2: the hierarchy analysis, B2, B6 and A6 stop at TAU, and those on P2 smooth with P2_LAMBDA."""
    kb, patients = load_kb(repo), load_validation(repo)
    counts = count_frequencies(kb, iter_train(repo))
    selected = _json(dev_out / run_dev.SELECTED)["P1"]["config"]
    main: dict[str, BayesPolicy] = {"P1": P1Config(selected["alpha"], selected["eps"], TAU).policy(kb, ())}
    main.update({v.name: p2_fixed(v).policy(kb, counts, ()) for v in P2_VARIANTS})

    def stop_of(policy: BayesPolicy) -> tuple[float, float]:
        alone = ScoreTask(policy, kb, ())(patients)
        return float(alone.column("gtpa1").mean()), float(alone.il.mean())

    def recorded(entry: dict[str, Any]) -> tuple[float, float]:
        return entry["stop"]["gtpa1"]["estimate"], entry["stop"]["il"]["mean"]

    hierarchy = _json(dev_out / "hierarchy.json")
    baselines = _json(dev_out / "baselines.json")
    for name, policy in main.items():
        assert recorded(hierarchy[name]) == stop_of(dataclasses.replace(policy, hierarchy=SENSITIVITY_HIERARCHY)), name
        assert recorded(baselines["B6"][name]) == stop_of(train_frequency_questions(policy, counts)), name
    for seed, entry in baselines["B2"]["seeds"].items():
        assert recorded(entry) == stop_of(random_questions(main["P1"], int(seed))), seed
    a6 = _json(dev_out / "ablations.json")["A6"]
    for age_sex in (False, True):
        binary, value = (P2Variant(mode, age_sex) for mode in (AnswerMode.BINARY, AnswerMode.VALUE))
        entry = a6[f"{binary.name} in value mode"]
        assert entry["config"] == p2_fixed(value).as_dict() == {**entry["config"], "lam": 1.0, "tau": 0.99}
        assert recorded(entry) == stop_of(main[value.name])


def test_every_paired_comparison_at_the_stop_has_an_interval_for_the_gtpa1_difference(dev_out: Path) -> None:
    def comparisons(node: Any) -> list[dict[str, Any]]:
        found = []
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "against":
                    found += list(value.values())
                else:
                    found += comparisons(value)
        return found

    for name in ("main.json", "ablations.json", "baselines.json", "hierarchy.json"):
        found = comparisons(_json(dev_out / name))
        assert found, name
        for comparison in found:
            for section in comparison.values():
                assert isinstance(section["gtpa1_diff"], float)
                interval = section["gtpa1_diff_interval"]
                assert interval["estimate"] == section["gtpa1_diff"]
                assert interval["low"] <= interval["estimate"] <= interval["high"]


def test_bed_records_its_answer_differences_reported_figures_and_rung(dev_out: Path) -> None:
    baselines = _json(dev_out / "baselines.json")
    bed_runs = baselines["B-BED"]
    main_run = bed_runs["stats_file"]
    differences = bed_runs["answer_differences"]
    assert (differences["run"], differences["patients"]) == ("stats_file", N_VALIDATION)
    assert differences["valued_questions"] >= differences["oracle_negative"] >= differences["both_negative"] >= 0
    assert differences["oracle_negative_bed_positive"] + differences["both_negative"] == differences["oracle_negative"]
    assert 0 <= differences["oracle_negative_bed_filled"] <= differences["oracle_negative_bed_positive"]
    assert set(differences) == {
        "run",
        "patients",
        "valued_questions",
        "oracle_negative",
        "oracle_negative_bed_positive",
        "oracle_negative_bed_filled",
        "both_negative",
        "patients_with_difference",
        "patients_with_difference_share",
    }
    assert 0 < differences["patients_with_difference"] <= N_VALIDATION
    assert differences["patients_with_difference_share"] == differences["patients_with_difference"] / N_VALIDATION
    # Every question asked up to the stop is in the patient-level file; the valued ones are counted.
    lines = gzip.decompress((dev_out / "patients/B-BED.jsonl.gz").read_bytes()).decode().splitlines()
    valued = {"douleur_intens", "douleur_lieu", "trav1"}
    assert differences["valued_questions"] == sum(q in valued for line in lines for q in json.loads(line)["asked"])
    reported = bed_runs["reported"]
    assert reported["bed"] == {"gtpa1": 0.9947, "il": 5.47} and reported["run"] == "stats_file"
    assert reported["this_minus_bed"] == {
        "gtpa1": main_run["stop"]["gtpa1"]["estimate"] - 0.9947,
        "gtpa1_official": main_run["stop"]["gtpa1_official"]["estimate"] - 0.9947,
        "il": main_run["stop"]["il"]["mean"] - 5.47,
    }
    assert set(reported) == {"run", "bed", "this_minus_bed"}
    assert list(bed_runs["ladder"]) == ["B-BED - P2-value"]
    rung = bed_runs["ladder"]["B-BED - P2-value"]
    assert (rung["this"], rung["other"]) == ("B-BED", "P2-value") and "tau_sensitivity" not in rung
    assert {k: rung[k] for k in ("stop", "5", "10", "15", "20", "30")} == main_run["against"]["P2-value"]
    assert rung["stop"]["gtpa1_diff_interval"]["low"] <= rung["stop"]["gtpa1_diff"]


POINTS = ("stop", "5", "10", "15", "20", "30")
ANSWER_RULE_POLICIES = ("P2-value", "P2-recorded", "B-BED")
ANSWER_RULE_FILES = {"answer_rule.json", "patients/P2-recorded.jsonl.gz"}
UNIQUENESS_POLICIES = ("P1", "P2-binary", "P2-value", "P2-recorded", "B-BED")
# The records a run of uniqueness over every validation row is checked against.
UNIQUENESS_REPRODUCES = {
    "strata": ["properties.json"],
    "P1": ["main.json"],
    "P2-binary": ["main.json"],
    "P2-value": ["main.json"],
    "P2-recorded": ["answer_rule.json"],
    "B-BED": ["baselines.json"],
}


def _patient_records(out: Path, name: str) -> list[dict[str, Any]]:
    lines = gzip.decompress((out / "patients" / f"{name}.jsonl.gz").read_bytes()).decode().splitlines()
    return [json.loads(line) for line in lines]


def _snapshot(out: Path) -> dict[str, bytes]:
    return {str(f.relative_to(out)): f.read_bytes() for f in sorted(out.rglob("*")) if f.is_file()}


def _edit(path: Path, change: Callable[[Any], None]) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def _edit_first_patient(path: Path) -> None:
    lines = gzip.decompress(path.read_bytes()).decode().splitlines()
    first = json.loads(lines[0])
    first["il"] += 1
    path.write_bytes(gzip.compress(("\n".join([json.dumps(first, ensure_ascii=False), *lines[1:]]) + "\n").encode()))


def _assert_reads_the_stats_file_and_every_train_row(meta: dict[str, Any]) -> None:
    """The provenance of a 보완 4 stage run on its own, where no earlier stage of the run recorded an input."""
    assert {bed.BED_STATS, TRAIN_CSV, VALIDATION_CSV} <= set(meta["inputs"])
    assert meta["train_rows"] == 150


@pytest.fixture
def bed_stats(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bed, "BED_STATS_SHA256", bed_stats_sha256(repo))


@pytest.fixture
def analysis_out(dev_out: Path, tmp_path: Path) -> Path:
    """An output directory holding what the stages of 보완 4 read of the earlier stages of a development run over
    every validation row."""
    read = (run_dev.SELECTED, "main.json", "baselines.json", "properties.json")
    for name in (*read, "patients/P2-value.jsonl.gz", "patients/B-BED.jsonl.gz"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dev_out / name, tmp_path / name)
    return tmp_path


@pytest.fixture
def uniqueness_out(dev_out: Path, analysis_out: Path) -> Path:
    """`analysis_out` once the answer_rule stage has run: with the answer_rule.json the uniqueness stage also reads."""
    shutil.copyfile(dev_out / "answer_rule.json", analysis_out / "answer_rule.json")
    return analysis_out


def test_answer_rule_runs_three_policies_on_the_same_patients(repo: Path, dev_out: Path) -> None:
    a = _json(dev_out / "answer_rule.json")
    assert set(a) == {
        *("limit", "budgets", "definitions", "comparisons", "reproduces", "provenance"),
        *ANSWER_RULE_POLICIES,
    }
    assert (a["limit"], a["budgets"]) == (None, [5, 10, 15, 20, 30])
    assert set(a["definitions"]) == {"answer_rule", "comparisons", "reproduces", *ANSWER_RULE_POLICIES}
    assert set(a["provenance"]) == PROVENANCE and a["provenance"]["stage"] == "answer_rule"
    # The policies of the earlier stages have the summaries those stages hold, their comparisons aside.
    main_json = _json(dev_out / "main.json")
    assert a["P2-value"] == {k: v for k, v in main_json["P2-value"].items() if k != "against"}
    bed_main = {k: v for k, v in _json(dev_out / "baselines.json")["B-BED"]["stats_file"].items() if k != "against"}
    assert a["B-BED"] == {**bed_main, "answers": "BED simulator (definitions.B-BED)"}
    assert a["reproduces"] == {
        "P2-value": ["main.json", "patients/P2-value.jsonl.gz"],
        "B-BED": ["baselines.json", "patients/B-BED.jsonl.gz"],
    }
    recorded = a["P2-recorded"]
    assert set(recorded) == MAIN_SUMMARY
    assert recorded["config"] == {"mode": "recorded", "age_sex": False, "lam": 1.0, "tau": 0.99}
    assert (recorded["inputs"], recorded["hierarchy"], recorded["answers"]) == (
        ["initial_evidence"],
        "none",
        "recorded",
    )
    assert (recorded["n"], recorded["failures"]) == (N_VALIDATION, 0)
    assert set(recorded["budgets"]) == {"5", "10", "15", "20", "30"} and len(recorded["curve"]["gtpa1"]) == 31
    assert {"low", "high"} <= set(recorded["stop"]["il"]) and {"low", "high"} <= set(recorded["stop"]["gtpa1"])
    assert list(recorded["tau_sensitivity"]) == ["0.9", "0.95", "0.99", "0.999"]
    assert recorded["tau_sensitivity"]["0.99"] == recorded["stop"]
    # Its stop is that of the fixed P2-recorded configuration run on its own, which the patient-level file holds.
    kb, patients = load_kb(repo), load_validation(repo)
    counts = count_frequencies(kb, iter_train(repo))
    alone = ScoreTask(p2_fixed(P2_RECORDED).policy(kb, counts, ()), kb, ())(patients)
    assert recorded["stop"]["gtpa1"]["estimate"] == float(alone.column("gtpa1").mean())
    assert recorded["stop"]["il"]["mean"] == float(alone.il.mean())
    records = _patient_records(dev_out, "P2-recorded")
    assert [r["row"] for r in records] == list(range(N_VALIDATION))
    assert [r["il"] for r in records] == alone.il.tolist()
    assert [r["asked"] for r in records] == [list(q) for q in alone.questions]
    # Whether an evidence is recorded changes what the policy asks.
    value = _patient_records(dev_out, "P2-value")
    assert any(r["asked"] != v["asked"] for r, v in zip(records, value, strict=True))


def test_the_answer_rule_comparisons_put_the_minuend_first(dev_out: Path) -> None:
    a = _json(dev_out / "answer_rule.json")
    comparisons = a["comparisons"]
    assert list(comparisons) == ["P2-recorded - P2-value", "B-BED - P2-recorded"]
    assert list(run_dev.ANSWER_RULE_LADDER) == [("P2-recorded", "P2-value"), ("B-BED", "P2-recorded")]
    for (this, other), factor in run_dev.ANSWER_RULE_LADDER.items():
        rung = comparisons[f"{this} - {other}"]
        assert (rung["this"], rung["other"], rung["factor"]) == (this, other, factor)
        thresholds = {"tau_sensitivity"} if this == "P2-recorded" else set()
        assert set(rung) == {"this", "other", "factor", *POINTS, *thresholds}
        for point in POINTS:
            section = rung[point]
            mine, theirs = (a[n]["stop"] if point == "stop" else a[n]["budgets"][point] for n in (this, other))
            difference = mine["gtpa1"]["estimate"] - theirs["gtpa1"]["estimate"]
            assert section["gtpa1_diff"] == pytest.approx(difference, abs=1e-12)
            interval = section["gtpa1_diff_interval"]
            assert interval["estimate"] == section["gtpa1_diff"]
            assert interval["low"] <= interval["estimate"] <= interval["high"]
            assert set(section["mcnemar"]) == {"metric", "only_this", "only_other", "p_value"}
            assert section["mcnemar"]["metric"] == "gtpa1_fixed"
            assert ("il_diff" in section) == (point == "stop")
        il = rung["stop"]["il_diff"]
        assert il["estimate"] == pytest.approx(
            a[this]["stop"]["il"]["mean"] - a[other]["stop"]["il"]["mean"], abs=1e-12
        )
        assert il["low"] <= il["estimate"] <= il["high"]
        # The two policies differ on the synthetic patients, so a subtraction the other way round would show.
        assert il["estimate"] != 0
    first = comparisons["P2-recorded - P2-value"]
    assert list(first["tau_sensitivity"]) == ["0.9", "0.95", "0.99", "0.999"]
    assert first["tau_sensitivity"]["0.99"] == first["stop"]
    for key, at_tau in first["tau_sensitivity"].items():
        assert set(at_tau) == set(first["stop"])
        mine, theirs = a["P2-recorded"]["tau_sensitivity"][key], a["P2-value"]["tau_sensitivity"][key]
        assert at_tau["il_diff"]["estimate"] == pytest.approx(mine["il"]["mean"] - theirs["il"]["mean"], abs=1e-12)
        difference = mine["gtpa1"]["estimate"] - theirs["gtpa1"]["estimate"]
        assert at_tau["gtpa1_diff"] == pytest.approx(difference, abs=1e-12)


def test_the_recorded_variant_stays_out_of_the_outputs_of_the_earlier_stages(dev_out: Path) -> None:
    assert "P2-recorded" not in _json(dev_out / "main.json")
    assert list(_json(dev_out / run_dev.SELECTED)["P2"]) == [v.name for v in P2_VARIANTS]
    assert all(r["policy"] != "P2-recorded" and r["config"].get("mode") != "recorded" for r in _tried(dev_out))
    for name in ("main.json", "ablations.json", "baselines.json", "hierarchy.json", "properties.json"):
        assert "P2-recorded" not in (dev_out / name).read_text(), name
    # LADDER's comparison of B-BED stays with P2-value in baselines.json.
    assert list(_json(dev_out / "baselines.json")["B-BED"]["ladder"]) == ["B-BED - P2-value"]


def test_answer_rule_writes_its_two_files_and_nothing_else(
    repo: Path, dev_out: Path, analysis_out: Path, bed_stats: None
) -> None:
    before = _snapshot(analysis_out)
    run_dev.run_dev(("answer_rule",), repo, analysis_out, workers=1)
    after = _snapshot(analysis_out)
    assert set(after) - set(before) == ANSWER_RULE_FILES
    assert all(after[name] == data for name, data in before.items())
    # Run on its own, the stage gives what it gives after the other stages of one run.
    mine, theirs = _json(analysis_out / "answer_rule.json"), _json(dev_out / "answer_rule.json")
    _assert_reads_the_stats_file_and_every_train_row(mine.pop("provenance"))
    del theirs["provenance"]
    assert mine == theirs
    patients = "patients/P2-recorded.jsonl.gz"
    assert after[patients] == (dev_out / patients).read_bytes()


def _other_il(data: Any) -> None:
    data["P2-value"]["stop"]["il"]["mean"] += 1e-9


def _other_gtpa1(data: Any) -> None:
    data["P2-value"]["stop"]["gtpa1"]["estimate"] -= 1 / N_VALIDATION


def _other_bed_gtpa1(data: Any) -> None:
    data["B-BED"]["stats_file"]["stop"]["gtpa1"]["estimate"] -= 1 / N_VALIDATION


def _other_bed_rows(data: Any) -> None:
    data["B-BED"]["stats_file"]["n"] += 1


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        (lambda out: _edit(out / "main.json", _other_il), RuntimeError, "P2-value: the stop of this run is not the"),
        (lambda out: _edit(out / "main.json", _other_gtpa1), RuntimeError, "P2-value: the stop of this run is not the"),
        (
            lambda out: _edit(out / "baselines.json", _other_bed_gtpa1),
            RuntimeError,
            "B-BED: the stop of this run is not the one baselines.json records",
        ),
        (
            lambda out: _edit_first_patient(out / "patients/P2-value.jsonl.gz"),
            RuntimeError,
            "P2-value: a patient of this run differs from patients/P2-value.jsonl.gz",
        ),
        (
            lambda out: _edit_first_patient(out / "patients/B-BED.jsonl.gz"),
            RuntimeError,
            "B-BED: a patient of this run differs from patients/B-BED.jsonl.gz",
        ),
        (lambda out: (out / "baselines.json").unlink(), FileNotFoundError, "baselines.json does not exist"),
        (lambda out: (out / "patients/B-BED.jsonl.gz").unlink(), FileNotFoundError, "B-BED.jsonl.gz does not exist"),
        (
            lambda out: _edit(out / "baselines.json", _other_bed_rows),
            ValueError,
            "baselines.json records a run over 46 patients, not over these 45",
        ),
    ],
)
def test_answer_rule_stops_before_writing_when_a_policy_run_again_is_not_its_record(
    repo: Path,
    analysis_out: Path,
    bed_stats: None,
    change: Callable[[Path], None],
    error: type[Exception],
    message: str,
) -> None:
    """보완 4 §2.4: the P2-value and B-BED runs of the stage must be those the development run recorded."""
    change(analysis_out)
    before = _snapshot(analysis_out)
    with pytest.raises(error, match=re.escape(message)):
        run_dev.run_dev(("answer_rule",), repo, analysis_out, workers=1)
    assert _snapshot(analysis_out) == before


def test_a_limited_run_is_checked_against_the_records_over_the_same_rows_only(
    repo: Path, dev_out: Path, tmp_path: Path, bed_stats: None
) -> None:
    run_dev.run_dev(("grid",), repo, tmp_path, workers=1, limit=10)
    # Records of a run over other rows are nothing to check a limited run against.
    for name in ("baselines.json", "properties.json"):
        shutil.copyfile(dev_out / name, tmp_path / name)
    run_dev.run_dev(("answer_rule", "uniqueness"), repo, tmp_path, workers=1, limit=10)
    a = _json(tmp_path / "answer_rule.json")
    assert (a["limit"], a["P2-recorded"]["n"], a["B-BED"]["n"]) == (10, 10, 10)
    assert a["reproduces"] == {"P2-value": ["main.json", "patients/P2-value.jsonl.gz"], "B-BED": []}
    assert a["P2-value"]["stop"] == _json(tmp_path / "main.json")["P2-value"]["stop"]
    u = _json(tmp_path / "uniqueness.json")
    assert (u["limit"], u["n"]) == (10, 10)
    main_policies = {name: ["main.json"] for name in ("P1", "P2-binary", "P2-value")}
    found = {**main_policies, "P2-recorded": ["answer_rule.json"], "B-BED": []}
    assert u["reproduces"] == {"strata": [], **found}
    assert u["strata"]["unique"]["n"] + u["strata"]["not_unique"]["n"] == 10
    run_dev.run_dev(("properties",), repo, tmp_path, workers=1, limit=10)
    run_dev.run_dev(("uniqueness",), repo, tmp_path, workers=1, limit=10)
    assert _json(tmp_path / "uniqueness.json")["reproduces"] == {"strata": ["properties.json"], **found}
    # A record over the same rows is checked in a limited run too.
    _edit(tmp_path / "main.json", _other_il)
    for stage in ("answer_rule", "uniqueness"):
        with pytest.raises(RuntimeError, match=r"P2-value: the stop of this run is not the one main\.json records"):
            run_dev.run_dev((stage,), repo, tmp_path, workers=1, limit=10)


def test_the_stages_of_the_fourth_revision_need_the_grid_selection(repo: Path, tmp_path: Path) -> None:
    for stage in ("answer_rule", "uniqueness"):
        with pytest.raises(FileNotFoundError, match="--stage grid"):
            run_dev.run_dev((stage,), repo, tmp_path, workers=1)
    assert not list(tmp_path.iterdir())


def test_uniqueness_splits_every_policy_by_the_list_match(dev_out: Path) -> None:
    u = _json(dev_out / "uniqueness.json")
    assert set(u) == {"limit", "definitions", "n", "strata", "policies", "reproduces", "provenance"}
    assert set(u["definitions"]) == {"strata", "policies", "lost_score", "question_limit", "reproduces"}
    assert set(u["provenance"]) == PROVENANCE and u["provenance"]["stage"] == "uniqueness"
    properties = _json(dev_out / "properties.json")
    assert (u["limit"], u["n"]) == (None, N_VALIDATION)
    assert u["reproduces"] == UNIQUENESS_REPRODUCES
    assert u["strata"]["unique"] == {"n": properties["unique"]["count"], "share": properties["unique"]["share"]}
    assert u["strata"]["unique"]["n"] + u["strata"]["not_unique"]["n"] == N_VALIDATION
    assert u["strata"]["unique"]["share"] + u["strata"]["not_unique"]["share"] == pytest.approx(1.0)
    assert tuple(u["policies"]) == UNIQUENESS_POLICIES
    main_json = _json(dev_out / "main.json")
    stops = {name: main_json[name]["stop"] for name in ("P1", "P2-binary", "P2-value")}
    stops["P2-recorded"] = _json(dev_out / "answer_rule.json")["P2-recorded"]["stop"]
    stops["B-BED"] = _json(dev_out / "baselines.json")["B-BED"]["stats_file"]["stop"]
    for name, entry in u["policies"].items():
        assert set(entry) - {"config"} == {"failures", "all", "strata", "lost_score", "question_limit"}
        assert ("config" in entry) == (name != "B-BED")
        # At its stop each policy is the run the other outputs summarize.
        everyone, stop = entry["all"], stops[name]
        assert (everyone["n"], entry["failures"]) == (N_VALIDATION, 0)
        assert (everyone["gtpa1"], everyone["il_mean"]) == (stop["gtpa1"], stop["il"]["mean"])
        assert [entry["strata"][k]["n"] for k in ("unique", "not_unique")] == [
            u["strata"][k]["n"] for k in ("unique", "not_unique")
        ]
        records = _patient_records(dev_out, name)
        reached = [r["gtpa1"] for r in records if r["il"] == MAX_IL]
        earlier = [r["gtpa1"] for r in records if r["il"] < MAX_IL]
        for group, scores in (("reached", reached), ("stopped_earlier", earlier)):
            got = entry["question_limit"][group]
            assert set(got) == {"n", "gtpa1"} and got["n"] == len(scores)
            if scores:
                assert got["gtpa1"]["estimate"] == pytest.approx(sum(scores) / len(scores), abs=1e-6)
                assert got["gtpa1"]["low"] <= got["gtpa1"]["estimate"] <= got["gtpa1"]["high"]
            else:
                assert got["gtpa1"] is None
        assert entry["lost_score"]["total"] == pytest.approx(sum(1 - r["gtpa1"] for r in records), abs=1e-4)
    assert main_json["P1"]["config"] == u["policies"]["P1"]["config"]
    assert u["policies"]["P2-recorded"]["config"] == p2_fixed(P2_RECORDED).as_dict()
    # Every synthetic validation patient matches one list only: the other stratum is empty and says so.
    assert u["strata"]["not_unique"] == {"n": 0, "share": 0.0}
    for entry in u["policies"].values():
        assert entry["strata"]["not_unique"] == {"n": 0, "gtpa1": None, "il_mean": None}
        assert entry["lost_score"]["not_unique"] == 0.0
        assert (entry["lost_score"]["not_unique_share"] is None) == (entry["lost_score"]["total"] == 0)
    assert u["policies"]["P1"]["lost_score"] == {"total": 0.0, "not_unique": 0.0, "not_unique_share": None}
    assert u["policies"]["P2-recorded"]["lost_score"]["not_unique_share"] == 0.0


def _two_list_rows() -> list[dict[str, Any]]:
    """Nine patients whose positive evidences are on the lists of two synthetic diseases: for each pair of
    neighbouring diseases, one patient of each with two of the symptoms the pair shares; and one more of the second
    disease, whose record also holds the default value of an evidence only that disease lists."""
    rows = []
    for k in range(4):
        shared = SYNTH_SYMPTOMS[6 * k + 6 : 6 * k + 8]
        for name in SYNTH_DISEASES[k : k + 2]:
            rows.append(
                {
                    "age": 40,
                    "sex": "F",
                    "pathology": name,
                    "evidences": [*shared, "trav1_@_N"],
                    "initial": shared[0],
                    "differential": [(name, 1.0)],
                }
            )
    rows.append({**rows[1], "evidences": [*SYNTH_SYMPTOMS[6:8], "douleur_intens_@_0", "trav1_@_N"]})
    return rows


@pytest.fixture(scope="module")
def mixed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, Path]]:
    """A synthetic repository whose validation patients are in both strata of 보완 4 §3 and whose BED stats file
    holds the statistics of every other train row, so B-BED from the stats file is not B-BED from the counted train
    split; with the outputs of a development run of every stage over it."""
    root = tmp_path_factory.mktemp("mixed")
    make_repo(root)
    write_csv(root / VALIDATION_CSV, [*synth_rows(30, seed=4), *_two_list_rows()])
    kb = load_kb(root)
    half = itertools.islice(iter_train(root), 0, None, 2)
    knowledge = bed.count_knowledge(kb, half, bed.text_default_codes(synth_kb_json()[1]))
    (root / bed.BED_STATS).write_text(json.dumps(bed_stats_json(kb, knowledge)))
    out = root / "out"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bed, "BED_STATS_SHA256", bed_stats_sha256(root))
        run_dev.run_dev(STAGES, root, out, workers=1)
        yield root, out


def test_uniqueness_reports_each_stratum_of_a_mixed_set(mixed: tuple[Path, Path]) -> None:
    root, out = mixed
    kb, patients = load_kb(root), load_validation(root)
    counts = count_frequencies(kb, iter_train(root))
    n = len(patients)
    matches = [kb.diseases_containing(positive_codes(kb, p)) for p in patients]
    unique = np.array(
        [m.sum() == 1 and m[kb.disease_index[p.pathology]] for m, p in zip(matches, patients, strict=True)]
    )
    # The nine hand-made patients match two lists each; the drawn ones may match one or more.
    assert all(m.sum() == 2 for m in matches[-9:]) and not unique[-9:].any()
    assert 9 <= int((~unique).sum()) < n
    assert [list_match_is_unique(kb, p) for p in patients] == unique.tolist()
    audit = support_audit(kb, patients)
    assert audit.unique == int(unique.sum()) == _json(out / "properties.json")["unique"]["count"]
    # A default value is not positive: the last patient's recorded codes fit one list, its positive evidences two.
    assert audit.unique_recorded == audit.unique + 1
    u = _json(out / "uniqueness.json")
    assert u["strata"] == {
        "unique": {"n": int(unique.sum()), "share": int(unique.sum()) / n},
        "not_unique": {"n": int((~unique).sum()), "share": int((~unique).sum()) / n},
    }
    assert (u["n"], u["reproduces"]) == (n, UNIQUENESS_REPRODUCES)
    p1 = P1Config.from_dict(_json(out / run_dev.SELECTED)["P1"]["config"])
    binary, value = (p2_fixed(P2Variant(mode, age_sex=False)) for mode in (AnswerMode.BINARY, AnswerMode.VALUE))
    policies: dict[str, Policy] = {
        "P1": p1.policy(kb, ()),
        "P2-binary": binary.policy(kb, counts, ()),
        "P2-value": value.policy(kb, counts, ()),
        "P2-recorded": p2_fixed(P2_RECORDED).policy(kb, counts, ()),
        "B-BED": bed.BedPolicy(bed.load_knowledge(kb, root)),
    }
    assert tuple(u["policies"]) == tuple(policies)
    for name, policy in policies.items():
        scored = ScoreTask(policy, kb, ())(patients)
        assert not scored.failed.any()
        gtpa1, il = scored.column("gtpa1"), scored.il
        entry = u["policies"][name]

        def expected(mask: np.ndarray, with_il: bool = True, scored: Scored = scored) -> Any:
            """The figures of a group: those of the stop summary of the run over the group's patients."""
            k = int(mask.sum())
            out: dict[str, Any] = {"n": k, "gtpa1": None}
            stop = report(scored.select(scored.rows[mask].tolist()), bootstrap=False)["stop"] if k else None
            if stop is not None:
                assert stop["gtpa1"]["estimate"] == pytest.approx(float(scored.column("gtpa1")[mask].mean()), abs=1e-12)
                out["gtpa1"] = stop["gtpa1"]
            if with_il:
                out["il_mean"] = None if stop is None else stop["il"]["mean"]
            return out

        assert entry["all"] == expected(np.ones(n, dtype=bool))
        assert entry["strata"] == {"unique": expected(unique), "not_unique": expected(~unique)}
        lost = 1.0 - gtpa1
        assert entry["lost_score"]["total"] == pytest.approx(float(lost.sum()), abs=1e-12)
        assert entry["lost_score"]["not_unique"] == pytest.approx(float(lost[~unique].sum()), abs=1e-12)
        if lost.sum() > 0:
            share = float(lost[~unique].sum() / lost.sum())
            assert entry["lost_score"]["not_unique_share"] == pytest.approx(share, abs=1e-12)
        else:
            assert entry["lost_score"]["not_unique_share"] is None
        assert entry["question_limit"] == {
            "reached": expected(il == MAX_IL, with_il=False),
            "stopped_earlier": expected(il < MAX_IL, with_il=False),
        }
    # Two patients with the same answers and different diseases cannot both be ranked first, so matching the lists
    # loses at least half of each of the four hand-made pairs, all of them in the stratum without a unique match.
    p1_entry = u["policies"]["P1"]
    assert p1_entry["strata"]["not_unique"]["gtpa1"]["estimate"] < p1_entry["strata"]["unique"]["gtpa1"]["estimate"]
    assert p1_entry["lost_score"]["total"] >= p1_entry["lost_score"]["not_unique"] >= 4.0
    assert p1_entry["lost_score"]["not_unique_share"] >= 0.5


def test_the_stages_of_the_fourth_revision_run_b_bed_from_the_stats_file(mixed: tuple[Path, Path]) -> None:
    """보완 4 §2.3, §3.2: B-BED is the main run of baselines.json, not the one from the counted train split."""
    out = mixed[1]
    recorded = _json(out / "baselines.json")["B-BED"]
    stats_file, counted = (recorded[source]["stop"] for source in ("stats_file", "counted"))
    assert (stats_file["gtpa1"], stats_file["il"]["mean"]) != (counted["gtpa1"], counted["il"]["mean"])
    a = _json(out / "answer_rule.json")
    assert a["B-BED"]["stop"] == stats_file
    assert a["reproduces"]["B-BED"] == ["baselines.json", "patients/B-BED.jsonl.gz"]
    u = _json(out / "uniqueness.json")
    everyone = u["policies"]["B-BED"]["all"]
    assert (everyone["gtpa1"], everyone["il_mean"]) == (stats_file["gtpa1"], stats_file["il"]["mean"])
    assert u["reproduces"]["B-BED"] == ["baselines.json"]


def test_uniqueness_stops_when_its_strata_are_not_those_of_the_properties(
    repo: Path, dev_out: Path, uniqueness_out: Path, bed_stats: None
) -> None:
    def fewer(data: Any) -> None:
        data["unique"] = {"count": N_VALIDATION - 1, "share": (N_VALIDATION - 1) / N_VALIDATION}

    _edit(uniqueness_out / "properties.json", fewer)
    before = _snapshot(uniqueness_out)
    with pytest.raises(RuntimeError, match=r"the unique stratum holds 45 of 45 patients, properties\.json records 44"):
        run_dev.run_dev(("uniqueness",), repo, uniqueness_out, workers=1)
    assert _snapshot(uniqueness_out) == before
    (uniqueness_out / "properties.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"properties\.json does not exist"):
        run_dev.run_dev(("uniqueness",), repo, uniqueness_out, workers=1)
    assert not (uniqueness_out / "uniqueness.json").exists()
    # With the records it checks against, the stage writes uniqueness.json and nothing else.
    shutil.copyfile(dev_out / "properties.json", uniqueness_out / "properties.json")
    before = _snapshot(uniqueness_out)
    run_dev.run_dev(("uniqueness",), repo, uniqueness_out, workers=1)
    after = _snapshot(uniqueness_out)
    assert set(after) - set(before) == {"uniqueness.json"}
    assert all(after[name] == data for name, data in before.items())
    mine, theirs = _json(uniqueness_out / "uniqueness.json"), _json(dev_out / "uniqueness.json")
    _assert_reads_the_stats_file_and_every_train_row(mine.pop("provenance"))
    del theirs["provenance"]
    assert mine == theirs


def _other_failures(data: Any) -> None:
    data["B-BED"]["stats_file"]["failures"] += 1


def _other_recorded_il(data: Any) -> None:
    data["P2-recorded"]["stop"]["il"]["mean"] += 1e-9


def _other_p1_bound(data: Any) -> None:
    data["P1"]["stop"]["gtpa1"]["low"] -= 1e-9


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        (
            lambda out: _edit(out / "main.json", _other_p1_bound),
            RuntimeError,
            "P1: the stop of this run is not the one",
        ),
        (
            lambda out: _edit(out / "main.json", _other_il),
            RuntimeError,
            "P2-value: the stop of this run is not the one",
        ),
        (
            lambda out: _edit(out / "answer_rule.json", _other_recorded_il),
            RuntimeError,
            "P2-recorded: the stop of this run is not the one answer_rule.json records",
        ),
        (
            lambda out: _edit(out / "baselines.json", _other_bed_gtpa1),
            RuntimeError,
            "B-BED: the stop of this run is not the one baselines.json records",
        ),
        (
            lambda out: _edit(out / "baselines.json", _other_failures),
            RuntimeError,
            "B-BED: the stop of this run is not the one baselines.json records",
        ),
        (lambda out: (out / "answer_rule.json").unlink(), FileNotFoundError, "answer_rule.json does not exist"),
        (lambda out: (out / "main.json").unlink(), FileNotFoundError, "main.json does not exist"),
        (
            lambda out: _edit(out / "baselines.json", _other_bed_rows),
            ValueError,
            "baselines.json records a run over 46 patients, not over these 45",
        ),
    ],
)
def test_uniqueness_stops_before_writing_when_a_policy_run_again_is_not_its_record(
    repo: Path,
    uniqueness_out: Path,
    bed_stats: None,
    change: Callable[[Path], None],
    error: type[Exception],
    message: str,
) -> None:
    """Each policy of the uniqueness stage must be the run the stage that reports it recorded."""
    change(uniqueness_out)
    before = _snapshot(uniqueness_out)
    with pytest.raises(error, match=re.escape(message)):
        run_dev.run_dev(("uniqueness",), repo, uniqueness_out, workers=1)
    assert _snapshot(uniqueness_out) == before


def test_strata_are_refused_for_a_run_over_other_patients(repo: Path) -> None:
    kb, patients = load_kb(repo), load_validation(repo, 6)
    scored = ScoreTask(P1Config(0.5, 1e-3, TAU).policy(kb, ()), kb, ())(patients)
    rows = [p.row for p in patients]
    unique = np.array([True, False] * 3)
    entry = run_dev._by_stratum(scored, rows, unique)
    assert [entry["strata"][k]["n"] for k in ("unique", "not_unique")] == [3, 3]
    # The same patients in another order, fewer rows, and a mask of another length.
    for other_rows, mask in (([*rows[1:], rows[0]], unique), (rows[:5], unique[:5]), (rows, unique[:5])):
        with pytest.raises(ValueError, match="the strata cover other patients"):
            run_dev._by_stratum(scored, other_rows, mask)


def test_the_figures_over_every_patient_are_the_stop_summary_to_the_last_bit() -> None:
    """The uniqueness stage compares `all` with the stop another stage recorded by ==: both average the same way."""
    n, m, rng = 3000, len(METRICS), np.random.default_rng(0)
    scored = Scored(
        diseases=("A", "B"),
        rows=np.arange(n),
        gold=rng.integers(0, 2, n),
        il=rng.integers(1, MAX_IL + 1, n),
        stop=rng.random((n, m)),
        budgets=(),
        at_budget=np.zeros((n, 0, m)),
        peak=np.zeros((n, 0)),
        max_gain=np.zeros((n, 0)),
        questions=((),) * n,
        failed=np.zeros(n, dtype=np.bool_),
        errors=(),
        top=None,
        top_p=None,
    )
    entry = run_dev._by_stratum(scored, list(range(n)), rng.random(n) < 0.9)
    stop = report(scored, bootstrap=False)["stop"]
    assert (entry["all"]["gtpa1"], entry["all"]["il_mean"]) == (stop["gtpa1"], stop["il"]["mean"])
    # The column averaged on its own sums in another order and differs in its last bits.
    assert float(scored.column("gtpa1").mean()) != stop["gtpa1"]["estimate"]


def test_properties(dev_out: Path) -> None:
    properties = _json(dev_out / "properties.json")
    assert (properties["n"], properties["outside_support"]) == (N_VALIDATION, 0)
    assert set(properties) == {
        "limit",
        "n",
        "outside_support",
        "unique",
        "unique_recorded",
        "definitions",
        "provenance",
    }
    assert properties["provenance"]["inputs"]["data/ddxplus/release_validate_patients.csv"]


def test_results_do_not_depend_on_the_number_of_workers(
    repo: Path, dev_out: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bed, "BED_STATS_SHA256", bed_stats_sha256(repo))
    run_dev.run_dev(STAGES, repo, tmp_path, workers=2)

    def strip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in r.items() if k not in ("run", "commit")} for r in rows]

    assert strip(_tried(tmp_path)) == strip(_tried(dev_out))
    written = sorted(f.name for f in (tmp_path / "patients").iterdir())
    assert written == sorted(f.name for f in (dev_out / "patients").iterdir())
    for name in written:
        assert (tmp_path / "patients" / name).read_bytes() == (dev_out / "patients" / name).read_bytes()
    for name in ("ablations.json", "baselines.json", "hierarchy.json", "answer_rule.json", "uniqueness.json"):
        mine, theirs = _json(tmp_path / name), _json(dev_out / name)
        del mine["provenance"], theirs["provenance"]
        assert mine == theirs


def test_help_opens_with_a_whole_sentence(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "No sealed set is read. options:" in " ".join(capsys.readouterr().out.split())


def test_a_limited_run_writes_beside_the_development_outputs(repo: Path) -> None:
    main(["--stage", "properties", "--limit", "10", "--root", str(repo)])
    out = repo / "paper1/outputs/v3/dev_limit10/properties.json"
    assert (_json(out)["n"], _json(out)["limit"]) == (10, 10)
    assert run_dev.OUTPUT_DIR == "paper1/outputs/v3/dev"


def test_ablations_need_the_grid_selection(repo: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--stage grid"):
        run_dev.run_dev(("ablations",), repo, tmp_path, workers=1)


def test_ablations_refuse_a_selection_made_on_other_rows(repo: Path, dev_out: Path, tmp_path: Path) -> None:
    selected = _json(dev_out / run_dev.SELECTED)
    (tmp_path / run_dev.SELECTED).write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="selected on 45 rows"):
        run_dev.run_dev(("ablations",), repo, tmp_path, workers=1, limit=10)
    (tmp_path / run_dev.SELECTED).write_text(json.dumps({**selected, "n": 10, "limit": 10}))
    with pytest.raises(ValueError, match="limit 10"):
        run_dev.run_dev(("ablations",), repo, tmp_path, workers=1)


def test_ablations_refuse_a_selection_with_other_fixed_settings(repo: Path, dev_out: Path, tmp_path: Path) -> None:
    selected = _json(dev_out / run_dev.SELECTED)
    selected["P2"]["P2-value"]["config"]["tau"] = 0.999
    (tmp_path / run_dev.SELECTED).write_text(json.dumps(selected))
    with pytest.raises(ValueError, match=r"P2-value stops at tau 0\.999"):
        run_dev.run_dev(("ablations",), repo, tmp_path, workers=1)


def test_frequencies_come_from_every_train_row(dev_out: Path) -> None:
    for name in (
        *("selected.json", "main.json", "ablations.json", "baselines.json", "hierarchy.json"),
        *("answer_rule.json", "uniqueness.json"),
    ):
        meta = _json(dev_out / name)["provenance"]
        assert meta["train_rows"] == 150
        assert meta["inputs"]["data/ddxplus/release_train_patients.csv"]
        assert set(meta) == PROVENANCE
        assert not any("manifests" in path for path in meta["inputs"])


def test_development_never_reads_the_test_split(repo: Path, dev_out: Path) -> None:
    assert (repo / TEST_CSV).is_dir()
    written = {f.name for f in dev_out.iterdir()}
    assert written == {
        *(f"{stage}.json" for stage in ("main", "ablations", "baselines", "hierarchy", "properties")),
        *(f"{stage}.json" for stage in ("answer_rule", "uniqueness")),
        "selected.json",
        "configs_tried.jsonl",
        "patients",
    }
    for name in written - {"configs_tried.jsonl", "patients"}:
        assert not any("test" in path for path in _json(dev_out / name)["provenance"]["inputs"]), name


@pytest.mark.parametrize(
    "inside",
    [
        "paper1/outputs/dev",
        "paper1/outputs/eval",
        "paper1/outputs/dev/patients",
        "paper1/outputs/v2",
        "paper1/outputs/v2/dev",
        "paper1/outputs/v2/dev_limit3000",
        "paper1/outputs/v2/eval/again",
    ],
)
def test_development_never_writes_into_the_outputs_of_an_earlier_run(repo: Path, inside: str) -> None:
    with pytest.raises(ValueError, match="outputs of an earlier run"):
        run_dev.run_dev(STAGES, repo, repo / inside, workers=1)
    with pytest.raises(ValueError, match="outputs of an earlier run"):
        main(["--stage", "properties", "--root", str(repo), "--out", str(repo / inside)])
    assert not (repo / inside).exists()


def test_the_analyses_of_the_fourth_revision_run_after_the_three_earlier_stages() -> None:
    assert STAGES == ("grid", "ablations", "properties", "answer_rule", "uniqueness") == tuple(run_dev.STAGE_RUNNERS)


def test_development_refuses_an_unsealed_environment(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_dev, "unseal_requested", lambda: True)
    with pytest.raises(RuntimeError, match="DDXKB_UNSEAL"):
        run_dev.run_dev(STAGES, repo, tmp_path, workers=1)


def test_provenance_records_the_git_state_at_the_start_of_the_run(
    toy_kb: KB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment, "git_head", lambda root: "start")
    monkeypatch.setattr(experiment, "git_changes", lambda root, *paths: [f"?? {paths[0]}/x"])
    ctx = experiment.Context(root=tmp_path, kb=toy_kb, patients=[], split="validate", out=tmp_path)
    monkeypatch.setattr(experiment, "git_head", lambda root: "end")
    monkeypatch.setattr(experiment, "git_changes", lambda root, *paths: [])
    meta = experiment.provenance(ctx, "grid", "t", 0.0, 0)
    assert (meta["commit"], meta["commit_at_end"]) == ("start", "end")
    assert (meta["paper1_changes"], meta["design_changes"]) == (["?? paper1/x"], ["?? docs/design/x"])
