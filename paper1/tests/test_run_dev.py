from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from ddxkb import experiment, run_dev
from ddxkb.data import TEST_CSV, iter_train, load_kb, load_validation
from ddxkb.evaluation import ScoreTask
from ddxkb.frequencies import count_frequencies
from ddxkb.kb import KB
from ddxkb.policies import bed
from ddxkb.policies.baselines import random_questions, train_frequency_questions
from ddxkb.policies.bayes import BayesPolicy
from ddxkb.protocol import (
    LADDER,
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
from ddxkb.simulator import AnswerMode
from helpers import bed_stats_sha256, make_repo

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
    run_dev.run_dev(("grid", "ablations"), repo, tmp_path, workers=2)

    def strip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in r.items() if k not in ("run", "commit")} for r in rows]

    assert strip(_tried(tmp_path)) == strip(_tried(dev_out))
    written = sorted(f.name for f in (tmp_path / "patients").iterdir())
    assert written == sorted(f.name for f in (dev_out / "patients").iterdir())
    for name in written:
        assert (tmp_path / "patients" / name).read_bytes() == (dev_out / "patients" / name).read_bytes()
    for name in ("ablations.json", "baselines.json", "hierarchy.json"):
        mine, theirs = _json(tmp_path / name), _json(dev_out / name)
        del mine["provenance"], theirs["provenance"]
        assert mine == theirs


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
    for name in ("selected.json", "main.json", "ablations.json", "baselines.json", "hierarchy.json"):
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


def test_the_stages_are_grid_ablations_and_properties() -> None:
    assert STAGES == ("grid", "ablations", "properties") == tuple(run_dev.STAGE_RUNNERS)


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
