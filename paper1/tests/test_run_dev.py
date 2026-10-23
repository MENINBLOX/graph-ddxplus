from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from ddxkb import run_dev
from ddxkb.data import TEST_CSV, iter_train, load_kb, load_validation
from ddxkb.evaluation import ScoreTask
from ddxkb.frequencies import AGE_BIN_EDGES, count_frequencies
from ddxkb.policies import bed
from ddxkb.protocol import (
    FIRST_AGE_BIN_EDGES,
    FIRST_MAX_IL,
    P2_VARIANTS,
    V1_TIE_SEEDS,
    P1Config,
    P2Config,
    p1_grid,
    p1_wider_grid,
    p2_grid,
    selection_key,
)
from ddxkb.run_dev import STAGES, main
from helpers import bed_stats_sha256, make_repo

N_VALIDATION = 45


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


def test_grid_logs_every_candidate_in_listed_order(dev_out: Path) -> None:
    tried = [r for r in _tried(dev_out) if r["grid"] == "original"]
    assert len(tried) == 27 + 4 * 8
    p1 = [r for r in tried if r["policy"] == "P1"]
    assert [r["order"] for r in p1] == list(range(27))
    assert [r["config"] for r in p1] == [c.as_dict() for c in p1_grid()]
    for variant in P2_VARIANTS:
        rows = [r for r in tried if r["policy"] == variant.name]
        assert [r["config"] for r in rows] == [c.as_dict() for c in p2_grid(variant)]
    for r in tried:
        stop = r["stop"]
        assert r["n"] == N_VALIDATION and r["limit"] is None
        assert r["selection_key"] == list(selection_key(stop["gtpa1"]["estimate"], stop["il"]["mean"], r["order"]))
        assert len(r["curve"]["gtpa1"]) == 31


def test_selection_follows_the_objective(dev_out: Path) -> None:
    tried = [r for r in _tried(dev_out) if r["grid"] == "original"]
    selected = _json(dev_out / run_dev.SELECTED)
    for name, chosen in [("P1", selected["P1"]), *selected["P2"].items()]:
        rows = [r for r in tried if r["policy"] == name]
        best = min(rows, key=lambda r: tuple(r["selection_key"]))
        assert chosen["config"] == best["config"] and chosen["order"] == best["order"]
    assert list(selected["P2"]) == [v.name for v in P2_VARIANTS]


def _check_wider(out: Path) -> dict[str, object]:
    """The wider-grid record of a grid run against its configs_tried lines; returns the boundary record."""
    selected = _json(out / run_dev.SELECTED)["P1"]
    boundary: dict[str, object] = selected["boundary"]
    tried = [r for r in _tried(out) if r["policy"] == "P1"]
    wider_lines = [r for r in tried if r["grid"] == "wider"]
    chosen = P1Config.from_dict(selected["config"])
    if not boundary["ends"]:
        assert boundary["wider_grid"] is None and not wider_lines
        return boundary
    wider = p1_wider_grid(chosen)
    assert [P1Config.from_dict(r["config"]) for r in wider_lines] == [c for c in wider if c not in set(p1_grid())]
    assert [r["order"] for r in wider_lines] == [wider.index(P1Config.from_dict(r["config"])) for r in wider_lines]
    record = boundary["wider_grid"]
    assert isinstance(record, dict) and (record["n"], record["new"]) == (len(wider), len(wider_lines))
    keys = {
        P1Config.from_dict(r["config"]): selection_key(
            r["stop"]["gtpa1"]["estimate"], r["stop"]["il"]["mean"], wider.index(P1Config.from_dict(r["config"]))
        )
        for r in tried
    }
    best = min(keys, key=keys.__getitem__)
    assert record["best"]["config"] == best.as_dict() and record["best_is_selected"] == (best == chosen)
    return boundary


def test_the_wider_grid_is_recorded_only_at_a_boundary(dev_out: Path) -> None:
    _check_wider(dev_out)


def test_a_selection_at_a_boundary_records_the_wider_grid_without_choosing_from_it(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corner = P1Config(0.3, 1e-4, 0.99)
    best = run_dev._best

    def forced(entries: Any) -> Any:
        return next((e for e in entries if e[0] == corner), None) or best(entries)

    monkeypatch.setattr(run_dev, "_best", forced)
    run_dev.run_dev(("grid",), repo, tmp_path, workers=1)
    boundary = _check_wider(tmp_path)
    assert boundary["ends"] == {"alpha": "first", "eps": "last", "tau": "last"}
    record = boundary["wider_grid"]
    assert isinstance(record, dict) and record["new"] == 4 * 4 * 4 - 27
    assert record["values"] == {
        "alpha": [0.1, 0.3, 0.5, 0.7],
        "eps": [1e-2, 1e-3, 1e-4, 1e-5],
        "tau": [0.9, 0.95, 0.99, 0.999],
    }
    assert _json(tmp_path / run_dev.SELECTED)["P1"]["config"] == corner.as_dict()
    # A wider line derived from an original trajectory is the stop of that configuration run on its own.
    held = P1Config(0.5, 1e-3, 0.999)
    line = next(r for r in _tried(tmp_path) if r["grid"] == "wider" and r["config"] == held.as_dict())
    kb = load_kb(repo)
    alone = ScoreTask(held.policy(kb, ()), kb, ())(load_validation(repo))
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


def test_first_protocol_changes_are_the_runs_they_stand_for(repo: Path, dev_out: Path) -> None:
    """The changes derived from the grid stage's runs equal the policies run with those settings on their own."""
    main_json = _json(dev_out / "main.json")
    selected = _json(dev_out / run_dev.SELECTED)
    kb, patients = load_kb(repo), load_validation(repo)
    first_counts = count_frequencies(kb, iter_train(repo, without="H2"))
    full_counts = count_frequencies(kb, iter_train(repo))

    def alone(policy: Any) -> tuple[float, float]:
        s = ScoreTask(policy, kb, ())(patients)
        return float(s.column("gtpa1").mean()), float(s.il.mean())

    def reported(entry: dict[str, Any]) -> tuple[float, float]:
        return entry["stop"]["gtpa1"]["estimate"], entry["stop"]["il"]["mean"]

    p1 = P1Config.from_dict(selected["P1"]["config"])
    changes = main_json["P1"]["first_protocol"]
    assert set(changes) == {"definitions", "cap30"}
    assert reported(changes["cap30"]) == alone(dataclasses.replace(p1.policy(kb, ()), max_il=FIRST_MAX_IL))
    assert set(changes["cap30"]["against"]) == {"P1"}
    for v in P2_VARIANTS:
        cfg = P2Config.from_dict(selected["P2"][v.name]["config"])
        changes = main_json[v.name]["first_protocol"]
        # Without age and sex the age groups change nothing, so the first setting needs only the frequencies.
        first = cfg.policy(kb, first_counts, (), FIRST_AGE_BIN_EDGES if v.age_sex else AGE_BIN_EDGES)
        capped = dataclasses.replace(cfg.policy(kb, full_counts, ()), max_il=FIRST_MAX_IL)
        assert reported(changes["cap30"]) == alone(capped)
        assert reported(changes["train_minus_h2"]) == alone(cfg.policy(kb, first_counts, ()))
        assert reported(changes["first_setting"]) == alone(dataclasses.replace(first, max_il=FIRST_MAX_IL))
        if v.age_sex:
            assert reported(changes["first_knowledge"]) == alone(first)
            assert reported(changes["first_age_groups"]) == alone(cfg.policy(kb, full_counts, (), FIRST_AGE_BIN_EDGES))
        assert set(changes["definitions"]) == set(changes) - {"definitions"}
        assert set(changes["cap30"]["against"]) == {v.name}


def test_ablations_and_baselines_cover_the_design(dev_out: Path) -> None:
    ablations = _json(dev_out / "ablations.json")
    assert set(ablations) >= {"A1", "A2", "A3", "A4", "A5", "A6", "references", "provenance"}
    assert set(ablations["A1"]) == {"both", "positive_only", "negative_only"}
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
    assert set(bed_runs) == {"definitions", "stats_file", "counted"}
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


def test_v1_and_properties(dev_out: Path) -> None:
    v1 = _json(dev_out / "v1_replay.json")
    assert set(v1["runs"]) == {
        "cui/trav1_bug",
        "cui",
        "code/trav1_bug",
        "code",
        "code_no_subattributes/trav1_bug",
        "code_no_subattributes",
    }
    assert set(v1["tie_seeds"]["seeds"]) == {str(s) for s in V1_TIE_SEEDS}
    assert v1["J4"]["observed"] == 100 * v1["runs"]["cui/trav1_bug"]["top1"]["estimate"]
    properties = _json(dev_out / "properties.json")
    assert (properties["n"], properties["outside_support"]) == (N_VALIDATION, 0)
    assert set(properties) >= {"unique", "unique_recorded", "definitions"}
    assert properties["kb_folded_to_cuis"]["edges"] == properties["kb_folded_to_cuis"]["shared"]
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
    out = repo / "paper1/outputs/v2/dev_limit10/properties.json"
    assert (_json(out)["n"], _json(out)["limit"]) == (10, 10)


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


def test_frequencies_come_from_every_train_row(dev_out: Path) -> None:
    # make_repo writes 150 train rows, every sixth of them in H2; every train row is counted, and the first
    # protocol's frequencies of the grid stage's changes count the 125 outside H2.
    for name in ("selected.json", "main.json", "ablations.json", "baselines.json", "hierarchy.json"):
        meta = _json(dev_out / name)["provenance"]
        assert meta["train_rows"] == 150
        assert meta["inputs"]["data/ddxplus/release_train_patients.csv"]
    assert _json(dev_out / "main.json")["provenance"]["first_protocol_train_rows"] == 125


def test_development_never_reads_the_test_split(repo: Path, dev_out: Path) -> None:
    assert (repo / TEST_CSV).is_dir()
    written = {f.name for f in dev_out.iterdir()}
    assert written >= {f"{stage}.json" for stage in ("main", "ablations", "baselines", "hierarchy", "properties")}
    assert written >= {"selected.json", "v1_replay.json", "configs_tried.jsonl"}
    for name in written - {"configs_tried.jsonl", "patients"}:
        assert not any("test" in path for path in _json(dev_out / name)["provenance"]["inputs"]), name


@pytest.mark.parametrize("inside", ["paper1/outputs/dev", "paper1/outputs/eval", "paper1/outputs/dev/patients"])
def test_development_never_writes_into_the_first_runs_outputs(repo: Path, inside: str) -> None:
    with pytest.raises(ValueError, match="first run's outputs"):
        run_dev.run_dev(STAGES, repo, repo / inside, workers=1)
    with pytest.raises(ValueError, match="first run's outputs"):
        main(["--stage", "properties", "--root", str(repo), "--out", str(repo / inside)])
    assert not (repo / inside).exists()


def test_development_refuses_an_unsealed_environment(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_dev, "unseal_requested", lambda: True)
    with pytest.raises(RuntimeError, match="DDXKB_UNSEAL"):
        run_dev.run_dev(STAGES, repo, tmp_path, workers=1)
