from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from ddxkb import run_dev
from ddxkb.protocol import P2_VARIANTS, V1_TIE_SEEDS, p1_grid, p2_grid, selection_key
from ddxkb.run_dev import STAGES, main
from helpers import make_repo

N_VALIDATION = 45


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("repo")
    make_repo(root, n_validation=N_VALIDATION)
    return root


@pytest.fixture(scope="module")
def dev_out(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("dev")
    run_dev.run_dev(STAGES, repo, out, workers=1)
    return out


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _tried(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / run_dev.TRIED).read_text().splitlines()]


def test_grid_logs_every_candidate_in_listed_order(dev_out: Path) -> None:
    tried = _tried(dev_out)
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
        assert len(r["curve"]["gtpa1"]) == 30


def test_selection_follows_the_objective(dev_out: Path) -> None:
    tried = _tried(dev_out)
    selected = _json(dev_out / run_dev.SELECTED)
    for name, chosen in [("P1", selected["P1"]), *selected["P2"].items()]:
        rows = [r for r in tried if r["policy"] == name]
        best = min(rows, key=lambda r: tuple(r["selection_key"]))
        assert chosen["config"] == best["config"] and chosen["order"] == best["order"]
    assert list(selected["P2"]) == [v.name for v in P2_VARIANTS]


def test_main_reruns_the_selected_configurations(dev_out: Path) -> None:
    selected = _json(dev_out / run_dev.SELECTED)
    main_json = _json(dev_out / "main.json")
    assert main_json["P1"]["config"] == selected["P1"]["config"]
    assert main_json["P1"]["stop"]["gtpa1"]["estimate"] == selected["P1"]["gtpa1"]
    assert main_json["P1"]["stop"]["il"]["mean"] == selected["P1"]["il_mean"]
    assert "against" not in main_json["P1"]
    for v in P2_VARIANTS:
        entry = main_json[v.name]
        assert entry["config"] == selected["P2"][v.name]["config"]
        assert entry["stop"]["il"]["mean"] == selected["P2"][v.name]["il_mean"]
        assert set(entry["against"]["P1"]) == {"stop", "5", "10", "15", "20", "29"}
        assert {"low", "high"} <= set(entry["against"]["P1"]["stop"]["il_diff"])
        assert set(entry["budgets"]) == {"5", "10", "15", "20", "29"}
        assert {"low", "high"} <= set(entry["stop"]["ddf1"])
    for name in ("P1", *(v.name for v in P2_VARIANTS)):
        lines = gzip.decompress((dev_out / "patients" / f"{name}.jsonl.gz").read_bytes()).decode().splitlines()
        records = [json.loads(line) for line in lines]
        assert [r["row"] for r in records] == list(range(N_VALIDATION))
        assert sum(r["gtpa1"] for r in records) / N_VALIDATION == pytest.approx(
            main_json[name]["stop"]["gtpa1"]["estimate"]
        )


def test_ablations_and_baselines_cover_the_design(dev_out: Path) -> None:
    ablations = _json(dev_out / "ablations.json")
    assert set(ablations) >= {"A1", "A2", "A3", "A4", "A5", "A6", "references", "provenance"}
    assert set(ablations["A1"]) == {"both", "positive_only", "negative_only"}
    assert set(ablations["A3"]) == {"0.1", "0.01", "0.001", "0.0001", "1e-06", "1e-09"}
    # 30 symptoms with 29 CUIs, douleur and trav1 mapped; e30-e33 and the two pain subattributes have no CUI.
    assert ablations["A4"]["n_units"] == {"code": 38, "cui_v1": 31, "cui_merged": 37, "no_subattributes": 36}
    assert ablations["A4"]["cui_merged"]["failures"] == 0
    assert set(ablations["A6"]) == {"P2-binary in value mode", "P2-binary-agesex in value mode"}
    assert set(ablations["A6"]["P2-binary in value mode"]["against"]) == {"P1", "P2-binary"}
    selected_eps = _json(dev_out / run_dev.SELECTED)["P1"]["config"]["eps"]
    same = ablations["A3"][f"{selected_eps:g}"]["stop"]
    assert same["gtpa1"] == ablations["references"]["P1"]["stop"]["gtpa1"]
    assert same["il"]["mean"] == ablations["references"]["P1"]["stop"]["il"]["mean"]
    baselines = _json(dev_out / "baselines.json")
    assert set(baselines["B0"]) == {"most_frequent", "initial_only"}
    assert set(baselines["B1"]) == {"set_inclusion", "kb_bayes", *(f"naive_bayes/{v.name}" for v in P2_VARIANTS)}
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


def test_results_do_not_depend_on_the_number_of_workers(repo: Path, dev_out: Path, tmp_path: Path) -> None:
    run_dev.run_dev(("grid",), repo, tmp_path, workers=2)

    def strip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in r.items() if k not in ("run", "commit")} for r in rows]

    assert strip(_tried(tmp_path)) == strip(_tried(dev_out))
    for f in sorted((dev_out / "patients").iterdir()):
        assert (tmp_path / "patients" / f.name).read_bytes() == f.read_bytes()


def test_a_limited_run_writes_beside_the_development_outputs(repo: Path) -> None:
    main(["--stage", "properties", "--limit", "10", "--root", str(repo)])
    out = repo / "paper1/outputs/dev_limit10/properties.json"
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


def test_frequencies_come_from_train_minus_h2(dev_out: Path) -> None:
    # make_repo writes 150 train rows and seals every sixth (25) into H2.
    for name in ("selected.json", "main.json", "ablations.json"):
        assert _json(dev_out / name)["provenance"]["train_minus_h2_rows"] == 125


def test_development_refuses_an_unsealed_environment(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_dev, "unseal_requested", lambda: True)
    with pytest.raises(RuntimeError, match="DDXKB_UNSEAL"):
        run_dev.run_dev(STAGES, repo, tmp_path, workers=1)
