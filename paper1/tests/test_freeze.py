from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ddxkb import freeze
from ddxkb.freeze import (
    FROZEN_CONFIG,
    SELECTED,
    FreezeError,
    config_digest,
    frozen_config,
    protocol_constants,
    write_frozen,
)
from ddxkb.protocol import A3_EPSILONS, BUDGETS, MIN_GAIN, P2_VARIANTS, P1Config, P2Config
from ddxkb.provenance import environment, file_sha256
from helpers import make_repo, write_selected


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    make_repo(tmp_path)
    write_selected(tmp_path, P1Config(0.7, 1e-4, 0.95))
    return tmp_path


def test_the_frozen_protocol_survives_its_yaml_round_trip(repo: Path) -> None:
    digest = write_frozen(repo)
    text = (repo / FROZEN_CONFIG).read_text()
    assert text.startswith("# The frozen protocol")
    config = yaml.safe_load(text)
    assert config["config_sha256"] == digest == config_digest(config)
    assert config == frozen_config(repo)
    assert config["P1"] == {"alpha": 0.7, "eps": 0.0001, "tau": 0.95}
    assert list(config["P2"]) == [v.name for v in P2_VARIANTS]
    assert config["selected"] == {"path": SELECTED, "sha256": file_sha256(repo / SELECTED)}
    assert config["manifests"]["H2"] == file_sha256(repo / "paper1/manifests/H2.json")
    assert config["budgets"] == list(BUDGETS)
    assert config["environment"] == environment()
    assert config["constants"] == protocol_constants()
    assert config["constants"]["min_gain"] == MIN_GAIN
    assert config["constants"]["a3_epsilons"] == list(A3_EPSILONS)


def test_a_frozen_protocol_is_never_overwritten(repo: Path) -> None:
    write_frozen(repo)
    with pytest.raises(FreezeError, match="never overwritten"):
        write_frozen(repo)


def test_refuses_a_smoke_selection(tmp_path: Path) -> None:
    make_repo(tmp_path)
    with pytest.raises(FreezeError, match="does not exist"):
        frozen_config(tmp_path)
    write_selected(tmp_path, limit=2000)
    with pytest.raises(FreezeError, match="limited to 2000"):
        write_frozen(tmp_path)
    assert not (tmp_path / FROZEN_CONFIG).exists()


def test_refuses_a_configuration_under_another_variant(tmp_path: Path) -> None:
    make_repo(tmp_path)
    p2 = {v.name: P2Config(v, 1.0, 0.99).as_dict() for v in P2_VARIANTS}
    p2["P2-binary"] = P2Config(P2_VARIANTS[2], 1.0, 0.99).as_dict()
    write_selected(tmp_path, p2=p2)
    with pytest.raises(FreezeError, match="P2-value configuration under P2-binary"):
        frozen_config(tmp_path)


def test_the_cli_prints_the_digest(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    freeze.main(["--root", str(repo)])
    digest = yaml.safe_load((repo / FROZEN_CONFIG).read_text())["config_sha256"]
    assert capsys.readouterr().out.strip().endswith(digest)
    with pytest.raises(SystemExit, match=r"freeze refused: .*never overwritten"):
        freeze.main(["--root", str(repo)])


def test_digest_ignores_key_order_and_the_recorded_digest() -> None:
    a = {"x": 1, "y": [1e-6, 0.5], "config_sha256": "anything"}
    b = {"y": [1e-6, 0.5], "x": 1}
    assert config_digest(a) == config_digest(b)
    assert config_digest(a) != config_digest({**b, "x": 2})
    assert json.loads(json.dumps(protocol_constants())) == protocol_constants()
