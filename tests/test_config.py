"""Tests for seg3d/config.py."""

import pytest

from seg3d.config import load_config

from conftest import REPO_ROOT


def test_default_loads():
    cfg = load_config()
    assert cfg.seed == 0
    assert cfg.cameras.n_views == 200
    assert cfg.masks.model == "sam2.1_hiera_large"


def test_run_config_overrides_defaults():
    cfg = load_config(REPO_ROOT / "configs" / "toy.yaml")
    assert cfg.cameras.n_views == 40
    assert cfg.cameras.fov_deg == 70.0  # inherited from default.yaml


def test_command_line_overrides():
    cfg = load_config(overrides=["seed=3", "cameras.n_views=100", "scene.up=[0,-1,0]"])
    assert cfg.seed == 3
    assert cfg.cameras.n_views == 100
    assert cfg.scene.up == [0, -1, 0]


def test_unknown_key_is_an_error(tmp_path):
    with pytest.raises(KeyError):
        load_config(overrides=["camera.n_views=100"])  # typo: camera, not cameras
    bad = tmp_path / "bad.yaml"
    bad.write_text("cameras:\n  n_veiws: 5\n")
    with pytest.raises(KeyError):
        load_config(bad)


def test_resolved_config_is_a_plain_dict():
    d = load_config().to_dict()
    assert d["train"]["dim"] == 16
