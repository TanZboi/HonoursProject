"""Tests for scripts/make_toy_scene.py and the committed fixture tests/fixtures/toy.ply."""

import json

import numpy as np

from seg3d.io.ply import read_ply, sh_degree_from_properties

from conftest import FIXTURES


def test_deterministic(toy_module):
    """Same seed and size give bitwise-identical output."""
    a, ids_a, _ = toy_module.make_toy_scene(n=2000, seed=7)
    b, ids_b, _ = toy_module.make_toy_scene(n=2000, seed=7)
    assert a.tobytes() == b.tobytes()
    assert np.array_equal(ids_a, ids_b)


def test_different_seed_differs(toy_module):
    a, _, _ = toy_module.make_toy_scene(n=2000, seed=0)
    b, _, _ = toy_module.make_toy_scene(n=2000, seed=1)
    assert a.tobytes() != b.tobytes()


def test_counts_and_properties(toy_module):
    data, ids, objects = toy_module.make_toy_scene(n=50_000, seed=0)
    assert len(data) == len(ids) == 50_000
    assert ids.dtype == np.int32
    assert set(np.unique(ids)) == set(objects) == {1, 2, 3, 4, 5}
    assert np.bincount(ids)[1] == 20_000  # floor takes 40%
    assert all(np.bincount(ids)[2:] == 7_500)  # each object 15%
    assert sh_degree_from_properties(data.dtype.names) == 3
    assert "confidence" in data.dtype.names
    assert data.dtype.names[-1] == "confidence"  # extra property is last
    assert all(data.dtype[n] == np.dtype("<f4") for n in data.dtype.names)


def test_colours_and_geometry(toy_module):
    """Each object has its intended colour, and points lie on its surface."""
    data, ids, _ = toy_module.make_toy_scene(n=20_000, seed=0)
    rgb = 0.5 + toy_module.SH_C0 * np.column_stack([data[f"f_dc_{i}"] for i in range(3)])
    means = np.column_stack([data["x"], data["y"], data["z"]])

    red_sphere = ids == 2
    assert np.allclose(rgb[red_sphere].mean(axis=0), [0.9, 0.1, 0.1], atol=0.02)
    radius = np.linalg.norm(means[red_sphere] - [-1, -1, 0.4], axis=1)
    assert np.allclose(radius, 0.4, atol=1e-5)

    floor = ids == 1
    assert np.all(means[floor, 2] == 0)
    assert np.all(means[:, 2] >= 0)  # nothing below the floor

    quats = np.column_stack([data[f"rot_{i}"] for i in range(4)])
    assert np.allclose(np.linalg.norm(quats, axis=1), 1, atol=1e-5)
    opacity = 1 / (1 + np.exp(-data["opacity"].astype(np.float64)))
    assert opacity.min() >= 0.699 and opacity.max() <= 0.981


def test_fixture_matches_generator(toy_module):
    """The committed fixture is what the script produces for --n 3000 --seed 0.

    Compared with a tiny tolerance rather than bitwise, because sin/cos/log can differ in
    the last bit between operating systems.
    """
    header, data = read_ply(FIXTURES / "toy.ply")
    expected, expected_ids, objects = toy_module.make_toy_scene(n=3000, seed=0)
    assert data.dtype == expected.dtype
    for name in data.dtype.names:
        assert np.allclose(data[name], expected[name], rtol=1e-6, atol=1e-6), name
    assert np.array_equal(np.load(FIXTURES / "toy_gt.npy"), expected_ids)
    with open(FIXTURES / "toy_objects.json") as f:
        assert json.load(f) == {str(k): v for k, v in objects.items()}


def test_fixture_is_small():
    """SPEC: fixtures must stay under 1 MB."""
    for path in FIXTURES.iterdir():
        assert path.stat().st_size < 1_000_000, path
