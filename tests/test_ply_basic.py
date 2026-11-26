"""Basic tests of the M0 PLY reader/writer. The full lossless tests (T1.x) come in M1."""

import numpy as np
import pytest

from seg3d.io.ply import read_ply, sh_degree_from_properties, write_ply


def _small_array():
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("flag", "u1"), ("count", "<i4"), ("w", "<f8")])
    data = np.zeros(3, dtype=dtype)
    data["x"] = [1.5, -0.0, np.nan]
    data["y"] = [2, 3, 4]
    data["flag"] = [0, 1, 255]
    data["count"] = [-5, 0, 2**31 - 1]
    data["w"] = [1e-300, np.inf, -np.inf]
    return data


def test_write_then_read(tmp_path):
    data = _small_array()
    write_ply(tmp_path / "a.ply", data, comments=["hello world"])
    header, back = read_ply(tmp_path / "a.ply")
    assert header.format == "binary_little_endian"
    assert header.comments == ["hello world"]
    assert back.dtype == data.dtype
    assert back.tobytes() == data.tobytes()  # bitwise, including -0.0, NaN and inf


def test_refuses_overwrite(tmp_path):
    path = tmp_path / "a.ply"
    write_ply(path, _small_array())
    with pytest.raises(FileExistsError):
        write_ply(path, _small_array())


def test_big_endian_and_ascii(tmp_path):
    """Hand-written big-endian and ASCII files read to the same values."""
    values = np.array([(1.25, -2.0), (3.5, 0.0)], dtype=[("x", ">f4"), ("y", ">f4")])
    header = b"ply\nformat binary_big_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nend_header\n"
    (tmp_path / "be.ply").write_bytes(header + values.tobytes())
    h, data = read_ply(tmp_path / "be.ply")
    assert h.format == "binary_big_endian"
    assert data.dtype["x"].byteorder == ">"
    assert data["x"].tolist() == [1.25, 3.5]

    text = b"ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nend_header\n1.25 -2\n3.5 0\n"
    (tmp_path / "ascii.ply").write_bytes(text)
    _, data = read_ply(tmp_path / "ascii.ply")
    assert data["x"].tolist() == [1.25, 3.5] and data["y"].tolist() == [-2.0, 0.0]


def test_count_mismatch_is_an_error(tmp_path):
    header = b"ply\nformat binary_little_endian 1.0\nelement vertex 5\nproperty float x\nend_header\n"
    (tmp_path / "short.ply").write_bytes(header + np.zeros(3, "<f4").tobytes())
    with pytest.raises(ValueError):
        read_ply(tmp_path / "short.ply")


@pytest.mark.parametrize("degree, n_rest", [(0, 0), (1, 9), (2, 24), (3, 45)])
def test_sh_degree(degree, n_rest):
    names = ["x", "f_dc_0"] + [f"f_rest_{i}" for i in range(n_rest)]
    assert sh_degree_from_properties(names) == degree
