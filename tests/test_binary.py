import numpy as np

from datasets.binary import read_binary, write_binary


def _sample():
    rows = np.array([0, 1], dtype=np.int32)
    cols = np.array([2, 0], dtype=np.int32)
    vals = np.array([0.5, -0.25], dtype=np.float64)
    return rows, cols, vals


def test_header_bytes(tmp_path):
    p = tmp_path / "m.bin"
    rows, cols, vals = _sample()
    write_binary(p, (2, 3), rows, cols, vals)
    data = p.read_bytes()
    assert data[:8] == b"GINKGODI"
    assert tuple(np.frombuffer(data[8:32], dtype="<u8")) == (2, 3, 2)
    assert len(data) == 32 + 2 * 16


def test_round_trip(tmp_path):
    p = tmp_path / "m.bin"
    rows, cols, vals = _sample()
    digest = write_binary(p, (2, 3), rows, cols, vals)
    assert len(digest) == 64
    shape, r, c, v = read_binary(p)
    assert shape == (2, 3)
    np.testing.assert_array_equal(r, rows)
    np.testing.assert_array_equal(c, cols)
    np.testing.assert_array_equal(v, vals)
    assert not (tmp_path / "m.bin.tmp").exists()
