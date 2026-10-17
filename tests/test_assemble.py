import numpy as np
import pytest

from datasets.assemble import Pattern, assemble, row_length_stats, sellp_padding_ratio


def _pat():
    return Pattern(
        (3, 3),
        np.array([1, 1, 2], dtype=np.int64),
        np.array([0, 0, 0], dtype=np.int64),
    )


def test_symmetrize_dedup_diagonal():
    r, c, v = assemble(_pat(), symmetric=True, add_diagonal=True, seed=1)
    assert r.dtype == np.int32 and c.dtype == np.int32 and v.dtype == np.float64
    assert r.tolist() == [0, 0, 0, 1, 1, 2, 2]
    assert c.tolist() == [0, 1, 2, 0, 1, 0, 2]


def test_values_depend_on_pattern_and_seed():
    a = assemble(_pat(), symmetric=True, add_diagonal=True, seed=1)[2]
    b = assemble(_pat(), symmetric=True, add_diagonal=True, seed=1)[2]
    d = assemble(_pat(), symmetric=True, add_diagonal=True, seed=2)[2]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, d)
    assert a.min() >= -1.0 and a.max() < 1.0


def test_rejects_overflow():
    with pytest.raises(ValueError):
        assemble(_pat(), symmetric=False, add_diagonal=False, seed=1, _max_nnz=2)


def test_sellp_ratio():
    rows = np.concatenate([np.arange(128), np.zeros(9, dtype=np.int64)])
    assert sellp_padding_ratio(128, rows) == pytest.approx((10 * 64 + 1 * 64) / 137)


def test_row_length_stats():
    s = row_length_stats(3, np.array([0, 0, 1]))
    assert s["min"] == 0 and s["max"] == 2
    assert s["mean"] == pytest.approx(1.0)
    assert s["cv"] == pytest.approx(s["std"] / s["mean"])
