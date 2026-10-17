"""Turn a sparsity pattern into sorted, unique COO data with random values."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_NNZ = 2**31


@dataclass
class Pattern:
    shape: tuple[int, int]
    rows: np.ndarray  # int64, 0-based, unsorted, may contain duplicates
    cols: np.ndarray


def assemble(p: Pattern, *, symmetric: bool, add_diagonal: bool, seed: int,
             _max_nnz: int = MAX_NNZ):
    """Return int32 rows, int32 cols, float64 values, sorted row-major, unique."""
    n_rows, n_cols = p.shape
    rows = np.asarray(p.rows, dtype=np.int64)
    cols = np.asarray(p.cols, dtype=np.int64)
    parts_r, parts_c = [rows], [cols]
    if symmetric:
        parts_r.append(cols)
        parts_c.append(rows)
    if add_diagonal:
        d = np.arange(min(n_rows, n_cols), dtype=np.int64)
        parts_r.append(d)
        parts_c.append(d)
    rows = np.concatenate(parts_r) if len(parts_r) > 1 else rows
    cols = np.concatenate(parts_c) if len(parts_c) > 1 else cols
    del parts_r, parts_c
    key = np.unique(rows * np.int64(n_cols) + cols)
    del rows, cols
    nnz = len(key)
    if nnz >= _max_nnz:
        raise ValueError(f"matrix has {nnz} entries, limit is {_max_nnz - 1}")
    r = (key // n_cols).astype(np.int32)
    c = (key % n_cols).astype(np.int32)
    del key
    v = np.random.default_rng(seed).uniform(-1.0, 1.0, nnz)
    return r, c, v


def _row_lengths(n_rows: int, rows: np.ndarray) -> np.ndarray:
    return np.bincount(rows, minlength=n_rows)


def row_length_stats(n_rows: int, rows: np.ndarray) -> dict:
    lens = _row_lengths(n_rows, rows)
    mean = float(lens.mean())
    std = float(lens.std())
    return {"min": int(lens.min()), "max": int(lens.max()), "mean": mean,
            "std": std, "cv": std / mean if mean > 0 else 0.0}


def sellp_padding_ratio(n_rows: int, rows: np.ndarray, slice_size: int = 64) -> float:
    lens = _row_lengths(n_rows, rows)
    pad = (-n_rows) % slice_size
    if pad:
        lens = np.concatenate([lens, np.zeros(pad, dtype=lens.dtype)])
    longest = lens.reshape(-1, slice_size).max(axis=1)
    nnz = int(lens.sum())
    return float(longest.sum()) * slice_size / nnz if nnz else 1.0
