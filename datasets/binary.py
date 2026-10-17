"""Ginkgo binary matrix format (see core/base/mtx_io.cpp, write_binary_raw)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

MAGIC = b"GINKGODI"  # double values, int32 indices
ENTRY_DTYPE = np.dtype([("r", "<i4"), ("c", "<i4"), ("v", "<f8")])
HEADER_BYTES = 32


def write_binary(path: Path, shape, rows, cols, values) -> str:
    """Write `<path>.tmp`, rename it to `path`, return the sha256 (hex)."""
    path = Path(path)
    n = len(rows)
    if not (len(cols) == n and len(values) == n):
        raise ValueError("rows, cols and values must have equal length")
    entries = np.empty(n, dtype=ENTRY_DTYPE)
    entries["r"] = rows
    entries["c"] = cols
    entries["v"] = values
    header = MAGIC + np.array([shape[0], shape[1], n], dtype="<u8").tobytes()
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(header)
        entries.tofile(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def read_binary(path: Path):
    """Return ((n_rows, n_cols), rows, cols, values)."""
    with open(path, "rb") as f:
        header = f.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES or header[:8] != MAGIC:
            raise ValueError(f"{path}: not a Ginkgo binary matrix")
        n_rows, n_cols, nnz = (int(x) for x in np.frombuffer(header[8:], dtype="<u8"))
        entries = np.fromfile(f, dtype=ENTRY_DTYPE, count=nnz)
    if len(entries) != nnz:
        raise ValueError(f"{path}: truncated, expected {nnz} entries")
    return (n_rows, n_cols), entries["r"].copy(), entries["c"].copy(), entries["v"].copy()
