"""Readers that turn downloaded graph/matrix files into sparsity patterns.

All parsing is vectorized with numpy over large chunks of the decompressed
byte stream (a loop over chunks, never over lines or entries).
"""
from __future__ import annotations

import bz2
import contextlib
import gzip
import zipfile
from pathlib import Path

import numpy as np

from .assemble import Pattern

CHUNK_BYTES = 1 << 28  # 256 MiB of text per chunk
_NL = 10


@contextlib.contextmanager
def _open(path):
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".bz2":
        f = bz2.open(path, "rb")
    elif suf == ".gz":
        f = gzip.open(path, "rb")
    elif suf == ".zip":
        z = zipfile.ZipFile(path)
        f = z.open(z.namelist()[0])
    else:
        f = open(path, "rb")
    try:
        yield f
    finally:
        f.close()


def _chunks(path, chunk_bytes):
    """Yield byte chunks that each consist of whole lines (end with a newline)."""
    with _open(path) as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                return
            if not b.endswith(b"\n"):
                b += f.readline()
                if not b.endswith(b"\n"):
                    b += b"\n"
            yield b


def _line_starts(buf):
    nl = np.flatnonzero(buf == _NL)
    starts = np.empty(len(nl), dtype=np.int64)
    if len(nl):
        starts[0] = 0
        starts[1:] = nl[:-1] + 1
    return starts, nl


def _drop_lines(buf, starts, nl, drop):
    """Remove the lines flagged in `drop` (a bool array per line)."""
    lens = nl + 1 - starts
    keep = np.repeat(~drop, lens)
    return buf[keep]


def _strip_comments(buf, comment: int):
    starts, nl = _line_starts(buf)
    if len(starts) == 0:
        return buf
    drop = buf[starts] == comment
    if not drop.any():
        return buf
    return _drop_lines(buf, starts, nl, drop)


def _first_line(buf):
    """Split off the first line: returns (line bytes, rest)."""
    i = int(np.argmax(buf == _NL))
    return bytes(buf[:i]), buf[i + 1:]


def _tokens(buf, dtype):
    """Parse all whitespace-separated numbers of `buf`."""
    if buf.size == 0 or not (buf > 32).any():
        return np.empty(0, dtype=dtype)
    out = np.fromstring(buf.tobytes(), dtype=dtype, sep=" ")
    return out


def _writable(chunk: bytes):
    return np.frombuffer(bytearray(chunk), dtype=np.uint8)


# --------------------------------------------------------------------- METIS

def read_metis(path, _chunk_bytes: int = CHUNK_BYTES) -> Pattern:
    n = None
    vw = ew = 0
    line0 = 0  # vertex lines consumed so far
    rows_parts, cols_parts = [], []
    for chunk in _chunks(path, _chunk_bytes):
        buf = _strip_comments(_writable(chunk), ord("%"))
        if n is None:
            if buf.size == 0:
                continue
            head, buf = _first_line(buf)
            h = [int(x) for x in head.split()]
            n = h[0]
            fmt = h[2] if len(h) > 2 else 0
            ncon = h[3] if len(h) > 3 else 1
            ew = 2 if fmt % 10 == 1 else 1
            vw = (ncon if (fmt // 10) % 10 == 1 else 0) + (1 if fmt // 100 else 0)
        starts, nl = _line_starts(buf)
        nlines = len(nl)
        vals = _tokens(buf, np.int64)
        if vals.size:
            m = buf > 32
            tok = np.flatnonzero(m[1:] & ~m[:-1]) + 1
            if m[0]:
                tok = np.concatenate(([0], tok))
            assert len(tok) == len(vals), "non-numeric token in METIS file"
            line = np.searchsorted(nl, tok)
            counts = np.bincount(line, minlength=nlines)
            first = np.cumsum(counts) - counts
            pos = np.arange(len(vals)) - first[line] - vw
            sel = (pos >= 0) & (pos % ew == 0) & (line0 + line < n)
            cols = vals[sel]
            if cols.size and (cols.min() < 1 or cols.max() > n):
                raise ValueError("METIS neighbour id out of range 1..n")
            rows_parts.append(line[sel] + line0)
            cols_parts.append(cols - 1)
        line0 += nlines
    if n is None:
        raise ValueError("empty METIS file")
    return _pattern((n, n), rows_parts, cols_parts)


def _pattern(shape, rows_parts, cols_parts):
    if rows_parts:
        rows = np.concatenate(rows_parts).astype(np.int64, copy=False)
        cols = np.concatenate(cols_parts).astype(np.int64, copy=False)
    else:
        rows = cols = np.empty(0, dtype=np.int64)
    return Pattern(shape, rows, cols)


# ------------------------------------------------------------------- DIMACS9

def read_dimacs9(path, _chunk_bytes: int = CHUNK_BYTES) -> Pattern:
    shape = None
    rows_parts, cols_parts = [], []
    for chunk in _chunks(path, _chunk_bytes):
        buf = _writable(chunk)
        starts, nl = _line_starts(buf)
        first = buf[starts]
        for s in starts[first == ord("p")]:
            e = int(s) + int(np.argmax(buf[s:] == _NL))
            parts = bytes(buf[s:e]).split()
            shape = (int(parts[2]), int(parts[2]))
        is_a = first == ord("a")
        buf[starts[is_a]] = 32
        buf = _drop_lines(buf, starts, nl, ~is_a)
        vals = _tokens(buf, np.int64)
        if vals.size % 3:
            raise ValueError("malformed 'a' line in DIMACS9 file")
        vals = vals.reshape(-1, 3)
        rows_parts.append(vals[:, 0] - 1)
        cols_parts.append(vals[:, 1] - 1)
    if shape is None:
        raise ValueError("no 'p' line in DIMACS9 file")
    return _pattern(shape, rows_parts, cols_parts)


# --------------------------------------------------------------- edge lists

def read_edge_list(path, min_id: int = 0, compact: bool = False,
                   _chunk_bytes: int = CHUNK_BYTES) -> Pattern:
    """Read `u v` lines (`#` comments). `min_id` is the smallest id the file
    uses (1 for 1-based files); ids are shifted by it. n = max_id + 1, so gaps
    in the ids still count as rows, unless `compact`: then the ids that occur
    are renumbered 0..n-1 in increasing order and the gaps disappear."""
    rows_parts, cols_parts = [], []
    for chunk in _chunks(path, _chunk_bytes):
        buf = _strip_comments(_writable(chunk), ord("#"))
        vals = _tokens(buf, np.int64)
        if vals.size % 2:
            raise ValueError("edge list line without exactly two ids")
        vals = vals.reshape(-1, 2)
        rows_parts.append(vals[:, 0] - min_id)
        cols_parts.append(vals[:, 1] - min_id)
    p = _pattern((0, 0), rows_parts, cols_parts)
    if compact:
        ids = np.unique(np.concatenate([p.rows, p.cols]))
        return Pattern((ids.size, ids.size), np.searchsorted(ids, p.rows),
                       np.searchsorted(ids, p.cols))
    n = int(max(p.rows.max(), p.cols.max())) + 1 if p.rows.size else 0
    return Pattern((n, n), p.rows, p.cols)


def webgraph_nodes(properties) -> int:
    """The node count (`nodes=`) of a BV graph's .properties file."""
    for line in Path(properties).read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "nodes":
            return int(value)
    raise ValueError(f"{properties}: no nodes= line")


def read_webgraph_arcs(path, n_nodes: int, min_id: int = 0,
                       _chunk_bytes: int = CHUNK_BYTES) -> Pattern:
    """Arcs from `webgraph to arcs`. The graph has n_nodes rows, also when its
    last nodes have no arcs (unlike a plain edge list, n isn't max_id + 1)."""
    p = read_edge_list(path, min_id, _chunk_bytes=_chunk_bytes)
    if p.shape[0] > n_nodes:
        raise ValueError(f"{path}: arc to node {p.shape[0] - 1}, but the graph has "
                         f"{n_nodes} nodes")
    return Pattern((n_nodes, n_nodes), p.rows, p.cols)


# ------------------------------------------------------------- Matrix Market

def read_mtx(path, _chunk_bytes: int = CHUNK_BYTES) -> Pattern:
    """Coordinate Matrix Market. Values are ignored; a symmetric file yields
    only the stored triangle."""
    shape = None
    width = 3
    rows_parts, cols_parts = [], []
    for chunk in _chunks(path, _chunk_bytes):
        buf = _writable(chunk)
        if shape is None:
            banner = bytes(buf[:int(np.argmax(buf == _NL))]).lower().split()
            if len(banner) < 5 or banner[0] != b"%%matrixmarket" \
                    or banner[2] != b"coordinate":
                raise ValueError("not a coordinate Matrix Market file")
            width = {b"pattern": 2, b"complex": 4}.get(banner[3], 3)
        buf = _strip_comments(buf, ord("%"))
        if shape is None:
            if buf.size == 0:
                continue
            head, buf = _first_line(buf)
            h = head.split()
            shape = (int(h[0]), int(h[1]))
        if width == 2:
            vals = _tokens(buf, np.int64).reshape(-1, 2)
        else:
            vals = _tokens(buf, np.float64).reshape(-1, width)
        rows_parts.append(vals[:, 0].astype(np.int64) - 1)
        cols_parts.append(vals[:, 1].astype(np.int64) - 1)
    if shape is None:
        raise ValueError("empty Matrix Market file")
    return _pattern(shape, rows_parts, cols_parts)
