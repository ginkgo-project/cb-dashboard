import bz2
import time
import zipfile

import numpy as np

import pytest

from datasets.readers import (read_dimacs9, read_edge_list, read_metis,
                              read_mtx, read_webgraph_arcs, webgraph_nodes)


def entries(p):
    return sorted(zip(p.rows.tolist(), p.cols.tolist()))


def test_metis_basic(tmp_path):
    f = tmp_path / "g.graph"
    f.write_text("3 2\n2\n1 3\n2\n")
    p = read_metis(f)
    assert p.shape == (3, 3)
    assert p.rows.dtype == np.int64 and p.cols.dtype == np.int64
    assert entries(p) == [(0, 1), (1, 0), (1, 2), (2, 1)]


def test_metis_comments_and_chunks(tmp_path):
    f = tmp_path / "g.graph"
    f.write_text("% hi\n3 2\n% mid\n2\n1 3\n% x\n2\n")
    assert entries(read_metis(f)) == [(0, 1), (1, 0), (1, 2), (2, 1)]
    import datasets.readers as r
    assert entries(r.read_metis(f, _chunk_bytes=4)) == \
        [(0, 1), (1, 0), (1, 2), (2, 1)]


def test_metis_empty_line_and_weights(tmp_path):
    f = tmp_path / "g.graph"
    # fmt 1: neighbour weight pairs; vertex 2 (index 1) is isolated
    f.write_text("3 1 1\n3 7\n\n1 7\n")
    p = read_metis(f)
    assert p.shape == (3, 3)
    assert entries(p) == [(0, 2), (2, 0)]


def test_metis_vertex_weights(tmp_path):
    f = tmp_path / "g.graph"
    f.write_text("2 1 011 2\n5 6 2 9\n7 8 1 9\n")
    assert entries(read_metis(f)) == [(0, 1), (1, 0)]


def test_dimacs9(tmp_path):
    f = tmp_path / "g.gr"
    f.write_text("c comment\np sp 4 2\na 1 2 10\na 3 4 5\n")
    p = read_dimacs9(f)
    assert p.shape == (4, 4)
    assert entries(p) == [(0, 1), (2, 3)]


def test_edge_list_gap_and_one_based(tmp_path):
    f = tmp_path / "e.txt"
    f.write_text("# c\n1\t3\n3 5\n")
    p = read_edge_list(f, min_id=1)
    assert p.shape == (5, 5)
    assert entries(p) == [(0, 2), (2, 4)]
    p0 = read_edge_list(f)
    assert p0.shape == (6, 6)


def test_edge_list_compact_ids(tmp_path):
    f = tmp_path / "e.txt"
    f.write_text("# c\n1\t3\n3 5\n")
    p = read_edge_list(f, min_id=1, compact=True)
    assert p.shape == (3, 3)
    assert entries(p) == [(0, 1), (1, 2)]


def test_webgraph_rows_from_node_count(tmp_path):
    f = tmp_path / "g.arcs"
    f.write_text("0\t2\n2\t4\n")
    # The last nodes of a BV graph have no arcs here, but they are still rows.
    assert read_webgraph_arcs(f, 7).shape == (7, 7)
    with pytest.raises(ValueError, match="4 nodes"):
        read_webgraph_arcs(f, 4)


def test_webgraph_nodes(tmp_path):
    f = tmp_path / "g.properties"
    f.write_text("#BVGraph properties\nversion=0\nnodes = 41652230\narcs=1468365182\n")
    assert webgraph_nodes(f) == 41652230
    f.write_text("arcs=3\n")
    with pytest.raises(ValueError, match="nodes"):
        webgraph_nodes(f)


def test_mtx_zip_symmetric(tmp_path):
    f = tmp_path / "m.zip"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("m.mtx", "%%MatrixMarket matrix coordinate real symmetric\n"
                   "% c\n3 3 3\n1 1 1.5\n2 1 2e0\n3 2 -1\n")
    p = read_mtx(f)
    assert p.shape == (3, 3)
    assert entries(p) == [(0, 0), (1, 0), (2, 1)]


def test_mtx_pattern(tmp_path):
    f = tmp_path / "m.mtx"
    f.write_text("%%MatrixMarket matrix coordinate pattern general\n2 3 2\n1 3\n2 1\n")
    p = read_mtx(f)
    assert p.shape == (2, 3)
    assert entries(p) == [(0, 2), (1, 0)]


def test_bz2_streaming(tmp_path):
    txt = b"3 2\n2\n1 3\n2\n"
    f = tmp_path / "g.graph.bz2"
    f.write_bytes(bz2.compress(txt))
    assert entries(read_metis(f)) == [(0, 1), (1, 0), (1, 2), (2, 1)]


def test_metis_speed(tmp_path):
    n, k = 1_000_000, 20
    rng = np.random.default_rng(0)
    nb = rng.integers(1, n + 1, size=(n, k)).astype(str).tolist()
    f = tmp_path / "big.graph"
    f.write_text(f"{n} {n * k // 2}\n" + "\n".join(" ".join(r) for r in nb) + "\n")
    t = time.perf_counter()
    p = read_metis(f)
    dt = time.perf_counter() - t
    print(f"\nmetis_speed: {dt:.2f}s")
    assert len(p.rows) == n * k and p.shape == (n, n)
    assert dt < 20
