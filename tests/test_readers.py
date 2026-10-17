import bz2
import time
import zipfile

import numpy as np

from datasets.readers import (read_dimacs9, read_edge_list, read_metis,
                              read_mtx, read_webgraph_arcs)


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
    assert read_webgraph_arcs(f, min_id=1).shape == (5, 5)


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
