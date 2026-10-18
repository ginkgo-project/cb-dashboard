import http.server
import importlib.metadata
import json
import os
import subprocess
import threading
import zipfile
import io

import pytest

from datasets import __main__ as cli
from datasets import build as B
from datasets.binary import read_binary
from datasets.build import BuildError, VersionError, build
from datasets.catalog import load_catalog

TRIANGLE = b"3 3\n2 3\n1 3\n1 2\n"
EDGES = b"# c\n1 2\n2 3\n3 1\n"
MTX = b"%%MatrixMarket matrix coordinate real symmetric\n3 3 2\n2 1 1.0\n3 3 2.0\n"


def _zip(name, data, *more):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr(name, data)
        for n, d in zip(more[::2], more[1::2]):
            z.writestr(n, d)
    return b.getvalue()


class _H(http.server.BaseHTTPRequestHandler):
    files = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = self.files.get(self.path)
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    _H.files = {"/tiny.graph": TRIANGLE, "/e.txt": EDGES,
                "/m.zip": _zip("m.mtx", MTX),
                # Network Repository: a Matrix Market file named .edges, and a readme.
                "/nr.zip": _zip("readme.html", b"<html>", "nr.edges", MTX),
                "/tw.graph": b"g", "/tw.properties": b"nodes=4\narcs=3\n",
                "/gaps.txt": b"1 3\n3 5\n5 1\n"}
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


def write_catalog(tmp_path, url, version="1", nnz=6, fem_size=3):
    p = tmp_path / "catalog.toml"
    p.write_text(f'''version = "{version}"
seed = 7

[[entry]]
name = "tiny"
set = "spmv"
source = "test"
url = "{url}/tiny.graph"
format = "metis"
symmetric = true
add_diagonal = false
expected_rows = 3
expected_nnz = {nnz}
nnz_check = "exact"

[[entry]]
name = "fem1"
set = "fem"
source = "generated"
format = "fem"
symmetric = true
add_diagonal = false
nnz_check = "none"
roles = ["spgemm"]
spd = true
fem = {{problem = "poisson_p1", size = {fem_size}}}
''')
    return p


def run(tmp_path, url, **kw):
    cat = load_catalog(write_catalog(tmp_path, url, **kw))
    root = tmp_path / "root"
    return build(cat, root, tmp_path / "manifest.json", None, None), root


def test_build_writes_manifest(tmp_path, server):
    m, root = run(tmp_path, server)
    assert json.loads((tmp_path / "manifest.json").read_text()) == m
    assert m["version"] == "1" and m["seed"] == 7 and len(m["catalog_sha256"]) == 64
    tiny, fem1 = m["matrices"]
    assert tiny["path"] == "matrices/spmv/tiny.bin"
    assert tiny["set"] == "spmv" and tiny["nonzeros"] == 6 and tiny["rows"] == 3
    assert tiny["download_sha256"].keys() == {"tiny.graph"}
    assert tiny["nnz_check"] == "exact" and tiny["expected_nnz"] == 6
    assert tiny["row_length"]["min"] == 2 and tiny["row_length"]["max"] == 2
    assert tiny["run_sellp"] == (tiny["sellp_padding_ratio"] <= 2.0)
    assert tiny["sellp_padding_ratio"] == 64 * 2 / 6  # one slice, longest row 2
    assert fem1["set"] == "fem" and fem1["roles"] == ["spgemm"] and fem1["spd"] is True
    assert fem1["url"] is None and fem1["expected_nnz"] is None
    for e in m["matrices"]:
        shape, r, c, v = read_binary(root / e["path"])
        assert shape == (e["rows"], e["cols"]) and len(r) == e["nonzeros"]
    # FEM values are the real ones, not random in [-1, 1]
    _, r, c, v = read_binary(root / fem1["path"])
    assert abs(v).max() > 1.0 or (v == v.round()).any()


def test_manifest_records_entry_hashes(tmp_path, server):
    m, _ = run(tmp_path, server)
    cat = load_catalog(tmp_path / "catalog.toml")
    assert [e["entry_sha256"] for e in m["matrices"]] == [e.sha256 for e in cat.entries]
    assert all(len(e.sha256) == 64 for e in cat.entries)


def age(root):
    """Set every binary's mtime to 1000; return {name: path}."""
    paths = {p.stem: p for p in (root / "matrices").glob("*/*.bin")}
    for p in paths.values():
        os.utime(p, (1000, 1000))
    return paths


def rebuilt(paths):
    return sorted(name for name, p in paths.items() if p.stat().st_mtime != 1000)


def strip_entry_hashes(manifest):
    m = json.loads(manifest.read_text())
    for e in m["matrices"]:
        del e["entry_sha256"]
    manifest.write_text(json.dumps(m))


def test_skip_when_sha_matches(tmp_path, server):
    _, root = run(tmp_path, server)
    paths = age(root)
    run(tmp_path, server)
    assert rebuilt(paths) == []


def test_skip_rebuilds_corrupt_bin(tmp_path, server):
    _, root = run(tmp_path, server)
    p = root / "matrices/spmv/tiny.bin"
    p.write_bytes(b"junk")
    m, _ = run(tmp_path, server)
    assert p.stat().st_size > 4 and m["matrices"][0]["nonzeros"] == 6


def test_stale_tmp_ignored(tmp_path, server):
    d = tmp_path / "root/matrices/spmv"
    d.mkdir(parents=True)
    (d / "tiny.bin.tmp").write_bytes(b"stale")
    m, root = run(tmp_path, server)
    assert read_binary(root / "matrices/spmv/tiny.bin")[0] == (3, 3)


def test_version_guard(tmp_path, server):
    run(tmp_path, server)
    p = write_catalog(tmp_path, server, nnz=7)  # same version, new content
    assert cli.main(["build", "--root", str(tmp_path / "root"), "--catalog", str(p),
                     "--manifest", str(tmp_path / "manifest.json")]) == 2


def test_new_version_rebuilds_only_changed_entries(tmp_path, server):
    m1, root = run(tmp_path, server)
    paths = age(root)
    m2, _ = run(tmp_path, server, version="2", fem_size=4)
    assert m2["version"] == "2" and [e["name"] for e in m2["matrices"]] == ["tiny", "fem1"]
    assert rebuilt(paths) == ["fem1"]
    assert m2["matrices"][0] == m1["matrices"][0]  # kept as built, converter_version too
    assert m2["matrices"][1]["rows"] != m1["matrices"][1]["rows"]


def test_new_version_rebuilds_a_kept_entry_without_its_bin(tmp_path, server):
    _, root = run(tmp_path, server)
    (root / "matrices/spmv/tiny.bin").unlink()
    m, _ = run(tmp_path, server, version="2")
    assert read_binary(root / "matrices/spmv/tiny.bin")[0] == (3, 3)
    assert [e["name"] for e in m["matrices"]] == ["tiny", "fem1"]


def test_new_version_rebuilds_entries_without_a_hash(tmp_path, server):
    _, root = run(tmp_path, server)
    strip_entry_hashes(tmp_path / "manifest.json")
    paths = age(root)
    run(tmp_path, server, version="2")
    assert rebuilt(paths) == ["fem1", "tiny"]


def test_same_version_fills_in_entry_hashes(tmp_path, server):
    m1, root = run(tmp_path, server)
    strip_entry_hashes(tmp_path / "manifest.json")
    paths = age(root)
    m2, _ = run(tmp_path, server)
    assert rebuilt(paths) == [] and m2 == m1
    assert json.loads((tmp_path / "manifest.json").read_text()) == m1


def test_nnz_check_fails_loudly(tmp_path, server):
    cat = load_catalog(write_catalog(tmp_path, server, nnz=99))
    with pytest.raises(BuildError, match=r"tiny.*6.*99|tiny.*99.*6"):
        build(cat, tmp_path / "root", tmp_path / "manifest.json", None, None)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert [e["name"] for e in m["matrices"]] == ["fem1"]  # failed entry absent
    assert not (tmp_path / "root/matrices/spmv/tiny.bin").exists()


def test_only(tmp_path, server):
    cat = load_catalog(write_catalog(tmp_path, server))
    m = build(cat, tmp_path / "root", tmp_path / "manifest.json", ["fem1"], None)
    assert [e["name"] for e in m["matrices"]] == ["fem1"]


def extra_catalog(tmp_path, url, body):
    p = tmp_path / "catalog.toml"
    p.write_text('version = "1"\nseed = 7\n' + body.replace("URL", url))
    return load_catalog(p)


def test_edgelist_index_base_and_mtx_zip(tmp_path, server):
    cat = extra_catalog(tmp_path, server, '''
[[entry]]
name = "edges"
set = "spmv"
source = "t"
url = "URL/e.txt"
format = "edgelist"
index_base = 1
symmetric = false
add_diagonal = false
expected_rows = 3
expected_nnz = 3
nnz_check = "exact"

[[entry]]
name = "mz"
set = "spmv"
source = "t"
url = "URL/m.zip"
format = "mtx-zip"
symmetric = false
add_diagonal = false
expected_rows = 3
expected_nnz = 3
nnz_check = "exact"
''')
    m = build(cat, tmp_path / "root", tmp_path / "manifest.json", None, None)
    assert m["matrices"][0]["rows"] == 3 and m["matrices"][0]["nonzeros"] == 3
    # symmetric MM file: (2,1) mirrored, (3,3) diagonal -> 3 entries
    assert m["matrices"][1]["nonzeros"] == 3


def test_mtx_zip_member_found_by_its_banner(tmp_path, server):
    cat = extra_catalog(tmp_path, server, '''
[[entry]]
name = "nr"
set = "spmv"
source = "t"
url = "URL/nr.zip"
format = "mtx-zip"
symmetric = false
add_diagonal = false
expected_rows = 3
expected_nnz = 3
nnz_check = "exact"
''')
    m = build(cat, tmp_path / "root", tmp_path / "manifest.json", None, None)
    assert m["matrices"][0]["nonzeros"] == 3


def test_mtx_zip_with_two_matrices_fails(tmp_path):
    z = tmp_path / "two.zip"
    z.write_bytes(_zip("a.mtx", MTX, "b.edges", MTX))
    with pytest.raises(ValueError, match="one Matrix Market member"):
        B._mtx_from_zip(z, tmp_path / "two.mtx")


def test_edgelist_compact_ids(tmp_path, server):
    cat = extra_catalog(tmp_path, server, '''
[[entry]]
name = "gaps"
set = "spmv"
source = "t"
url = "URL/gaps.txt"
format = "edgelist"
index_base = 1
compact_ids = true
symmetric = false
add_diagonal = false
expected_rows = 3
expected_nnz = 3
nnz_check = "exact"
''')
    m = build(cat, tmp_path / "root", tmp_path / "manifest.json", None, None)
    assert m["matrices"][0]["rows"] == 3 and m["matrices"][0]["nonzeros"] == 3


WEB = '''
[[entry]]
name = "web"
set = "spmv"
source = "t"
url = "URL/tw"
format = "webgraph"
symmetric = false
add_diagonal = false
expected_rows = 4
expected_nnz = 3
nnz_check = "exact"
'''


def test_webgraph_with_fake_command(tmp_path, server):
    cat = extra_catalog(tmp_path, server, WEB)
    root = tmp_path / "root"
    cmd = "python3 -c \"import sys; open(sys.argv[2],'w').write('0 1\\n1 2\\n2 0\\n')\" {basename} {out}"
    m = build(cat, root, tmp_path / "manifest.json", None, cmd)
    e = m["matrices"][0]
    # 4 rows: the .properties has 4 nodes, the last one without arcs.
    assert e["rows"] == 4 and e["nonzeros"] == 3
    assert set(e["download_sha256"]) == {"tw.graph", "tw.properties"}
    assert (root / "raw/tw.graph").exists() and (root / "raw/web.arcs").exists()


def test_webgraph_without_command_fails(tmp_path, server):
    cat = extra_catalog(tmp_path, server, WEB)
    with pytest.raises(BuildError, match="webgraph-cmd"):
        build(cat, tmp_path / "root", tmp_path / "manifest.json", None, None)


def test_rows_check_fails_loudly(tmp_path, server):
    cat = extra_catalog(tmp_path, server, '''
[[entry]]
name = "badrows"
set = "spmv"
source = "t"
url = "URL/tiny.graph"
format = "metis"
symmetric = true
add_diagonal = false
expected_rows = 5
nnz_check = "none"
''')
    manifest = tmp_path / "manifest.json"
    with pytest.raises(BuildError, match=r"badrows.*3.*5"):
        build(cat, tmp_path / "root", manifest, None, None)
    assert json.loads(manifest.read_text())["matrices"] == []
    assert not (tmp_path / "root/matrices/spmv/badrows.bin").exists()


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                           *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repository standing in for this one, with a tracked manifest."""
    r = tmp_path / "repo"
    (r / "datasets").mkdir(parents=True)
    git(r, "init", "-q")
    (r / "datasets" / "fem.py").write_text("x = 1\n")
    (r / "datasets" / "manifest.json").write_text("{}\n")
    git(r, "add", ".")
    git(r, "commit", "-q", "-m", "init")
    monkeypatch.setattr(B, "REPO", r)
    return r


def test_converter_version(tmp_path, server, repo):
    cat = load_catalog(write_catalog(tmp_path, server))
    manifest = repo / "datasets" / "manifest.json"
    m = build(cat, tmp_path / "root", manifest, None, None)
    expected = {"repository": git(repo, "rev-parse", "HEAD"),
                **{p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-fem")}}
    # Rewriting the tracked manifest doesn't make the tree dirty.
    assert git(repo, "status", "--porcelain") != ""
    assert [e["converter_version"] for e in m["matrices"]] == [expected, expected]
    assert json.loads(manifest.read_text()) == m


def test_converter_version_dirty(tmp_path, server, repo):
    (repo / "datasets" / "fem.py").write_text("x = 2\n")
    cat = load_catalog(write_catalog(tmp_path, server))
    m = build(cat, tmp_path / "root", tmp_path / "manifest.json", ["fem1"], None)
    assert m["matrices"][0]["converter_version"]["repository"] == (
        git(repo, "rev-parse", "HEAD") + "-dirty")


def test_converter_version_outside_git(tmp_path, server, monkeypatch):
    monkeypatch.setattr(B, "REPO", tmp_path / "not-a-repo")
    cat = load_catalog(write_catalog(tmp_path, server))
    m = build(cat, tmp_path / "root", tmp_path / "manifest.json", ["fem1"], None)
    assert m["matrices"][0]["converter_version"]["repository"] is None


def test_webgraph_converter_version(tmp_path, server, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    wg = bindir / "webgraph"
    wg.write_text('#!/bin/sh\n[ "$1" = --version ] && echo "webgraph-cli 0.5.0"\n')
    wg.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    cat = extra_catalog(tmp_path, server, WEB)
    cmd = "python3 -c \"import sys; open(sys.argv[2],'w').write('0 1\\n1 2\\n2 0\\n')\" {basename} {out}"
    m = build(cat, tmp_path / "root", tmp_path / "manifest.json", None, cmd)
    assert m["matrices"][0]["converter_version"]["webgraph-cli"] == "webgraph-cli 0.5.0"


def test_changed_data_needs_version_bump(tmp_path, server, monkeypatch):
    run(tmp_path, server)
    manifest = tmp_path / "manifest.json"
    before = json.loads(manifest.read_text())
    # Same catalog, but the converter now produces other values for fem1.
    generate = B.fem.generate
    monkeypatch.setattr(B.fem, "generate", lambda problem, size: 2.0 * generate(problem, size))
    (tmp_path / "root/matrices/fem/fem1.bin").unlink()
    with pytest.raises(VersionError, match="fem1.*version bump"):
        build(load_catalog(tmp_path / "catalog.toml"), tmp_path / "root", manifest, None, None)
    assert json.loads(manifest.read_text()) == before
    (tmp_path / "root/matrices/fem/fem1.bin").unlink()
    assert cli.main(["build", "--root", str(tmp_path / "root"), "--catalog",
                     str(tmp_path / "catalog.toml"), "--manifest", str(manifest)]) == 2


def test_same_data_rebuilt_is_fine(tmp_path, server):
    m1, root = run(tmp_path, server)
    (root / "matrices/fem/fem1.bin").unlink()
    m2, _ = run(tmp_path, server)
    assert m2["matrices"][1]["binary_sha256"] == m1["matrices"][1]["binary_sha256"]
