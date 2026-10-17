import http.server
import json
import os
import threading
import zipfile
import io

import pytest

from datasets import __main__ as cli
from datasets.binary import read_binary
from datasets.build import BuildError, build
from datasets.catalog import load_catalog

TRIANGLE = b"3 3\n2 3\n1 3\n1 2\n"
EDGES = b"# c\n1 2\n2 3\n3 1\n"
MTX = b"%%MatrixMarket matrix coordinate real symmetric\n3 3 2\n2 1 1.0\n3 3 2.0\n"


def _zip(name, data):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr(name, data)
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
                "/tw.graph": b"g", "/tw.properties": b"p", "/tw.offsets": b"o"}
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


def write_catalog(tmp_path, url, version="1", nnz=6):
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
fem = {{problem = "poisson_p1", size = 3}}
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


def test_skip_when_sha_matches(tmp_path, server):
    _, root = run(tmp_path, server)
    paths = [root / "matrices/spmv/tiny.bin", root / "matrices/fem/fem1.bin"]
    for p in paths:
        os.utime(p, (1000, 1000))
    run(tmp_path, server)
    assert all(p.stat().st_mtime == 1000 for p in paths)


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


def test_new_version_rebuilds(tmp_path, server):
    run(tmp_path, server)
    m, _ = run(tmp_path, server, version="2", nnz=6)
    assert m["version"] == "2" and len(m["matrices"]) == 2


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


WEB = '''
[[entry]]
name = "web"
set = "spmv"
source = "t"
url = "URL/tw"
format = "webgraph"
symmetric = false
add_diagonal = false
expected_rows = 3
expected_nnz = 3
nnz_check = "exact"
'''


def test_webgraph_with_fake_command(tmp_path, server):
    cat = extra_catalog(tmp_path, server, WEB)
    root = tmp_path / "root"
    cmd = "python3 -c \"import sys; open(sys.argv[2],'w').write('0 1\\n1 2\\n2 0\\n')\" {basename} {out}"
    m = build(cat, root, tmp_path / "manifest.json", None, cmd)
    e = m["matrices"][0]
    assert e["nonzeros"] == 3 and set(e["download_sha256"]) == {"tw.graph", "tw.properties", "tw.offsets"}
    assert (root / "raw/tw.graph").exists() and (root / "raw/web.arcs").exists()


def test_webgraph_without_command_fails(tmp_path, server):
    cat = extra_catalog(tmp_path, server, WEB)
    with pytest.raises(BuildError, match="webgraph-cmd"):
        build(cat, tmp_path / "root", tmp_path / "manifest.json", None, None)
