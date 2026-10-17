import hashlib
import http.server
import threading

import pytest

from datasets.fetch import download, fetch_webgraph

PAYLOAD = bytes(range(256)) * 4000  # ~1 MB


class _Handler(http.server.BaseHTTPRequestHandler):
    files = {}
    requests = []
    honour_range = True

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = self.files.get(self.path)
        if body is None:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        self.requests.append((self.path, rng))
        start = 0
        if rng and self.honour_range:
            start = int(rng.split("=")[1].rstrip("-"))
            if start >= len(body):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(body)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body)-1}/{len(body)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        self.wfile.write(body[start:])


@pytest.fixture
def server():
    _Handler.files = {"/a.bin": PAYLOAD}
    _Handler.requests = []
    _Handler.honour_range = True
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}", _Handler
    srv.shutdown()
    srv.server_close()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def test_download_and_sha(server, tmp_path):
    base, h = server
    dest = tmp_path / "a.bin"
    assert download(base + "/a.bin", dest) == sha(PAYLOAD)
    assert dest.read_bytes() == PAYLOAD
    assert not dest.with_suffix(".bin.part").exists()


def test_resume(server, tmp_path):
    base, h = server
    dest = tmp_path / "a.bin"
    dest.with_suffix(".bin.part").write_bytes(PAYLOAD[:500_000])
    assert download(base + "/a.bin", dest) == sha(PAYLOAD)
    assert dest.read_bytes() == PAYLOAD
    assert h.requests == [("/a.bin", "bytes=500000-")]


def test_resume_server_ignores_range(server, tmp_path):
    base, h = server
    h.honour_range = False
    dest = tmp_path / "a.bin"
    dest.with_suffix(".bin.part").write_bytes(PAYLOAD[:1000])
    assert download(base + "/a.bin", dest) == sha(PAYLOAD)
    assert dest.read_bytes() == PAYLOAD


def test_resume_complete_part(server, tmp_path):
    base, h = server
    dest = tmp_path / "a.bin"
    dest.with_suffix(".bin.part").write_bytes(PAYLOAD)
    assert download(base + "/a.bin", dest) == sha(PAYLOAD)
    assert dest.read_bytes() == PAYLOAD


def test_skip_existing(server, tmp_path):
    base, h = server
    dest = tmp_path / "a.bin"
    dest.write_bytes(PAYLOAD)
    assert download(base + "/a.bin", dest) == sha(PAYLOAD)
    assert h.requests == []


def test_error_leaves_no_dest(server, tmp_path):
    base, h = server
    dest = tmp_path / "missing.bin"
    with pytest.raises(Exception):
        download(base + "/missing.bin", dest)
    assert not dest.exists()


def test_fetch_webgraph(server, tmp_path):
    base, h = server
    h.files = {f"/g/web.{e}": e.encode() * 10 for e in ("graph", "properties", "offsets")}
    out = fetch_webgraph(base + "/g/web", tmp_path / "d")
    assert set(out) == {"graph", "properties", "offsets"}
    for e, s in out.items():
        assert (tmp_path / "d" / f"web.{e}").exists()
        assert s == sha(e.encode() * 10)
