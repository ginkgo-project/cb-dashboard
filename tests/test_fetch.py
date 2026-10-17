import hashlib
import http.server
import threading
import time
import urllib.error

import pytest

from datasets import fetch
from datasets.fetch import download, fetch_webgraph

PAYLOAD = bytes(range(256)) * 4000  # ~1 MB


class _Handler(http.server.BaseHTTPRequestHandler):
    files = {}
    requests = []
    honour_range = True
    stall = 0.0  # seconds to wait before answering

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.stall:
            time.sleep(self.stall)
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
    _Handler.stall = 0.0
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


def test_urlopen_has_timeout(server, tmp_path, monkeypatch):
    base, h = server
    seen = []
    real = fetch.urllib.request.urlopen

    def urlopen(req, *a, **kw):
        seen.append(kw.get("timeout"))
        return real(req, *a, **kw)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen)
    download(base + "/a.bin", tmp_path / "a.bin")
    assert seen == [60] and fetch.TIMEOUT_S == 60


def test_stalled_server_times_out(server, tmp_path, monkeypatch):
    base, h = server
    h.stall = 3.0
    monkeypatch.setattr(fetch, "TIMEOUT_S", 0.5)
    dest = tmp_path / "a.bin"
    t = time.monotonic()
    with pytest.raises((TimeoutError, urllib.error.URLError)):
        download(base + "/a.bin", dest)
    assert time.monotonic() - t < 2.5
    assert not dest.exists()
