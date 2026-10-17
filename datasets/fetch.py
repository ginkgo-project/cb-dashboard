"""Resumable, streaming downloads of source matrices."""
import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 1 << 20
WEBGRAPH_EXTENSIONS = ("graph", "properties", "offsets")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> str:
    """Download url to dest, resuming from dest.part; return the sha256.

    Data goes to ``<dest>.part`` and is renamed only once complete, so dest
    is never half-written. An existing dest is not downloaded again.
    """
    dest = Path(dest)
    if dest.exists():
        return _sha256_file(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    complete = False
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:
            complete = True  # .part already holds the whole file
            resp = None
        else:
            raise
    if not complete:
        with resp:
            mode = "ab"
            if have and resp.status != 206:
                mode, have = "wb", 0  # server ignored Range: start over
            expected = resp.headers.get("Content-Length")
            written = 0
            with open(part, mode) as f:
                while chunk := resp.read(CHUNK):
                    f.write(chunk)
                    written += len(chunk)
            if expected is not None and written != int(expected):
                raise IOError(f"short download of {url}: {written} of {expected} bytes")
    sha = _sha256_file(part)
    os.replace(part, dest)
    return sha


def fetch_webgraph(basename_url: str, dest_dir: Path) -> dict[str, str]:
    """Download <basename>.{graph,properties,offsets}; return sha256 per extension."""
    dest_dir = Path(dest_dir)
    base = basename_url.rsplit("/", 1)[-1]
    return {
        ext: download(f"{basename_url}.{ext}", dest_dir / f"{base}.{ext}")
        for ext in WEBGRAPH_EXTENSIONS
    }
