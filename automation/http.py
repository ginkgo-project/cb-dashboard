"""HTTP on urllib: the automation runs with the standard library only."""
from __future__ import annotations

import base64
import json
import urllib.request
from urllib.error import HTTPError, URLError  # noqa: F401  (re-exported)

AGENT = "cb-dashboard-automation"
TIMEOUT = 60  # seconds, per request


def _request(url: str, data: bytes | None = None, headers: dict | None = None):
    return urllib.request.Request(url, data=data, headers={"User-Agent": AGENT, **(headers or {})})


def get_json(url: str, token: str | None = None) -> object:
    """GET a JSON document; HTTPError on a status other than 2xx."""
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(_request(url, headers=headers), timeout=TIMEOUT) as r:
        return json.load(r)


def post(url: str, data: bytes, auth: tuple[str, str]) -> tuple[int, str]:
    """POST with basic auth, like `curl -u user:secret --data-binary`; (status, text)."""
    cred = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
    req = _request(url, data=data, headers={"Authorization": f"Basic {cred}"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, r.read().decode(errors="replace")
    except HTTPError as e:
        return e.code, e.read().decode(errors="replace") if e.fp else ""


def download(url: str, limit: int) -> bytes:
    """GET at most `limit` bytes; ValueError above it, HTTPError on a status other than 2xx."""
    with urllib.request.urlopen(_request(url), timeout=TIMEOUT) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{url}: more than {limit} bytes")
    return data
