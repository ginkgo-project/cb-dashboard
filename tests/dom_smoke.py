"""Load each tab of the built page in headless Chrome and check its DOM.

    python -m dashboard build          # writes site/data
    python tests/dom_smoke.py [--require-browser]

A page fails when its DOM contains "[object ", has a text node that is
exactly "null" or "undefined", or lacks its landmark: the heatmap on the
first three tabs, the setup on the machines tab, the selected cell and its
click box when a cell is selected. Every flag link found on the tabs is
followed too. Without a browser the test skips, unless --require-browser is
given (as in CI). Standard library only.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
BROWSERS = ("google-chrome", "chromium", "chromium-browser")
FLAG_LINKS_PER_TAB = 3


class Dom(HTMLParser):
    """The parts of a dumped page the checks look at."""

    def __init__(self):
        super().__init__()
        self.texts: list[str] = []
        self.tags: list[tuple[str, dict]] = []
        self.flag_links: list[str] = []
        self._in_flags = 0

    def handle_starttag(self, tag, attrs):
        attrs = {k: v or "" for k, v in attrs}
        self.tags.append((tag, attrs))
        classes = attrs.get("class", "").split()
        if tag == "ul" and "flags" in classes:
            self._in_flags += 1
        elif tag == "a" and self._in_flags and attrs.get("href"):
            self.flag_links.append(attrs["href"])

    def handle_endtag(self, tag):
        if tag == "ul" and self._in_flags:
            self._in_flags -= 1

    def handle_data(self, data):
        self.texts.append(data)

    def has(self, tag, *classes, **attrs) -> bool:
        for t, a in self.tags:
            have = a.get("class", "").split()
            if (t == tag and all(c in have for c in classes)
                    and all(a.get(k) == v for k, v in attrs.items())):
                return True
        return False


def find_browser() -> str | None:
    for name in BROWSERS:
        path = shutil.which(name)
        if path:
            return path
    return None


def serve(site: Path):
    handler = functools.partial(Quiet, directory=str(site))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def dump(browser: str, profile: Path, url: str) -> str:
    proc = subprocess.run(
        [browser, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--no-first-run", "--no-default-browser-check",
         "--disable-extensions", f"--user-data-dir={profile}",
         "--virtual-time-budget=10000", "--dump-dom", url],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0 or "<html" not in proc.stdout:
        raise RuntimeError(f"{url}: the browser failed "
                           f"(exit {proc.returncode}): {proc.stderr[-500:]}")
    return proc.stdout


def problems(page: str, dom: Dom, landmarks) -> list[str]:
    out = []
    at = page.find("[object ")
    if at >= 0:
        out.append(f'"[object " in the page: ...{page[max(0, at - 60):at + 60]}...')
    for text in dom.texts:
        if text.strip() in ("null", "undefined"):
            out.append(f'a text node "{text.strip()}"')
    for name, check in landmarks:
        if not check(dom):
            out.append(f"no {name}")
    return out


HEATMAP = ("heatmap", lambda d: d.has("table", "heatmap"))
SELECTED = ("selected heatmap cell", lambda d: d.has("td", "selected"))
CLICKBOX = ("filled click box",
            lambda d: d.has("section", "clickbox") and any(
                t.strip() == "Command to reproduce" for t in d.texts))
SETUP = ("machine setup", lambda d: d.has("dl", "values", "setup"))


def default_cell(site: Path) -> str | None:
    """A cell of the default basic view: SpMV csr on the first machine."""
    data = json.loads((site / "data" / "basic.json").read_text())
    for machine, m in data["machines"].items():
        for c in data["cases"]:
            lab = c["labels"]
            row = m["cells"].get(c["key"]) or []
            if (lab.get("operation") == "spmv" and lab.get("format") == "csr"
                    and row and row[-1] is not None):
                cell = quote(f"{c['key']}@{len(row) - 1}", safe="")
                return f"#basic?cell={cell}&machine={quote(machine, safe='')}"
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", default=str(ROOT / "site"))
    parser.add_argument("--require-browser", action="store_true",
                        help="fail instead of skipping without a browser")
    args = parser.parse_args(argv)
    site = Path(args.site)
    if not (site / "data" / "meta.json").is_file():
        print(f"FAIL: no {site / 'data' / 'meta.json'}; "
              "run python -m dashboard build first")
        return 1
    browser = find_browser()
    if browser is None:
        print(f"{'FAIL' if args.require_browser else 'SKIP'}: "
              f"no browser found ({', '.join(BROWSERS)})")
        return 1 if args.require_browser else 0

    pages = [("#basic", [HEATMAP]), ("#solvers", [HEATMAP]),
             ("#preconditioners", [HEATMAP]),
             ("#machines", [SETUP])]
    cell = default_cell(site)
    if cell is None:
        print("FAIL: basic.json has no SpMV csr cell to select")
        return 1
    pages.append((cell, [HEATMAP, SELECTED, CLICKBOX]))

    server = serve(site)
    base = f"http://127.0.0.1:{server.server_address[1]}/index.html"
    failed, seen = 0, set()
    with tempfile.TemporaryDirectory(prefix="dom-smoke-") as tmp:
        i = 0
        while i < len(pages):
            hash_, landmarks = pages[i]
            i += 1
            seen.add(hash_)
            page = dump(browser, Path(tmp) / "profile", base + hash_)
            dom = Dom()
            dom.feed(page)
            found = problems(page, dom, landmarks)
            print(f"{'FAIL' if found else 'ok  '} {hash_}")
            for p in found:
                print(f"     {p}")
            failed += bool(found)
            if "cell=" not in hash_:  # follow the tab's flag links
                new = [h for h in dict.fromkeys(dom.flag_links)
                       if h not in seen][:FLAG_LINKS_PER_TAB]
                for h in new:
                    seen.add(h)
                    pages.append((h, [HEATMAP, CLICKBOX]))
    server.shutdown()
    print(f"{len(pages) - failed} of {len(pages)} pages ok ({browser})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
