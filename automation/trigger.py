"""Start Ginkgo's CSCS CI pipeline "benchmark" for a commit.

Only on the machines without a run of it on cb-dashboard's results branch,
unless forced. Run by .github/workflows/benchmark.yml in ginkgo-project/ginkgo,
twice a week, with develop's head; with --distributed, on Saturdays, for the
distributed jobs, whose runs are under distributed/<machine>/. The CSCS
credentials come from the environment and are never printed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Callable, Mapping

from automation.http import HTTPError, URLError, get_json, post
from bench.checkout import DEVELOP
from bench.machine import DEFAULT, _known

MACHINES = _known(DEFAULT)
TRIGGER_URL = "https://cicd-ext-mw.cscs.ch/ci/pipeline/trigger"
CONTENTS = ("https://api.github.com/repos/ginkgo-project/cb-dashboard/contents/"
            "{path}?ref=results")
PIPELINE = "benchmark"
REPOSITORY_ID, SECRET = "CSCS_CI_REPOSITORY_ID", "CSCS_CI_WEBHOOK_SECRET"


def has_run(names: list[str], sha: str) -> bool:
    """Whether a results directory name `<date>_<sha10>[_n]` is one of `sha`."""
    pattern = re.compile(rf"\d{{4}}-\d{{2}}-\d{{2}}_{re.escape(sha[:10])}(_\d+)?")
    return any(pattern.fullmatch(n) for n in names)


def pending(sha: str, listing: Callable[[str], list[str]], machines: list[str]) -> list[str]:
    """The machines whose results have no run of `sha`, whatever its outcome."""
    return [m for m in machines if not has_run(listing(m), sha)]


def body(sha: str, machines: list[str], distributed: bool = False) -> dict:
    """The trigger request; YAML, of which JSON is a subset. CB_BENCHMARK starts
    the single-node jobs, CB_DISTRIBUTED the distributed ones."""
    kind = "CB_DISTRIBUTED" if distributed else "CB_BENCHMARK"
    return {"ref": f"sha:{sha}", "pipeline": PIPELINE,
            "variables": {kind: "1", "CB_MACHINES": ",".join(machines), "CB_REF": DEVELOP}}


def _listing(path: str, token: str | None) -> list[str]:
    try:
        entries = get_json(CONTENTS.format(path=path), token)
    except HTTPError as e:
        if e.code == 404:  # no run on this machine yet
            return []
        raise
    return [e["name"] for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []


def main(argv: list[str], env: Mapping[str, str]) -> int:
    p = argparse.ArgumentParser(prog="python -m automation trigger",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--sha", required=True, help="the full commit sha")
    p.add_argument("--machines", default=",".join(MACHINES), help="comma-separated")
    p.add_argument("--force", action="store_true", help="also if the results have a run")
    p.add_argument("--dry-run", action="store_true", help="print the request, send nothing")
    p.add_argument("--distributed", action="store_true",
                   help="the distributed jobs, checked against distributed/<machine>/")
    args = p.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        print(f"error: --sha must be a full commit sha, not {args.sha!r}", file=sys.stderr)
        return 2
    asked = {m for m in args.machines.split(",") if m}
    if asked - set(MACHINES):
        print(f"error: unknown machines {sorted(asked - set(MACHINES))}; known: "
              f"{', '.join(MACHINES)}", file=sys.stderr)
        return 2
    machines = [m for m in MACHINES if m in asked]
    short = args.sha[:10]
    prefix, what = ("distributed/", "distributed ") if args.distributed else ("", "")
    try:
        if not args.force:
            token = env.get("GITHUB_TOKEN") or None
            todo = pending(args.sha, lambda m: _listing(prefix + m, token), machines)
            if not todo:
                print(f"nothing to do: {short} has {what}runs on {', '.join(machines)}")
                return 0
            machines = todo
        data = body(args.sha, machines, distributed=args.distributed)
        if args.dry_run:
            print(f"would trigger {PIPELINE} ({what or 'single-node '}jobs) for {short} on "
                  f"{', '.join(machines)}:")
            print(json.dumps(data, indent=2))
            return 0
        missing = [k for k in (REPOSITORY_ID, SECRET) if not env.get(k)]
        if missing:
            print(f"error: {', '.join(missing)} not set", file=sys.stderr)
            return 2
        secret = env[SECRET]
        status, text = post(TRIGGER_URL, json.dumps(data).encode(), (env[REPOSITORY_ID], secret))
    except (HTTPError, URLError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"triggered {PIPELINE} for {short} on {', '.join(machines)}: HTTP {status}")
    if text.strip():
        print(text.strip()[:2000].replace(secret, "***"))
    return 0 if 200 <= status < 300 else 1
