"""Command line of the dashboard: python -m dashboard <command>."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .comments import check_comments
from .frozen import freeze, load_frozen, structures
from .load import load_runs
from .models import load_machines
from .setup import git_script_changed
from .site_data import build_site

REPO = Path(__file__).resolve().parent.parent


def _freeze(args) -> int:
    frozen_path = Path(args.frozen)
    old = load_frozen(frozen_path)
    new = freeze(load_runs(Path(args.results)), old)
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(json.dumps(new, indent=1, sort_keys=True) + "\n")
    added = sum(len(v) for v in new.values()) - sum(len(v) for v in old.values())
    print(f"froze {added} new entries into {frozen_path}")
    return 0


def _build(args) -> int:
    results = Path(args.results)
    runs = load_runs(results)
    if not runs:
        print(f"no runs in {results}", file=sys.stderr)
        return 1
    frozen, warnings = structures(runs, load_frozen(Path(args.frozen)))
    if args.ginkgo:
        warnings += check_comments(Path(args.ginkgo))
    out = Path(args.out)
    build_site(results, Path(args.manifest), Path(args.catalog), out,
               machines=load_machines(), frozen=frozen,
               script_changed=git_script_changed(REPO), warnings=warnings)
    files = sorted(out.rglob("*.json"))
    size = sum(f.stat().st_size for f in files)
    counts = json.loads((out / "meta.json").read_text())["flag_counts"]
    print(f"wrote {len(files)} files ({size / 1e6:.2f} MB) to {out}")
    print("flags: " + ", ".join(f"{tab} {n}" for tab, n in counts.items()))
    for w in warnings:
        print(f"warning: {w}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dashboard")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze", help="freeze structure from the first runs")
    p.add_argument("--results", default="results")
    p.add_argument("--frozen", default="dashboard/frozen.json")
    p.set_defaults(func=_freeze)
    p = sub.add_parser("build", help="write the page data into site/data")
    p.add_argument("--results", default="results")
    p.add_argument("--manifest", default="datasets/manifest.json")
    p.add_argument("--catalog", default="datasets/catalog.toml")
    p.add_argument("--out", default="site/data")
    p.add_argument("--frozen", default="dashboard/frozen.json")
    p.add_argument("--ginkgo", default=None,
                   help="Ginkgo checkout whose copied comments are checked")
    p.set_defaults(func=_build)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
