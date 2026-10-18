"""Command line of the dashboard: python -m dashboard <command>."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .frozen import freeze, load_frozen
from .load import load_runs


def _freeze(args) -> int:
    frozen_path = Path(args.frozen)
    old = load_frozen(frozen_path)
    new = freeze(load_runs(Path(args.results)), old)
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(json.dumps(new, indent=1, sort_keys=True) + "\n")
    added = sum(len(v) for v in new.values()) - sum(len(v) for v in old.values())
    print(f"froze {added} new entries into {frozen_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dashboard")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze", help="freeze structure from the first runs")
    p.add_argument("--results", default="results")
    p.add_argument("--frozen", default="dashboard/frozen.json")
    p.set_defaults(func=_freeze)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
