"""Command line: `python -m automation trigger|ingest ...`."""
from __future__ import annotations

import importlib
import os
import sys

COMMANDS = {"trigger": "start the CSCS benchmark pipeline for a commit"}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print("usage: python -m automation {%s} ..." % ",".join(COMMANDS), file=sys.stderr)
        for name, text in COMMANDS.items():
            print(f"  {name}: {text}", file=sys.stderr)
        return 2
    return importlib.import_module(f"automation.{argv[0]}").main(argv[1:], os.environ)


if __name__ == "__main__":
    sys.exit(main())
