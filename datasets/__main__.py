import argparse
import sys
from pathlib import Path

from .build import BuildError, VersionError, build
from .catalog import DEFAULT, load_catalog

DEFAULT_MANIFEST = Path(__file__).with_name("manifest.json")

# `webgraph to arcs <basename>` writes TAB-separated arcs to stdout (webgraph-cli
# 0.5.0, docs.rs/webgraph-cli/latest/src/webgraph_cli/to/arcs.rs.html). The build
# does not use a shell, so wrap in `sh -c` for the redirect. NOTE: confirm with
# `webgraph to arcs --help` on the cluster; `webgraph` must be on PATH
# ($ROOT/tools/cargo/bin, which build.sbatch adds).
DEFAULT_WEBGRAPH_CMD = 'sh -c "webgraph to arcs {basename} > {out}"'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m datasets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="download, convert and write the manifest")
    b.add_argument("--root", required=True, type=Path)
    b.add_argument("--only", nargs="+", metavar="NAME")
    b.add_argument("--catalog", type=Path, default=DEFAULT)
    b.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    b.add_argument("--webgraph-cmd", default=DEFAULT_WEBGRAPH_CMD,
                   help="format string with {basename} and {out} that converts a "
                        "BV graph to a text arc list")
    args = ap.parse_args(argv)
    try:
        build(load_catalog(args.catalog), args.root, args.manifest, args.only,
              args.webgraph_cmd)
    except VersionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (BuildError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
