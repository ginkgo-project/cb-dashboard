"""Command line: `python -m bench run <machine>` and `python -m bench sbatch-args <machine>`."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from bench.build import build
from bench.checkout import (DEVELOP, ensure_clone, prepare_worktree, read_last, resolve,
                            should_run, write_last)
from bench.groups import GROUPS, invocations
from bench.machine import load_machine
from bench.record import gpu_info, results_dir_name, run_record, toolchain_info
from bench.runner import run_group

GINKGO_URL = "https://github.com/ginkgo-project/ginkgo.git"
DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "datasets" / "manifest.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_results_dir(base: Path, commit) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    while True:
        path = results_dir_name(base, commit)
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            continue


def cmd_sbatch_args(args) -> int:
    m = load_machine(args.machine, env={**os.environ, "SCRATCH": os.environ.get("SCRATCH", "/unused")})
    print(" ".join([*m.sbatch, f"--uenv={m.uenv}"]))
    return 0


def cmd_run(args) -> int:
    m = load_machine(args.machine)
    groups = [g for g in args.groups.split(",") if g]
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        print(f"error: unknown groups {bad}; known: {', '.join(GROUPS)}", file=sys.stderr)
        return 2
    manifest = json.loads(Path(args.manifest).read_text())
    if args.dry_run:
        print(f"machine: {m.name}")
        print(f"ref: {args.ref}")
        for inv in invocations(m, manifest, groups):
            print(f"{inv.group} {inv.label} {inv.exe} {' '.join(inv.args)}")
        return 0
    if platform.machine() != m.arch:
        print(f"error: {m.name} needs {m.arch} but this host is {platform.machine()}", file=sys.stderr)
        return 2
    clone = m.root / "ginkgo"
    worktree = m.root / "worktrees" / m.name
    state = m.root / "state" / m.name
    ensure_clone(GINKGO_URL, clone)
    commit = resolve(clone, args.ref)
    if not should_run(commit, read_last(state), args.ref, args.force):
        print(f"nothing to do: {commit.sha} already benchmarked")
        return 0
    started = _now()
    gpu, toolchain = gpu_info(m), toolchain_info(m)
    prepare_worktree(clone, worktree, commit.sha)
    out = _make_results_dir(m.results_dir, commit)

    def record(build_info, group_results):
        rec = run_record(m, commit, args.ref, {"cmake_cache": {}}, manifest, group_results,
                         started, _now(), gpu=gpu, toolchain=toolchain)
        rec["build"] = build_info
        rec["cmake_cache"] = build_info.get("cmake_cache", {})
        (out / "run.json").write_text(json.dumps(rec, indent=2))

    try:
        info = build(m, worktree)
    except Exception as e:
        record({"error": f"{type(e).__name__}: {e}"}, {})
        print(f"error: build failed: {e}", file=sys.stderr)
        return 1
    results = {}
    for g in groups:
        invs = invocations(m, manifest, [g])
        results[g] = run_group(g, invs, m.srun, out)
    record(info, results)
    if args.ref == DEVELOP:
        write_last(state, commit.sha)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("machine")
    r.add_argument("--ref", default=DEVELOP)
    r.add_argument("--force", action="store_true")
    r.add_argument("--groups", default=",".join(GROUPS))
    r.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    r.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("sbatch-args")
    s.add_argument("machine")
    args = p.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_sbatch_args(args)


if __name__ == "__main__":
    sys.exit(main())
