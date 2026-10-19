"""Command line: `python -m bench run <machine>` and `python -m bench sbatch-args <machine>`."""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bench.build import build
from bench.checkout import (DEVELOP, ensure_clone, lacks, prepare_worktree, read_last,
                            resolve, should_run, write_last)
from bench.energy import snapshot as pm_counters
from bench.groups import ADDED, GROUPS, invocations
from bench.machine import load_machine
from bench.record import (REPO, gpu_info, results_dir_name, run_record, script_version,
                          toolchain_info)
from bench.runner import Killed, job_deadline, run_group
from bench.stream import measure as measure_stream

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


def _write_json(path: Path, data) -> None:
    """Write atomically: a reader (or a kill) never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def cmd_sbatch_args(args) -> int:
    m = load_machine(args.machine, env={**os.environ, "SCRATCH": os.environ.get("SCRATCH", "/unused")})
    print(" ".join([*m.sbatch, f"--uenv={m.uenv}"]))
    return 0


def cmd_run(args) -> int:
    m = load_machine(args.machine)
    requested = [g for g in args.groups.split(",") if g]
    bad = [g for g in requested if g not in GROUPS]
    if bad:
        print(f"error: unknown groups {bad}; known: {', '.join(GROUPS)}", file=sys.stderr)
        return 2
    groups = [g for g in GROUPS if g in requested]  # the spec's order, no repeats
    manifest = json.loads(Path(args.manifest).read_text())
    if args.dry_run:
        print(f"machine: {m.name}")
        print(f"ref: {args.ref}")
        for inv in invocations(m, manifest, groups):
            note = f"  # {inv.disabled}" if inv.disabled else ""
            print(f"{inv.group} {inv.label} {inv.exe} {' '.join(inv.args)}{note}")
        return 0
    if platform.machine() != m.arch:
        print(f"error: {m.name} needs {m.arch} but this host is {platform.machine()}", file=sys.stderr)
        return 2
    # Slurm jobs on Alps start with SIGCHLD blocked. Children inherit the mask, and
    # CMake (libuv) then never sees its own children exit and hangs in configure.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
    clone = m.root / "ginkgo"
    worktree = m.root / "worktrees" / m.name
    state = m.root / "state" / m.name
    ensure_clone(GINKGO_URL, clone)
    commit = resolve(clone, args.ref)
    if not should_run(commit, read_last(state), args.ref, args.force):
        print(f"nothing to do: {commit.sha} already benchmarked")
        return 0
    unavailable = lacks(clone, commit.sha, ADDED)  # e.g. on an older release
    started = _now()
    gpu, toolchain = gpu_info(m), toolchain_info(m)
    counters = pm_counters()  # which energy counters this node has
    # Once: a sync of this repository during the run changes HEAD, not the code running.
    script = script_version(subprocess.run, REPO)
    prepare_worktree(clone, worktree, commit.sha)
    out = _make_results_dir(m.results_dir, commit)

    results = {g: {"status": "pending"} if g in groups else
               {"status": "skipped", "reason": "not requested"} for g in GROUPS}
    build_info: dict | None = None
    stream_info: dict | None = None

    def record(finished=None):
        info = build_info if build_info is not None else {}
        rec = run_record(m, commit, args.ref, {"cmake_cache": info.get("cmake_cache", {})},
                         manifest, results, started, finished, gpu=gpu, toolchain=toolchain,
                         script=script)
        rec["build"] = info
        rec["pm_counters"] = counters
        if stream_info is not None:
            rec["stream"] = stream_info
        _write_json(out / "run.json", rec)

    def skip_pending(reason):
        for g, r in results.items():
            if r["status"] in ("pending", "running"):
                results[g] = {"status": "skipped", "reason": reason}

    killed: list[int] = []

    def on_sigterm(signum, frame):
        if not killed:  # raise once; the handling below must not be interrupted
            killed.append(signum)
            raise Killed(signum)

    previous = signal.signal(signal.SIGTERM, on_sigterm)
    current = None
    try:
        try:
            build_info = build(m, worktree)
        except Exception as e:
            build_info = {"error": f"{type(e).__name__}: {e}"}
            skip_pending("build failed")
            record(_now())
            print(f"error: build failed: {e}", file=sys.stderr)
            return 1
        record()
        stream_info = measure_stream(m, out)  # the node's bandwidth, before the groups
        record()
        deadline = job_deadline()
        for g in groups:
            current = g
            results[g] = {"status": "running"}
            record()
            t = time.monotonic()
            try:
                results[g] = run_group(g, invocations(m, manifest, [g], unavailable),
                                       m.srun, out, deadline=deadline)
            except Exception as e:
                results[g] = {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                              "seconds": time.monotonic() - t, "invocations": []}
            current = None
            record()
        record(_now())
    except Killed as k:
        if build_info is None:
            build_info = {"error": "killed (SIGTERM) during the build"}
        if current is not None:
            results[current] = k.group_result or {"status": "killed",
                                                  "reason": "job killed (SIGTERM)"}
        skip_pending("job killed")
        record(_now())
        print(f"error: killed by signal {k.signum}; run.json written", file=sys.stderr)
        return 128 + k.signum
    finally:
        signal.signal(signal.SIGTERM, previous)
    if args.ref == DEVELOP and groups == GROUPS:
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
