"""Command line: `python -m bench run|distributed|sbatch-args|uenv <machine>`."""
from __future__ import annotations

import argparse
import dataclasses
import getpass
import json
import os
import platform
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from bench import distributed
from bench.build import build
from bench.checkout import (DEVELOP, ensure_clone, lacks, prepare_worktree, read_last,
                            resolve, should_run, write_last)
from bench.energy import snapshot as pm_counters
from bench.groups import ADDED, GROUPS, invocations
from bench.machine import load_machine
from bench.mpibench import build as build_mpibench
from bench.record import (REPO, ci_info, gpu_info, results_dir_name, run_record, script_version,
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


@contextmanager
def _sigterm_raises():
    """Inside, the first SIGTERM raises Killed; later ones are ignored, so the
    handling of the first (writing run.json) isn't interrupted."""
    killed: list[int] = []

    def on_sigterm(signum, frame):
        if not killed:
            killed.append(signum)
            raise Killed(signum)

    previous = signal.signal(signal.SIGTERM, on_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _machine_without_scratch(name: str):
    return load_machine(name, env={**os.environ, "SCRATCH": os.environ.get("SCRATCH", "/unused")})


def cmd_sbatch_args(args) -> int:
    m = _machine_without_scratch(args.machine)
    print(" ".join([*m.sbatch, f"--uenv={m.uenv}"]))
    return 0


def cmd_uenv(args) -> int:
    print(_machine_without_scratch(args.machine).uenv)
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
    # one per account: the CSCS CI's service account and the maintainer run on the
    # same store, and sbatch's singleton dependency only covers one user's jobs
    worktree = m.root / "worktrees" / getpass.getuser() / m.name
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
    if args.path_file:  # for bench/ci.sh, which copies the run into the job's artifacts
        Path(args.path_file).write_text(f"{out}\n")
    ref = args.record_ref or args.ref
    ci = ci_info(os.environ)

    results = {g: {"status": "pending"} if g in groups else
               {"status": "skipped", "reason": "not requested"} for g in GROUPS}
    build_info: dict | None = None
    stream_info: dict | None = None

    def record(finished=None):
        info = build_info if build_info is not None else {}
        rec = run_record(m, commit, ref, {"cmake_cache": info.get("cmake_cache", {})},
                         manifest, results, started, finished, gpu=gpu, toolchain=toolchain,
                         script=script)
        rec["build"] = info
        rec["pm_counters"] = counters
        if stream_info is not None:
            rec["stream"] = stream_info
        if ci is not None:
            rec["ci"] = ci
        _write_json(out / "run.json", rec)

    def skip_pending(reason):
        for g, r in results.items():
            if r["status"] in ("pending", "running"):
                results[g] = {"status": "skipped", "reason": reason}

    current = None
    with _sigterm_raises():
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
    if ref == DEVELOP and groups == GROUPS:
        write_last(state, commit.sha)
    return 0


def cmd_distributed(args) -> int:
    """The weekly distributed run: the MPI benchmark, then weak scaling of the
    distributed Ginkgo benchmarks, at each GPU count in turn, in one job."""
    m = load_machine(args.machine)
    d = m.distributed
    if d is None:
        print(f"error: {m.name} has no [distributed] settings", file=sys.stderr)
        return 2
    try:
        requested = {int(n) for n in args.gpus.split(",") if n}
    except ValueError:
        requested = {-1}
    bad = sorted(requested - set(distributed.GPU_COUNTS))
    if bad:
        print(f"error: unknown GPU counts {bad}; known: "
              f"{', '.join(map(str, distributed.GPU_COUNTS))}", file=sys.stderr)
        return 2
    gpus = [n for n in distributed.GPU_COUNTS if n in requested]
    if args.dry_run:
        print(f"machine: {m.name}")
        print(f"ref: {args.ref}")
        mpibench = d.mpibench_build_dir / "mpibench"
        for n in gpus:
            for inv in distributed.invocations(m, n, mpibench):
                print(f"{n} {inv.group} {inv.label} srun "
                      f"{' '.join(distributed.srun_args(m, n))} {inv.exe} {' '.join(inv.args)}")
        return 0
    if platform.machine() != m.arch:
        print(f"error: {m.name} needs {m.arch} but this host is {platform.machine()}", file=sys.stderr)
        return 2
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})  # see cmd_run
    clone = m.root / "ginkgo"
    # not the single-node worktree: the two kinds of job may run at once
    worktree = m.root / "worktrees" / getpass.getuser() / f"{m.name}-distributed"
    state = m.root / "state" / "distributed" / m.name
    ensure_clone(GINKGO_URL, clone)
    commit = resolve(clone, args.ref)
    if not should_run(commit, read_last(state), args.ref, args.force):
        print(f"nothing to do: {commit.sha} already benchmarked")
        return 0
    started = _now()
    gpu, toolchain = gpu_info(m), toolchain_info(m)
    counters = pm_counters()
    script = script_version(subprocess.run, REPO)
    prepare_worktree(clone, worktree, commit.sha)
    out = _make_results_dir(d.results_dir, commit)
    if args.path_file:
        Path(args.path_file).write_text(f"{out}\n")
    ref = args.record_ref or args.ref
    ci = ci_info(os.environ)

    results = {str(n): {g: {"status": "pending"} for g in distributed.GROUPS} for n in gpus}
    build_info: dict | None = None
    mpibench_info: dict = {}

    def record(finished=None):
        info = build_info if build_info is not None else {}
        rec = run_record(m, commit, ref, {"cmake_cache": info.get("cmake_cache", {})},
                         {"version": None}, {}, started, finished, gpu=gpu,
                         toolchain=toolchain, script=script)
        rec["kind"] = "distributed"
        rec["distributed"] = {"gpu_counts": gpus, "gpus_per_node": d.gpus_per_node,
                              "nodes": os.environ.get("SLURM_JOB_NODELIST"),
                              "rows_per_gpu": distributed.ROWS_PER_GPU}
        rec["gpus"] = results
        rec["build"] = info
        rec["mpibench_build"] = mpibench_info
        rec["pm_counters"] = counters
        if ci is not None:
            rec["ci"] = ci
        _write_json(out / "run.json", rec)

    def skip_pending(reason):
        for groups in results.values():
            for g, r in groups.items():
                if r["status"] in ("pending", "running"):
                    groups[g] = {"status": "skipped", "reason": reason}

    current = None
    with _sigterm_raises():
        try:
            try:
                # the machine's flags, then the distributed ones (a later -D wins)
                build_info = build(dataclasses.replace(
                    m, build_dir=d.build_dir, cmake_flags=m.cmake_flags + d.cmake_flags),
                    worktree, targets=distributed.TARGETS)
            except Exception as e:
                build_info = {"error": f"{type(e).__name__}: {e}"}
                skip_pending("build failed")
                record(_now())
                print(f"error: build failed: {e}", file=sys.stderr)
                return 1
            mpibench = None
            try:
                mpibench = build_mpibench(m)
            except Exception as e:  # the Ginkgo benchmarks still run
                mpibench_info = {"error": f"{type(e).__name__}: {e}"}
                print(f"error: mpibench build failed: {e}", file=sys.stderr)
            record()
            os.environ.update(d.env)  # srun passes it on (--export=ALL)
            deadline = job_deadline()
            for n in gpus:
                for g in distributed.GROUPS:
                    if g == "mpi" and mpibench is None:
                        results[str(n)][g] = {"status": "failed",
                                              "reason": "mpibench build failed",
                                              "seconds": 0.0, "invocations": []}
                        continue
                    current = (str(n), g)
                    results[str(n)][g] = {"status": "running"}
                    record()
                    t = time.monotonic()
                    try:
                        invs = [i for i in distributed.invocations(m, n, mpibench)
                                if i.group == g]
                        results[str(n)][g] = run_group(g, invs, distributed.srun_args(m, n),
                                                       out / f"gpus-{n}", deadline=deadline)
                    except Exception as e:
                        results[str(n)][g] = {"status": "failed",
                                              "reason": f"{type(e).__name__}: {e}",
                                              "seconds": time.monotonic() - t,
                                              "invocations": []}
                    current = None
                    record()
            record(_now())
        except Killed as k:
            if build_info is None:
                build_info = {"error": "killed (SIGTERM) during the build"}
            if current is not None:
                n, g = current
                results[n][g] = k.group_result or {"status": "killed",
                                                   "reason": "job killed (SIGTERM)"}
            skip_pending("job killed")
            record(_now())
            print(f"error: killed by signal {k.signum}; run.json written", file=sys.stderr)
            return 128 + k.signum
    if ref == DEVELOP and gpus == distributed.GPU_COUNTS:
        write_last(state, commit.sha)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("machine")
    r.add_argument("--ref", default=DEVELOP)
    # benchmark --ref but record this ref, e.g. a sha the CSCS CI was triggered
    # with for develop's head
    r.add_argument("--record-ref", default=None)
    r.add_argument("--path-file", default=None,
                   help="write the path of the run directory to this file")
    r.add_argument("--force", action="store_true")
    r.add_argument("--groups", default=",".join(GROUPS))
    r.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    r.add_argument("--dry-run", action="store_true")
    dist = sub.add_parser("distributed", help="the weekly distributed run, every GPU count")
    dist.add_argument("machine")
    dist.add_argument("--ref", default=DEVELOP)
    dist.add_argument("--record-ref", default=None)
    dist.add_argument("--path-file", default=None,
                      help="write the path of the run directory to this file")
    dist.add_argument("--force", action="store_true")
    dist.add_argument("--gpus", default=",".join(map(str, distributed.GPU_COUNTS)),
                      help="comma-separated GPU counts")
    dist.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("sbatch-args")
    s.add_argument("machine")
    u = sub.add_parser("uenv", help="print the uenv of the machine")
    u.add_argument("machine")
    args = p.parse_args(argv)
    return {"run": cmd_run, "distributed": cmd_distributed, "sbatch-args": cmd_sbatch_args,
            "uenv": cmd_uenv}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
