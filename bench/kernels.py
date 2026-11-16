"""The kernel counter stage of a single-node run (README "Kernel counters").

After the five timing groups, each selected case (bench.kernel_cases) runs in its
passes (bench.profilers) under srun; its rows (bench.attribute) go to
<run>/kernels.json, rewritten after every case. The stage's record goes to
run.json["kernels"], outside "groups": nothing here fails the run, changes the
timing history or keeps the import from taking the run.

The time pass's JSON and each pass's stderr stay in <run>/logs/kernels/<case>/;
the profilers' outputs (large: an ncu pass of a multigrid case is ~14 MB) go to
raw_dir(), outside the run directory, which bench/ci.sh copies into the CI
artifacts and the import takes only up to 50 MB.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Callable

from bench import attribute as A
from bench.kernel_cases import KernelCase
from bench.machine import Machine
from bench.profilers import (measured_ranges, passes, read_detailed, read_ncu, read_nsys,
                             read_rocprof)
from bench.runner import srun_time_minutes


def raw_dir(m: Machine, out: Path) -> Path:
    """$SCRATCH/cb/kernels-raw/<machine>/<run dir>."""
    return m.results_dir.parent.parent / "kernels-raw" / m.name / Path(out).name


def _aggregate(statuses: list[str]) -> str:
    if not statuses or all(s == "skipped" for s in statuses):
        return "skipped"
    if all(s in ("ok", "not_available") for s in statuses):
        return "ok"
    if not any(s in ("ok", "partial") for s in statuses):
        return "failed"
    return "partial"


def _run_pass(run: Callable, argv: list[str], stdin: str | None, stdout: Path, stderr: Path,
              timeout: int) -> tuple[int | None, str | None]:
    """(exit code, problem) of one command; the problem is None if it exited 0."""
    with open(stdout, "w") as so, open(stderr, "w") as se:
        try:
            r = run(argv, input=stdin.encode() if stdin is not None else None, stdout=so, stderr=se,
                    timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, f"timed out after {timeout} s"
        except OSError as e:
            return None, f"cannot start: {e}"
    return r.returncode, None if r.returncode == 0 else f"exit {r.returncode}"


def _rows(m: Machine, case: KernelCase, done: dict, problems: list[str]) -> list[dict]:
    """The case's rows from the passes that exited 0; a pass that can't be read adds a problem."""
    tool = m.kernels.tool
    measured, stages = measured_ranges(case), case.stages

    def read(p, f):
        try:
            return f(p.out)
        except (OSError, ValueError, KeyError, IndexError, sqlite3.Error) as e:
            problems.append(f"cannot read {p.name}: {type(e).__name__}: {e}")
            return None

    times = read(done["time"], lambda path: read_detailed(Path(path).read_text(), stages)) or {}
    trace: dict = {}
    parts = []
    if tool == "ncu":
        if "trace" in done:
            got = read(done["trace"], read_nsys)
            if got is not None:
                trace = A.attribute_trace(*got, measured, stages)
        for name, p in done.items():
            if name.startswith("counters-"):
                got = read(p, lambda path: read_ncu(Path(path).read_text()))
                if got is not None:
                    parts.append(A.attribute_counters(tool, got, measured, stages))
    else:
        for name, p in done.items():
            if name.startswith("counters-"):
                got = read(p, read_rocprof)
                if got is None:
                    continue
                ranges, kernels, copies = got
                if not ranges and not kernels:
                    problems.append(f"cannot read {name}: no ranges or kernels")
                    continue
                if name == "counters-0":  # the pass that also traced the copies
                    trace = A.attribute_trace(ranges, kernels, copies, measured, stages)
                parts.append(A.attribute_counters(tool, kernels, measured, stages, ranges))
    return A.rows(trace, parts, times)


def _run_case(m: Machine, case: KernelCase, out: Path, deadline: float | None,
              run: Callable) -> tuple[dict, list[dict]]:
    k = m.kernels
    logdir = out / "logs" / "kernels" / case.label
    raw = raw_dir(m, out) / case.label
    logdir.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    srun = ["srun", *m.srun, f"--time={srun_time_minutes(k.timeout_s)}"]
    rec_passes, problems, done = [], [], {}
    for p in passes(m, case, logdir, raw):
        if deadline is not None and deadline - time.time() < k.timeout_s:
            problems.append(f"{p.name}: not enough job time")
            break
        t = time.monotonic()
        code, problem = _run_pass(run, srun + p.argv, p.stdin, p.stdout, p.stderr, k.timeout_s + 60)
        if problem is None and p.post:
            _, post_problem = _run_pass(run, p.post, None, logdir / f"{p.name}-post.out",
                                        logdir / f"{p.name}-post.err", k.timeout_s)
            problem = post_problem and f"export: {post_problem}"
        rec_passes.append({"name": p.name, "exit_code": code, "seconds": time.monotonic() - t})
        if problem is not None:
            problems.append(f"{p.name}: {problem}")
            if p.name == "time":  # the benchmark itself fails: nothing to profile
                return ({"status": "failed", "reason": "; ".join(problems), "passes": rec_passes}, [])
            continue
        done[p.name] = p
    rows = _rows(m, case, done, problems) if "time" in done else []
    status = "ok" if not problems else ("partial" if "time" in done else "failed")
    return {"status": status, "reason": "; ".join(problems) or None, "passes": rec_passes}, rows


def run_kernels(m: Machine, cases: list[KernelCase], out: Path, deadline: float | None,
                run: Callable = subprocess.run) -> dict:
    """Run the cases in order within the machine's budget; write <out>/kernels.json after
    every case; return the stage's record for run.json["kernels"]."""
    k = m.kernels
    if k is None:
        raise ValueError(f"{m.name} has no [kernels] settings")
    out = Path(out)
    t0 = time.monotonic()
    doc = {"tool": k.tool, "cases": {}}
    records = []
    for case in cases:
        t = time.monotonic()
        rows: list[dict] = []
        if case.unavailable is not None:
            rec = {"status": "not_available", "reason": case.unavailable, "passes": []}
        elif case.missing_input is not None:
            rec = {"status": "failed", "reason": "missing input", "passes": []}
        elif time.monotonic() - t0 >= k.budget_s:
            rec = {"status": "skipped", "reason": f"the kernel counter budget ({k.budget_s} s) is used up",
                   "passes": []}
        elif deadline is not None and deadline - time.time() < k.timeout_s:
            rec = {"status": "skipped", "reason": "not enough job time", "passes": []}
        else:
            rec, rows = _run_case(m, case, out, deadline, run)
        records.append({"key": case.key, **rec, "seconds": time.monotonic() - t})
        entry = {"status": rec["status"], "operations": rows}
        if rec.get("reason"):
            entry["reason"] = rec["reason"]
        doc["cases"][case.key] = entry
        out.mkdir(parents=True, exist_ok=True)
        (out / "kernels.json").write_text(json.dumps(doc, separators=(",", ":")))
    return {"status": _aggregate([r["status"] for r in records]),
            "seconds": time.monotonic() - t0, "cases": records}
