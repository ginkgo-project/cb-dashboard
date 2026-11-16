"""The kernel counters' passes and the profilers' output.

Each case runs in several passes (README "Kernel counters"):
- "time": the benchmark with Ginkgo's -detailed timings, no profiler; the only
  times we keep, since profiling distorts them;
- "trace" (Daint): Nsight Systems, for the kernel launches, allocations and
  copies per Ginkgo operation;
- "counters-<i>": one per metrics group of the machine: Nsight Compute on Daint,
  rocprofv3 on Beverin (whose first counters pass also traces the ranges, kernels
  and copies).

The profiled passes run the benchmark with -profile (1 repetition, no warm-up,
Ginkgo's NVTX/ROCTX ranges on), after the suite's own flags so that it wins.
"""
from __future__ import annotations

import csv
import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from bench.kernel_cases import KernelCase
from bench.machine import Machine

# The benchmarks time their work in ranges named "repetition" (with the stage,
# "repetition generate" and "repetition apply", in the preconditioner benchmark);
# the matrix reading, the conversions and the preconditioner benchmark's second
# generate for its storage ("storage generate") are outside them.
MEASURED = "repetition"
NCU_FILTER = "regex:repetition.*/"


@dataclass
class Pass:
    name: str
    argv: list[str]  # after srun's options
    out: Path  # what the parsers read: a file or a directory
    stdin: str | None
    stdout: Path
    stderr: Path
    post: list[str] | None = None  # run after the pass, e.g. exporting the trace


def measured_ranges(case: KernelCase) -> tuple[str, ...]:
    """The names of the benchmark's ranges whose kernels count, one per stage."""
    if case.stages:
        return tuple(f"{MEASURED} {s}" for s in case.stages)
    return (MEASURED,)


def passes(m: Machine, case: KernelCase, logdir: Path, rawdir: Path | None = None) -> list[Pass]:
    """The case's passes. The time pass's JSON and every pass's stderr go to `logdir`; the
    profilers' outputs and the profiled benchmark's stdout to `rawdir` (default `logdir`)."""
    k = m.kernels
    if k is None:
        raise ValueError(f"{m.name} has no [kernels] settings")
    logdir = Path(logdir)
    rawdir = Path(rawdir) if rawdir is not None else logdir
    bench = [str(case.exe), *case.args]
    profiled = bench + ["-profile"]

    def p(name, argv, out, stdout=None, post=None):
        return Pass(name, argv, out, case.stdin, stdout or rawdir / f"{name}.out",
                    logdir / f"{name}.err", post)

    out = [p("time", bench + ["-detailed"], logdir / "time.json", stdout=logdir / "time.json")]
    if k.tool == "ncu":
        rep = rawdir / "trace"
        out.append(p("trace", [*k.tracer, "profile", "-t", "cuda,nvtx", "--force-overwrite", "true",
                               "-o", str(rep), *profiled],
                     rawdir / "trace.sqlite",
                     post=[*k.tracer, "export", "--type", "sqlite", "--force-overwrite", "true",
                           "-o", str(rawdir / "trace.sqlite"), str(rep) + ".nsys-rep"]))
        for i, group in enumerate(k.metrics):
            csv = rawdir / f"counters-{i}.csv"
            out.append(p(f"counters-{i}",
                         [*k.profiler, "--nvtx", "--nvtx-include", NCU_FILTER,
                          "--cache-control", "none", "--clock-control", "none",
                          # the symbol: nsys and ncu print readable names differently
                          "--print-kernel-base", "mangled",
                          "--metrics", ",".join(group), "--csv", "--log-file", str(csv), *profiled],
                         csv))
    else:  # rocprofv3: the first counters pass also traces what the counts need
        for i, group in enumerate(k.metrics):
            d = rawdir / f"counters-{i}"
            # Ginkgo's ranges don't synchronize, so a kernel may run after its range ends: it is
            # placed by its launch call (the HIP runtime trace), not by when it ran
            trace = ["--marker-trace", "--kernel-trace", "--hip-runtime-trace"] + (
                ["--memory-copy-trace"] if i == 0 else [])
            out.append(p(f"counters-{i}",
                         [*k.profiler, "--pmc", *group, *trace, "-d", str(d), "-o", "run",
                          "--output-format", "csv", "--", *profiled],
                         d))
    return out


# --- reading the profilers' output ------------------------------------------------------------


@dataclass
class Range:
    """One of Ginkgo's (or the benchmark's) NVTX/ROCTX ranges, on the CPU."""
    name: str
    start: int  # ns
    end: int
    thread: int
    parent: int | None = None  # index of the enclosing range of the same thread


@dataclass
class Kernel:
    """One kernel launch. `launch` and `thread` are the CPU call that launched it (nsys,
    rocprofv3); `stack` is the range stack at the launch, outermost first (ncu)."""
    name: str
    launch: int | None = None
    thread: int | None = None
    stack: list[str] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    correlation: int | None = None
    display: str | None = None  # a readable name where `name` is the mangled symbol (nsys)


@dataclass
class Copy:
    launch: int | None
    thread: int | None
    bytes: int | None  # None where the trace has no size (rocprofv3 of ROCm 6.3)
    kind: str = ""


def _number(text: str) -> float | None:
    """A metric value as ncu prints it: digit grouping with ' or , and a . for decimals."""
    t = text.strip().replace("'", "").replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


_RANGE = re.compile(r'"([^"]*)"')
# Ginkgo's ranges are in NVTX's default domain. Thrust and CUB (CCCL) put their own ranges,
# in their own domain, inside Ginkgo's operations; they don't place kernels, so a sort's
# kernels stay with the Ginkgo operation (components::sort_row_major) that has the time.
DEFAULT_DOMAIN = "<default domain>:"


def _ncu_stack(text: str) -> list[str]:
    """`"<default domain>:dot:none:...:ASCII:dot"  "..."` -> ["dot", ...], outermost first;
    the ranges of other NVTX domains are left out."""
    names = []
    for part in _RANGE.findall(text):
        if not part.startswith(DEFAULT_DOMAIN):
            continue
        names.append(part.split(":ASCII:", 1)[1].strip() if ":ASCII:" in part
                     else part[len(DEFAULT_DOMAIN):].split(":", 1)[0].strip())
    return names


def read_ncu(csv_text: str) -> list[Kernel]:
    """ncu --csv --nvtx: one row per kernel and metric; one Kernel per launch ("ID")."""
    lines = [l for l in csv_text.splitlines() if l and not l.startswith("==")]
    rows = list(csv.reader(lines))
    if not rows:
        return []
    head = rows[0]
    kid, name, mname, mval = (head.index(h) for h in ("ID", "Kernel Name", "Metric Name",
                                                      "Metric Value"))
    stack = next((i for i, h in enumerate(head) if "Push/Pop_Range" in h), None)
    kernels: dict[str, Kernel] = {}
    for r in rows[1:]:
        if len(r) != len(head):
            continue
        k = kernels.get(r[kid])
        if k is None:
            k = kernels[r[kid]] = Kernel(r[name], stack=_ncu_stack(r[stack]) if stack is not None else [])
        v = _number(r[mval])
        if v is not None:
            k.metrics[r[mname]] = v
    return list(kernels.values())


def _nest(ranges: list[Range]) -> list[Range]:
    """Sort by thread and start (longest first on ties) and set each range's parent."""
    ranges.sort(key=lambda r: (r.thread, r.start, -r.end))
    open_: list[int] = []
    thread = None
    for i, r in enumerate(ranges):
        if r.thread != thread:
            open_, thread = [], r.thread
        while open_ and ranges[open_[-1]].end < r.start:
            open_.pop()
        r.parent = open_[-1] if open_ else None
        open_.append(i)
    return ranges


NVTX_PUSH_POP = 59  # NVTX_EVENTS.eventType of push/pop ranges; domainId 0 is the default domain


def read_nsys(sqlite_path: Path) -> tuple[list[Range], list[Kernel], list[Copy]]:
    """The tables of `nsys export --type sqlite`: ranges, kernels and copies, the latter two
    with the CPU time and thread of the call that launched them."""
    import sqlite3  # here, not at the top: Beverin's venv Python has no sqlite3, and only nsys needs it
    con = sqlite3.connect(str(sqlite_path))
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    strings = dict(con.execute("select id, value from StringIds")) if "StringIds" in tables else {}
    launch = {}
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        launch = {c: (s, t) for c, s, t in con.execute(
            "select correlationId, start, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME")}
    ranges = [Range(text if text is not None else strings.get(tid, ""), start, end, thread)
              for start, end, text, tid, thread in con.execute(
                  "select start, end, text, textId, globalTid from NVTX_EVENTS "
                  "where eventType = ? and end is not null and coalesce(domainId, 0) = 0",
                  (NVTX_PUSH_POP,))]
    kernels = []
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        cols = {r[1] for r in con.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        symbol = "mangledName" if "mangledName" in cols else "demangledName"
        for corr, sym, name in con.execute(
                f"select correlationId, {symbol}, demangledName from CUPTI_ACTIVITY_KIND_KERNEL "
                "order by start"):
            at, thread = launch.get(corr, (None, None))
            readable = strings.get(name, str(name))
            # the symbol joins with ncu (--print-kernel-base mangled); the readable name is shown
            kernels.append(Kernel(strings.get(sym, readable) or readable, at, thread,
                                  correlation=corr, display=readable))
    copies = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for corr, nbytes, kind in con.execute(
                "select correlationId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start"):
            at, thread = launch.get(corr, (None, None))
            copies.append(Copy(at, thread, int(nbytes), str(kind)))
    con.close()
    return _nest(ranges), kernels, copies


def _find(node, key):
    """The first dict under `key` anywhere in `node` (depth first)."""
    if isinstance(node, dict):
        if isinstance(node.get(key), dict):
            return node[key]
        for v in node.values():
            found = _find(v, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for v in node:
            found = _find(v, key)
            if found is not None:
                return found
    return None


def _timed(node):
    """The first dict with a numeric "time" anywhere in `node`."""
    if isinstance(node, dict):
        if isinstance(node.get("time"), (int, float)):
            return node
        for v in node.values():
            found = _timed(v)
            if found is not None:
                return found
    elif isinstance(node, list):
        for v in node:
            found = _timed(v)
            if found is not None:
                return found
    return None


def read_detailed(json_text: str, stages: tuple[str, ...]) -> dict[str, dict]:
    """The time pass's output: per stage ("" if the case has none) the total and the exclusive
    seconds of each operation (Ginkgo's "components"; BLAS has none)."""
    data = json.loads(json_text)
    out = {}
    for stage in stages or ("",):
        node = _find(data, stage) if stage else _timed(data)
        if node is None:
            continue
        comps = node.get("components") if isinstance(node.get("components"), dict) else {}
        out[stage] = {"total": node.get("time"),
                      "components": {k: float(v) for k, v in comps.items()
                                     if isinstance(v, (int, float))}}
    return out


def _csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _int(text) -> int | None:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def read_rocprof(directory: Path, prefix: str = "run") -> tuple[list[Range], list[Kernel], list[Copy]]:
    """One rocprofv3 pass (-d <directory> -o run --output-format csv): the ROCTX ranges, the
    kernels with their counters, and the copies; kernels and copies carry the CPU time and
    thread of the HIP call that launched them (--hip-runtime-trace)."""
    d = Path(directory)
    ranges = [Range(r["Function"], int(r["Start_Timestamp"]), int(r["End_Timestamp"]),
                    int(r["Thread_Id"]))
              for r in _csv_rows(d / f"{prefix}_marker_api_trace.csv")]
    launch = {r["Correlation_Id"]: (int(r["Start_Timestamp"]), int(r["Thread_Id"]))
              for r in _csv_rows(d / f"{prefix}_hip_api_trace.csv")}
    metrics: dict[str, dict[str, float]] = defaultdict(dict)
    for r in _csv_rows(d / f"{prefix}_counter_collection.csv"):
        v = _number(r["Counter_Value"])
        if v is not None:
            m = metrics[r["Dispatch_Id"]]
            m[r["Counter_Name"]] = m.get(r["Counter_Name"], 0.0) + v
    kernels = []
    for r in _csv_rows(d / f"{prefix}_kernel_trace.csv"):
        at, thread = launch.get(r["Correlation_Id"], (None, None))
        kernels.append(Kernel(r["Kernel_Name"], at, thread, metrics=dict(metrics.get(r["Dispatch_Id"], {})),
                              correlation=_int(r["Correlation_Id"])))
    copies = []
    for r in _csv_rows(d / f"{prefix}_memory_copy_trace.csv"):
        at, thread = launch.get(r["Correlation_Id"], (None, None))
        size = r.get("Bytes", r.get("Size"))
        copies.append(Copy(at, thread, _int(size) if size is not None else None, r.get("Direction", "")))
    return _nest(ranges), kernels, copies
