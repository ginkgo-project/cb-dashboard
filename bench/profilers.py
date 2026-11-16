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

from dataclasses import dataclass
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


def passes(m: Machine, case: KernelCase, logdir: Path) -> list[Pass]:
    k = m.kernels
    if k is None:
        raise ValueError(f"{m.name} has no [kernels] settings")
    logdir = Path(logdir)
    bench = [str(case.exe), *case.args]
    profiled = bench + ["-profile"]

    def p(name, argv, out, stdout=None, post=None):
        return Pass(name, argv, out, case.stdin, stdout or logdir / f"{name}.out",
                    logdir / f"{name}.err", post)

    out = [p("time", bench + ["-detailed"], logdir / "time.json", stdout=logdir / "time.json")]
    if k.tool == "ncu":
        rep = logdir / "trace"
        out.append(p("trace", [*k.tracer, "profile", "-t", "cuda,nvtx", "--force-overwrite", "true",
                               "-o", str(rep), *profiled],
                     logdir / "trace.sqlite",
                     post=[*k.tracer, "export", "--type", "sqlite", "--force-overwrite", "true",
                           "-o", str(logdir / "trace.sqlite"), str(rep) + ".nsys-rep"]))
        for i, group in enumerate(k.metrics):
            csv = logdir / f"counters-{i}.csv"
            out.append(p(f"counters-{i}",
                         [*k.profiler, "--nvtx", "--nvtx-include", NCU_FILTER,
                          "--cache-control", "none", "--clock-control", "none",
                          "--metrics", ",".join(group), "--csv", "--log-file", str(csv), *profiled],
                         csv))
    else:  # rocprofv3: the first counters pass also traces what the counts need
        for i, group in enumerate(k.metrics):
            d = logdir / f"counters-{i}"
            trace = ["--marker-trace", "--kernel-trace"] + (["--memory-copy-trace"] if i == 0 else [])
            out.append(p(f"counters-{i}",
                         [*k.profiler, "--pmc", *group, *trace, "-d", str(d), "-o", "run",
                          "--output-format", "csv", "--", *profiled],
                         d))
    return out
