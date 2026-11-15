"""The weekly distributed run: weak scaling of Ginkgo's distributed benchmarks
from 1 to 64 GPUs, after the MPI benchmark (mpibench/), all in one allocation."""
from __future__ import annotations

import json
import math
import re
import subprocess
from collections.abc import Collection, Mapping
from pathlib import Path

from bench.groups import ADDED, MIN, TIMEOUTS, Invocation, _base
from bench.machine import Machine
from bench.runner import srun_time_minutes

GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64]
ROWS_PER_GPU = 256 ** 3
GROUPS = ["mpi", "spmv", "spgemm", "solver", "preconditioner"]
TARGETS = ["spmv_distributed", "sparse_blas_distributed", "solver_distributed",
           "preconditioner_distributed"]
SOLVERS = ["cg", "gmres"]
PRECONDITIONERS = ["mg-pgm", "mg-uniform", "mg-pmis"]
# the newer ones, which a Ginkgo commit without their develop merge lacks
PRECONDITIONERS_ADDED = {p: ADDED[p] for p in PRECONDITIONERS if p in ADDED}
# the solvers' preconditioner: the solver benchmark names each solver
# <solver>-<preconditioner> in its output
SOLVER_PRECONDITIONER = "schwarz-jacobi"
# the single-node solver group's settings, with that preconditioner
SOLVER_SETTINGS = [
    "-preconditioners", SOLVER_PRECONDITIONER,
    "-rel_res_goal", "1e-10", "-max_iters", "1000",
    "-rhs_generation", "sinus", "-initial_guess_generation", "0",
]
# the ping-pong runs at one GPU count only: 8 has a pair in a node and one across
PINGPONG_GPUS = 8
TIMEOUT_MPI = 20 * MIN
TIMEOUT_GPU_CHECK = 5 * MIN
# srun's line for ranks that died, e.g.
# "srun: error: nid005939: task 4: Segmentation fault (core dumped)"
SRUN_FAILURE = re.compile(r"^srun: error: (\S+): (tasks? [\d,-]+): (.+)$", re.M)


def case(n: int) -> list[dict]:
    """The stdin of every Ginkgo benchmark at n GPUs. The distributed generator
    divides "size" by the ranks, so each GPU gets ROWS_PER_GPU rows."""
    return [{"size": ROWS_PER_GPU * n, "stencil": "7pt", "comm_pattern": "stencil",
             "optimal": {"spmv": "csr-csr"}}]


def srun_args(m: Machine, n: int) -> list[str]:
    """The srun arguments of a step with n ranks, one per GPU."""
    d = _settings(m)
    nodes = math.ceil(n / d.gpus_per_node)
    return [f"--nodes={nodes}", f"--ntasks={n}",
            f"--ntasks-per-node={min(n, d.gpus_per_node)}", *d.srun]


def gpu_check_command(m: Machine, n: int, mpibench: Path) -> list[str]:
    """`mpibench --check-gpus` on the ranks of n GPUs, launched as the benchmarks are."""
    return ["srun", *srun_args(m, n), f"--time={srun_time_minutes(TIMEOUT_GPU_CHECK)}",
            *_settings(m).wrapper, str(mpibench), "--check-gpus"]


def gpu_check(m: Machine, n: int, mpibench: Path, env: Mapping[str, str] | None = None,
              run=subprocess.run) -> list[str]:
    """Check that every rank of n GPUs has a working GPU. Returns the problems,
    one per failed rank or node, or [] if every GPU works. A rank that dies
    before mpibench's report (e.g. in MPI_Init) is named by srun's error line."""
    limit = (srun_time_minutes(TIMEOUT_GPU_CHECK) + 1) * MIN  # srun's own limit ends it first
    try:
        p = run(gpu_check_command(m, n, mpibench), capture_output=True, text=True,
                timeout=limit, env=env)
    except subprocess.TimeoutExpired:
        return [f"no answer within {limit // MIN} min"]
    except OSError as e:
        return [f"cannot start: {e}"]
    report = _gpu_report(p.stdout)
    if report is not None and report["failed"]:
        return [f"{f['node']} (rank {f['rank']}): {f['error']}" for f in report["failed"]]
    if report is not None and p.returncode == 0:
        return []
    died = [f"{node} ({tasks}): {reason}" for node, tasks, reason in SRUN_FAILURE.findall(p.stderr)
            if reason != "Terminated"]  # the ranks srun stopped because another died
    if died:
        return list(dict.fromkeys(died))
    last = next((ln.strip() for ln in reversed(p.stderr.splitlines()) if ln.strip()), "no output")
    return [f"exit code {p.returncode}: {last}"]


def _gpu_report(stdout: str) -> dict | None:
    """mpibench --check-gpus's report: its last line starting with "{"."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("{"):
            try:
                report = json.loads(line)
            except ValueError:
                return None
            return report if isinstance(report, dict) and isinstance(report.get("failed"), list) \
                else None
    return None


def exe(m: Machine, target: str) -> Path:
    """The executable of a distributed benchmark target, e.g. spmv_distributed."""
    name = target.removesuffix("_distributed")
    return _settings(m).build_dir / "benchmark" / name / "distributed" / target


def missing_targets(worktree: Path) -> set[str]:
    """The TARGETS whose benchmark the Ginkgo checkout lacks, e.g. the
    distributed SpGEMM on v1.11.0."""
    return {t for t in TARGETS
            if not (worktree / "benchmark" / t.removesuffix("_distributed")
                    / "distributed").is_dir()}


def invocations(m: Machine, n: int, mpibench: Path,
                unavailable: Collection[str] = ()) -> list[Invocation]:
    """Every invocation at n GPUs, in GROUPS order. `unavailable`: the TARGETS
    and PRECONDITIONERS the Ginkgo commit lacks; their invocations aren't run."""
    def na(target: str, precond: str | None = None) -> str | None:
        if target in unavailable:
            return f"not in this Ginkgo commit (no {target})"
        if precond in unavailable:
            return f"not in this Ginkgo commit (added by {ADDED[precond][:10]})"
        return None

    stdin = json.dumps(case(n))
    mpi_args = ["--ranks-per-node", str(_settings(m).gpus_per_node)]
    if n == PINGPONG_GPUS:
        mpi_args.append("--pingpong")
    out = [Invocation("mpi", "mpibench", mpibench, mpi_args, None, TIMEOUT_MPI)]
    out.append(Invocation("spmv", "stencil", exe(m, "spmv_distributed"),
                          _base(m) + ["-formats", "csr-csr"], stdin, TIMEOUTS["spmv"],
                          unavailable=na("spmv_distributed")))
    out.append(Invocation("spgemm", "stencil", exe(m, "sparse_blas_distributed"),
                          _base(m) + ["-operations", "spgemm"], stdin, TIMEOUTS["spgemm"],
                          unavailable=na("sparse_blas_distributed")))
    for s in SOLVERS:
        out.append(Invocation("solver", s, exe(m, "solver_distributed"),
                              _base(m) + ["-detailed=false"] + SOLVER_SETTINGS + ["-solvers", s],
                              stdin, TIMEOUTS["solver"], unavailable=na("solver_distributed")))
    for p in PRECONDITIONERS:
        out.append(Invocation("preconditioner", p, exe(m, "preconditioner_distributed"),
                              _base(m) + ["-detailed=false", "-preconditioners", p],
                              stdin, TIMEOUTS["preconditioner"],
                              unavailable=na("preconditioner_distributed", p)))
    return out


def _settings(m: Machine):
    if m.distributed is None:
        raise ValueError(f"{m.name} has no [distributed] settings")
    return m.distributed
