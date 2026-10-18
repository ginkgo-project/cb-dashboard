"""Ginkgo benchmark groups and their exact invocations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bench.build import exe
from bench.machine import Machine

GROUPS = ["blas", "spmv", "spgemm", "solver", "preconditioner"]

VECTOR_LENGTHS = [1048576, 16777216, 134217728]
# The solver benchmark rejects a test case without "optimal": {"spmv": ...}
# (benchmark/solver/solver_common.hpp, validate_config) and then exits with 2.
STENCILS = [{"size": 8000000, "stencil": s, "optimal": {"spmv": "csr"}}
            for s in ("5pt", "7pt", "27pt")]
PRECOND_SPD = "jacobi,ilu,ilu-syncfree,ic,ic-syncfree,spd-isai,mg-pgm,mg-pmis,mg-uniform"
PRECOND_GENERAL = "jacobi,ilu,ilu-syncfree,general-isai,mg-pgm,mg-pmis,mg-uniform"
SOLVER_SETTINGS = [
    "-preconditioners", "jacobi", "-jacobi_max_block_size", "1",
    "-rel_res_goal", "1e-10", "-max_iters", "1000",
    "-rhs_generation", "sinus", "-initial_guess_generation", "0",
]
# The 8 MB vectors take about 10 us per operation, and after one warm-up their
# first timed repetition was still up to 1.7x slower than the others.
BLAS_WARMUP = 3
MIN = 60
# The preconditioner's is for one preconditioner on one matrix. On the MI300A,
# ic-syncfree on porous_pressure took 12 min (7 generates of 99 s).
TIMEOUTS = {"blas": 15 * MIN, "spmv": 20 * MIN, "spgemm": 20 * MIN,
            "solver": 60 * MIN, "preconditioner": 15 * MIN}


@dataclass
class Invocation:
    group: str
    label: str
    exe: Path
    args: list[str]
    stdin: str | None
    timeout_s: int
    missing_input: Path | None = None
    disabled: str | None = None  # why it isn't run on this machine


def _base(m: Machine, warmup: int = 1) -> list[str]:
    # A growth factor above 1 (Ginkgo's default is 1.5) times the repetitions in
    # growing batches, even with a fixed count; at 1 each one is timed alone.
    # "time" is the median, which one slow repetition doesn't move; all of them
    # are still in "repetition_times".
    return ["-executor", m.executor, "-warmup", str(warmup), "-repetitions", "5", "-gpu_timer",
            "-repetition_growth_factor", "1", "-timer_method", "median", "-seed", "42"]


def _matrix(m: Machine, entry: dict) -> tuple[list[str], Path | None]:
    path = m.root / entry["path"]
    return ["-input_matrix", str(path)], (None if path.is_file() else path)


def invocations(m: Machine, manifest: dict, groups: list[str] = GROUPS) -> list[Invocation]:
    mats = manifest["matrices"]
    known = set(PRECOND_SPD.split(",")) | set(PRECOND_GENERAL.split(","))
    unknown = sorted(set(m.disabled_preconditioners) - known)
    if unknown:
        raise ValueError(f"{m.name}: unknown disabled_preconditioners {unknown}")
    out: list[Invocation] = []
    for g in GROUPS:
        if g not in groups:
            continue
        t = TIMEOUTS[g]
        if g == "blas":
            out.append(Invocation(g, "vectors", exe(m, "blas"),
                                  _base(m, BLAS_WARMUP) + ["-operations", "copy,dot,norm,axpy"],
                                  json.dumps([{"n": n} for n in VECTOR_LENGTHS]), t))
        elif g == "solver":
            for label, ortho, solvers in (("mgs", "mgs", "cg,bicgstab,gmres,ir"),
                                          ("cgs", "cgs", "gmres"), ("cgs2", "cgs2", "gmres")):
                args = _base(m) + ["-detailed=false"] + SOLVER_SETTINGS + [
                    "-solvers", solvers, "-gmres_ortho_method", ortho]
                out.append(Invocation(g, label, exe(m, "solver"), args, json.dumps(STENCILS), t))
        else:
            for e in mats:
                inp, missing = _matrix(m, e)
                if g == "spmv" and e["set"] == "spmv":
                    fmts = "csr,coo,hybrid" + (",sellp" if e["run_sellp"] else "")
                    args = _base(m) + ["-formats", fmts] + inp
                    target = "spmv"
                elif g == "spgemm" and "spgemm" in e["roles"]:
                    args = _base(m) + ["-operations", "spgemm", "-spgemm_mode", "normal"] + inp
                    target = "sparse_blas"
                elif g == "preconditioner" and "precond" in e["roles"]:
                    # One invocation per preconditioner, so one that hangs or crashes
                    # loses only its own result. Its log is <matrix>/<preconditioner>.log.
                    # A disabled one is still listed, so run.json says it wasn't run.
                    lst = PRECOND_SPD if e.get("spd") else PRECOND_GENERAL
                    for p in lst.split(","):
                        args = _base(m) + ["-detailed=false", "-formats", "csr",
                                           "-preconditioners", p] + inp
                        off = (f"disabled on {m.name}" if p in m.disabled_preconditioners
                               else None)
                        out.append(Invocation(g, f"{e['name']}/{p}", exe(m, "preconditioner"),
                                              args, None, t, missing, off))
                    continue
                else:
                    continue
                out.append(Invocation(g, e["name"], exe(m, target), args, None, t, missing))
    return out
