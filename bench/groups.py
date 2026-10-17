"""Ginkgo benchmark groups and their exact invocations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bench.build import exe
from bench.machine import Machine

GROUPS = ["blas", "spmv", "spgemm", "solver", "preconditioner"]

VECTOR_LENGTHS = [1048576, 16777216, 134217728]
STENCILS = [{"size": 8000000, "stencil": s} for s in ("5pt", "7pt", "27pt")]
PRECOND_SPD = "jacobi,ilu,ilu-syncfree,ic,ic-syncfree,spd-isai,mg-pgm,mg-pmis,mg-uniform"
PRECOND_GENERAL = "jacobi,ilu,ilu-syncfree,general-isai,mg-pgm,mg-pmis,mg-uniform"
SOLVER_SETTINGS = [
    "-preconditioners", "jacobi", "-jacobi_max_block_size", "1",
    "-rel_res_goal", "1e-10", "-max_iters", "1000",
    "-rhs_generation", "sinus", "-initial_guess_generation", "0",
]
MIN = 60
TIMEOUTS = {"blas": 15 * MIN, "spmv": 20 * MIN, "spgemm": 20 * MIN,
            "solver": 60 * MIN, "preconditioner": 30 * MIN}


@dataclass
class Invocation:
    group: str
    label: str
    exe: Path
    args: list[str]
    stdin: str | None
    timeout_s: int
    missing_input: Path | None = None


def _base(m: Machine) -> list[str]:
    return ["-executor", m.executor, "-warmup", "1", "-repetitions", "5", "-gpu_timer"]


def _matrix(m: Machine, entry: dict) -> tuple[list[str], Path | None]:
    path = m.root / entry["path"]
    return ["-input_matrix", str(path)], (None if path.is_file() else path)


def invocations(m: Machine, manifest: dict, groups: list[str] = GROUPS) -> list[Invocation]:
    mats = manifest["matrices"]
    out: list[Invocation] = []
    for g in GROUPS:
        if g not in groups:
            continue
        t = TIMEOUTS[g]
        if g == "blas":
            out.append(Invocation(g, "vectors", exe(m, "blas"),
                                  _base(m) + ["-operations", "copy,dot,norm,axpy"],
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
                    lst = PRECOND_SPD if e.get("spd") else PRECOND_GENERAL
                    args = _base(m) + ["-detailed=false", "-formats", "csr",
                                       "-preconditioners", lst] + inp
                    target = "preconditioner"
                else:
                    continue
                out.append(Invocation(g, e["name"], exe(m, target), args, None, t, missing))
    return out
