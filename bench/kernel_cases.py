"""The cases of the kernel counters: a few per group, chosen to cover different
structures (see README "Kernel counters"), each run as its own invocation so a
crash or a hang loses one case and the profilers never have to tell cases apart.

The arguments are the suite's (bench.groups); the passes (bench.profilers) add
their own flags. Keys are the dashboard's case keys, without the stage for the
preconditioners, whose one invocation measures both generate and apply.
"""
from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from bench.build import exe
from bench.groups import ADDED, BLAS_WARMUP, PRECOND_GENERAL, PRECOND_SPD, VECTOR_LENGTHS, _base, _matrix
from bench.machine import Machine

SPMV_MATRICES = ("ecology2", "audikw1", "TSOPF_RS_b2383_c1", "com-Orkut")
SPMV_FORMATS = ("csr", "coo", "hybrid", "sellp")
MG_MATRICES = ("curl_curl", "elasticity_p2")
MG = ("mg-pgm", "mg-uniform", "mg-pmis")
BLAS_OPERATIONS = ("dot", "axpy")
# a dense matrix times one vector: in the L2, in the MI300A's Infinity Cache only, in memory
GEMV_SIZES = (2048, 8192, 32768)
STAGES = ("generate", "apply")


@dataclass
class KernelCase:
    key: str
    label: str  # the key, safe as a directory name
    exe: Path
    args: list[str]
    stdin: str | None
    stages: tuple[str, ...]
    missing_input: Path | None = None
    unavailable: str | None = None  # why this Ginkgo commit can't run it


def parse_case_keys(text: str) -> list[str]:
    """`a,b` from the command line, blanks dropped."""
    return [k.strip() for k in text.split(",") if k.strip()]


def _case(key: str, m: Machine, target: str, args: list[str], stdin: str | None = None,
          stages: tuple[str, ...] = (), missing: Path | None = None,
          unavailable: str | None = None) -> KernelCase:
    return KernelCase(key, key.replace("/", "_"), exe(m, target), args, stdin, stages,
                      missing, unavailable)


def _spmv(m: Machine, e: dict, fmt: str) -> KernelCase:
    inp, missing = _matrix(m, e)
    return _case(f"spmv/{fmt}/{e['name']}", m, "spmv", _base(m) + ["-formats", fmt] + inp,
                 missing=missing)


def _preconditioner(m: Machine, e: dict, p: str, unavailable: Collection[str]) -> KernelCase:
    inp, missing = _matrix(m, e)
    na = (f"not in this Ginkgo commit (added by {ADDED[p][:10]})" if p in unavailable else None)
    args = _base(m) + ["-detailed=false", "-formats", "csr", "-preconditioners", p] + inp
    return _case(f"preconditioner/{p}/{e['name']}", m, "preconditioner", args, stages=STAGES,
                 missing=missing, unavailable=na)


def _spgemm(m: Machine, e: dict) -> KernelCase:
    inp, missing = _matrix(m, e)
    return _case(f"spgemm/{e['name']}", m, "sparse_blas",
                 _base(m) + ["-operations", "spgemm", "-spgemm_mode", "normal"] + inp,
                 missing=missing)


def _blas(m: Machine, op: str, n: int) -> KernelCase:
    if op == "gemv":  # Ginkgo's BLAS benchmark has no gemv: mm with one column
        return _case(f"blas/gemv/{n}", m, "blas", _base(m, BLAS_WARMUP) + ["-operations", "mm"],
                     json.dumps([{"n": n, "k": n, "m": 1}]))
    return _case(f"blas/{op}/{n}", m, "blas", _base(m, BLAS_WARMUP) + ["-operations", op],
                 json.dumps([{"n": n}]))


def _from_key(key: str, m: Machine, manifest: dict, unavailable: Collection[str]) -> KernelCase:
    """Any case of the suite, by its dashboard key (a preconditioner's with or
    without the stage). KeyError if the suite has no such case."""
    mats = {e["name"]: e for e in manifest["matrices"]}
    parts = key.split("/")
    group = parts[0]
    if group == "preconditioner" and len(parts) == 4 and parts[1] in STAGES:
        parts = [parts[0], parts[2], parts[3]]
    if group == "spmv" and len(parts) == 3:
        e = mats.get(parts[2])
        if e and e["set"] == "spmv" and parts[1] in SPMV_FORMATS \
                and (parts[1] != "sellp" or e["run_sellp"]):
            return _spmv(m, e, parts[1])
    elif group == "preconditioner" and len(parts) == 3:
        e = mats.get(parts[2])
        if e and "precond" in e["roles"]:
            known = (PRECOND_SPD if e.get("spd") else PRECOND_GENERAL).split(",")
            if parts[1] in known:
                return _preconditioner(m, e, parts[1], unavailable)
    elif group == "spgemm" and len(parts) == 2:
        e = mats.get(parts[1])
        if e and "spgemm" in e["roles"]:
            return _spgemm(m, e)
    elif group == "blas" and len(parts) == 3 and parts[2].isdigit():
        n = int(parts[2])
        if parts[1] in ("copy", "dot", "norm", "axpy") and n in VECTOR_LENGTHS:
            return _blas(m, parts[1], n)
        if parts[1] == "gemv":
            return _blas(m, "gemv", n)
    raise KeyError(f"no such case in the suite: {key!r}")


def kernel_cases(m: Machine, manifest: dict, unavailable: Collection[str] = (),
                 extra: Collection[str] = ()) -> list[KernelCase]:
    """The selected cases, multigrid first (they are the expensive ones and the
    budget is spent in this order), then `extra` cases named by their keys."""
    mats = {e["name"]: e for e in manifest["matrices"]}
    out: list[KernelCase] = []
    for name in MG_MATRICES:
        if name in mats:
            out += [_preconditioner(m, mats[name], p, unavailable) for p in MG]
    for name in SPMV_MATRICES:
        if name in mats:
            out += [_spmv(m, mats[name], f) for f in SPMV_FORMATS
                    if f != "sellp" or mats[name]["run_sellp"]]
    out += [_spgemm(m, e) for e in manifest["matrices"] if "spgemm" in e["roles"]]
    out += [_blas(m, op, n) for op in BLAS_OPERATIONS for n in VECTOR_LENGTHS]
    out += [_blas(m, "gemv", n) for n in GEMV_SIZES]
    have = {c.key for c in out}
    for key in extra:
        case = _from_key(key, m, manifest, unavailable)
        if case.key not in have:
            out.append(case)
            have.add(case.key)
    return out
