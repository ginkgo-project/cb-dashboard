"""The shell command that reruns one case of a run on the cluster."""
from __future__ import annotations

import json
import shlex

from .cases import Measurement
from .load import Run

TARGETS = {"blas": "blas", "spmv": "spmv", "spgemm": "sparse_blas",
           "solver": "solver", "preconditioner": "preconditioner"}


def _narrow(args: list, flag: str, value: str) -> list:
    """args with the token after `flag` replaced by `value`."""
    out = list(args)
    for i, a in enumerate(out[:-1]):
        if a == flag:
            out[i + 1] = value
            break
    return out


def command(run: Run, m: Measurement) -> str | None:
    invs = run.record.get("groups", {}).get(m.group, {}).get(
        "invocations") or []
    inv = next((i for i in invs
                if f"{m.group}:{i['label']}" == m.invocation), None)
    if inv is None or m.group not in TARGETS:
        return None
    args = list(inv.get("args") or [])
    stdin = None
    if m.group == "blas":
        args = _narrow(args, "-operations", m.labels["operation"])
        stdin = json.dumps([{"n": m.extra["n"]}])
    elif m.group == "spmv":
        args = _narrow(args, "-formats", m.labels["format"])
    elif m.group == "solver":
        solver = m.labels["solver"]
        if solver.startswith("gmres-"):
            solver = "gmres"
        args = _narrow(args, "-solvers", solver)
        stdin = json.dumps([{"size": m.extra["size"],
                             "stencil": m.labels["stencil"],
                             "optimal": m.extra["optimal"]}])
    elif m.group == "preconditioner":
        args = _narrow(args, "-preconditioners", m.labels["preconditioner"])
    t = TARGETS[m.group]
    binding = run.record["gpu"]["binding"]["srun"]
    exe = f"$SCRATCH/cb/build/{run.machine}/benchmark/{t}/{t}"
    cmd = shlex.join(binding) + " " + exe + " " + shlex.join(args)
    if stdin is not None:
        cmd += " <<< " + shlex.quote(stdin)
    return cmd
