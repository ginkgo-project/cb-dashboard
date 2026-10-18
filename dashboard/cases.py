"""One Measurement per case of a run, with a status for every key."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from bench.groups import STENCILS, VECTOR_LENGTHS

from .load import Run

STATUSES = ("ok", "failed", "not_available", "disabled", "missing")
PRECONDITIONERS = ("jacobi", "ilu", "ilu-syncfree", "ic", "ic-syncfree",
                   "spd-isai", "general-isai", "mg-pgm", "mg-pmis",
                   "mg-uniform")
STAGES = ("apply", "generate")


@dataclass
class Measurement:
    key: str
    group: str
    labels: dict[str, str]
    status: str
    time: float | None
    repetition_times: list[float] = field(default_factory=list)
    invocation: str | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)


def plain_preconditioner(encoded: str) -> str:
    """The preconditioner name without the parameters Ginkgo appends."""
    if encoded in PRECONDITIONERS:
        return encoded
    for name in sorted(PRECONDITIONERS, key=len, reverse=True):
        if re.match(re.escape(name) + r"-\d", encoded):
            return name
    return encoded


def _stem(path: str) -> str:
    return Path(path).stem


def _arg(args: list, name: str) -> str | None:
    for i, a in enumerate(args[:-1]):
        if a == name:
            return args[i + 1]
    return None


def _measure(key, group, labels, entry, stage, invocation, extra):
    """A Measurement from a Ginkgo result dict (one with `completed`)."""
    if entry.get("completed"):
        status, error = "ok", None
    elif entry.get("error_type") == "gko::NotImplemented":
        status, error = "not_available", entry.get("error")
    else:
        status, error = "failed", entry.get("error")
    ok = status == "ok"
    return Measurement(
        key, group, labels, status,
        stage.get("time") if ok else None,
        list(stage.get("repetition_times") or []) if ok else [],
        invocation, error, extra)


def measurements(run: Run) -> dict[str, Measurement]:
    out: dict[str, Measurement] = {}
    invs = {g: v.get("invocations") or []
            for g, v in run.record.get("groups", {}).items()}
    labels = {g: {i["label"] for i in v} for g, v in invs.items()}

    for e in run.data.get("blas") or []:
        n = e["n"]
        for op, v in e.get("blas", {}).items():
            key = f"blas/{op}/{n}"
            out[key] = _measure(
                key, "blas", {"operation": op, "n": str(n)}, v, v,
                "blas:vectors",
                {"n": n, "flops_rate": v.get("flops"),
                 "bandwidth_rate": v.get("bandwidth")})

    for e in run.data.get("spmv") or []:
        matrix = _stem(e["filename"])
        for fmt, v in e.get("spmv", {}).items():
            key = f"spmv/{fmt}/{matrix}"
            out[key] = _measure(
                key, "spmv",
                {"operation": "spmv", "format": fmt, "matrix": matrix},
                v, v, f"spmv:{matrix}",
                {"rows": e.get("rows"), "cols": e.get("cols"),
                 "nonzeros": e.get("nonzeros"),
                 "format_parameters": v.get("format_parameters")})

    for e in run.data.get("spgemm") or []:
        matrix = _stem(e["filename"])
        for v in e.get("sparse_blas", {}).values():
            key = f"spgemm/{matrix}"
            out[key] = _measure(
                key, "spgemm", {"operation": "spgemm", "matrix": matrix},
                v, v, f"spgemm:{matrix}",
                {"rows": e.get("rows"), "cols": e.get("cols"),
                 "nonzeros": e.get("nonzeros"),
                 "flops_rate": v.get("flops"),
                 "bandwidth_rate": v.get("bandwidth")})

    for e in run.data.get("solver") or []:
        stencil = e["stencil"]
        for name, v in e.get("solver", {}).items():
            solver = name.removesuffix("-jacobi")
            ortho = (v.get("parameters") or {}).get("ortho_method")
            label = "mgs"
            if solver == "gmres":
                label = ortho or "mgs"
                solver = f"gmres-{label}"
            key = f"solver/{solver}/{stencil}"
            apply = v.get("apply") or {}
            rhs = v.get("rhs_norm")
            res = v.get("residual_norm")
            out[key] = _measure(
                key, "solver", {"solver": solver, "stencil": stencil},
                v, apply, f"solver:{label}",
                {"iterations": apply.get("iterations"),
                 "rows": e.get("rows"), "size": e.get("size"),
                 "optimal": e.get("optimal"),
                 "generate_time": (v.get("generate") or {}).get("time"),
                 "rel_residual": (res / rhs if res is not None and rhs
                                  else None),
                 "parameters": v.get("parameters")})

    pre_labels = labels.get("preconditioner", set())

    def pre_invocation(matrix, plain):
        lab = f"{matrix}/{plain}"
        return "preconditioner:" + (lab if lab in pre_labels else matrix)

    for e in run.data.get("preconditioner") or []:
        matrix = _stem(e["filename"])
        for encoded, v in e.get("preconditioner", {}).items():
            plain = plain_preconditioner(encoded)
            for stage in STAGES:
                key = f"preconditioner/{stage}/{plain}/{matrix}"
                out[key] = _measure(
                    key, "preconditioner",
                    {"stage": stage, "preconditioner": plain,
                     "matrix": matrix},
                    v, v.get(stage) or {}, pre_invocation(matrix, plain),
                    {"rows": e.get("rows"), "nonzeros": e.get("nonzeros"),
                     "storage": v.get("storage"), "levels": v.get("levels"),
                     "parameters": v.get("parameters"), "encoded": encoded})

    for group, group_invs in invs.items():
        for inv in group_invs:
            status = {"disabled": "disabled", "skipped": "missing"}.get(
                inv["status"], "failed")
            if inv["status"] not in ("failed", "killed", "skipped",
                                     "disabled"):
                continue
            reason = inv.get("reason")
            args = inv.get("args") or []
            label = inv["label"]
            for key, labs, name in _covered(group, label, args):
                if key in out:
                    continue
                out[key] = Measurement(
                    key, group, labs, status, None, [],
                    f"{group}:{name}", reason, {})
    return out


def _csv(value: str | None) -> list[str]:
    return [x for x in (value or "").split(",") if x]


def _covered(group, label, args):
    """(key, labels, invocation label) of every key an invocation covers."""
    if group == "blas":
        for op in _csv(_arg(args, "-operations")):
            for n in VECTOR_LENGTHS:
                yield (f"blas/{op}/{n}", {"operation": op, "n": str(n)},
                       label)
    elif group == "spmv":
        for fmt in _csv(_arg(args, "-formats")):
            yield (f"spmv/{fmt}/{label}",
                   {"operation": "spmv", "format": fmt, "matrix": label},
                   label)
    elif group == "spgemm":
        yield (f"spgemm/{label}", {"operation": "spgemm", "matrix": label},
               label)
    elif group == "solver":
        ortho = _arg(args, "-gmres_ortho_method") or label
        for s in _csv(_arg(args, "-solvers")):
            solver, name = (s, "mgs")
            if s == "gmres":
                solver, name = f"gmres-{ortho}", ortho
            for st in STENCILS:
                yield (f"solver/{solver}/{st['stencil']}",
                       {"solver": solver, "stencil": st["stencil"]}, name)
    elif group == "preconditioner":
        if "/" in label:
            matrix, plain = label.split("/", 1)
            names = [plain]
        else:
            matrix = label
            names = _csv(_arg(args, "-preconditioners"))
        for encoded in names:
            plain = plain_preconditioner(encoded)
            for stage in STAGES:
                yield (f"preconditioner/{stage}/{plain}/{matrix}",
                       {"stage": stage, "preconditioner": plain,
                        "matrix": matrix}, label)
