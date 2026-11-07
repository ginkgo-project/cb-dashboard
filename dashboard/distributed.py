"""The Distributed tab's data: the weekly distributed runs under
results/distributed/<machine>/<dir>/ (python -m bench distributed).

distributed.json lists each machine's runs, oldest first, and per row (a
benchmark and phase, or an MPI operation and size) and GPU count the time of
each run with its noise band and status. distributed/<machine>/<dir>.json has
what an opened row shows besides: the problem size, the solver iterations and
the multigrid hierarchy per GPU count.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bench.distributed import GPU_COUNTS, PRECONDITIONERS, SOLVERS

from .changes import BAND_FLOOR, BASELINE_WINDOW, noise_band
from .load import _read_json, _tracked

GROUPS = ("mpi", "spmv", "spgemm", "solver", "preconditioner")
PHASES = ("generate", "apply")
BENCHMARK_ROWS = ["spmv", "spgemm",
                  *(f"{s}/{p}" for s in SOLVERS for p in PHASES),
                  *(f"{p}/{ph}" for p in PRECONDITIONERS for ph in PHASES)]
MPI_OPS = ("allreduce", "alltoallv", "neighbor_alltoallv", "pingpong_intra", "pingpong_inter")


@dataclass
class DistRun:
    machine: str
    dir: str
    record: dict
    data: dict[str, dict[str, list | None]]  # GPU count -> group -> output

    @property
    def sort_key(self):
        g = self.record.get("ginkgo") or {}
        return g.get("commit_date") or "", self.record.get("started") or "", self.dir


def load_distributed(results: Path) -> dict[str, list[DistRun]]:
    """The runs per machine, oldest first; in a work tree only the tracked ones."""
    results = Path(results)
    tracked = _tracked(results) if results.is_dir() else None
    out: dict[str, list[DistRun]] = {}
    for run_json in sorted(results.glob("distributed/*/*/run.json")):
        d = run_json.parent
        if tracked is not None and f"distributed/{d.parent.name}/{d.name}" not in tracked:
            continue
        record = _read_json(run_json)
        if not isinstance(record, dict):
            continue
        data = {str(n): {g: _read_json(d / f"gpus-{n}" / f"{g}.json") for g in GROUPS}
                for n in GPU_COUNTS}
        out.setdefault(d.parent.name, []).append(DistRun(d.parent.name, d.name, record, data))
    for runs in out.values():
        runs.sort(key=lambda r: r.sort_key)
    return out


def _completed(result) -> bool:
    return isinstance(result, dict) and result.get("completed") is True


def _time(value) -> float | None:
    return value if isinstance(value, (int, float)) and value > 0 else None


def _measurements(run: DistRun, n: str) -> dict[str, float | None]:
    """Row key -> the compared time at n GPUs; None for a failed measurement.
    Solver apply is per iteration, so a change in iterations isn't a slowdown."""
    out: dict[str, float | None] = {}
    data = run.data.get(n, {})
    for case in data.get("spmv") or []:
        r = (case.get("spmv") or {}).get("csr-csr")
        out["spmv"] = _time(r.get("time")) if _completed(r) else None
    for case in data.get("spgemm") or []:
        r = (case.get("sparse_blas") or {}).get("spgemm")
        out["spgemm"] = _time(r.get("time")) if _completed(r) else None
    for case in data.get("solver") or []:
        for name, r in (case.get("solver") or {}).items():
            ok = _completed(r)
            gen = _time((r.get("generate") or {}).get("time")) if ok else None
            apply = r.get("apply") or {} if ok else {}
            its = apply.get("iterations")
            t = _time(apply.get("time"))
            out[f"{name}/generate"] = gen
            out[f"{name}/apply"] = t / its if t and isinstance(its, int) and its > 0 else None
    for case in data.get("preconditioner") or []:
        for name, r in (case.get("preconditioner") or {}).items():
            ok = _completed(r)
            for phase in PHASES:
                out[f"{name}/{phase}"] = (_time((r.get(phase) or {}).get("time"))
                                          if ok else None)
    for case in data.get("mpi") or []:
        op = case.get("operation")
        r = (case.get("mpi") or {}).get(op)
        out[f"mpi/{op}/{case.get('bytes')}"] = _time(r.get("time")) if _completed(r) else None
    return out


def _group_of(key: str) -> str:
    if key.startswith("mpi/"):
        return "mpi"
    if key in ("spmv", "spgemm"):
        return key
    return "solver" if key.split("/")[0] in SOLVERS else "preconditioner"


def _status(run: DistRun, n: str, key: str, measured: dict) -> str:
    """ok; failed (it ran and failed); skipped or killed (the group's status); or
    missing (this run or GPU count has no such measurement)."""
    if key in measured:
        return "ok" if measured[key] is not None else "failed"
    group = ((run.record.get("gpus") or {}).get(n) or {}).get(_group_of(key)) or {}
    status = group.get("status")
    if status in ("failed", "skipped", "killed"):
        return status
    return "missing"


def _mpi_key_order(key: str) -> tuple:
    _, op, size = key.split("/")
    return MPI_OPS.index(op) if op in MPI_OPS else len(MPI_OPS), op, int(size)


def _rows(runs: list[DistRun]) -> dict:
    measured = [{str(n): _measurements(r, str(n)) for n in GPU_COUNTS} for r in runs]
    mpi_keys = sorted({k for m in measured for per_n in m.values() for k in per_n
                       if k.startswith("mpi/")}, key=_mpi_key_order)
    rows = {}
    for key in BENCHMARK_ROWS + mpi_keys:
        row = {}
        for n in map(str, GPU_COUNTS):
            times = [m[n].get(key) for m in measured]
            row[n] = {
                "time": times,
                "band": [_band(times[:i]) for i in range(len(times))],
                "status": [_status(r, n, key, m[n]) for r, m in zip(runs, measured)],
            }
        rows[key] = row
    return rows


def _band(earlier: list[float | None]) -> float:
    """The noise band of a run, from the measured runs before it (as changes.py)."""
    values = [1 / t for t in earlier if t][-BASELINE_WINDOW:]
    band = noise_band(values)
    return BAND_FLOOR if band is None else band


def _details(run: DistRun) -> dict:
    gpus = {}
    for n in map(str, GPU_COUNTS):
        data = run.data.get(n, {})
        entry: dict = {"iterations": {}, "solver_time": {}, "levels": {}}
        for case in data.get("spgemm") or data.get("preconditioner") or []:
            entry["rows"], entry["nonzeros"] = case.get("rows"), case.get("nonzeros")
            break
        for case in data.get("solver") or []:
            for name, r in (case.get("solver") or {}).items():
                apply = (r.get("apply") or {}) if _completed(r) else {}
                entry["iterations"][name] = apply.get("iterations")
                entry["solver_time"][name] = _time(apply.get("time"))
        for case in data.get("preconditioner") or []:
            fine = case.get("nonzeros")
            for name, r in (case.get("preconditioner") or {}).items():
                levels = r.get("levels") if _completed(r) else None
                if not isinstance(levels, list) or not levels:
                    continue
                coarse = [lv.get("coarse_nonzeros") for lv in levels]
                oc = (None if not fine or any(not isinstance(c, int) for c in coarse)
                      else (fine + sum(coarse)) / fine)
                entry["levels"][name] = {"levels": len(levels),
                                         "coarsest_rows": levels[-1].get("coarse_rows"),
                                         "operator_complexity": oc}
        gpus[n] = entry
    rec = run.record
    return {"dir": run.dir, "machine": run.machine, "ginkgo": rec.get("ginkgo"),
            "ci": rec.get("ci"), "distributed": rec.get("distributed"),
            "started": rec.get("started"), "gpus": gpus}


def build_distributed(results: Path, out: Path, warnings: list[str]) -> None:
    """Write out/distributed.json and out/distributed/<machine>/<dir>.json."""
    out = Path(out)
    machines = {}
    for machine, runs in load_distributed(results).items():
        machines[machine] = {
            "runs": [{"dir": r.dir, "commit": (r.record.get("ginkgo") or {}).get("commit"),
                      "date": (r.record.get("ginkgo") or {}).get("commit_date"),
                      "started": r.record.get("started"), "ci": r.record.get("ci"),
                      "nodes": (r.record.get("distributed") or {}).get("nodes")}
                     for r in runs],
            "rows": _rows(runs),
        }
        for r in runs:
            _write(out / "distributed" / machine / f"{r.dir}.json", _details(r))
            if not r.record.get("finished"):
                warnings.append(f"{machine}: distributed run {r.dir} did not finish")
    _write(out / "distributed.json", {"machines": machines})


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")))
