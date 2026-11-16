"""The Kernels tab's data: site/data/kernels.json, from the kernels.json of the runs.

Only runs whose run.json git tracks (as the other tabs) and that have a
kernels.json; a machine without any is listed with no runs, so the tab can say
so. SpMV and BLAS cases get the dashboard's modelled bytes of the same run, to
set the measured DRAM bytes against.
"""
from __future__ import annotations

import json
from pathlib import Path

from .load import load_runs

_FORMATS = {"csr": "CSR", "coo": "COO", "hybrid": "hybrid", "sellp": "SELL-P"}


def case_group(key: str) -> tuple[str, str]:
    """(group, label) of a case key for the tab's case picker."""
    parts = key.split("/")
    if parts[0] == "preconditioner":
        return "multigrid", " · ".join(parts[1:])
    if parts[0] == "spmv":
        return "spmv", f"{_FORMATS.get(parts[1], parts[1])} · {parts[2]}"
    if parts[0] == "spgemm":
        return "spgemm", parts[1]
    if parts[0] == "blas":
        return ("gemv" if parts[1] == "gemv" else "blas"), " · ".join(parts[1:])
    return parts[0], " · ".join(parts[1:])


def build_kernels(results: Path, out: Path, warnings: list[str],
                  model_bytes: dict[str, dict[str, dict[str, float]]] | None = None) -> None:
    """Write out/kernels.json. `model_bytes[machine][run dir][case key]`: the modelled bytes."""
    results, out = Path(results), Path(out)
    model_bytes = model_bytes or {}
    machines: dict[str, dict] = {}
    for r in load_runs(results):
        m = machines.setdefault(r.machine, {"tool": None, "runs": [], "cases": {}})
        path = results / r.machine / r.dir / "kernels.json"
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text())
            cases = doc["cases"]
        except (ValueError, KeyError, TypeError) as e:
            warnings.append(f"{r.machine}/{r.dir}: unreadable kernels.json ({type(e).__name__})")
            continue
        m["tool"] = doc.get("tool")
        m["runs"].append({"dir": r.dir, "commit": r.commit, "date": r.commit_date,
                          "started": r.started})
        bytes_of = model_bytes.get(r.machine, {}).get(r.dir, {})
        for key, case in cases.items():
            group, label = case_group(key)
            entry = m["cases"].setdefault(key, {"group": group, "label": label, "runs": []})
            run = {"dir": r.dir, "status": case.get("status"), "operations": case.get("operations", [])}
            if case.get("reason"):
                run["reason"] = case["reason"]
            if bytes_of.get(key):
                run["model_bytes"] = bytes_of[key]
            entry["runs"].append(run)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernels.json").write_text(json.dumps({"machines": machines}, separators=(",", ":")))
