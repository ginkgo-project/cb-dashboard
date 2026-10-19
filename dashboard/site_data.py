"""Write the JSON files the page reads into site/data."""
from __future__ import annotations

import json
import re
import tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Callable

from .cases import Measurement, measurements
from .changes import Cell, Point, cells_for_case, flags, unchanged
from .load import GROUPS, Column, Run, columns, load_runs
from .models import MODELS, compare_kind, cost, model_name, stencil_nonzeros
from .reproduce import command
from .setup import setup_changes

TABS = {"basic": ("blas", "spmv", "spgemm"), "solvers": ("solver",),
        "preconditioners": ("preconditioner",)}
SETTINGS_OF = {"solvers": "solver", "preconditioners": "preconditioner"}
STATUS_ORDER = ("ok", "partial", "failed", "killed")
BENCH_SETTINGS = ("warmup", "repetitions", "timer_method", "gpu_timer", "seed")
_MACHINE_KEYS = ("label", "fp64_peak", "largest_cache", "cache_name")


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))


def _sig(x: float) -> float:
    """x to 4 significant digits, for the derived matrix ratios."""
    return float(f"{x:.4g}")


def _natural(text: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", text))


def _case_order(key: str, groups: tuple) -> tuple:
    group, *labels = key.split("/")
    return groups.index(group), tuple(_natural(p) for p in labels)


# --- cases -------------------------------------------------------------------

def _matrix_props(e: dict) -> dict:
    rl = e.get("row_length") or {}
    rows, nnz = e.get("rows"), e.get("nonzeros")
    return _drop_none({
        "rows": rows, "nonzeros": nnz,
        "nnz_per_row": _sig(nnz / rows) if rows and nnz is not None else None,
        "max_row": rl.get("max"),
        "row_length_cv": _sig(rl["cv"]) if rl.get("cv") is not None else None})


def _props(m: Measurement, matrices: dict) -> dict:
    if m.group == "blas":
        return {"n": int(m.labels["n"])}
    if m.group == "solver":
        rows = m.extra.get("rows")
        if not rows:
            return {}
        z = stencil_nonzeros(m.labels["stencil"], rows)
        return {"rows": rows, "nnz_per_row": _sig(z / rows)}
    e = matrices.get(m.labels.get("matrix"))
    return _matrix_props(e) if e else {}


def _points_of(cols: list[Column], points: dict, key: str) -> list[Point]:
    return [points[r.dir][key] for c in cols for r in c.runs
            if key in points[r.dir]]


def _compare(key, rep, cols, points, machine: dict) -> str:
    """From the bytes of the machine's latest ok point (or latest point)."""
    pts = _points_of(cols, points, key)
    ok = [p for p in pts if p.m.status == "ok"]
    p = (ok or pts or [None])[-1]
    if p is None:
        return compare_kind(rep, None, machine)
    return compare_kind(p.m, p.nbytes, machine)


def _cell_json(cell: Cell | None):
    return None if cell is None else _drop_none(asdict(cell))


def _values(cell: Cell) -> dict:
    return _drop_none({"time": cell.time, "gbs": cell.gbs,
                       "gflops": cell.gflops, "pct": cell.pct})


def _latest(cols, points, cells, keys) -> dict:
    last = len(cols) - 1
    col = cols[last]
    triads = [r.triad for r in col.runs if r.triad]
    out = []
    for key in keys:
        row = cells[key]
        cell = row[last]
        if cell is None or cell.time is None:
            continue
        point = {"key": key, **_values(cell)}
        ratios = [p.flops / p.nbytes for p in _points_of([col], points, key)
                  if p.m.status == "ok" and p.flops is not None and p.nbytes]
        if ratios:
            point["intensity"] = median(ratios)
        prev = row[last - 1] if last else None
        if prev is not None and prev.time is not None:
            point["prev"] = _values(prev)
        out.append(point)
    return _drop_none({"column": last,
                       "triad": median(triads) if triads else None,
                       "points": out})


# --- settings ----------------------------------------------------------------

def _flatten(value, path=""):
    if isinstance(value, dict) and value:
        for k, v in value.items():
            yield from _flatten(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, list) and value:
        for i, v in enumerate(value):
            yield from _flatten(v, f"{path}[{i}]")
    else:
        yield path, value


def _flat_diff(before, after) -> list[dict]:
    """The leaves that differ, by path: a.b for keys, a[0] for list items."""
    a, b = dict(_flatten(before)), dict(_flatten(after))
    return [_drop_none({"path": p, "before": a.get(p), "after": b.get(p)})
            for p in sorted(a.keys() | b.keys())
            if p not in a or p not in b or a[p] != b[p]]


def _parameters(col: Column, points, keys) -> tuple[str, dict] | None:
    """The case and parameters of the column's latest run that has any."""
    for r in reversed(col.runs):
        for key in keys:
            p = points[r.dir].get(key)
            if p is not None and p.m.extra.get("parameters") is not None:
                return key, p.m.extra["parameters"]
    return None


def _settings(label: str, keys, rep, machines: dict) -> dict:
    """Per solver or preconditioner, from the first machine that has it."""
    by_name: dict[str, list[str]] = {}
    for key in keys:
        by_name.setdefault(rep[key].labels[label], []).append(key)
    out = {}
    for name, name_keys in by_name.items():
        for cols, points in machines.values():
            found = _parameters(cols[-1], points, name_keys)
            if found is None:
                continue
            key, cur = found
            # the same case in the column before, so the diff is a change
            prev = (_parameters(cols[-2], points, [key])
                    if len(cols) > 1 else None)
            changed = [] if prev is None else _flat_diff(prev[1], cur)
            out[name] = {"parameters": cur, "changed": changed}
            break
    return out


# --- dataset tables ----------------------------------------------------------

def _blank(row: list) -> list:
    return ["" if v is None else v for v in row]


def _criteria_text(criteria) -> str:
    parts = []
    for c in criteria or []:
        fields = ", ".join(f"{k} {v}" for k, v in sorted(c.items())
                           if k not in ("type", "value_type"))
        parts.append(f"{c.get('type')}: {fields}")
    return "; ".join(parts)


def _dataset(tab: str, manifest: dict, problems: dict, runs: list[Run]) -> dict:
    version = manifest.get("version")
    mats = manifest.get("matrices") or []
    if tab == "basic":
        spmv = [e for e in mats if e.get("set") == "spmv"]
        spgemm = [e for e in mats if "spgemm" in (e.get("roles") or [])]
        rows = []
        for e in spmv + spgemm:
            p = _matrix_props(e)
            rows.append(_blank([e["name"], e.get("source"), p.get("rows"),
                                p.get("nonzeros"), p.get("nnz_per_row"),
                                p.get("max_row"), p.get("row_length_cv"),
                                e.get("run_sellp")]))
        return {"title": f"Dataset version {version}: {len(spmv)} SpMV and "
                         f"{len(spgemm)} SpGEMM matrices",
                "columns": ["matrix", "source", "rows", "nonzeros", "nnz/row",
                            "max row", "row length CV", "SELL-P"],
                "rows": rows}
    if tab == "solvers":
        with_solver = [r for r in runs if r.data.get("solver")]
        entries = (max(with_solver, key=lambda r: (r.commit_date, r.started))
                   .data["solver"] if with_solver else [])
        rows, seen, criteria = [], set(), None
        for e in entries:
            if criteria is None:
                first = next(iter((e.get("solver") or {}).values()), {})
                criteria = (first.get("parameters") or {}).get("criteria")
            if e["stencil"] in seen:
                continue
            seen.add(e["stencil"])
            n = e.get("rows")
            z = stencil_nonzeros(e["stencil"], n) if n else None
            rows.append(_blank([e["stencil"], n, z]))
        if criteria:
            rows.append(["criteria", _criteria_text(criteria), ""])
        return {"title": f"{len(seen)} stencils",
                "columns": ["stencil", "rows", "nonzeros"], "rows": rows}
    pre = [e for e in mats if "precond" in (e.get("roles") or [])]
    rows = []
    for e in pre:
        p = _matrix_props(e)
        rows.append(_blank([e["name"], problems.get(e["name"]), p.get("rows"),
                            p.get("nonzeros"), p.get("nnz_per_row"),
                            p.get("row_length_cv"), e.get("spd")]))
    return {"title": f"Dataset version {version}: {len(pre)} FEM matrices",
            "columns": ["matrix", "problem", "rows", "nonzeros", "nnz/row",
                        "row length CV", "SPD"],
            "rows": rows}


# --- runs and machines -------------------------------------------------------

def _status(run: Run) -> str:
    """The worst group status; groups not requested don't count.

    Any other status (skipped for another reason, pending) counts as failed.
    """
    rank = 0
    for g in run.record.get("groups", {}).values():
        s = g.get("status")
        if s == "skipped" and g.get("reason") == "not requested":
            continue
        rank = max(rank, STATUS_ORDER.index(
            s if s in STATUS_ORDER else "failed"))
    return STATUS_ORDER[rank]


def _seconds(run: Run) -> float | None:
    try:
        return (datetime.fromisoformat(run.record["finished"])
                - datetime.fromisoformat(run.started)).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None


def _invocations(run: Run, group: str) -> list[dict]:
    return (run.record.get("groups", {}).get(group) or {}).get(
        "invocations") or []


def _command(run: Run, m: Measurement) -> str | None:
    try:
        return command(run, m)
    except (KeyError, TypeError):  # a case without what its command needs
        return None


def _run_detail(run: Run, points: dict[str, Point]) -> dict:
    rec = run.record
    invs = {f"{g}:{i['label']}": _drop_none({
                "status": i.get("status"), "seconds": i.get("seconds"),
                "reason": i.get("reason"), "energy_j": i.get("energy_j")})
            for g in rec.get("groups", {}) for i in _invocations(run, g)}
    cases = {}
    for key, p in points.items():
        case = {"invocation": p.m.invocation, "command": _command(run, p.m)}
        if (model_name(p.m) == "amg" and p.m.labels.get("stage") == "apply"
                and p.m.extra.get("levels") is not None):
            case["levels"] = p.m.extra["levels"]
        cases[key] = _drop_none(case)
    return _drop_none({
        "machine": run.machine, "dir": run.dir, "node": rec.get("node"),
        "started": rec.get("started"), "finished": rec.get("finished"),
        "ginkgo": rec.get("ginkgo"), "triad": run.triad,
        "invocations": invs, "cases": cases})


def _nodes(runs: list[Run]) -> dict[str, dict]:
    by_node: dict[str, list[Run]] = {}
    for r in runs:
        by_node.setdefault(r.node, []).append(r)
    out = {}
    for node, rs in by_node.items():
        t = sorted(r.triad for r in rs if r.triad is not None)
        out[node] = {"runs": len(rs)}
        if t:
            out[node].update(triad_min=t[0], triad_median=median(t),
                             triad_max=t[-1])
    return out


def _stream(run: Run) -> dict:
    s = run.record.get("stream")
    return s if isinstance(s, dict) else {}


def _bench_settings(runs: list[Run]) -> dict | None:
    """warm-up, repetitions, ... of the latest run with invocations.

    From its first spmv invocation, or that of another group but blas, with
    the blas value added where blas differs, as blas warms up 3 times:
    "1 (blas: 3)". A setting only blas has is taken from blas.
    """
    for r in reversed(runs):
        first = {g: _invocations(r, g)[0].get("args") or []
                 for g in GROUPS if _invocations(r, g)}
        if first:
            break
    else:
        return None
    blas = _parse_args(first.get("blas", []))
    main = next((_parse_args(first[g]) for g in GROUPS
                 if g != "blas" and g in first), {})
    out = {}
    for k in BENCH_SETTINGS:
        if k in main:
            differs = k in blas and blas[k] != main[k]
            out[k] = f"{main[k]} (blas: {blas[k]})" if differs else main[k]
        elif k in blas:
            out[k] = blas[k]
    return out


def _parse_args(args: list[str]) -> dict:
    """-name value, -name=value and bare -name (True) options."""
    flag = re.compile(r"-+[A-Za-z]")
    parsed, i = {}, 0
    while i < len(args):
        if flag.match(args[i]):
            name, eq, value = args[i].lstrip("-").partition("=")
            if eq:
                parsed[name] = value
            elif i + 1 < len(args) and not flag.match(args[i + 1]):
                parsed[name], i = args[i + 1], i + 1
            else:
                parsed[name] = True
        i += 1
    return parsed


def _setup(runs: list[Run]) -> dict:
    rec = runs[-1].record
    gpu = rec.get("gpu") or {}
    after = _stream(runs[-1]).get("gpu_after") or {}
    power = (gpu.get("clocks") or {}).get("power_limit_w")
    if power is None:
        amd = ((after.get("rocm-smi") or {}).get("card0") or {}).get(
            "Max Graphics Package Power (W)")
        try:
            power = float(amd) if amd is not None else None
        except ValueError:
            power = None
    part = (after.get("partition") or {}).get("card0") or {}
    partition = None
    if "Compute Partition" in part and "Memory Partition" in part:
        partition = f"{part['Compute Partition']} / {part['Memory Partition']}"
    setup = _drop_none({
        "gpu": gpu.get("gpu"), "driver": gpu.get("driver"),
        "power_limit_w": power, "binding": gpu.get("binding"),
        "uenv": rec.get("uenv"), "toolchain": rec.get("toolchain"),
        "cmake_cache": rec.get("cmake_cache"),
        "settings": _bench_settings(runs),
        "dataset_version": rec.get("dataset_version"),
        "script_version": rec.get("script_version")})
    setup["partition"] = partition
    return setup


def _last_gpu_after(runs: list[Run], node: str) -> dict | None:
    for r in reversed(runs):
        after = _stream(r).get("gpu_after")
        if r.node == node and after:
            return after
    return None


def _energy(invs: list[dict], counter: str):
    vals = [(i.get("energy_j") or {}).get(counter) for i in invs]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def _machine_history(runs: list[Run]) -> dict:
    """machines.json for one machine; `runs` in the order they started."""
    nodes = []
    for node, stats in _nodes(runs).items():
        nodes.append(_drop_none({"node": node, **stats,
                                 "gpu_after": _last_gpu_after(runs, node)}))
    groups, history = [], []
    for r in runs:
        date = r.started[:10]
        for g in GROUPS:
            v = r.record.get("groups", {}).get(g)
            if v is None:
                continue
            invs = v.get("invocations") or []
            groups.append(_drop_none({
                "dir": r.dir, "date": date, "group": g,
                "status": v.get("status"), "seconds": v.get("seconds"),
                "gpu_j": _energy(invs, "accel0_energy"),
                "node_j": _energy(invs, "energy"),
                "cpu_j": _energy(invs, "cpu_energy")}))
        status = _status(r)
        failed = sum(i.get("status") == "failed"
                     for g in r.record.get("groups", {})
                     for i in _invocations(r, g))
        history.append(_drop_none({
            "dir": r.dir, "date": date, "status": status,
            "seconds": _seconds(r), "failed": failed,
            "killed": status == "killed"}))
    return {"setup": _setup(runs),
            "triad": [{"dir": r.dir, "date": r.started[:10], "node": r.node,
                       "triad": r.triad} for r in runs if r.triad is not None],
            "nodes": nodes, "groups": groups, "runs": history}


# --- the build ---------------------------------------------------------------

def build_site(results: Path, manifest: Path, catalog: Path, out: Path, *,
               machines: dict, frozen: dict,
               script_changed: Callable[[str, str], bool],
               warnings: list[str]) -> None:
    """Write meta.json, the tab files, machines.json and runs/<m>/<dir>.json.

    Warnings found here are appended to `warnings`.
    """
    out = Path(out)
    runs = load_runs(Path(results))
    for name in sorted({r.machine for r in runs} - set(machines)):
        warnings.append(f"{name}: not in dashboard/machines.toml; "
                        "its runs are left out")
    runs = [r for r in runs if r.machine in machines]
    man = json.loads(Path(manifest).read_text())
    matrices = {e["name"]: e for e in man.get("matrices") or []}
    with open(catalog, "rb") as f:
        problems = {e["name"]: (e.get("fem") or {}).get("problem")
                    for e in tomllib.load(f).get("entry", [])}

    rep: dict[str, Measurement] = {}
    points: dict[str, dict[str, dict[str, Point]]] = {}
    for r in runs:
        structure = frozen.get(r.machine, {})
        pts = points.setdefault(r.machine, {})[r.dir] = {}
        for key, m in measurements(r).items():
            nbytes = flops = None
            if m.status == "ok":
                nbytes, flops = cost(m, structure.get(key))
            pts[key] = Point(r, m, nbytes, flops)
            if m.extra or key not in rep:
                rep[key] = m
    # in the order of machines.toml
    known = {name: (columns(runs, name), points[name])
             for name in machines if name in points}
    cols_json = {}
    for name, (cols, _) in known.items():
        cols_json[name] = [
            {"commit": c.commit, "date": c.date, "ref": c.ref,
             "runs": [r.dir for r in c.runs],
             "setup_changes": (setup_changes(cols[i - 1].runs[-1], c.runs[0],
                                             script_changed) if i else [])}
            for i, c in enumerate(cols)]

    flag_counts = {}
    for tab, groups in TABS.items():
        keys = sorted((k for k, m in rep.items() if m.group in groups),
                      key=lambda k: _case_order(k, groups))
        cases, cells = [], {name: {} for name in known}
        for key in keys:
            compare = {}
            for name, (cols, pts) in known.items():
                compare[name] = _compare(key, rep[key], cols, pts,
                                         machines[name])
                cells[name][key] = cells_for_case(cols, key, compare[name],
                                                  pts)
            m = rep[key]
            cases.append({"key": key, "group": m.group, "labels": m.labels,
                          "model": model_name(m), "compare": compare,
                          "props": _props(m, matrices)})
        tab_flags = [_drop_none(asdict(f)) for f in flags(cells)]
        flag_counts[tab] = len(tab_flags)
        used = {c["model"] for c in cases}
        data = {
            "tab": tab, "cases": cases,
            "machines": {name: {"columns": cols_json[name],
                                "cells": {k: [_cell_json(c) for c in row]
                                          for k, row in cells[name].items()}}
                         for name in known},
            "flags": tab_flags,
            "unchanged": unchanged(cells, {k: rep[k].labels for k in keys}),
            "latest": {name: _latest(cols, pts, cells[name], keys)
                       for name, (cols, pts) in known.items()},
            "dataset": _dataset(tab, man, problems, runs),
            "models": [{"name": md.name, "bytes": md.bytes_text,
                        "flops": md.flops_text, "source": md.source,
                        "bandwidth_bound": md.bandwidth_bound}
                       for md in MODELS.values() if md.name in used],
        }
        if tab in SETTINGS_OF:
            data["settings"] = _settings(SETTINGS_OF[tab], keys, rep, known)
        _write(out / f"{tab}.json", data)

    meta_machines, history = {}, {}
    for name, (_, pts) in known.items():
        mruns = sorted((r for r in runs if r.machine == name),
                       key=lambda r: r.started)
        for r in mruns:
            _write(out / "runs" / name / f"{r.dir}.json",
                   _run_detail(r, pts[r.dir]))
        last = mruns[-1]
        meta_machines[name] = {
            **{k: machines[name][k] for k in _MACHINE_KEYS},
            "last_run": _drop_none({
                "dir": last.dir, "commit": last.commit,
                "commit_date": last.commit_date, "ref": last.ref,
                "node": last.node, "status": _status(last),
                "seconds": _seconds(last)}),
            "nodes": _nodes(mruns)}
        history[name] = _machine_history(mruns)
    _write(out / "machines.json", {"machines": history})
    _write(out / "meta.json", {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machines": meta_machines, "flag_counts": flag_counts,
        "warnings": warnings})
