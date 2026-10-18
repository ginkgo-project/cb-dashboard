"""Compare each case with its own history per machine and list the flags."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from .cases import Measurement
from .load import Column, Run

BASELINE_WINDOW, MIN_HISTORY, BAND_FLOOR, BAND_SIGMAS = 8, 4, 0.03, 3.0
NOT_OK_ORDER = ("failed", "not_available", "disabled", "missing")


@dataclass
class Point:
    run: Run
    m: Measurement
    nbytes: float | None
    flops: float | None


@dataclass
class Cell:
    state: str
    change: float | None = None
    band: float | None = None
    raw: bool = False
    time: float | None = None
    gbs: float | None = None
    gflops: float | None = None
    pct: float | None = None
    iterations: int | None = None
    iterations_prev: int | None = None
    residual: float | None = None
    storage: int | None = None
    levels: int | None = None
    params_changed: bool = False
    runs: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class Flag:
    key: str
    machine: str
    column: int
    state: str
    change: float | None
    band: float | None


def _med(values):
    values = [v for v in values if v is not None]
    return median(values) if values else None


def _is_ok(p: Point) -> bool:
    return p.m.status == "ok" and p.m.time is not None and p.m.time > 0


def _per_iteration(p: Point) -> float:
    its = p.m.extra.get("iterations")
    if p.m.group == "solver" and its:
        return p.m.time / its
    return p.m.time


def _column_cell(col: Column, key: str, points) -> tuple[Cell | None, dict | None]:
    """The values of one column; the second item is the internal detail."""
    pts = [points[r.dir][key] for r in col.runs
           if key in points.get(r.dir, {})]
    if not pts:
        return None, None
    runs = [p.run.dir for p in pts]
    ok = [p for p in pts if _is_ok(p)]
    if not ok:
        for state in NOT_OK_ORDER:
            hit = next((p for p in pts if p.m.status == state), None)
            if hit is not None:
                return Cell(state, runs=runs, error=hit.m.error), None
        return Cell(pts[0].m.status, runs=runs, error=pts[0].m.error), None
    time = _med(_per_iteration(p) for p in ok)
    gbs_list = [p.nbytes / p.m.time if p.nbytes is not None else None
                for p in ok]
    gbs = _med(gbs_list)
    gflops = _med(p.flops / p.m.time if p.flops is not None else None
                  for p in ok)
    pct = _med(g / p.run.triad for g, p in zip(gbs_list, ok)
               if g is not None and p.run.triad)
    its = _med(p.m.extra.get("iterations") for p in ok)
    levels = _med(len(p.m.extra["levels"]) for p in ok
                  if p.m.extra.get("levels") is not None)
    cell = Cell(
        "ok", time=time, gbs=gbs, gflops=gflops, pct=pct,
        iterations=None if its is None else int(round(its)),
        residual=_med(p.m.extra.get("rel_residual") for p in ok),
        storage=_med(p.m.extra.get("storage") for p in ok),
        levels=None if levels is None else int(round(levels)),
        runs=runs)
    detail = {"p_raw": 1.0 / time, "pct": pct,
              "params": ok[0].m.extra.get("parameters")}
    return cell, detail


def cells_for_case(cols: list[Column], key: str, kind: str,
                   points: dict[str, dict[str, Point]]) -> list[Cell | None]:
    cells: list[Cell | None] = []
    details: list[dict | None] = []
    for i, col in enumerate(cols):
        cell, detail = _column_cell(col, key, points)
        if cell is not None and detail is not None:
            prev_cell = cells[i - 1] if i else None
            prev_detail = details[i - 1] if i else None
            if (prev_cell is not None and prev_detail is not None
                    and prev_cell.iterations is not None
                    and prev_cell.iterations != cell.iterations):
                cell.iterations_prev = prev_cell.iterations
            if prev_detail is not None:
                cell.params_changed = prev_detail["params"] != detail["params"]
            _compare(cell, detail, kind, [d for d in details if d is not None])
        cells.append(cell)
        details.append(detail)
    return cells


def _compare(cell: Cell, detail: dict, kind: str, prev: list[dict]) -> None:
    window = prev[-BASELINE_WINDOW:]
    with_pct = [d["pct"] for d in window if d["pct"] is not None]
    if (kind == "bandwidth" and detail["pct"] is not None
            and len(with_pct) >= MIN_HISTORY):
        cell.raw = False
        current, values = detail["pct"], with_pct
    else:
        cell.raw = kind == "bandwidth"
        current, values = detail["p_raw"], [d["p_raw"] for d in window]
    if len(values) < MIN_HISTORY:
        cell.state = "history"
        return
    baseline = median(values)
    mad = median(abs(x - baseline) for x in values)
    cell.band = max(BAND_FLOOR, BAND_SIGMAS * 1.4826 * mad / baseline)
    cell.change = current / baseline - 1
    if cell.change < -cell.band:
        cell.state = "slower"
    elif cell.change > cell.band:
        cell.state = "faster"
    else:
        cell.state = "same"


def flags(cells: dict[str, dict[str, list[Cell | None]]]) -> list[Flag]:
    new_failures, slower, faster = [], [], []
    for machine, by_key in cells.items():
        for key, row in by_key.items():
            if not row:
                continue
            last = len(row) - 1
            cell = row[last]
            if cell is None:
                continue
            prev = row[last - 1] if last > 0 else None
            flag = Flag(key, machine, last, cell.state, cell.change, cell.band)
            if cell.state == "failed":
                if prev is None or prev.state != "failed":
                    flag.state = "new_failure"
                    new_failures.append(flag)
            elif cell.state == "slower":
                slower.append(flag)
            elif cell.state == "faster":
                faster.append(flag)
    slower.sort(key=lambda f: f.change)
    faster.sort(key=lambda f: -f.change)
    return new_failures + slower + faster


def unchanged(cells: dict[str, dict[str, list[Cell | None]]],
              labels: dict[str, dict[str, str]]) -> list[dict]:
    groups: dict[str, dict] = {}
    for machine, by_key in cells.items():
        for key, row in by_key.items():
            if len(row) < 2 or row[-1] is None or row[-2] is None:
                continue
            state = row[-1].state
            if (state not in ("failed", "not_available", "disabled")
                    or row[-2].state != state):
                continue
            lab = labels[key]
            name = (lab.get("preconditioner") or lab.get("solver")
                    or lab.get("format") or lab["operation"])
            g = groups.setdefault(state, {"count": 0, "names": set(),
                                          "machines": set()})
            g["count"] += 1
            g["names"].add(name)
            g["machines"].add(machine)
    return [{"state": s, "count": g["count"], "names": sorted(g["names"]),
             "machines": sorted(g["machines"])} for s, g in groups.items()]
