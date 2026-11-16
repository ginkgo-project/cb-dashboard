"""Kernels, ranges and copies of a profiled case -> rows of counts and counters.

A row is a range name inside one of the case's measured ranges ("repetition",
or "repetition generate"/"repetition apply" with stages), as in Ginkgo's own
-detailed "components": an operation (csr::spgemm), a linear operator's range
(generate(...)), or allocate/copy/free. A kernel belongs to the innermost range
open on its thread when it was launched (or the innermost of its ncu stack); a
kernel in a measured range but in no range below it goes to OUTSIDE; kernels in
no measured range (reading the matrix, conversions) are dropped. Each row's time
is Ginkgo's exclusive time of that name, from the -detailed pass.
"""
from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass, field

from bench.profilers import Copy, Kernel, Range

OUTSIDE = "(outside operations)"
COUNTERS = ("dram_read_bytes", "dram_write_bytes", "fp64_ops", "l2_hits", "l2_misses")


def normalize(tool: str, metrics: dict[str, float]) -> dict[str, float]:
    """The vendor's metrics -> the common counters; missing ones are left out."""
    g = metrics.get
    out = {}
    if tool == "ncu":
        pairs = {"dram_read_bytes": g("dram__bytes_read.sum"),
                 "dram_write_bytes": g("dram__bytes_write.sum"),
                 "l2_hits": g("lts__t_sectors_lookup_hit.sum"),
                 "l2_misses": g("lts__t_sectors_lookup_miss.sum")}
        fp = [g(f"sm__sass_thread_inst_executed_op_d{op}_pred_on.sum") for op in ("fma", "add", "mul")]
        if any(v is not None for v in fp):
            pairs["fp64_ops"] = 2 * (fp[0] or 0) + (fp[1] or 0) + (fp[2] or 0)
    elif tool == "rocprofv3":
        pairs = {"dram_read_bytes": None if g("FETCH_SIZE") is None else g("FETCH_SIZE") * 1024,
                 "dram_write_bytes": None if g("WRITE_SIZE") is None else g("WRITE_SIZE") * 1024,
                 "l2_hits": g("TCC_HIT_sum"), "l2_misses": g("TCC_MISS_sum")}
        fp = [g(f"SQ_INSTS_VALU_{op}_F64") for op in ("FMA", "ADD", "MUL")]
        if any(v is not None for v in fp):  # counted per wavefront of 64 lanes
            pairs["fp64_ops"] = (2 * (fp[0] or 0) + (fp[1] or 0) + (fp[2] or 0)) * 64
    else:
        raise ValueError(f"unknown tool {tool!r}")
    for k, v in pairs.items():
        if v is not None:
            out[k] = float(v)
    return out


@dataclass
class Row:
    parent: str | None = None
    calls: int = 0
    launches: int = 0
    allocations: int = 0
    copies: int = 0
    copy_bytes: int | None = 0
    kernels: dict[str, dict] = field(default_factory=dict)  # name -> {"launches": n}


class _Index:
    """The ranges per thread, for the innermost range containing a time."""

    def __init__(self, ranges: list[Range]):
        self.ranges = ranges
        self.by_thread: dict[int, list[int]] = defaultdict(list)
        for i, r in enumerate(ranges):  # ranges come sorted by (thread, start) from _nest
            self.by_thread[r.thread].append(i)
        self.starts = {t: [ranges[i].start for i in ix] for t, ix in self.by_thread.items()}

    def innermost(self, thread: int | None, t: int | None) -> int | None:
        if thread is None or t is None or thread not in self.by_thread:
            return None
        ix = self.by_thread[thread]
        j = bisect.bisect_right(self.starts[thread], t) - 1
        if j < 0:
            return None
        i = ix[j]
        while i is not None and self.ranges[i].end < t:
            i = self.ranges[i].parent
        return i


def _place(ranges: list[Range], i: int | None, measured: tuple[str, ...],
           stages: tuple[str, ...]) -> tuple[str, str, str | None] | None:
    """(stage, row, parent row) of range i, or None outside every measured range."""
    if i is None:
        return None
    path = []
    while i is not None:
        path.append(ranges[i].name)
        i = ranges[i].parent
    for depth, name in enumerate(path):
        if name in measured:
            stage = stages[measured.index(name)] if stages else ""
            if depth == 0:
                return stage, OUTSIDE, None
            parent = path[1] if depth > 1 else None
            return stage, path[0], parent
    return None


def attribute_trace(ranges: list[Range], kernels: list[Kernel], copies: list[Copy],
                    measured: tuple[str, ...], stages: tuple[str, ...]) -> dict[tuple[str, str], Row]:
    """The counts per row: calls of the range, kernel launches, allocations and copies."""
    rows: dict[tuple[str, str], Row] = defaultdict(Row)
    for i, r in enumerate(ranges):
        if r.name in measured:
            continue
        place = _place(ranges, i, measured, stages)
        if place is None:
            continue
        stage, name, parent = place
        row = rows[(stage, name)]
        row.calls += 1
        row.parent = row.parent or parent
        if name == "allocate":
            outer = _place(ranges, r.parent, measured, stages)
            if outer is not None:
                rows[(outer[0], outer[1])].allocations += 1
    index = _Index(ranges)
    for k in kernels:
        place = _place(ranges, index.innermost(k.thread, k.launch), measured, stages)
        if place is None:
            continue
        row = rows[(place[0], place[1])]
        row.launches += 1
        row.kernels.setdefault(k.name, {"launches": 0})["launches"] += 1
    for c in copies:
        place = _place(ranges, index.innermost(c.thread, c.launch), measured, stages)
        if place is None:
            continue
        row = rows[(place[0], place[1])]
        row.copies += 1
        row.copy_bytes = None if c.bytes is None or row.copy_bytes is None else row.copy_bytes + c.bytes
    return dict(rows)


def attribute_counters(tool: str, kernels: list[Kernel], measured: tuple[str, ...],
                       stages: tuple[str, ...], ranges: list[Range] | None = None
                       ) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    """One counters pass: per row and kernel name, the summed common counters. ncu kernels
    carry their range stack; rocprofv3 kernels are placed in `ranges` by their launch."""
    out: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    index = _Index(ranges) if ranges is not None else None
    for k in kernels:
        if k.stack is not None:
            at = next((j for j, n in enumerate(k.stack) if n in measured), None)
            if at is None:
                continue
            stage = stages[measured.index(k.stack[at])] if stages else ""
            name = k.stack[-1] if len(k.stack) > at + 1 else OUTSIDE
        else:
            place = _place(ranges or [], index.innermost(k.thread, k.launch) if index else None,
                           measured, stages)
            if place is None:
                continue
            stage, name, _ = place
        sums = out.setdefault((stage, name), {}).setdefault(k.name, {})
        for c, v in normalize(tool, k.metrics).items():
            sums[c] = sums.get(c, 0.0) + v
    return out


def _with_rate(d: dict) -> dict:
    hits, misses = d.get("l2_hits"), d.get("l2_misses")
    if hits is not None and misses is not None and hits + misses > 0:
        d["l2_hit_rate"] = hits / (hits + misses)
    return d


def rows(trace: dict[tuple[str, str], Row], counters: list[dict], times: dict[str, dict]) -> list[dict]:
    """The rows of a case for kernels.json: the counts of the trace, the counters of every
    counters pass, and the time of the -detailed pass."""
    keys = list(trace)
    for part in counters:
        keys += [k for k in part if k not in trace and k not in keys]
    per_stage: dict[str, int] = defaultdict(int)
    for stage, name in keys:
        if name != OUTSIDE:
            per_stage[stage] += 1
    out = []
    for key in keys:
        stage, name = key
        t = trace.get(key, Row())
        kernels: dict[str, dict] = {n: {"launches": v["launches"]} for n, v in t.kernels.items()}
        for part in counters:
            for kname, sums in part.get(key, {}).items():
                kd = kernels.setdefault(kname, {"launches": None})
                for c, v in sums.items():
                    kd[c] = kd.get(c, 0.0) + v
        stage_times = times.get(stage, {})
        comps = stage_times.get("components") or {}
        if name in comps:
            time = comps[name]
        elif not comps and name != OUTSIDE and per_stage[stage] == 1:
            time = stage_times.get("total")
        else:
            time = None
        row = {"stage": stage, "name": name, "parent": t.parent, "time": time, "calls": t.calls,
               "launches": t.launches, "allocations": t.allocations, "copies": t.copies,
               "copy_bytes": t.copy_bytes if t.copies else 0}
        for c in COUNTERS:
            vals = [kd[c] for kd in kernels.values() if c in kd]
            if vals:
                row[c] = sum(vals)
        _with_rate(row)
        row["kernels"] = [_with_rate({"name": n, **kd}) for n, kd in
                          sorted(kernels.items(), key=lambda kv: -(kv[1].get("launches") or 0))]
        out.append(row)
    return out
