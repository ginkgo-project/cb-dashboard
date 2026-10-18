import pytest

from dashboard.cases import Measurement
from dashboard.changes import Cell, Point, cells_for_case, flags, unchanged
from dashboard.load import Column, Run

KEY = "k"


def run(name, triad=3.0e12):
    rec = {"ginkgo": {"commit": name, "commit_date": "d", "ref": "r"}}
    if triad is not None:
        rec["stream"] = {"results": {"triad": {"bandwidth": triad}}}
    return Run("m", name, rec, {})


def pt(r, time, status="ok", nbytes=None, error=None, extra=None,
       group="blas"):
    m = Measurement(KEY, group, {}, status, time, error=error,
                    extra=extra or {})
    return Point(r, m, nbytes, None)


def build(specs, kind="time"):
    """specs: one list of (r, Point) per column."""
    cols, points = [], {}
    for i, col in enumerate(specs):
        runs = []
        for r, p in col:
            runs.append(r)
            if p is not None:
                points.setdefault(r.dir, {})[KEY] = p
            else:
                points.setdefault(r.dir, {})
        cols.append(Column(f"c{i}", "d", "r", runs))
    return cells_for_case(cols, KEY, kind, points)


def series(ps, kind="time"):
    specs = []
    for i, p in enumerate(ps):
        r = run(f"r{i}")
        specs.append([(r, pt(r, 1.0 / p))])
    return build(specs, kind)


def test_history_until_four_previous():
    cells = series([1.0] * 5)
    assert [c.state for c in cells] == ["history"] * 4 + ["same"]


PREV = [1.0, 1.04, 0.96, 1.0, 1.04, 0.96, 1.0, 1.0]


def test_band_and_states():
    same = series(PREV + [0.92])[-1]
    assert same.state == "same"
    assert same.band == pytest.approx(0.088956, abs=1e-5)
    slow = series(PREV + [0.90])[-1]
    assert slow.state == "slower"
    assert slow.change == pytest.approx(-0.10)
    assert series(PREV + [1.10])[-1].state == "faster"
    flat = series([1.0] * 8 + [0.96])[-1]
    assert flat.band == pytest.approx(0.03)
    assert flat.state == "slower"


def bw_specs(last_triad):
    specs = []
    for i in range(8):
        r = run(f"r{i}")
        specs.append([(r, pt(r, 1e-3, nbytes=1e9))])
    r = run("r8", last_triad)
    specs.append([(r, pt(r, 1.1e-3, nbytes=1e9))])
    return specs


def test_bandwidth_normalized_by_triad():
    last = build(bw_specs(2.7e12), "bandwidth")[-1]
    assert last.state == "same" and last.raw is False
    assert build(bw_specs(2.7e12), "time")[-1].state == "slower"


def test_no_stream_falls_back_to_raw():
    last = build(bw_specs(None), "bandwidth")[-1]
    assert last.state == "slower" and last.raw is True


def test_repeats_use_median_of_ok_runs():
    rs = [run("a"), run("b"), run("c")]
    cell = build([[(rs[0], pt(rs[0], 1.0)),
                   (rs[1], pt(rs[1], None, "failed", error="x")),
                   (rs[2], pt(rs[2], 1.2))]])[0]
    assert cell.time == pytest.approx(1.1)
    assert len(cell.runs) == 3


def test_failed_column():
    rs = [run("a"), run("b")]
    cell = build([[(rs[0], pt(rs[0], None, "failed", error="boom")),
                   (rs[1], pt(rs[1], None, "failed", error="bang"))]])[0]
    assert cell.state == "failed" and cell.error == "boom"


def test_solver_iterations_change():
    specs = []
    for i, its in enumerate([37, 37, 37, 37, 37, 40]):
        r = run(f"r{i}")
        specs.append([(r, pt(r, 0.1 * its, group="solver",
                             extra={"iterations": its}))])
    cells = build(specs)
    assert cells[-1].iterations == 40
    assert cells[-1].iterations_prev == 37
    assert cells[-1].time == pytest.approx(0.1)
    assert cells[-1].state == "same"


def test_new_case_is_history_not_flagged():
    specs = []
    for i in range(7):
        r = run(f"r{i}")
        specs.append([(r, pt(r, 1.0) if i == 6 else None)])
    cells = build(specs)
    assert cells[:6] == [None] * 6
    assert cells[6].state == "history"
    assert flags({"m": {KEY: cells}}) == []


def test_flags_order():
    def row(*states):
        return [Cell(s[0], s[1], 0.03) if isinstance(s, tuple) else Cell(s)
                for s in states]
    cells = {"m": {
        "fast": row("same", ("faster", 0.1)),
        "slow2": row("same", ("slower", -0.1)),
        "fail": row("same", "failed"),
        "slow1": row("same", ("slower", -0.2)),
        "oldfail": row("failed", "failed"),
    }}
    out = flags(cells)
    assert [f.key for f in out] == ["fail", "slow1", "slow2", "fast"]
    assert [f.state for f in out] == ["failed", "slower", "slower", "faster"]


def test_unchanged_groups_by_state():
    row = [Cell("not_available"), Cell("not_available")]
    cells = {"a": {"x": row, "y": [Cell("same"), Cell("same")]},
             "b": {"x": row}}
    labels = {"x": {"preconditioner": "mg-pmis", "operation": "apply"},
              "y": {"operation": "dot"}}
    assert unchanged(cells, labels) == [
        {"state": "not_available", "count": 2, "names": ["mg-pmis"],
         "machines": ["a", "b"]}]
