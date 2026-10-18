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
    assert [f.state for f in out] == ["new_failure", "slower", "slower", "faster"]


def test_unchanged_groups_by_state():
    row = [Cell("not_available"), Cell("not_available")]
    cells = {"a": {"x": row, "y": [Cell("same"), Cell("same")]},
             "b": {"x": row}}
    labels = {"x": {"preconditioner": "mg-pmis", "operation": "apply"},
              "y": {"operation": "dot"}}
    assert unchanged(cells, labels) == [
        {"state": "not_available", "count": 2, "names": ["mg-pmis"],
         "machines": ["a", "b"]}]


def test_window_limited_to_eight_previous():
    # an old outlier outside the last 8 must not change the baseline
    cells = series([1.0] * 9 + [2.0] * 8 + [2.0])
    assert cells[-1].state == "same"
    assert cells[-1].change == pytest.approx(0.0)


def with_params(params_list):
    specs = []
    for i, prm in enumerate(params_list):
        r = run(f"r{i}")
        specs.append([(r, pt(r, 1.0, extra={"parameters": prm}))])
    return build(specs)


def test_params_changed():
    assert with_params([{"a": 1}, {"a": 2}])[-1].params_changed is True
    assert with_params([{"a": 1}, {"a": 1}])[-1].params_changed is False


def test_failure_priority_order():
    for order in (("failed", "not_available"), ("not_available", "failed")):
        rs = [run("a"), run("b")]
        cell = build([[(r, pt(r, None, s, error=s))
                       for r, s in zip(rs, order)]])[0]
        assert cell.state == "failed" and cell.error == "failed"


def bw_history(n_with_pct):
    specs = []
    for i in range(8):
        r = run(f"r{i}", 3.0e12 if i < n_with_pct else None)
        specs.append([(r, pt(r, 1e-3, nbytes=1e9))])
    r = run("r8")
    specs.append([(r, pt(r, 1e-3, nbytes=1e9))])
    return build(specs, "bandwidth")[-1]


def test_bandwidth_needs_min_history_with_pct():
    assert bw_history(3).raw is True
    assert bw_history(4).raw is False


def test_pct_from_points_with_triad_only():
    a, b = run("a", 3.0e12), run("b", None)
    cell = build([[(a, pt(a, 1e-3, nbytes=1e9)),
                   (b, pt(b, 1e-3, nbytes=1e9))]])[0]
    assert cell.pct == pytest.approx(1e12 / 3.0e12)


def test_iterations_prev_none_when_unchanged():
    specs = []
    for i in range(2):
        r = run(f"r{i}")
        specs.append([(r, pt(r, 3.7, group="solver",
                             extra={"iterations": 37}))])
    assert build(specs)[-1].iterations_prev is None


def test_unchanged_requires_same_state_before():
    cells = {"a": {"x": [Cell("same"), Cell("not_available")]}}
    assert unchanged(cells, {"x": {"operation": "dot"}}) == []


def test_unchanged_names_fallback():
    row = [Cell("failed"), Cell("failed")]
    cells = {"a": {"s": row, "f": row, "o": row, "p": row}}
    labels = {"s": {"solver": "cg", "operation": "solve"},
              "f": {"format": "csr", "operation": "spmv"},
              "o": {"operation": "dot"},
              "p": {"preconditioner": "jacobi", "solver": "cg",
                    "operation": "solve"}}
    out = unchanged(cells, labels)
    assert out[0]["names"] == ["cg", "csr", "dot", "jacobi"]
    assert out[0]["count"] == 4


def test_speedups_largest_first():
    cells = {"m": {"a": [Cell("same"), Cell("faster", 0.1, 0.03)],
                   "b": [Cell("same"), Cell("faster", 0.3, 0.03)]}}
    assert [f.key for f in flags(cells)] == ["b", "a"]


def test_solver_gbs_uses_whole_solve_time():
    r = run("r")
    cell = build([[(r, pt(r, 2.0, group="solver", nbytes=8e9,
                         extra={"iterations": 4}))]])[0]
    assert cell.gbs == pytest.approx(8e9 / 2.0)
    assert cell.time == pytest.approx(0.5)


def test_disabled_in_latest_run_wins():
    a, b = run("a"), run("b")
    cell = build([[(a, pt(a, 1e-3)),
                   (b, pt(b, None, status="disabled", error="off"))]])[0]
    assert cell.state == "disabled" and cell.error == "off"
    assert (cell.time, cell.gbs, cell.gflops, cell.pct) == (None,) * 4
    assert cell.runs == ["a", "b"]
    # the other way round, the newer run measured it: its value is shown
    later = build([[(a, pt(a, None, status="disabled")), (b, pt(b, 1e-3))]])[0]
    assert later.state == "history" and later.time == 1e-3
