from pathlib import Path

import pytest
from dash_fakes import write_run

from dashboard.cases import measurements, plain_preconditioner
from dashboard.load import load_runs

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _run(tmp_path, data=None, invocations=None):
    groups = {}
    for inv in invocations or []:
        g = inv.pop("group")
        groups.setdefault(g, {"status": "ok", "seconds": 1.0, "invocations": []})
        groups[g]["invocations"].append(inv)
    write_run(tmp_path, "m", "A", commit="c", date="2026-10-18",
              started="2026-10-18T10:00", groups=groups, data=data)
    return load_runs(tmp_path)[0]


def _inv(group, label, status="ok", reason=None, args=()):
    return {"group": group, "label": label, "status": status,
            "reason": reason, "args": list(args)}


def test_spmv_keys_and_failed_format(tmp_path):
    entry = {"filename": "/x/cage15.bin", "rows": 5, "cols": 5, "nonzeros": 9,
             "spmv": {
                 "csr": {"time": 1e-3, "repetition_times": [1e-3],
                         "format_parameters": {}, "completed": True},
                 "coo": {"completed": False, "error": "boom"}}}
    run = _run(tmp_path, {"spmv": [entry]}, [_inv("spmv", "cage15")])
    m = measurements(run)
    assert m["spmv/csr/cage15"].status == "ok"
    assert m["spmv/csr/cage15"].time == 1e-3
    assert m["spmv/coo/cage15"].status == "failed"
    assert m["spmv/coo/cage15"].error == "boom"
    assert m["spmv/coo/cage15"].invocation == "spmv:cage15"
    assert m["spmv/csr/cage15"].labels == {
        "operation": "spmv", "format": "csr", "matrix": "cage15"}


def _precond_entry(name, **kw):
    return {"filename": "/x/porous_pressure.bin", "rows": 4, "cols": 4,
            "nonzeros": 8, "preconditioner": {name: kw}}


def test_not_implemented_is_not_available(tmp_path):
    entry = _precond_entry("mg-pmis", completed=False,
                           error_type="gko::NotImplemented", error="nope",
                           generate={}, apply={})
    run = _run(tmp_path, {"preconditioner": [entry]},
               [_inv("preconditioner", "porous_pressure/mg-pmis", "failed")])
    m = measurements(run)
    for stage in ("apply", "generate"):
        k = f"preconditioner/{stage}/mg-pmis/porous_pressure"
        assert m[k].status == "not_available"
    assert len(m) == 2


def test_disabled_invocation_without_output(tmp_path):
    inv = {"group": "preconditioner",
           "label": "porous_pressure/ilu-syncfree", "status": "disabled",
           "reason": "disabled on m", "args": []}
    m = measurements(_run(tmp_path, invocations=[inv]))
    for stage in ("apply", "generate"):
        x = m[f"preconditioner/{stage}/ilu-syncfree/porous_pressure"]
        assert x.status == "disabled"
        assert x.error == "disabled on m"
        assert x.time is None


def test_failed_invocation_covers_its_formats(tmp_path):
    inv = _inv("spmv", "cage15", "failed", "timeout",
               ["-formats", "csr,coo", "-input_matrix", "/x/cage15.bin"])
    m = measurements(_run(tmp_path, invocations=[inv]))
    assert set(m) == {"spmv/csr/cage15", "spmv/coo/cage15"}
    for x in m.values():
        assert x.status == "failed"
        assert x.error == "timeout"


def test_skipped_is_missing_and_old_layout_precond(tmp_path):
    inv = _inv("preconditioner", "porous_pressure", "skipped", "no time",
               ["-preconditioners", "jacobi,ilu-syncfree"])
    m = measurements(_run(tmp_path, invocations=[inv]))
    assert len(m) == 4
    x = m["preconditioner/apply/ilu-syncfree/porous_pressure"]
    assert x.status == "missing"
    assert x.invocation == "preconditioner:porous_pressure"


def _solver(name, ortho):
    return {"size": 8, "stencil": "7pt", "optimal": {"spmv": "csr"},
            "solver": {name: {
                "rhs_norm": 2.0, "residual_norm": 1e-10, "completed": True,
                "generate": {"time": 0.1},
                "apply": {"time": 0.5, "iterations": 7,
                          "repetition_times": [0.5]},
                "parameters": {"ortho_method": ortho}}}}


def test_solver_keys(tmp_path):
    data = {"solver": [_solver("gmres-jacobi", "cgs"),
                       _solver("cg-jacobi", "mgs")]}
    invs = [_inv("solver", "cgs"), _inv("solver", "mgs")]
    m = measurements(_run(tmp_path, data, invs))
    g = m["solver/gmres-cgs/7pt"]
    assert g.invocation == "solver:cgs"
    assert g.time == 0.5
    c = m["solver/cg/7pt"]
    assert c.invocation == "solver:mgs"
    assert c.extra["rel_residual"] == 1e-10 / 2.0
    assert c.extra["iterations"] == 7
    assert c.extra["generate_time"] == 0.1


def test_blas_and_spgemm(tmp_path):
    blas = [{"n": 1048576, "blas": {"dot": {
        "time": 2.0, "repetition_times": [2.0], "flops": 1.0,
        "bandwidth": 3.0, "completed": True}}}]
    gemm = [{"filename": "/x/p.bin", "rows": 1, "cols": 1, "nonzeros": 1,
             "sparse_blas": {"spgemm": {"time": 4.0, "completed": True,
                                        "flops": 5.0, "bandwidth": 6.0}}}]
    m = measurements(_run(tmp_path, {"blas": blas, "spgemm": gemm},
                          [_inv("blas", "vectors"), _inv("spgemm", "p")]))
    b = m["blas/dot/1048576"]
    assert b.extra["flops_rate"] == 1.0 and b.extra["bandwidth_rate"] == 3.0
    assert b.invocation == "blas:vectors"
    assert m["spgemm/p"].time == 4.0


def test_plain_preconditioner():
    assert plain_preconditioner("jacobi-32-0,0") == "jacobi"
    assert plain_preconditioner("spd-isai-1") == "spd-isai"
    assert plain_preconditioner("general-isai-1") == "general-isai"
    assert plain_preconditioner("ilu-syncfree") == "ilu-syncfree"
    assert plain_preconditioner("mg-pgm") == "mg-pgm"
    assert plain_preconditioner("ilu") == "ilu"


def _real(machine, name):
    d = RESULTS / machine / name
    if not d.is_dir():
        pytest.skip("run missing")
    return next(r for r in load_runs(RESULTS)
                if r.machine == machine and r.dir == name)


@pytest.mark.skipif(not RESULTS.is_dir(), reason="no results/")
def test_real_runs_measure():
    daint = measurements(_real("daint-gh200", "2026-10-18_9cb13ee6ab_3"))
    counts = {}
    for k in daint:
        counts[k.split("/")[0]] = counts.get(k.split("/")[0], 0) + 1
    assert counts == {"blas": 12, "spmv": 36, "spgemm": 2, "solver": 18,
                      "preconditioner": 86}
    assert sum(x.status == "not_available" for x in daint.values()) == 10

    bev = measurements(_real("beverin-mi300a", "2026-10-18_9cb13ee6ab_3"))
    assert sum(x.status == "disabled" for x in bev.values()) == 18
    assert bev["spgemm/elasticity_p2_graded"].status == "failed"

    bev2 = measurements(_real("beverin-mi300a", "2026-10-18_9cb13ee6ab_2"))
    k = bev2["preconditioner/apply/ilu-syncfree/anisotropic_heat"]
    assert k.status == "failed" and k.error == "timeout"
