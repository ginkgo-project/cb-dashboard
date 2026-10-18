import pytest

from dashboard import models
from dashboard.cases import Measurement
from dashboard.models import (MODELS, cost, compare_kind, load_machines,
                              model_name, stencil_nonzeros)


def _m(group, labels, extra, time=1.0):
    return Measurement("k", group, labels, "ok", time, extra=extra)


def _spmv(fmt, n=10, c=10, z=28):
    return _m("spmv", {"format": fmt}, {"rows": n, "cols": c, "nonzeros": z})


def _solver(name, params=None, it=10):
    return _m("solver", {"solver": name, "stencil": "5pt"},
              {"iterations": it, "rows": 100, "parameters": params or {}})


def _pre(name, stage="apply", extra=None):
    return _m("preconditioner", {"preconditioner": name, "stage": stage},
              {"rows": 100, "nonzeros": 500, **(extra or {})})


@pytest.fixture
def z500(monkeypatch):
    monkeypatch.setattr(models, "stencil_nonzeros", lambda s, r: 500)


def test_stencil_nonzeros():
    assert stencil_nonzeros("5pt", 7997584) == 39976608
    assert stencil_nonzeros("7pt", 8000000) == 55760000
    assert stencil_nonzeros("27pt", 8000000) == 213847192


def test_spmv_formats():
    assert cost(_spmv("csr"), None) == (540, 56)
    assert cost(_spmv("coo"), None) == (608, 56)
    s = {"ell_num_stored_elements_per_row": 2, "coo_num_stored_elements": 8}
    assert cost(_spmv("hybrid"), s) == (528, 56)
    assert cost(_spmv("sellp"), {"slice_size": 64, "total_cols": 1}) == (928, 56)
    assert cost(_spmv("hybrid"), None) == (None, None)
    assert cost(_spmv("sellp"), None) == (None, None)


def test_cg_bicgstab_ir(z500):
    assert cost(_solver("cg"), None) == (216040, 23000)
    assert cost(_solver("bicgstab"), None) == (392080, 46000)
    assert cost(_solver("ir"), None) == (144040, 16000)


def test_gmres_variants(z500):
    p = {"krylov_dim": 3}
    assert cost(_solver("gmres-mgs", p, 4), None) == (119224, 13200)
    assert cost(_solver("gmres-cgs", p, 4), None) == (116824, 13200)
    assert cost(_solver("gmres-cgs2", p, 4), None) == (142424, 16000)


def test_preconditioner_applies():
    assert cost(_pre("jacobi", extra={}), {"storage": 8000}) == (9600, 2000)
    assert cost(_pre("jacobi"), None) == (None, None)
    for name in ("ilu", "ic-syncfree", "spd-isai"):
        assert cost(_pre(name), None) == (11208, 1200)
    assert cost(_pre("general-isai"), None) == (8004, 1000)


def test_amg():
    m = _pre("mg-pgm", extra={"parameters": {
        "smoother_iters": 1,
        "coarsest_solver": [{"criteria": [{"max_iters": 2}]}]}})
    s = {"levels": [{"fine_rows": 100, "coarse_rows": 40,
                     "coarse_nonzeros": 160}]}
    assert cost(m, s) == (60024, 6180)
    assert cost(m, None) == (None, None)


def test_rates_and_generate():
    m = _m("blas", {"operation": "dot"},
           {"bandwidth_rate": 2e9, "flops_rate": 1e9}, time=0.5)
    assert cost(m, None) == (1e9, 5e8)
    for name in ("jacobi", "mg-pgm", "ilu"):
        assert cost(_pre(name, "generate"), None) == (None, None)
    assert model_name(_pre("ilu", "generate")) == "generate"
    assert model_name(_pre("ic-syncfree")) == "ilu"
    assert model_name(_pre("mg-pmis")) == "amg"
    assert model_name(_solver("gmres-cgs")) == "gmres-cgs"


def test_compare_kind():
    mach = {"largest_cache": 50e6}
    assert compare_kind(_spmv("csr"), 80e6, mach) == "time"
    assert compare_kind(_spmv("csr"), 195e6, mach) == "bandwidth"
    assert compare_kind(_spmv("csr"), None, mach) == "time"
    assert compare_kind(_solver("cg"), 1.0, mach) == "bandwidth"
    assert compare_kind(_pre("ilu"), 1e9, mach) == "time"
    assert compare_kind(_m("spgemm", {}, {}), 1e9, mach) == "time"
    assert compare_kind(_pre("ilu", "generate"), None, mach) == "time"


def test_models_have_text():
    for name, mod in MODELS.items():
        assert mod.name == name and mod.source and mod.bytes_text


def test_machines_file():
    mach = load_machines()
    assert set(mach) == {"daint-gh200", "beverin-mi300a"}
    for v in mach.values():
        assert v["fp64_peak"] > 0 and v["largest_cache"] > 0
        assert v["citations"] and v["label"] and v["cache_name"]
