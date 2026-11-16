import json

import pytest

from bench import groups as G
from bench import kernel_cases as K
from bench.build import exe
from bench.machine import load_machine


def entry(name, set_, roles=(), spd=None, sellp=True):
    return {"name": name, "set": set_, "roles": list(roles), "spd": spd,
            "path": f"matrices/{set_}/{name}.bin", "run_sellp": sellp}


MANIFEST = {"matrices": [
    entry("ecology2", "spmv"),
    entry("TSOPF_RS_b2383_c1", "spmv"),
    entry("audikw1", "spmv"),
    entry("USA-road-d.USA", "spmv"),
    entry("com-Orkut", "spmv", sellp=False),
    entry("curl_curl", "fem", ["precond"], spd=True),
    entry("elasticity_p2", "fem", ["precond"], spd=True),
    entry("porous_pressure", "fem", ["precond"], spd=True),
    entry("poisson_p1", "fem", ["spgemm"], spd=True),
    entry("elasticity_p2_graded", "fem", ["spgemm"], spd=True),
]}


@pytest.fixture
def m(tmp_path):
    mach = load_machine("daint-gh200", env={"SCRATCH": str(tmp_path / "s")})
    mach.root = tmp_path / "root"
    for e in MANIFEST["matrices"]:
        p = mach.root / e["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    return mach


def flag(case, name):
    return case.args[case.args.index(name) + 1]


def keys(cases, prefix):
    return [c.key for c in cases if c.key.startswith(prefix)]


def test_counts(m):
    cases = K.kernel_cases(m, MANIFEST)
    assert len(keys(cases, "preconditioner/")) == 6
    assert len(keys(cases, "spmv/")) == 15
    assert "spmv/sellp/com-Orkut" not in keys(cases, "spmv/")
    assert len(keys(cases, "spgemm/")) == 2
    assert sorted(keys(cases, "blas/dot/") + keys(cases, "blas/axpy/")) == sorted(
        f"blas/{op}/{n}" for op in ("dot", "axpy") for n in G.VECTOR_LENGTHS)
    assert keys(cases, "blas/gemv/") == [f"blas/gemv/{n}" for n in K.GEMV_SIZES]
    # multigrid first, as the budget spends on them first
    assert all(c.key.startswith("preconditioner/") for c in cases[:6])
    assert len({c.label for c in cases}) == len(cases)


def test_spmv_case_args(m):
    case = next(c for c in K.kernel_cases(m, MANIFEST) if c.key == "spmv/sellp/audikw1")
    assert case.exe == exe(m, "spmv")
    assert flag(case, "-formats") == "sellp"
    assert flag(case, "-input_matrix") == str(m.root / "matrices/spmv/audikw1.bin")
    assert "-gpu_timer" in case.args and flag(case, "-seed") == "42"
    assert case.stages == () and case.stdin is None and case.missing_input is None


def test_multigrid_stages_and_unavailable(m):
    case = next(c for c in K.kernel_cases(m, MANIFEST, unavailable={"mg-pmis"})
                if c.key == "preconditioner/mg-pmis/curl_curl")
    assert case.exe == exe(m, "preconditioner")
    assert case.stages == ("generate", "apply")
    assert flag(case, "-preconditioners") == "mg-pmis"
    assert flag(case, "-formats") == "csr"
    assert G.ADDED["mg-pmis"][:10] in case.unavailable
    pgm = next(c for c in K.kernel_cases(m, MANIFEST) if c.key == "preconditioner/mg-pgm/curl_curl")
    assert pgm.unavailable is None


def test_spgemm_case(m):
    case = next(c for c in K.kernel_cases(m, MANIFEST) if c.key == "spgemm/poisson_p1")
    assert case.exe == exe(m, "sparse_blas")
    assert flag(case, "-operations") == "spgemm" and flag(case, "-spgemm_mode") == "normal"


def test_blas_case(m):
    case = next(c for c in K.kernel_cases(m, MANIFEST) if c.key == "blas/axpy/16777216")
    assert case.exe == exe(m, "blas")
    assert flag(case, "-operations") == "axpy"
    assert json.loads(case.stdin) == [{"n": 16777216}]


def test_gemv_is_mm_with_one_column(m):
    case = next(c for c in K.kernel_cases(m, MANIFEST) if c.key == "blas/gemv/8192")
    assert flag(case, "-operations") == "mm"
    assert json.loads(case.stdin) == [{"n": 8192, "k": 8192, "m": 1}]


def test_missing_matrix(m):
    (m.root / "matrices/spmv/ecology2.bin").unlink()
    case = next(c for c in K.kernel_cases(m, MANIFEST) if c.key == "spmv/csr/ecology2")
    assert case.missing_input == m.root / "matrices/spmv/ecology2.bin"


def test_extra_case_from_the_suite(m):
    cases = K.kernel_cases(m, MANIFEST, extra=["spmv/csr/USA-road-d.USA",
                                                "preconditioner/mg-pgm/porous_pressure"])
    assert [c.key for c in cases[-2:]] == ["spmv/csr/USA-road-d.USA",
                                          "preconditioner/mg-pgm/porous_pressure"]
    # a selected case named again isn't repeated
    again = K.kernel_cases(m, MANIFEST, extra=["spmv/csr/audikw1"])
    assert len(again) == len(K.kernel_cases(m, MANIFEST))
    with pytest.raises(KeyError):
        K.kernel_cases(m, MANIFEST, extra=["spmv/csr/nope"])
    with pytest.raises(KeyError):
        K.kernel_cases(m, MANIFEST, extra=["solver/cg/7pt"])


def test_parse_case_keys():
    assert K.parse_case_keys(" spmv/csr/a, blas/dot/1048576 ,") == ["spmv/csr/a", "blas/dot/1048576"]
    assert K.parse_case_keys("") == []
