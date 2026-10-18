import json

import pytest

from bench import groups as G
from bench.build import exe
from bench.machine import load_machine

SPD = "jacobi,ilu,ilu-syncfree,ic,ic-syncfree,spd-isai,mg-pgm,mg-pmis,mg-uniform"
NONSPD = "jacobi,ilu,ilu-syncfree,general-isai,mg-pgm,mg-pmis,mg-uniform"


def entry(name, set_, roles=(), spd=None, sellp=True):
    return {"name": name, "set": set_, "roles": list(roles), "spd": spd,
            "path": f"matrices/{set_}/{name}.bin", "run_sellp": sellp}


MANIFEST = {"matrices": [
    entry("web", "spmv"),
    entry("road", "spmv", sellp=False),
    entry("lap", "fem", ["precond"], spd=True),
    entry("conv", "fem", ["precond"], spd=False),
    entry("mm", "fem", ["spgemm"]),
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


def by_group(m, g):
    return G.invocations(m, MANIFEST, [g])


def flag(inv, name):
    return inv.args[inv.args.index(name) + 1]


def test_groups_constant():
    assert G.GROUPS == ["blas", "spmv", "spgemm", "solver", "preconditioner"]


def test_blas(m):
    (inv,) = by_group(m, "blas")
    assert inv.label == "vectors" and inv.exe == exe(m, "blas")
    assert flag(inv, "-operations") == "copy,dot,norm,axpy"
    assert json.loads(inv.stdin) == [{"n": 1048576}, {"n": 16777216}, {"n": 134217728}]
    assert inv.timeout_s == 15 * 60
    assert inv.args[:2] == ["-executor", m.executor]


def test_spmv_sellp_only_when_suitable(m):
    a, b = by_group(m, "spmv")
    assert (a.label, flag(a, "-formats")) == ("web", "csr,coo,hybrid,sellp")
    assert (b.label, flag(b, "-formats")) == ("road", "csr,coo,hybrid")
    assert flag(a, "-input_matrix") == str(m.root / "matrices/spmv/web.bin")
    assert a.timeout_s == 20 * 60 and a.exe == exe(m, "spmv") and a.stdin is None


def test_spgemm(m):
    (inv,) = by_group(m, "spgemm")
    assert inv.label == "mm" and inv.exe == exe(m, "sparse_blas")
    assert flag(inv, "-operations") == "spgemm" and flag(inv, "-spgemm_mode") == "normal"
    assert inv.timeout_s == 20 * 60


def test_solver_three_invocations(m):
    invs = by_group(m, "solver")
    assert len(invs) == 3
    assert [i.label for i in invs] == ["mgs", "cgs", "cgs2"]
    assert [flag(i, "-gmres_ortho_method") for i in invs] == ["mgs", "cgs", "cgs2"]
    assert flag(invs[0], "-solvers") == "cg,bicgstab,gmres,ir"
    assert flag(invs[1], "-solvers") == "gmres"
    for i in invs:
        assert "-detailed=false" in i.args and i.timeout_s == 3600
        assert flag(i, "-rel_res_goal") == "1e-10" and flag(i, "-max_iters") == "1000"
        assert flag(i, "-preconditioners") == "jacobi" and flag(i, "-jacobi_max_block_size") == "1"
        assert flag(i, "-rhs_generation") == "sinus" and flag(i, "-initial_guess_generation") == "0"
        assert json.loads(i.stdin) == [{"size": 8000000, "stencil": s, "optimal": {"spmv": "csr"}}
                                       for s in ("5pt", "7pt", "27pt")]


def test_solver_cases_name_the_optimal_format(m):
    # Ginkgo's solver benchmark exits with 2 on a case without optimal.spmv.
    for i in by_group(m, "solver"):
        for case in json.loads(i.stdin):
            assert isinstance(case["optimal"]["spmv"], str) and case["optimal"]["spmv"] == "csr"


def test_precond_lists_by_spd(m):
    a, b = by_group(m, "preconditioner")
    assert (a.label, flag(a, "-preconditioners")) == ("lap", SPD)
    assert (b.label, flag(b, "-preconditioners")) == ("conv", NONSPD)
    assert "-detailed=false" in a.args and flag(a, "-formats") == "csr" and a.timeout_s == 1800


def test_missing_matrix_flagged(m):
    (m.root / "matrices/spmv/web.bin").unlink()
    a, b = by_group(m, "spmv")
    assert a.missing_input == m.root / "matrices/spmv/web.bin"
    assert b.missing_input is None
    assert all(i.missing_input is None for i in by_group(m, "blas"))


def test_common_flags_everywhere(m):
    invs = G.invocations(m, MANIFEST)
    assert len(invs) == 1 + 2 + 1 + 3 + 2
    for i in invs:
        j = " ".join(i.args)
        warmup = 3 if i.group == "blas" else 1
        assert f"-warmup {warmup} -repetitions 5 -gpu_timer" in j
        assert j.startswith(f"-executor {m.executor} ")
        assert flag(i, "-repetition_growth_factor") == "1"
        assert flag(i, "-timer_method") == "median"
        assert flag(i, "-seed") == "42"
        assert i.args.count("-seed") == 1 and i.args.count("-repetition_growth_factor") == 1
