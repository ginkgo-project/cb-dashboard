import json
from pathlib import Path

import pytest

from bench import distributed
from bench.machine import load_machine

MACHINES = ["daint-gh200", "beverin-mi300a"]


@pytest.fixture(params=MACHINES)
def m(request):
    return load_machine(request.param, env={"SCRATCH": "/s"})


def test_constants():
    assert distributed.GPU_COUNTS == [1, 2, 4, 8, 16, 32, 64]
    assert distributed.ROWS_PER_GPU == 256 ** 3
    assert distributed.GROUPS == ["mpi", "spmv", "spgemm", "solver", "preconditioner"]
    assert distributed.TARGETS == ["spmv_distributed", "sparse_blas_distributed",
                                   "solver_distributed", "preconditioner_distributed"]


def test_srun_args(m):
    per_rank = m.distributed.srun
    assert distributed.srun_args(m, 1) == ["--nodes=1", "--ntasks=1", "--ntasks-per-node=1",
                                           *per_rank]
    assert distributed.srun_args(m, 2)[:3] == ["--nodes=1", "--ntasks=2", "--ntasks-per-node=2"]
    assert distributed.srun_args(m, 8)[:3] == ["--nodes=2", "--ntasks=8", "--ntasks-per-node=4"]
    assert distributed.srun_args(m, 64)[:3] == ["--nodes=16", "--ntasks=64",
                                                "--ntasks-per-node=4"]


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
def test_case_size_is_weak_scaling(n):
    # the generator divides "size" by the ranks: 256^3 rows per GPU
    assert distributed.case(n) == [{"size": 16777216 * n, "stencil": "7pt",
                                    "comm_pattern": "stencil",
                                    "optimal": {"spmv": "csr-csr"}}]


def test_exe(m):
    b = m.distributed.build_dir
    assert distributed.exe(m, "spmv_distributed") == b / "benchmark/spmv/distributed/spmv_distributed"
    assert (distributed.exe(m, "preconditioner_distributed")
            == b / "benchmark/preconditioner/distributed/preconditioner_distributed")


@pytest.mark.parametrize("n", [1, 8, 64])
def test_invocations_per_gpu_count(m, n):
    mpibench = Path("/mb/mpibench")
    invs = distributed.invocations(m, n, mpibench)
    assert [(i.group, i.label) for i in invs] == [
        ("mpi", "mpibench"), ("spmv", "stencil"), ("spgemm", "stencil"),
        ("solver", "cg"), ("solver", "gmres"),
        ("preconditioner", "mg-pgm"), ("preconditioner", "mg-uniform"),
        ("preconditioner", "mg-pmis")]
    mpi, rest = invs[0], invs[1:]
    assert mpi.exe == mpibench and mpi.stdin is None
    assert ("--pingpong" in mpi.args) == (n == 8)
    assert all(i.stdin == json.dumps(distributed.case(n)) for i in rest)
    assert all("/distributed/" in str(i.exe) for i in rest)
    assert all(i.args[:2] == ["-executor", m.executor] for i in rest)
    solver = [i for i in rest if i.group == "solver"]
    assert all("schwarz-jacobi" in i.args and "1e-10" in i.args and "1000" in i.args
               for i in solver)
    assert [i.args[i.args.index("-solvers") + 1] for i in solver] == ["cg", "gmres"]
    assert [i.args[i.args.index("-operations") + 1] for i in rest if i.group == "spgemm"] == ["spgemm"]
    assert all(i.timeout_s > 0 for i in invs)


def test_machine_without_distributed_settings(tmp_path):
    (tmp_path / "m.toml").write_text(
        'executor = "reference"\nuenv = "u"\narch = "x86_64"\nsbatch = []\nsrun = []\n')
    m = load_machine("m", machines_dir=tmp_path, env={"SCRATCH": "/s"})
    with pytest.raises(ValueError, match="no \\[distributed\\] settings"):
        distributed.srun_args(m, 1)
