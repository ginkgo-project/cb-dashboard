from dashboard.cases import Measurement
from dashboard.load import Run
from dashboard.reproduce import command

BINDING = ["srun", "--ntasks=1", "--gpus-per-task=1"]


def _run(group, label, args, machine="daint-gh200"):
    record = {"gpu": {"binding": {"srun": BINDING}},
              "groups": {group: {"invocations": [
                  {"label": label, "args": args}]}}}
    return Run(machine, "d", record, {})


def _m(group, labels, invocation, extra=None):
    return Measurement("k", group, labels, "ok", 1.0, [], invocation, None,
                       extra or {})


def test_spmv_command_narrows_formats():
    args = ["-executor", "cuda", "-formats", "csr,coo,hybrid",
            "-input_matrix", "/m/cage15.bin"]
    run = _run("spmv", "cage15", args)
    m = _m("spmv", {"operation": "spmv", "format": "coo",
                    "matrix": "cage15"}, "spmv:cage15")
    assert command(run, m) == (
        "srun --ntasks=1 --gpus-per-task=1 "
        "$SCRATCH/cb/build/daint-gh200/benchmark/spmv/spmv "
        "-executor cuda -formats coo -input_matrix /m/cage15.bin")


def test_solver_command_has_stdin():
    args = ["-executor", "cuda", "-preconditioners", "jacobi",
            "-solvers", "gmres", "-gmres_ortho_method", "cgs"]
    run = _run("solver", "cgs", args)
    m = _m("solver", {"solver": "gmres-cgs", "stencil": "7pt"},
           "solver:cgs", {"size": 8000000, "optimal": {"spmv": "csr"}})
    cmd = command(run, m)
    assert "-solvers gmres " in cmd
    assert "-preconditioners jacobi" in cmd
    assert cmd.endswith(
        """<<< '[{"size": 8000000, "stencil": "7pt", "optimal": """
        """{"spmv": "csr"}}]'""")


def test_blas_command():
    run = _run("blas", "vectors", ["-executor", "cuda", "-operations",
                                   "copy,dot,norm,axpy"])
    m = _m("blas", {"operation": "dot", "n": "16777216"}, "blas:vectors",
           {"n": 16777216})
    cmd = command(run, m)
    assert "-operations dot" in cmd
    assert "blas/blas" in cmd
    assert cmd.endswith("""<<< '[{"n": 16777216}]'""")


def test_spgemm_unchanged():
    args = ["-operations", "spgemm", "-input_matrix", "/m/a.bin"]
    run = _run("spgemm", "a", args)
    m = _m("spgemm", {"operation": "spgemm", "matrix": "a"}, "spgemm:a")
    assert command(run, m).endswith(
        "sparse_blas/sparse_blas -operations spgemm -input_matrix /m/a.bin")


def test_preconditioner_narrowed_in_old_and_new_runs():
    args = ["-formats", "csr", "-preconditioners", "jacobi,ilu,ic",
            "-input_matrix", "/m/a.bin"]
    m = _m("preconditioner", {"stage": "apply", "preconditioner": "ilu",
                              "matrix": "a"}, "preconditioner:a")
    cmd = command(_run("preconditioner", "a", args), m)
    assert "-preconditioners ilu -input_matrix" in cmd
    assert "-formats csr" in cmd


def test_unknown_invocation_is_none():
    run = _run("spmv", "cage15", ["-formats", "csr"])
    m = _m("spmv", {"format": "csr"}, "spmv:other")
    assert command(run, m) is None
