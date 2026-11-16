import json
from pathlib import Path

import pytest

from bench import kernel_cases as K
from bench import profilers as P
from bench.machine import load_machine

MANIFEST = {"matrices": [
    {"name": "elasticity_p2", "set": "fem", "roles": ["precond"], "spd": True,
     "path": "matrices/fem/elasticity_p2.bin", "run_sellp": True},
]}


def machine(name, tmp_path):
    m = load_machine(name, env={"SCRATCH": str(tmp_path / "s")})
    m.root = tmp_path / "root"
    return m


def case(m, key):
    return next(c for c in K.kernel_cases(m, MANIFEST) if c.key == key)


def test_daint_passes(tmp_path):
    m = machine("daint-gh200", tmp_path)
    c = case(m, "blas/dot/1048576")
    log = tmp_path / "logs"
    ps = P.passes(m, c, log)
    assert [p.name for p in ps] == ["time", "trace", "counters-0", "counters-1", "counters-2",
                                    "counters-3"]
    bench = [str(c.exe), *c.args]
    time, trace, c0 = ps[0], ps[1], ps[2]
    assert time.argv == bench + ["-detailed"]
    assert time.stdout == time.out == log / "time.json" and time.post is None
    assert trace.argv == ["nsys", "profile", "-t", "cuda,nvtx", "--force-overwrite", "true",
                          "-o", str(log / "trace")] + bench + ["-profile"]
    assert trace.post == ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                          "-o", str(log / "trace.sqlite"), str(log / "trace.nsys-rep")]
    assert trace.out == log / "trace.sqlite"
    assert c0.argv == ["ncu", "--nvtx", "--nvtx-include", "regex:repetition.*/",
                       "--cache-control", "none", "--clock-control", "none",
                       "--metrics", "dram__bytes_read.sum,dram__bytes_write.sum",
                       "--csv", "--log-file", str(log / "counters-0.csv")] + bench + ["-profile"]
    assert c0.out == log / "counters-0.csv"
    assert ps[3].argv[ps[3].argv.index("--metrics") + 1].count(",") == 2
    for p in ps:  # -profile (1 repetition, no warm-up) comes after the suite's -repetitions 5
        if p.name != "time":
            assert p.argv[-1] == "-profile"
        assert p.stdin == c.stdin
        assert p.stdout.parent == log and p.stderr.parent == log


def test_beverin_passes(tmp_path):
    m = machine("beverin-mi300a", tmp_path)
    c = case(m, "preconditioner/mg-pgm/elasticity_p2")
    log = tmp_path / "logs"
    ps = P.passes(m, c, log)
    assert [p.name for p in ps] == ["time", "counters-0", "counters-1", "counters-2", "counters-3"]
    rp = "/opt/rocm-6.3.0/bin/rocprofv3"
    bench = [str(c.exe), *c.args]
    c0, c2 = ps[1], ps[3]
    assert c0.argv == [rp, "--pmc", "FETCH_SIZE", "--marker-trace", "--kernel-trace",
                       "--memory-copy-trace", "-d", str(log / "counters-0"), "-o", "run",
                       "--output-format", "csv", "--"] + bench + ["-profile"]
    assert c0.out == log / "counters-0"
    assert c2.argv[:6] == [rp, "--pmc", "SQ_INSTS_VALU_FMA_F64", "SQ_INSTS_VALU_ADD_F64",
                           "SQ_INSTS_VALU_MUL_F64", "--marker-trace"]
    # only the first counter pass traces the copies
    assert all("--memory-copy-trace" not in p.argv for p in ps[2:])
    # -detailed after the suite's -detailed=false
    assert ps[0].argv[-1] == "-detailed" and "-detailed=false" in ps[0].argv


def test_measured_ranges(tmp_path):
    m = machine("daint-gh200", tmp_path)
    assert P.measured_ranges(case(m, "preconditioner/mg-pgm/elasticity_p2")) == (
        "repetition generate", "repetition apply")
    assert P.measured_ranges(case(m, "blas/dot/1048576")) == ("repetition",)


def test_no_kernels_settings(tmp_path):
    m = machine("daint-gh200", tmp_path)
    m.kernels = None
    with pytest.raises(ValueError):
        P.passes(m, case(m, "blas/dot/1048576"), tmp_path)
