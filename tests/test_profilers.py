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
        assert p.stderr.parent == log
        assert p.stdout.parent == log  # rawdir defaults to the log directory


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
                       "--hip-runtime-trace", "--memory-copy-trace", "-d", str(log / "counters-0"),
                       "-o", "run",
                       "--output-format", "csv", "--"] + bench + ["-profile"]
    assert c0.out == log / "counters-0"
    assert c2.argv[:6] == [rp, "--pmc", "SQ_INSTS_VALU_FMA_F64", "SQ_INSTS_VALU_ADD_F64",
                           "SQ_INSTS_VALU_MUL_F64", "--marker-trace"]
    # only the first counter pass traces the copies; every pass traces the launches, which
    # place each kernel in the range open when it was launched
    assert all("--memory-copy-trace" not in p.argv for p in ps[2:])
    assert all("--hip-runtime-trace" in p.argv and "--marker-trace" in p.argv for p in ps[1:])
    # -detailed after the suite's -detailed=false
    assert ps[0].argv[-1] == "-detailed" and "-detailed=false" in ps[0].argv


def test_raw_outputs_elsewhere(tmp_path):
    m = machine("daint-gh200", tmp_path)
    log, raw = tmp_path / "logs", tmp_path / "raw"
    ps = P.passes(m, case(m, "blas/dot/1048576"), log, raw)
    assert ps[0].out == log / "time.json" and ps[0].stderr == log / "time.err"
    assert all(p.out.parent == raw and p.stdout.parent == raw and p.stderr.parent == log
               for p in ps[1:])
    assert str(raw / "trace.nsys-rep") in ps[1].post


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


DATA = Path(__file__).parent / "data" / "profiles"


def test_read_ncu_dot():
    ks = P.read_ncu((DATA / "daint" / "dot-ncu.csv").read_text())
    assert len(ks) == 4
    stacks = {tuple(k.stack) for k in ks}
    assert ("n = 1048576", "dot", "repetition", "dense::compute_dot_dispatch") in stacks
    # the operation runs cuBLAS's kernels, so names and operations differ
    k = next(k for k in ks if k.name.startswith("void dot_kernel<"))
    assert k.stack[-1] == "dense::compute_dot_dispatch"
    assert set(k.metrics) >= {"dram__bytes_read.sum", "dram__bytes_write.sum",
                              "lts__t_sectors_lookup_hit.sum"}
    assert all(isinstance(v, float) for v in k.metrics.values())
    assert k.launch is None


def test_read_ncu_pgm_names_intact():
    ks = P.read_ncu((DATA / "daint" / "pgm-ncu.csv").read_text())
    assert len(ks) == 40
    assert sum("repetition generate" in k.stack for k in ks) == 30
    assert sum("repetition apply" in k.stack for k in ks) == 10
    # templated names with commas, angle brackets and lambdas survive the CSV
    assert any("," in k.name and "<" in k.name for k in ks)
    assert all(k.metrics["dram__bytes_read.sum"] >= 0 for k in ks)


def test_ncu_numbers():
    assert P._number("266'924'800") == 266924800.0
    assert P._number("1,234.5") == 1234.5
    assert P._number("99.56") == 99.56
    assert P._number("n/a") is None


def test_read_nsys_dot():
    ranges, kernels, copies = P.read_nsys(DATA / "daint" / "dot-nsys.sqlite")
    assert len(kernels) == 2 and copies == []
    names = {r.name for r in ranges}
    assert {"dot", "repetition"} <= names
    for k in kernels:
        assert k.launch is not None and k.thread is not None and k.stack is None
        # launched inside the "repetition" range of its thread
        assert any(r.name == "repetition" and r.thread == k.thread and r.start <= k.launch <= r.end
                   for r in ranges)


def test_read_nsys_pgm():
    ranges, kernels, copies = P.read_nsys(DATA / "daint" / "pgm-nsys.sqlite")
    assert len(kernels) == 60
    assert len(copies) == 22 and all(c.bytes > 0 and c.launch is not None for c in copies)
    assert any(r.name == "repetition generate" for r in ranges)
    assert any(r.name == "allocate" for r in ranges)
    # parents: every range but the outermost ones has its enclosing range as parent
    inner = [r for r in ranges if r.parent is not None]
    assert inner and all(ranges[r.parent].start <= r.start and r.end <= ranges[r.parent].end
                         for r in inner)


def test_read_detailed():
    pgm = P.read_detailed((DATA / "daint" / "pgm-detailed.json").read_text(), ("generate", "apply"))
    assert set(pgm) == {"generate", "apply"}
    assert pgm["generate"]["components"]["csr::transpose"] > 0
    assert abs(pgm["generate"]["total"] - 0.11827667236328125) < 1e-12
    dot = P.read_detailed((DATA / "daint" / "dot-detailed.json").read_text(), ())
    assert dot == {"": {"total": 1.3248000293970109e-05, "components": {}}}
