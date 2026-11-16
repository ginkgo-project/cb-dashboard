from pathlib import Path

import pytest

from bench import attribute as A
from bench.profilers import Copy, Kernel, Range, _nest, read_detailed, read_ncu, read_nsys

DATA = Path(__file__).parent / "data" / "profiles"
STAGES = ("generate", "apply")
MEASURED = ("repetition generate", "repetition apply")


def ranges(*spec):
    """(name, start, end) on thread 1, nested by _nest."""
    return _nest([Range(n, s, e, 1) for n, s, e in spec])


def test_normalize_daint():
    m = A.normalize("ncu", {"dram__bytes_read.sum": 100.0, "dram__bytes_write.sum": 50.0,
                            "sm__sass_thread_inst_executed_op_dfma_pred_on.sum": 10.0,
                            "sm__sass_thread_inst_executed_op_dadd_pred_on.sum": 3.0,
                            "sm__sass_thread_inst_executed_op_dmul_pred_on.sum": 2.0,
                            "lts__t_sectors_lookup_hit.sum": 7.0,
                            "lts__t_sectors_lookup_miss.sum": 1.0})
    assert m == {"dram_read_bytes": 100.0, "dram_write_bytes": 50.0, "fp64_ops": 25.0,
                 "l2_hits": 7.0, "l2_misses": 1.0}
    # a pass with only some metrics gives only those
    assert A.normalize("ncu", {"dram__bytes_read.sum": 1.0}) == {"dram_read_bytes": 1.0}


def test_normalize_beverin():
    m = A.normalize("rocprofv3", {"FETCH_SIZE": 1048595.0, "WRITE_SIZE": 2.0,
                                  "SQ_INSTS_VALU_FMA_F64": 1048576.0,
                                  "TCC_HIT_sum": 4.0, "TCC_MISS_sum": 12.0})
    assert m["dram_read_bytes"] == 1048595.0 * 1024 and m["dram_write_bytes"] == 2048.0
    # per wavefront of 64 lanes; an FMA is two operations
    assert m["fp64_ops"] == 2 * 1048576 * 64
    assert (m["l2_hits"], m["l2_misses"]) == (4.0, 12.0)


def test_nested_ranges():
    rs = ranges(("repetition generate", 0, 100), ("generate(gko::multigrid::Pgm)", 10, 90),
                ("csr::spgemm", 20, 40), ("allocate", 22, 24), ("allocate", 26, 27),
                ("pgm::match_edge", 50, 60))
    ks = [Kernel("spgemm_a", 21, 1), Kernel("spgemm_b", 30, 1), Kernel("match", 55, 1),
          Kernel("pgm_other", 70, 1)]
    cs = [Copy(23, 1, 64), Copy(80, 1, 32)]
    rows = A.attribute_trace(rs, ks, cs, MEASURED, STAGES)
    spgemm = rows[("generate", "csr::spgemm")]
    assert spgemm.calls == 1 and spgemm.launches == 2 and spgemm.allocations == 2
    assert spgemm.parent == "generate(gko::multigrid::Pgm)"
    assert set(spgemm.kernels) == {"spgemm_a", "spgemm_b"}
    # the copy launched inside an allocate range counts for the allocate's row
    assert rows[("generate", "allocate")].calls == 2
    assert rows[("generate", "allocate")].copies == 1 and rows[("generate", "allocate")].copy_bytes == 64
    lin = rows[("generate", "generate(gko::multigrid::Pgm)")]
    assert lin.launches == 1 and lin.copies == 1 and lin.copy_bytes == 32  # pgm_other, the late copy


def test_kernel_outside_operations():
    rs = ranges(("repetition apply", 0, 100), ("csr::spmv", 10, 20))
    ks = [Kernel("spmv", 15, 1), Kernel("stray", 50, 1), Kernel("before", -5, 1)]
    rows = A.attribute_trace(rs, ks, [], MEASURED, STAGES)
    assert rows[("apply", A.OUTSIDE)].launches == 1
    assert rows[("apply", A.OUTSIDE)].kernels == {"stray": {"launches": 1}}
    # outside every measured range: dropped
    assert sum(r.launches for r in rows.values()) == 2


def test_other_thread_not_attributed():
    rs = _nest([Range("repetition", 0, 100, 1), Range("dense::fill", 10, 20, 1)])
    rows = A.attribute_trace(rs, [Kernel("k", 15, 2)], [], ("repetition",), ())
    # the range is a row (one call), but the other thread's kernel isn't in it
    assert rows[("", "dense::fill")].calls == 1 and rows[("", "dense::fill")].launches == 0
    assert sum(r.launches for r in rows.values()) == 0


def test_counters_by_stack_and_by_launch():
    ks = [Kernel("a", stack=["case", "mg-pgm", "repetition generate", "generate(x)", "csr::spgemm"],
                 metrics={"dram__bytes_read.sum": 10.0, "lts__t_sectors_lookup_hit.sum": 3.0}),
          Kernel("a", stack=["case", "mg-pgm", "repetition generate", "generate(x)", "csr::spgemm"],
                 metrics={"dram__bytes_read.sum": 5.0, "lts__t_sectors_lookup_hit.sum": 1.0}),
          Kernel("b", stack=["case", "mg-pgm", "storage generate", "csr::spgemm"],
                 metrics={"dram__bytes_read.sum": 99.0})]
    c = A.attribute_counters("ncu", ks, MEASURED, STAGES)
    assert c == {("generate", "csr::spgemm"): {"a": {"dram_read_bytes": 15.0, "l2_hits": 4.0}}}
    rs = ranges(("repetition", 0, 100), ("dense::fill", 10, 20))
    c2 = A.attribute_counters("rocprofv3", [Kernel("f", 12, 1, metrics={"FETCH_SIZE": 1.0})],
                              ("repetition",), (), rs)
    assert c2 == {("", "dense::fill"): {"f": {"dram_read_bytes": 1024.0}}}


def test_rows_merge_and_l2_rate():
    trace = A.attribute_trace(ranges(("repetition", 0, 100), ("dense::fill", 10, 20)),
                              [Kernel("f", 12, 1), Kernel("f", 14, 1)], [], ("repetition",), ())
    counters = [{("", "dense::fill"): {"f": {"dram_read_bytes": 10.0, "l2_hits": 30.0}}},
                {("", "dense::fill"): {"f": {"l2_misses": 10.0}}}]
    times = {"": {"total": 2.0, "components": {"dense::fill": 1.5}}}
    [row] = A.rows(trace, counters, times)
    assert row["stage"] == "" and row["name"] == "dense::fill" and row["time"] == 1.5
    assert row["launches"] == 2 and row["dram_read_bytes"] == 10.0
    # the rate from the sums, not a mean of rates
    assert row["l2_hit_rate"] == pytest.approx(0.75)
    assert row["kernels"] == [{"name": "f", "launches": 2, "dram_read_bytes": 10.0,
                               "l2_hits": 30.0, "l2_misses": 10.0, "l2_hit_rate": 0.75}]


def test_single_row_without_components_gets_the_total():
    trace = A.attribute_trace(ranges(("repetition", 0, 100), ("dense::compute_dot_dispatch", 10, 20)),
                              [Kernel("d", 12, 1)], [], ("repetition",), ())
    [row] = A.rows(trace, [], {"": {"total": 2.0, "components": {}}})
    assert row["time"] == 2.0


def test_daint_fixtures_end_to_end():
    stages, measured = STAGES, MEASURED
    rs, ks, cs = read_nsys(DATA / "daint" / "pgm-nsys.sqlite")
    trace = A.attribute_trace(rs, ks, cs, measured, stages)
    counters = [A.attribute_counters("ncu", read_ncu((DATA / "daint" / "pgm-ncu.csv").read_text()),
                                     measured, stages)]
    times = read_detailed((DATA / "daint" / "pgm-detailed.json").read_text(), stages)
    out = A.rows(trace, counters, times)
    names = {(r["stage"], r["name"]) for r in out}
    assert ("generate", "allocate") in names
    assert all(r["stage"] in stages for r in out)
    # every kernel the trace kept inside the generate stage is in some row
    launched = sum(r["launches"] for r in out)
    assert 0 < launched <= 60
    assert any(r.get("dram_read_bytes") for r in out)
    assert any(r["time"] is not None and r["time"] > 0 for r in out)


def test_beverin_fixtures_end_to_end():
    from bench.profilers import read_rocprof
    b = DATA / "beverin"
    ranges, kernels, copies = read_rocprof(b / "pgm-0")
    trace = A.attribute_trace(ranges, kernels, copies, MEASURED, STAGES)
    counters = []
    for i in range(4):
        r, k, _ = read_rocprof(b / f"pgm-{i}")
        counters.append(A.attribute_counters("rocprofv3", k, MEASURED, STAGES, r))
    times = read_detailed((b / "pgm-detailed.json").read_text(), STAGES)
    out = A.rows(trace, counters, times)
    assert out and all(r["stage"] == "generate" for r in out)
    assert sum(r["launches"] for r in out) == 40
    measured = [r for r in out if r["launches"]]
    assert all("dram_read_bytes" in r and "dram_write_bytes" in r and "fp64_ops" in r
               and "l2_hit_rate" in r for r in measured)
    assert any(r["time"] for r in out)


def test_trace_and_ncu_kernels_meet_by_symbol():
    """nsys and ncu print a kernel's name differently; both give its mangled symbol, which
    joins them, and the trace's readable name is the one shown."""
    rs = ranges(("repetition", 0, 100), ("csr::spmv", 10, 20))
    trace = A.attribute_trace(rs, [Kernel("_Z4spmvv", 12, 1, display="void gko::csr::spmv<(int)32>()")],
                              [], ("repetition",), ())
    counters = [A.attribute_counters("ncu", [Kernel("_Z4spmvv", stack=["repetition", "csr::spmv"],
                                                    metrics={"dram__bytes_read.sum": 8.0})],
                                     ("repetition",), ())]
    [row] = A.rows(trace, counters, {"": {"total": 1.0, "components": {}}})
    assert row["kernels"] == [{"name": "void gko::csr::spmv<(int)32>()", "launches": 1,
                               "dram_read_bytes": 8.0}]


def test_normalize_beverin_matrix_cores():
    """rocBLAS runs GEMV as a GEMM on the MI300A's matrix cores: those FP64 operations are
    counted by SQ_INSTS_VALU_MFMA_MOPS_F64, in units of 512."""
    m = A.normalize("rocprofv3", {"SQ_INSTS_VALU_FMA_F64": 0.0, "SQ_INSTS_VALU_ADD_F64": 6180.0,
                                  "SQ_INSTS_VALU_MUL_F64": 2060.0,
                                  "SQ_INSTS_VALU_MFMA_MOPS_F64": 4218880.0})
    assert m["fp64_ops"] == (6180 + 2060) * 64 + 4218880 * 512


def test_total_time_to_the_row_that_launches():
    """SpMV and BLAS write only the case's time, no per-range times: it goes to the one row
    that launched kernels, not to the linear operator's apply(...) around it."""
    rs = ranges(("repetition", 0, 100), ("apply(gko::matrix::Csr)", 5, 50), ("csr::spmv", 10, 20))
    trace = A.attribute_trace(rs, [Kernel("k", 12, 1)], [], ("repetition",), ())
    out = {r["name"]: r["time"] for r in A.rows(trace, [], {"": {"total": 2.0, "components": {}}})}
    assert out == {"apply(gko::matrix::Csr)": None, "csr::spmv": 2.0}
