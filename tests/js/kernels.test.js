import { test } from "node:test";
import assert from "node:assert/strict";
import * as K from "../../site/kernels-logic.js";
import { TABS, PARAMS, parseHash } from "../../site/logic.js";

// one row of a case's run, as bench/attribute.py writes it
const row = (name, time, extra = {}) => ({
  stage: "generate", name, parent: null, time, calls: 1, launches: 1, allocations: 0,
  copies: 0, copy_bytes: 0, kernels: [], ...extra,
});
const run = (dir, operations, extra = {}) => ({ dir, status: "ok", operations, ...extra });

test("the tab is listed and its case is a parameter", () => {
  assert.ok(TABS.includes("kernels"));
  assert.ok(PARAMS.includes("case"));
  assert.equal(parseHash("#kernels?case=spmv%2Fcsr%2Fa").params.case, "spmv/csr/a");
});

test("bars keep an operation's colour when another appears", () => {
  const runs = [
    run("r1", [row("csr::spgemm", 0.6), row("pgm::match_edge", 0.4)]),
    run("r2", [row("csr::spgemm", 0.3), row("csr::transpose", 0.3), row("pgm::match_edge", 0.4)]),
  ];
  const bars = K.barModel(runs, "generate");
  const colour = (b, name) => b.segments.find((s) => s.name === name).colour;
  assert.equal(colour(bars[0], "csr::spgemm"), colour(bars[1], "csr::spgemm"));
  assert.equal(colour(bars[0], "pgm::match_edge"), colour(bars[1], "pgm::match_edge"));
  assert.notEqual(colour(bars[1], "csr::transpose"), colour(bars[1], "csr::spgemm"));
  assert.equal(bars[1].total, 1.0);
});

test("small operations fold into other", () => {
  const bars = K.barModel([run("r1", [row("a", 0.98), row("b", 0.01), row("c", 0.01)])], "generate");
  assert.deepEqual(bars[0].segments.map((s) => s.name), ["a", K.OTHER]);
  assert.equal(bars[0].segments[1].time, 0.02);
  assert.equal(bars[0].segments[1].colour, "other");
});

test("bars of the other stage and failed runs", () => {
  const runs = [run("r1", [row("a", 1.0), { ...row("b", 2.0), stage: "apply" }]),
    { dir: "r2", status: "failed", reason: "time: exit 1", operations: [] }];
  const bars = K.barModel(runs, "apply");
  assert.deepEqual(bars[0].segments.map((s) => s.name), ["b"]);
  assert.equal(bars[1].total, null);
  assert.equal(bars[1].reason, "time: exit 1");
});

test("the table marks a changed count, a moved counter, new and gone operations", () => {
  const before = run("r1", [row("csr::spgemm", 1.0, { allocations: 2, dram_read_bytes: 100 }),
    row("old::op", 0.1)]);
  const after = run("r2", [row("csr::spgemm", 1.02, { allocations: 3, dram_read_bytes: 150 }),
    row("new::op", 0.2)]);
  const rows = K.tableModel(after, before, "generate");
  const get = (name) => rows.find((r) => r.name === name);
  const spgemm = get("csr::spgemm");
  assert.equal(spgemm.changes.allocations, "changed");  // a count: any change
  assert.equal(spgemm.changes.time, null);              // +2 %: within 5 %
  assert.equal(spgemm.changes.dram_read_bytes, "up");   // +50 %
  assert.equal(get("new::op").isNew, true);
  assert.equal(get("old::op").isGone, true);
  // by time, slowest first; gone ones last
  assert.deepEqual(rows.map((r) => r.name), ["csr::spgemm", "new::op", "old::op"]);
});

test("the table without a previous run marks nothing", () => {
  const rows = K.tableModel(run("r1", [row("a", 1.0)]), null, "generate");
  assert.equal(rows[0].isNew, false);
  assert.deepEqual(Object.values(rows[0].changes).filter(Boolean), []);
});

test("short kernel names", () => {
  assert.equal(K.shortKernel("void gko::kernels::cuda::csr::kernel::abstract_classical_spmv<4ul, double>(long, double const*)"),
    "csr::kernel::abstract_classical_spmv");
  assert.equal(K.shortKernel("void gko::kernels::cuda::generic_kernel_1d<void gko::kernels::cuda::components::aos_to_soa<double, int>(std::shared_ptr<gko::CudaExecutor const>)::[lambda(auto, auto) (instance 1)], long>(long, auto)"),
    "components::aos_to_soa (generic kernel)");
  assert.equal(K.shortKernel("void dot_kernel<double, 128, 0, cublasDotParams<double, double>>(cublasDotParams<double, double>)"),
    "dot_kernel");
  assert.equal(K.shortKernel("gko::kernels::hip::coo::(anonymous namespace)::abstract_spmv<double, int>(...)"),
    "coo::abstract_spmv");
});

test("the hit rate and bytes against the model", () => {
  assert.equal(K.modelRatio({ dram_read_bytes: 150, dram_write_bytes: 50 }, 100), 2);
  assert.equal(K.modelRatio({ dram_read_bytes: 150 }, undefined), null);
});

test("short range names keep operations and fold linear operators' types", () => {
  assert.equal(K.shortRange("csr::transpose"), "csr::transpose");
  assert.equal(K.shortRange("generate(gko::multigrid::Pgm<double, int>::Factory)"),
    "generate(multigrid::Pgm<…>::Factory)");
  assert.equal(K.shortRange("advanced_apply(gko::matrix::Dense<double> * gko::matrix::Csr<double, int> * gko::matrix::Identity<double> + gko::matrix::Dense<double> * gko::matrix::Csr<double, int>)"),
    "advanced_apply(matrix::Dense<…> * matrix::Csr<…> * matrix::Identity<…> + matrix::Dense<…> * matrix::Csr<…>)");
});

test("cub's versioned namespace is dropped", () => {
  assert.equal(K.shortKernel("cub::CUB_200101_750_800_860_NS::DeviceRadixSortOnesweepKernel<cub::A, true>(int)"),
    "cub::DeviceRadixSortOnesweepKernel");
});

test("a case timed only as a whole has no bar, and says why", () => {
  // a hybrid SpMV: two operations launch kernels, and the benchmark times only the whole case
  const ops = [row("apply(gko::matrix::Hybrid<double, int>)", null, { launches: 0 }),
    row("ell::spmv", null), row("coo::spmv2", null)].map((o) => ({ ...o, stage: "" }));
  const bars = K.barModel([run("r1", ops)], "");
  assert.equal(bars[0].total, null);
  assert.equal(bars[0].reason, K.UNSPLIT);
});
