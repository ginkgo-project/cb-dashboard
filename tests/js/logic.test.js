import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as logic from "../../site/logic.js";

test("parseHash defaults to the basic tab", () => {
  assert.deepEqual(logic.parseHash(""), { tab: "basic", params: {} });
  assert.deepEqual(logic.parseHash("#"), { tab: "basic", params: {} });
  assert.deepEqual(logic.parseHash("#nonsense"), { tab: "basic", params: {} });
});

test("parseHash and formatHash round trip", () => {
  const hash = "#solvers?cell=solver%2Fcg%2F5pt%402&machine=daint-gh200";
  const parsed = logic.parseHash(hash);
  assert.equal(parsed.tab, "solvers");
  assert.deepEqual(parsed.params, { cell: "solver/cg/5pt@2", machine: "daint-gh200" });
  assert.equal(logic.formatHash(parsed.tab, parsed.params), hash);
});

test("formatHash sorts keys and drops empty values", () => {
  assert.equal(logic.formatHash("basic", { x: "n", machine: "m", format: "", stage: null }), "#basic?machine=m&x=n");
  assert.equal(logic.formatHash("machines", {}), "#machines");
});

test("parseHash ignores unknown params", () => {
  assert.deepEqual(logic.parseHash("#basic?foo=1&format=csr").params, { format: "csr" });
});

test("formatValue", () => {
  assert.equal(logic.formatValue(8.33e-4, "time"), "0.833 ms");
  assert.equal(logic.formatValue(0.01234, "time"), "12.3 ms");
  assert.equal(logic.formatValue(1.5, "time"), "1.50 s");
  assert.equal(logic.formatValue(1.412e12, "gbs"), "1412 GB/s");
  assert.equal(logic.formatValue(2.65e11, "gflops"), "265 GFLOP/s");
  assert.equal(logic.formatValue(4.56e10, "gflops"), "45.6 GFLOP/s");
  assert.equal(logic.formatValue(0.384, "pct"), "38%");
  assert.equal(logic.formatValue(null, "time"), "–");
  assert.equal(logic.formatValue(undefined, "pct"), "–");
});

test("formatChange", () => {
  assert.equal(logic.formatChange(-0.12), "−12.0%");
  assert.equal(logic.formatChange(0.083), "+8.3%");
  assert.equal(logic.formatChange(null), "");
  assert.equal(logic.formatChange(undefined), "");
});

test("formatCount", () => {
  assert.equal(logic.formatCount(1048576), "1Mi");
  assert.equal(logic.formatCount(16777216), "16Mi");
  assert.equal(logic.formatCount(134217728), "128Mi");
  assert.equal(logic.formatCount(5000), "5000");
});

test("caseLabel", () => {
  assert.equal(logic.caseLabel({ group: "blas", labels: { operation: "dot", n: "16777216" } }), "dot · 16Mi");
  assert.equal(logic.caseLabel({ group: "spmv", labels: { operation: "spmv", format: "csr", matrix: "cage15" } }), "SpMV csr · cage15");
  assert.equal(logic.caseLabel({ group: "spgemm", labels: { operation: "spgemm", matrix: "poisson_p1" } }), "SpGEMM · poisson_p1");
  assert.equal(logic.caseLabel({ group: "solver", labels: { solver: "gmres-cgs", stencil: "7pt" } }), "gmres-cgs · 7pt");
  assert.equal(logic.caseLabel({ group: "preconditioner", labels: { preconditioner: "mg-pgm", matrix: "porous_pressure", stage: "apply" } }), "mg-pgm · porous_pressure (apply)");
});

test("flagText", () => {
  const c = { group: "spmv", labels: { operation: "spmv", format: "csr", matrix: "cage15" } };
  assert.equal(logic.flagText({ state: "slower", change: -0.12, band: 0.03 }, c, "Daint GH200"),
    "−12.0% (noise ±3.0%) · SpMV csr · cage15 · Daint GH200");
  assert.equal(logic.flagText({ state: "new_failure" }, c, "Daint GH200"),
    "new failure · SpMV csr · cage15 · Daint GH200");
});

test("app.js never uses innerHTML", () => {
  const source = readFileSync(new URL("../../site/app.js", import.meta.url), "utf8");
  assert.ok(!source.includes("innerHTML"));
});

test("parseHash keeps a malformed escape as typed", () => {
  assert.deepEqual(logic.parseHash("#basic?machine=%E0"), { tab: "basic", params: { machine: "%E0" } });
});

test("renderTokens: only the latest token is current", () => {
  const tokens = logic.renderTokens();
  const a = tokens.next();
  assert.ok(tokens.isLatest(a));
  const b = tokens.next();
  assert.ok(!tokens.isLatest(a));
  assert.ok(tokens.isLatest(b));
});

const spmv = (matrix, format = "csr", operation = "spmv") =>
  ({ key: `${operation}/${format}/${matrix}`, group: operation === "spmv" ? "spmv" : "blas",
     labels: { operation, format, matrix }, compare: { m1: "bandwidth" } });

test("rowFilter defaults per tab", () => {
  assert.ok(logic.rowFilter("basic", spmv("a"), {}));
  assert.ok(!logic.rowFilter("basic", spmv("a", "coo"), {}));
  assert.ok(logic.rowFilter("basic", spmv("a", "coo"), { format: "coo" }));
  assert.ok(!logic.rowFilter("basic", spmv("a", "csr", "spgemm"), {}));
  assert.ok(logic.rowFilter("basic", spmv("a", "csr", "spgemm"), { operation: "spgemm" }));
  const dot = { key: "dot/1", group: "blas", labels: { operation: "dot", n: "1" } };
  assert.ok(logic.rowFilter("basic", dot, { operation: "dot", format: "ell" }));
  const pre = (stage) => ({ key: stage, group: "preconditioner", labels: { stage } });
  assert.ok(logic.rowFilter("preconditioners", pre("apply"), {}));
  assert.ok(!logic.rowFilter("preconditioners", pre("generate"), {}));
  assert.ok(logic.rowFilter("preconditioners", pre("generate"), { stage: "generate" }));
  assert.ok(logic.rowFilter("solvers", { key: "s", group: "solver", labels: {} }, { operation: "x" }));
});

test("cellClass", () => {
  const slow = (change) => ({ state: "slower", change, band: 0.03 });
  assert.equal(logic.cellClass(null), "cell empty");
  assert.equal(logic.cellClass(slow(-0.05)), "cell slower s1");
  assert.equal(logic.cellClass(slow(-0.09)), "cell slower s2");
  assert.equal(logic.cellClass(slow(-0.2)), "cell slower s3");
  assert.equal(logic.cellClass({ state: "faster", change: 0.05, band: 0.03 }), "cell faster s1");
  assert.equal(logic.cellClass({ state: "faster", change: 0.2, band: 0.03 }), "cell faster s3");
  assert.equal(logic.cellClass({ state: "same" }), "cell same");
  assert.equal(logic.cellClass({ state: "history" }), "cell history");
  assert.equal(logic.cellClass({ state: "failed" }), "cell failed");
  assert.equal(logic.cellClass({ state: "not_available" }), "cell na");
  assert.equal(logic.cellClass({ state: "disabled" }), "cell disabled");
  assert.equal(logic.cellClass({ state: "missing" }), "cell empty");
});

const fixture = {
  tab: "basic",
  cases: [
    { key: "spmv/csr/a", group: "spmv", labels: { operation: "spmv", format: "csr", matrix: "a" }, compare: { m1: "bandwidth" } },
    { key: "spmv/csr/b", group: "spmv", labels: { operation: "spmv", format: "csr", matrix: "b" }, compare: { m1: "time" } },
    { key: "spmv/coo/a", group: "spmv", labels: { operation: "spmv", format: "coo", matrix: "a" }, compare: { m1: "time" } },
  ],
  machines: { m1: {
    columns: [
      { commit: "aaaaaaaaaa", date: "2026-10-01", ref: "r1", runs: ["d1"], setup_changes: [] },
      { commit: "bbbbbbbbbb", date: "2026-10-02", ref: "r2", runs: ["d2", "d3"], setup_changes: ["driver 1 to 2", "uenv x"] },
      { commit: "cccccccccc", date: "2026-10-03", ref: "r3", runs: ["d4"], setup_changes: [] },
    ],
    cells: {
      "spmv/csr/a": [{ state: "history", time: 1e-3, runs: ["d1"] }, { state: "same", time: 1e-3, change: 0, band: 0.03 },
                      { state: "slower", time: 2e-3, change: -0.2, band: 0.03, iterations_prev: 10 }],
      "spmv/csr/b": [null, { state: "failed" }, { state: "disabled" }],
      "spmv/coo/a": [{ state: "history" }, { state: "history" }, { state: "history" }],
    },
  } },
};

test("heatmapModel", () => {
  const model = logic.heatmapModel(fixture, "m1", {});
  assert.deepEqual(model.columns.map((c) => c.label), ["2026-10-01", "2026-10-02 ·2", "2026-10-03"]);
  assert.equal(model.columns[0].setup, null);
  assert.equal(model.columns[1].setup, "driver 1 to 2; uenv x");
  assert.ok(model.columns[1].title.includes("bbbbbbb") && model.columns[1].title.includes("r2") && model.columns[1].title.includes("d3"));
  assert.deepEqual(model.rows.map((r) => r.key), ["spmv/csr/a", "spmv/csr/b"]);
  assert.equal(model.rows[0].label, "SpMV csr · a");
  const cells = model.rows[0].cells;
  assert.deepEqual(cells.map((c) => c.cls), ["cell history", "cell same", "cell slower s3"]);
  assert.deepEqual(cells.map((c) => c.dot), [false, false, true]);
  assert.deepEqual(cells.map((c) => c.col), [0, 1, 2]);
  assert.ok(cells[2].title.includes("SpMV csr · a") && cells[2].title.includes("slower"));
  assert.equal(model.rows[1].cells[0].cls, "cell empty");
  assert.equal(model.legend, "bandwidth-limited rows: % of node bandwidth; others: time");
});

test("heatmapModel legend for uniform rows, unknown machine", () => {
  assert.equal(logic.heatmapModel(fixture, "m1", { format: "coo" }).legend, "time");
  const only = { ...fixture, cases: [fixture.cases[0]] };
  assert.equal(logic.heatmapModel(only, "m1", {}).legend, "% of node bandwidth");
  const none = logic.heatmapModel(fixture, "zz", {});
  assert.deepEqual(none.columns, []);
});

test("cellTitle", () => {
  const c = fixture.cases[0];
  const title = logic.cellTitle({ state: "slower", change: -0.2, band: 0.03, time: 2e-3 }, c);
  assert.ok(title.includes("SpMV csr · a") && title.includes("slower") && title.includes("−20.0%") && title.includes("±3.0%") && title.includes("2.00 ms"));
});
