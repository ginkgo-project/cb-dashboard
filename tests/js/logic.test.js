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
