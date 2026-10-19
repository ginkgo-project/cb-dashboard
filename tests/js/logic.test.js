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

test("logScale maps the endpoints and the log midpoint", () => {
  const s = logic.logScale(1, 100, 0, 200);
  assert.equal(s(10), 100);
  assert.equal(s(1), 0);
  assert.equal(s(100), 200);
});

const scatterCases = [
  { key: "a", group: "spmv", labels: { format: "csr", matrix: "a" }, props: { row_length_cv: 0.1, n: 10 } },
  { key: "b", group: "spmv", labels: { format: "csr", matrix: "b" }, props: { row_length_cv: 1.5 } },
  { key: "c", group: "spmv", labels: { format: "csr", matrix: "c" }, props: {} },
];

test("scatterModel: a previous column adds a faint dot", () => {
  const points = [
    { key: "a", time: 1e-3, pct: 0.5, prev: { time: 1e-3, pct: 0.4 } },
    { key: "b", time: 1e-3, pct: 0.8 },
    { key: "c", time: 1e-3, pct: 0.9 },
  ];
  const model = logic.scatterModel(points, scatterCases, { x: "row_length_cv", y: "pct" });
  assert.equal(model.dots.length, 3);
  assert.equal(model.dots.filter((d) => d.faint).length, 1);
  assert.ok(model.dots.every((d) => Number.isFinite(d.cx) && Number.isFinite(d.cy) && d.title));
  assert.ok(model.xTicks.length > 0 && model.yTicks.length > 0);
  const [a, b] = ["a", "b"].map((k) => model.dots.find((d) => d.key === k && !d.faint));
  assert.ok(b.cx > a.cx); // larger x to the right
  assert.ok(b.cy < a.cy); // larger y up
});

test("scatterModel skips points missing x or y", () => {
  const points = [{ key: "b", time: 1e-3 }, { key: "a", time: 1e-3, pct: 0.5 }];
  const model = logic.scatterModel(points, scatterCases, { x: "n", y: "pct" });
  assert.equal(model.dots.length, 1);
  assert.equal(logic.scatterModel([], scatterCases, {}).dots.length, 0);
});

test("rooflineModel: the knee is at intensity 10", () => {
  const latest = { m1: { triad: 3e12, points: [{ key: "a", time: 1, gflops: 1e12, intensity: 0.5 }, { key: "b", time: 1, gflops: 2e13, intensity: 20 }] } };
  const model = logic.rooflineModel(latest, { m1: { fp64_peak: 3e13 } }, scatterCases);
  assert.equal(model.dots.length, 2);
  const pts = model.ceilings[0].points.split(" ").map((p) => p.split(",").map(Number));
  assert.equal(pts.length, 3);
  assert.equal(model.ceilings[0].knee, 10);
  assert.ok(pts.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y)));
  assert.equal(pts[1][1], pts[2][1]); // flat after the knee
  assert.ok(pts[0][1] > pts[1][1]); // rising before it
  assert.ok(model.dots[0].cx < pts[1][0] && model.dots[1].cx > pts[1][0]);
});

test("test_roofline_without_triad", () => {
  const latest = { m1: { points: [{ key: "a", time: 1, gflops: 1e12, intensity: 0.5 }] } };
  const model = logic.rooflineModel(latest, { m1: { fp64_peak: 3e13 } }, scatterCases);
  const pts = model.ceilings[0].points.split(" ").map((p) => p.split(",").map(Number));
  assert.equal(pts.length, 2);
  assert.equal(pts[0][1], pts[1][1]);
});

test("scatterModel: time axis is logarithmic", () => {
  const points = [{ key: "a", time: 1e-6 }, { key: "b", time: 1e-3 }, { key: "c", time: 0 }];
  const cases = [{ key: "a", props: { n: 10 } }, { key: "b", props: { n: 100 } }, { key: "c", props: { n: 100 } }];
  const model = logic.scatterModel(points, cases, { x: "n", y: "time" });
  assert.equal(model.dots.length, 2); // the non-positive time is dropped
  const [a, b] = model.dots;
  const height = model.plot.y0 - model.plot.y1;
  // three decades span the plot: the dots are the whole height apart
  assert.ok(Math.abs(Math.abs(a.cy - b.cy) - height) < 1e-6);
  assert.ok(model.yTicks.every((t) => Number.isFinite(t.pos)));
});

test("rooflineModel: the ceiling stays inside the plot", () => {
  const latest = { m1: { triad: 1e9, points: [{ key: "a", time: 1, gflops: 1e13, intensity: 1000 }, { key: "b", time: 1, gflops: 2e13, intensity: 5000 }] } };
  const model = logic.rooflineModel(latest, { m1: { fp64_peak: 3e13 } }, scatterCases);
  const { plot } = model;
  for (const p of model.ceilings[0].points.split(" ")) {
    const y = Number(p.split(",")[1]);
    assert.ok(y <= plot.y0 + 1e-6 && y >= plot.y1 - 1e-6, `y ${y} outside`);
  }
});

test("scatterModel: gbs axis is logarithmic", () => {
  const points = [{ key: "a", gbs: 1e9 }, { key: "b", gbs: 1e12 }, { key: "c", gbs: -1 }];
  const cases = [{ key: "a", props: { n: 10 } }, { key: "b", props: { n: 100 } }, { key: "c", props: { n: 100 } }];
  const model = logic.scatterModel(points, cases, { x: "n", y: "gbs" });
  assert.equal(model.dots.length, 2);
  const [a, b] = model.dots;
  assert.ok(Math.abs(Math.abs(a.cy - b.cy) - (model.plot.y0 - model.plot.y1)) < 1e-6);
  assert.ok(model.yTicks.some((t) => t.label === "1 GB/s") && model.yTicks.some((t) => t.label === "1000 GB/s"));
  assert.equal(model.yLabel, "GB/s");
});

const solverData = {
  cases: [
    { key: "solver/cg/5pt", group: "solver", labels: { solver: "cg", stencil: "5pt" } },
    { key: "solver/ir/7pt", group: "solver", labels: { solver: "ir", stencil: "7pt" } },
  ],
  machines: { m1: { columns: [{}, {}], cells: {
    "solver/cg/5pt": [{ state: "history", time: 1 }, { state: "same", time: 1.5e-3, gbs: 2e12, pct: 0.5, iterations: 97, residual: 1e-10 }],
    "solver/ir/7pt": [null, null],
  } } },
};

test("solverRows: ms per iteration from the last column, in case order", () => {
  const rows = logic.solverRows(solverData, "m1");
  assert.equal(rows.length, 2);
  assert.deepEqual(rows[0], { solver: "cg", stencil: "5pt", iterations: 97, msPerIteration: 1.5, gbs: 2e12, pct: 0.5, residual: 1e-10 });
  assert.equal(rows[1].solver, "ir");
  assert.equal(rows[1].msPerIteration, null);
});

const precData = {
  cases: ["apply", "generate"].flatMap((stage) => ["jacobi", "ilu"].map((p) => (
    { key: `preconditioner/${stage}/${p}/a`, group: "preconditioner", labels: { stage, preconditioner: p, matrix: "a" } }))),
  machines: { m1: { columns: [{}], cells: {
    "preconditioner/apply/jacobi/a": [{ state: "history", time: 2e-3, storage: 3e6, levels: 4 }],
    "preconditioner/generate/jacobi/a": [{ state: "history", time: 5e-2 }],
    "preconditioner/apply/ilu/a": [{ state: "disabled", error: "disabled on m1" }],
    "preconditioner/generate/ilu/a": [{ state: "disabled" }],
  } } },
};

test("preconditionerRows: joins apply and generate", () => {
  const rows = logic.preconditionerRows(precData, "m1", "a");
  assert.deepEqual(rows[0], { preconditioner: "jacobi", generateMs: 50, applyMs: 2, storageMb: 3, levels: 4, state: "history" });
});

test("preconditionerRows: a disabled preconditioner has a state and null numbers", () => {
  const rows = logic.preconditionerRows(precData, "m1", "a");
  assert.deepEqual(rows[1], { preconditioner: "ilu", generateMs: null, applyMs: null, storageMb: null, levels: null, state: "disabled" });
  assert.deepEqual(logic.preconditionerRows(precData, "m1", "zzz"), []);
});

test("parameterPaths: paths as the build writes them", () => {
  const params = { coarsest_solver: [{ criteria: [{ max_iters: 5 }] }], storage_optimization: [0, 1], empty: {} };
  const leaves = logic.parameterPaths(params);
  assert.deepEqual(leaves.map((l) => l.path), ["coarsest_solver[0].criteria[0].max_iters", "storage_optimization[0]", "storage_optimization[1]", "empty"]);
  const changed = new Map([["coarsest_solver[0].criteria[0].max_iters", { before: 4, after: 5 }], ["storage_optimization[1]", { before: 0, after: 1 }]]);
  assert.deepEqual(leaves.filter((l) => changed.has(l.path)).map((l) => l.path),
    ["coarsest_solver[0].criteria[0].max_iters", "storage_optimization[1]"]);
  assert.equal(logic.childPath("", "a", false), "a");
  assert.equal(logic.childPath("a", "0", true), "a[0]");
});

test("triadChartModel: classes cycle in order of first appearance of the node", () => {
  const history = [
    { dir: "d1", date: "2026-10-01", node: "nidA", triad: 3.6e12 },
    { dir: "d2", date: "2026-10-08", node: "nidB", triad: 3.7e12 },
    { dir: "d3", date: "2026-10-15", node: "nidA", triad: 3.65e12 },
  ];
  const model = logic.triadChartModel(history, { width: 480, height: 160 });
  assert.deepEqual(model.dots.map((d) => d.cls), ["n0", "n1", "n0"]);
  assert.deepEqual(model.dots.map((d) => d.node), ["nidA", "nidB", "nidA"]);
  assert.ok(model.dots[0].cx < model.dots[1].cx && model.dots[1].cx < model.dots[2].cx);
  assert.ok(model.dots[1].cy < model.dots[0].cy); // larger triad is higher up
  assert.ok(model.yTicks.length >= 2);
  for (const d of model.dots) assert.ok(Number.isFinite(d.cx) && Number.isFinite(d.cy) && d.title.includes(d.node));
});

test("triadChartModel: one run, no runs, and more nodes than classes", () => {
  assert.deepEqual(logic.triadChartModel([]).dots, []);
  const one = logic.triadChartModel([{ dir: "d", date: "x", node: "n", triad: 3e12 }]);
  assert.equal(one.dots.length, 1);
  assert.ok(Number.isFinite(one.dots[0].cx) && Number.isFinite(one.dots[0].cy));
  const many = logic.triadChartModel(Array.from({ length: 7 }, (_, i) => ({ dir: `d${i}`, date: "x", node: `n${i}`, triad: 3e12 + i })));
  assert.equal(many.dots[6].cls, "n0");
});

test("energyRows: the latest run directory, gpuW = gpu_j / seconds", () => {
  const groups = [
    { dir: "r1", group: "spmv", seconds: 100, gpu_j: 1 },
    { dir: "r2", group: "spmv", seconds: 213, gpu_j: 28359, node_j: 150262, cpu_j: 33703 },
    { dir: "r2", group: "blas", seconds: 3.5, gpu_j: 392, node_j: 2312 },
    { dir: "r2", group: "solver", seconds: 70, status: "ok" },
  ];
  const rows = logic.energyRows(groups);
  assert.equal(rows.length, 3);
  assert.deepEqual(rows[0], { group: "spmv", seconds: 213, gpuJ: 28359, gpuW: 28359 / 213, nodeJ: 150262, cpuJ: 33703 });
  assert.deepEqual(rows[1], { group: "blas", seconds: 3.5, gpuJ: 392, gpuW: 392 / 3.5, nodeJ: 2312, cpuJ: null });
  assert.deepEqual(rows[2], { group: "solver", seconds: 70, gpuJ: null, gpuW: null, nodeJ: null, cpuJ: null });
  assert.deepEqual(logic.energyRows([]), []);
});

test("shortVersion: 7 characters of a sha, unknown otherwise", () => {
  const sha = "0123456789abcdef0123456789abcdef01234567";
  assert.deepEqual(logic.shortVersion(sha), { text: "0123456", title: sha });
  assert.deepEqual(logic.shortVersion({ error: "x" }), { text: "unknown", title: "unknown" });
  assert.deepEqual(logic.shortVersion(undefined), { text: "unknown", title: "unknown" });
});

test("flatChildren: nested arrays are flattened, null, undefined and false dropped", () => {
  assert.deepEqual(logic.flatChildren(["a", [null, ["b", undefined]], false, [[0, ""]]]), ["a", "b", 0, ""]);
  assert.deepEqual(logic.flatChildren([]), []);
});

test("app.js appends children only through put()", () => {
  const source = readFileSync(new URL("../../site/app.js", import.meta.url), "utf8");
  // the one native append and replaceChildren are those of put() and replace()
  assert.equal(source.match(/\.append\(/g)?.length, 1);
  assert.equal(source.match(/\.replaceChildren\(/g)?.length, 1);
});

test("cellLink: the case's view is in the link, so its row is shown", () => {
  const coo = spmv("a", "coo");
  assert.equal(logic.cellLink("basic", coo, "m1", 2), "#basic?cell=spmv%2Fcoo%2Fa%402&format=coo&machine=m1&operation=spmv");
  const dot = { key: "blas/dot/1", group: "blas", labels: { operation: "dot", n: "1" } };
  const spgemm = { key: "spgemm/x", group: "spgemm", labels: { operation: "spgemm", matrix: "x" } };
  const gen = { key: "preconditioner/generate/jacobi/a", group: "preconditioner", labels: { stage: "generate", preconditioner: "jacobi", matrix: "a" } };
  const cg = { key: "solver/cg/5pt", group: "solver", labels: { solver: "cg", stencil: "5pt" } };
  assert.equal(logic.cellLink("solvers", cg, "m1", 0), "#solvers?cell=solver%2Fcg%2F5pt%400&machine=m1");
  for (const [tab, c] of [["basic", coo], ["basic", dot], ["basic", spgemm], ["preconditioners", gen], ["solvers", cg]]) {
    const { params } = logic.parseHash(logic.cellLink(tab, c, "m1", 3));
    assert.ok(logic.rowFilter(tab, c, params), c.key);
    assert.equal(params.cell, `${c.key}@3`);
    assert.equal(params.format, c.labels.operation === "spmv" ? c.labels.format : undefined);
  }
});

test("revealCell: only a newly selected cell is scrolled to", () => {
  const route = (tab, params) => ({ tab, params });
  const sel = route("basic", { machine: "m", cell: "k@1" });
  assert.equal(logic.revealCell(null, sel), true);
  assert.equal(logic.revealCell(route("basic", {}), sel), true);
  assert.equal(logic.revealCell(route("basic", { machine: "m", cell: "k@0" }), sel), true);
  assert.equal(logic.revealCell(route("solvers", { machine: "m", cell: "k@1" }), sel), true);
  assert.equal(logic.revealCell(route("basic", { machine: "n", cell: "k@1" }), sel), true);
  assert.equal(logic.revealCell(sel, route("basic", { machine: "m", cell: "k@1", x: "n" })), false);
  assert.equal(logic.revealCell(sel, route("basic", { machine: "m" })), false);
});

test("rawNote: a comparison in raw time is marked, history is not", () => {
  const c = fixture.cases[0];
  const noStream = { state: "slower", change: -0.2, band: 0.03, raw: true, time: 2e-3 };
  assert.equal(logic.rawNote(noStream), "compared in raw time: no node bandwidth for this run");
  assert.equal(logic.rawNote({ ...noStream, pct: 0.4 }), "compared in raw time: too few earlier runs with a node bandwidth");
  assert.equal(logic.rawNote({ ...noStream, raw: false }), null);
  assert.equal(logic.rawNote({ state: "history", raw: true }), null); // not compared at all
  assert.ok(logic.cellTitle(noStream, c).includes("compared in raw time: no node bandwidth for this run"));
  assert.equal(logic.cellClass(noStream), "cell slower s3 raw");
  assert.equal(logic.cellClass({ state: "same", raw: true }), "cell same raw");
  assert.equal(logic.cellClass({ state: "history", raw: true }), "cell history");
});

test("flagText: a case now not available or missing", () => {
  const c = { group: "spmv", labels: { operation: "spmv", format: "csr", matrix: "cage15" } };
  assert.equal(logic.flagText({ state: "new_not_available" }, c, "Daint GH200"), "now not available · SpMV csr · cage15 · Daint GH200");
  assert.equal(logic.flagText({ state: "new_missing" }, c, "Daint GH200"), "now missing · SpMV csr · cage15 · Daint GH200");
});

test("baselineText: the median of the previous N columns, and the previous column", () => {
  const cols = fixture.machines.m1.columns;
  assert.equal(logic.baselineText({ state: "slower", change: -0.2, band: 0.03, baseline_n: 5 }, cols, 2),
    "Change against the median of the previous 5 columns; previous column: bbbbbbb (2026-10-02).");
  assert.equal(logic.baselineText({ state: "history", baseline_n: 1 }, cols, 1),
    "Not enough history yet: 1 earlier column with a value, 4 needed; previous column: aaaaaaa (2026-10-01).");
  assert.equal(logic.baselineText({ state: "failed" }, cols, 1), "Previous column: aaaaaaa (2026-10-01).");
  assert.equal(logic.baselineText({ state: "history", baseline_n: 0 }, cols, 0), "First column, nothing to compare with.");
});
