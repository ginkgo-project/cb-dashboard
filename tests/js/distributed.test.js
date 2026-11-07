import { test } from "node:test";
import assert from "node:assert/strict";
import * as D from "../../site/distributed-logic.js";
import { parseHash, formatHash } from "../../site/logic.js";

const N = D.GPU_COUNTS.map(String);

// a machine's data as dashboard/distributed.py writes it
function machine(timesByRow, { bands = {}, statuses = {}, runs = 2 } = {}) {
  const rows = {};
  for (const [key, perN] of Object.entries(timesByRow)) {
    rows[key] = {};
    for (const n of N) {
      const time = perN[n] ?? Array(runs).fill(null);
      rows[key][n] = {
        time,
        band: bands[key]?.[n] ?? Array(runs).fill(0.03),
        status: statuses[key]?.[n] ?? time.map((t) => (t === null ? "missing" : "ok")),
      };
    }
  }
  return { runs: Array.from({ length: runs }, (_, i) => ({ dir: `run${i}`, commit: `c${i}`, date: `2026-11-${14 + 7 * i}` })), rows };
}

const all = (pair) => Object.fromEntries(N.map((n) => [n, pair]));

test("classify: within the floor is the same, beyond it slower or faster", () => {
  assert.equal(D.classify(1, 1.02, 0.03).state, "same");
  const slow = D.classify(1, 1.2, 0.03);
  assert.equal(slow.state, "slower");
  assert.ok(Math.abs(slow.change - 0.2) < 1e-12); // the change in time
  assert.equal(D.classify(1, 0.8, 0.03).state, "faster");
});

test("classify: a wide noise band keeps a change the same", () => {
  assert.equal(D.classify(1, 1.2, 0.25).state, "same");
});

test("classify: strength steps by change over band", () => {
  assert.match(D.classify(1, 1.05, 0.03).cls, /\bs1\b/);
  assert.match(D.classify(1, 1.1, 0.03).cls, /\bs2\b/);
  assert.match(D.classify(1, 1.5, 0.03).cls, /\bs3\b/);
});

test("classify: failed, missing and the first run", () => {
  assert.equal(D.classify(1, null, 0.03, "ok", "failed").state, "failed");
  assert.equal(D.classify(1, null, 0.03, "ok", "missing").state, "missing");
  assert.equal(D.classify(null, 1, 0.03, "missing", "ok").state, "first");
  assert.equal(D.classify(null, 1, 0.03, null, "ok").state, "first");
});

test("rowLabel names benchmarks and MPI rows", () => {
  assert.equal(D.rowLabel("spmv"), "SpMV");
  assert.equal(D.rowLabel("cg/apply"), "CG apply (per iteration)");
  assert.equal(D.rowLabel("mg-pmis/generate"), "PMIS generate");
  assert.equal(D.rowLabel("mpi/allreduce/131072"), "MPI_Allreduce, 128 KiB");
  assert.equal(D.rowLabel("mpi/pingpong_inter/8"), "Ping-pong across nodes, 8 B");
});

test("formatBytes", () => {
  assert.equal(D.formatBytes(8), "8 B");
  assert.equal(D.formatBytes(131072), "128 KiB");
  assert.equal(D.formatBytes(2 ** 31), "2 GiB");
  assert.equal(D.formatBytes(33554432), "32 MiB");
});

test("changeTable: order, seven cells, and the largest swept MPI size", () => {
  const rows = {};
  for (const key of D.BENCH_ROWS) rows[key] = all([1, 1.2]);
  for (const k of [0, 7, 14]) rows[`mpi/allreduce/${8 * 4 ** k}`] = all([1, 1]);
  for (const k of [0, 7, 11]) rows[`mpi/alltoallv/${8 * 4 ** k}`] = all([1, 1]);
  rows["mpi/alltoallv/524288"] = all([1, 1]); // not one of the three
  const table = D.changeTable(machine(rows), 0, 1);
  const keys = table.map((r) => r.key);
  assert.deepEqual(keys.slice(0, D.BENCH_ROWS.length), D.BENCH_ROWS);
  assert.deepEqual(keys.slice(D.BENCH_ROWS.length), [
    "mpi/allreduce/8", "mpi/allreduce/131072", `mpi/allreduce/${2 ** 31}`,
    "mpi/alltoallv/8", "mpi/alltoallv/131072", `mpi/alltoallv/${8 * 4 ** 11}`]);
  assert.ok(table.every((r) => r.cells.length === 7));
  assert.equal(table[0].cells[0].state, "slower");
});

test("changeTable with one run: every measured cell is a first run", () => {
  const table = D.changeTable(machine({ spmv: all([1]) }, { runs: 1 }), null, 0);
  assert.ok(table[0].cells.every((c) => c.state === "first"));
  assert.ok(table[0].cells.every((c) => !/undefined|null|NaN/.test(c.text + c.title)));
});

test("efficiency: one at one GPU, corrected per row", () => {
  assert.deepEqual(D.efficiency([1, 2, null], [100, 100, 100]), [1, 0.5, null]);
  // twice the rows in twice the time is perfect
  assert.deepEqual(D.efficiency([1, 2], [100, 200]), [1, 1]);
  assert.deepEqual(D.efficiency([null, 1], [100, 100]), [null, null]);
});

test("mpiMap: sizes ascending, blank cells where a size wasn't swept", () => {
  const rows = { "mpi/alltoallv/32": all([1, 1.5]), "mpi/alltoallv/8": all([1, 1]) };
  rows["mpi/alltoallv/32"]["64"] = [null, null]; // not swept at 64 GPUs
  const map = D.mpiMap(machine(rows), "alltoallv", 0, 1);
  assert.deepEqual(map.sizes, [8, 32]);
  assert.equal(map.rows[1].cells[0].state, "slower");
  assert.equal(map.rows[1].cells[6].state, "missing");
});

test("defaultPair", () => {
  assert.deepEqual(D.defaultPair([]), [null, null]);
  assert.deepEqual(D.defaultPair([{}]), [null, 0]);
  assert.deepEqual(D.defaultPair([{}, {}, {}]), [1, 2]);
});

test("pickRun: a dir in the hash, else the default", () => {
  const runs = [{ dir: "x" }, { dir: "y" }, { dir: "z" }];
  assert.equal(D.pickRun(runs, "x", 2), 0);
  assert.equal(D.pickRun(runs, "nope", 2), 2);
  assert.equal(D.pickRun(runs, undefined, null), null);
});

test("logPlot: ticks inside the range and one path per series", () => {
  const m = D.logPlot([[1e-3, 2e-3, null], [1e-3, 3e-3, 4e-3]], { width: 300, height: 200 });
  assert.equal(m.paths.length, 2);
  assert.ok(m.paths[0].points.length === 2); // the null is left out
  assert.ok(m.yTicks.every((t) => t.pos >= m.plot.y1 && t.pos <= m.plot.y0));
  assert.equal(m.xTicks.length, 3);
});

test("the distributed tab's hash round trips", () => {
  const params = { machine: "daint-gh200", a: "2026-11-14_x", b: "2026-11-21_y", row: "cg/apply", n: "64" };
  assert.deepEqual(parseHash(formatHash("distributed", params)), { tab: "distributed", params });
});

test("formatRate stays short", () => {
  assert.equal(D.formatRate(1.5e12), "1.5 TB/s");
  assert.equal(D.formatRate(1e11), "100 GB/s");
  assert.equal(D.formatRate(8e6), "8.0 MB/s");
});

test("efficiencyAxis: 110% by default, above a higher value", () => {
  assert.deepEqual(D.efficiencyAxis([1, 0.5, null]), { min: 0, max: 1.25, ticks: [0, 0.25, 0.5, 0.75, 1, 1.25] });
  assert.equal(D.efficiencyAxis([1, 1.4]).max, 1.5);
  assert.ok(D.efficiencyAxis([1, 3]).max >= 3.1);
});
