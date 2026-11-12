// Pure functions of the Distributed tab (site/distributed.js): its change
// table, the cells' classification, efficiency and the plots' geometry. No DOM
// access, so node can test them. The data is site/data/distributed.json
// (dashboard/distributed.py).

import { formatValue } from "./logic.js";

export const GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64];
export const GPUS_PER_NODE = 4;
const SOLVERS = ["cg", "gmres"];
const PRECONDITIONERS = ["mg-pgm", "mg-uniform", "mg-pmis"];
const PHASES = ["generate", "apply"];
export const BENCH_ROWS = ["spmv", "spgemm",
  ...SOLVERS.flatMap((s) => PHASES.map((p) => `${s}/${p}`)),
  ...PRECONDITIONERS.flatMap((p) => PHASES.map((ph) => `${p}/${ph}`))];
export const MPI_OPS = ["allreduce", "alltoallv", "neighbor_alltoallv", "pingpong_intra", "pingpong_inter"];
// the sizes of an operation's rows in the change table: small, medium, its largest
const TABLE_SIZES = [8, 131072];

const NAMES = { spmv: "SpMV", spgemm: "SpGEMM", cg: "CG", gmres: "GMRES", "mg-pgm": "PGM",
  "mg-uniform": "Uniform", "mg-pmis": "PMIS" };
const MPI_NAMES = { allreduce: "MPI_Allreduce", alltoallv: "MPI_Alltoallv",
  neighbor_alltoallv: "MPI_Neighbor_alltoallv", pingpong_intra: "Ping-pong within a node",
  pingpong_inter: "Ping-pong across nodes" };
const MINUS = "−";

export function formatBytes(bytes) {
  for (const [unit, size] of [["GiB", 2 ** 30], ["MiB", 2 ** 20], ["KiB", 2 ** 10]]) {
    if (bytes >= size) {
      const v = bytes / size;
      return `${Number.isInteger(v) ? v : v.toFixed(1)} ${unit}`;
    }
  }
  return `${bytes} B`;
}

// short enough for an axis label: at most three digits
export function formatRate(bytesPerSecond) {
  if (bytesPerSecond === null || bytesPerSecond === undefined) return "–";
  for (const [unit, size] of [["TB/s", 1e12], ["GB/s", 1e9], ["MB/s", 1e6], ["kB/s", 1e3]]) {
    if (bytesPerSecond >= size) {
      const v = bytesPerSecond / size;
      return `${v >= 10 ? Math.round(v) : v.toFixed(1)} ${unit}`;
    }
  }
  return `${Math.round(bytesPerSecond)} B/s`;
}

// the efficiency axis: up to 110%, or above the highest value
export function efficiencyAxis(values) {
  const top = Math.max(1, ...values.filter((v) => v !== null && v !== undefined));
  const max = Math.ceil((top + 0.1) * 4) / 4;
  const ticks = [];
  for (let t = 0; t <= max + 1e-9; t += max > 2 ? 0.5 : 0.25) ticks.push(Number(t.toFixed(2)));
  return { min: 0, max, ticks };
}

export const mpiParts = (key) => {
  const [, op, bytes] = key.split("/");
  return { op, bytes: Number(bytes) };
};

export function rowLabel(key) {
  if (key.startsWith("mpi/")) {
    const { op, bytes } = mpiParts(key);
    return `${MPI_NAMES[op] ?? op}, ${formatBytes(bytes)}`;
  }
  const [name, phase] = key.split("/");
  const label = NAMES[name] ?? name;
  if (!phase) return label;
  return SOLVERS.includes(name) && phase === "apply" ? `${label} apply (per iteration)` : `${label} ${phase}`;
}

const pct = (x) => `${x < 0 ? MINUS : x > 0 ? "+" : ""}${Math.round(Math.abs(x) * 100)}%`;

const NOT_MEASURED = {
  failed: ["cell failed", "failed"], killed: ["cell failed", "killed (SIGTERM)"],
  skipped: ["cell na", "skipped"], missing: ["cell empty", "not measured"],
};

// The cell of one row and GPU count, from run a's time to run b's. `band` is
// run b's noise band, on performance (1 / time) as on the other tabs; the
// change shown is the change in time, so + is slower.
export function classify(a, b, band, statusA = "ok", statusB = "ok") {
  if (statusB !== "ok" || b === null || b === undefined) {
    const state = NOT_MEASURED[statusB] ? statusB : "missing";
    const [cls, title] = NOT_MEASURED[state];
    return { state, cls, text: "", title, change: null };
  }
  if (a === null || a === undefined || statusA !== "ok") {
    return { state: "first", cls: "cell history", text: "",
      title: `${formatValue(b, "time")}; no earlier run to compare with`, change: null };
  }
  const perf = a / b - 1;
  const change = b / a - 1;
  const state = perf < -band ? "slower" : perf > band ? "faster" : "same";
  const r = Math.abs(perf) / band;
  const strength = state === "same" ? "" : ` ${r < 2 ? "s1" : r < 4 ? "s2" : "s3"}`;
  return { state, cls: `cell ${state}${strength}`, text: pct(change), change,
    title: `${formatValue(a, "time")} → ${formatValue(b, "time")}: ${pct(change)} (noise ±${(band * 100).toFixed(1)}%)` };
}

function cellsOf(row, ia, ib) {
  return GPU_COUNTS.map(String).map((n) => {
    const r = row[n];
    if (!r || ib === null || ib === undefined) return classify(null, null, 0, null, "missing");
    const a = ia === null || ia === undefined ? null : r.time[ia];
    return classify(a, r.time[ib], r.band[ib], ia === null || ia === undefined ? null : r.status[ia], r.status[ib]);
  });
}

// the swept sizes of an MPI operation, ascending
export function mpiSizes(rows, op) {
  return Object.keys(rows).filter((k) => k.startsWith(`mpi/${op}/`))
    .map((k) => mpiParts(k).bytes).sort((x, y) => x - y);
}

// The change table: the benchmark rows, then for each MPI operation the rows
// of 8 B, 128 KiB and its largest swept size.
export function changeTable(m, ia, ib) {
  const out = [];
  for (const key of BENCH_ROWS) {
    if (m.rows[key]) out.push({ key, label: rowLabel(key), mpi: false, cells: cellsOf(m.rows[key], ia, ib) });
  }
  for (const op of MPI_OPS) {
    const sizes = mpiSizes(m.rows, op);
    if (!sizes.length) continue;
    const pick = [...new Set([...TABLE_SIZES.filter((s) => sizes.includes(s)), sizes[sizes.length - 1]])];
    for (const bytes of pick) {
      const key = `mpi/${op}/${bytes}`;
      out.push({ key, label: rowLabel(key), mpi: true, op, cells: cellsOf(m.rows[key], ia, ib) });
    }
  }
  return out;
}

// The change map of an MPI operation: its sizes by GPU count.
export function mpiMap(m, op, ia, ib) {
  const sizes = mpiSizes(m.rows, op);
  return { sizes, rows: sizes.map((bytes) => ({ bytes, cells: cellsOf(m.rows[`mpi/${op}/${bytes}`], ia, ib) })) };
}

// Weak-scaling efficiency per GPU count: the time per row at 1 GPU over the
// time per row at N, so that slightly different rows per GPU (the generator
// rounds the global problem to a cube) don't count. Without rows, plain t1 / tN.
export function efficiency(times, rows = []) {
  const t1 = times[0];
  if (!t1) return times.map(() => null);
  return times.map((t, i) => {
    if (!t) return null;
    const r1 = rows[0];
    const r = rows[i];
    return r1 && r ? (t1 / r1) / (t / r) : t1 / t;
  });
}

export function defaultPair(runs) {
  if (!runs.length) return [null, null];
  if (runs.length === 1) return [null, 0];
  return [runs.length - 2, runs.length - 1];
}

// one line per compared run (a is null without an earlier run): the nodes of
// its job, to tell a change of the network apart from a change of placement
export function nodeLines(runs, a, b) {
  return [a, b].filter((i) => i !== null && i !== undefined).map((i) => {
    const r = runs[i];
    return `Nodes of ${r.date} ${(r.commit || "").slice(0, 7)}: ${r.nodes || "not recorded"}`;
  });
}

// the index of the run named in the hash, else `fallback`
export function pickRun(runs, dir, fallback) {
  const i = runs.findIndex((r) => r.dir === dir);
  return i >= 0 ? i : fallback;
}

const PAD = { left: 70, right: 26, top: 24, bottom: 36 };

// The geometry of a line plot over evenly spaced x positions (GPU counts or
// message sizes, both powers): `series` are arrays of y values (null for
// none), on a log y axis, or a linear one from `linear.min` to `linear.max`.
export function logPlot(series, { width, height, xLabels, labelEvery = 1, unit = "time",
  linear = null, format = null } = {}) {
  const count = Math.max(...series.map((s) => s.length));
  const plot = { x0: PAD.left, x1: width - PAD.right, y0: height - PAD.bottom, y1: PAD.top };
  const xs = (i) => plot.x0 + ((plot.x1 - plot.x0) * i) / Math.max(1, count - 1);
  const values = series.flat().filter((v) => v !== null && v !== undefined && v > 0);
  const label = format ?? ((v) => formatValue(v, unit));
  let yOf;
  let yTicks = [];
  if (linear) {
    yOf = (v) => plot.y0 - ((plot.y0 - plot.y1) * (v - linear.min)) / (linear.max - linear.min);
    yTicks = (linear.ticks ?? []).map((v) => ({ pos: yOf(v), label: label(v) }));
  } else {
    let lo = values.length ? Math.log10(Math.min(...values)) : -3;
    let hi = values.length ? Math.log10(Math.max(...values)) : 0;
    if (hi - lo < 0.3) { // a flat curve: a third of a decade around it
      const mid = (hi + lo) / 2;
      [lo, hi] = [mid - 0.15, mid + 0.15];
    }
    lo -= 0.05 * (hi - lo);
    hi += 0.05 * (hi - lo);
    yOf = (v) => plot.y0 - ((plot.y0 - plot.y1) * (Math.log10(v) - lo)) / (hi - lo);
    const span = hi - lo;
    // a few labelled ticks in any range
    const steps = span > 4 ? [1] : span > 1.5 ? [1, 2, 5] : span > 0.5 ? [1, 1.5, 2, 3, 5, 7]
      : [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9];
    for (let k = Math.floor(lo); k <= Math.ceil(hi); k++) {
      for (const s of steps) {
        const v = s * 10 ** k;
        const e = Math.log10(v);
        if (e >= lo && e <= hi) yTicks.push({ pos: yOf(v), label: label(v) });
      }
    }
  }
  const labels = xLabels ?? Array.from({ length: count }, (_, i) => String(i));
  const xTicks = labels.map((text, i) => ({ pos: xs(i), label: i % labelEvery === 0 || i === count - 1 ? text : "" }));
  const paths = series.map((s, index) => ({
    index,
    points: s.map((v, i) => (v === null || v === undefined || (!linear && v <= 0) ? null : [xs(i), yOf(v), v, i]))
      .filter(Boolean),
  }));
  return { plot, xTicks, yTicks, paths, size: { width, height }, x: xs };
}
