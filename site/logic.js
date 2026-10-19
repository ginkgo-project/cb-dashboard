// Pure functions of the dashboard page: hash handling, number and label
// formatting. No DOM access, so node can test them.

export const TABS = ["basic", "solvers", "preconditioners", "machines"];
export const PARAMS = ["machine", "operation", "format", "stage", "matrix", "values", "x", "cell"];

// a malformed escape such as %E0 stays as typed instead of throwing
function safeDecode(text) {
  try { return decodeURIComponent(text); } catch { return text; }
}

// "#solvers?machine=m&cell=k%402" -> {tab: "solvers", params: {machine: "m", cell: "k@2"}}
export function parseHash(hash) {
  const text = (hash || "").replace(/^#/, "");
  const [name, query = ""] = text.split("?", 2);
  const tab = TABS.includes(name) ? name : "basic";
  const params = {};
  for (const part of query.split("&")) {
    if (!part) continue;
    const at = part.indexOf("=");
    const key = safeDecode(at < 0 ? part : part.slice(0, at));
    const value = at < 0 ? "" : safeDecode(part.slice(at + 1).replace(/\+/g, " "));
    if (PARAMS.includes(key) && value !== "") params[key] = value;
  }
  return { tab, params };
}

export function formatHash(tab, params = {}) {
  const query = Object.keys(params).sort()
    .filter((key) => params[key] !== undefined && params[key] !== null && params[key] !== "")
    .map((key) => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`);
  return `#${tab}${query.length ? `?${query.join("&")}` : ""}`;
}

const DASH = "–";
const MINUS = "−";
const DOT = " · ";

function sig3(x) {
  const digits = x >= 100 ? 0 : x >= 10 ? 1 : x >= 1 ? 2 : 3;
  return x.toFixed(digits);
}

// value in base units (s, bytes/s, flops/s, fraction)
export function formatValue(value, kind) {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  switch (kind) {
    case "time":
      if (value >= 1) return `${sig3(value)} s`;
      if (value >= 1e-4) return `${sig3(value * 1e3)} ms`;
      if (value >= 1e-7) return `${sig3(value * 1e6)} µs`;
      return `${sig3(value * 1e9)} ns`;
    case "gbs":
      return `${Math.round(value / 1e9)} GB/s`;
    case "gflops": {
      const g = value / 1e9;
      return `${g < 100 ? g.toFixed(1) : Math.round(g)} GFLOP/s`;
    }
    case "pct":
      return `${Math.round(value * 100)}%`;
    default:
      return String(value);
  }
}

export function formatChange(change) {
  if (change === null || change === undefined) return "";
  const text = (Math.abs(change) * 100).toFixed(1);
  return `${change < 0 ? MINUS : "+"}${text}%`;
}

export function formatCount(n) {
  n = Number(n);
  for (const [unit, size] of [["Gi", 2 ** 30], ["Mi", 2 ** 20], ["Ki", 2 ** 10]]) {
    if (n >= size && n % size === 0) return `${n / size}${unit}`;
  }
  return String(n);
}

export function caseLabel(c) {
  const l = c.labels || {};
  switch (c.group) {
    case "blas": return `${l.operation}${DOT}${formatCount(l.n)}`;
    case "spmv": return `SpMV ${l.format}${DOT}${l.matrix}`;
    case "spgemm": return `SpGEMM${DOT}${l.matrix}`;
    case "solver": return `${l.solver}${DOT}${l.stencil}`;
    case "preconditioner": return `${l.preconditioner}${DOT}${l.matrix} (${l.stage})`;
    default: return c.key || "";
  }
}

export function flagText(flag, c, machineLabel) {
  const what = flag.state === "new_failure"
    ? "new failure"
    : `${formatChange(flag.change)}${flag.band === undefined || flag.band === null ? "" : ` (noise ±${(flag.band * 100).toFixed(1)}%)`}`;
  return [what, caseLabel(c), machineLabel].join(DOT);
}

// One render at a time: each render takes a token, and after any await it
// may touch the page only while its token is still the latest.
export function renderTokens() {
  let latest = 0;
  return { next: () => ++latest, isLatest: (token) => token === latest };
}

// Which cases the heatmap shows. Basic: one operation (and, for spmv, one
// format) at a time; preconditioners: one stage; solvers: all.
export function rowFilter(tab, c, params = {}) {
  const l = c.labels || {};
  if (tab === "basic") {
    const operation = params.operation || "spmv";
    if (l.operation !== operation) return false;
    return operation !== "spmv" || l.format === (params.format || "csr");
  }
  if (tab === "preconditioners") return l.stage === (params.stage || "apply");
  return true;
}

// r = |change| / band: s1 below 2, s2 below 4, s3 above
export function cellClass(cell) {
  if (!cell) return "cell empty";
  switch (cell.state) {
    case "same": case "history": case "failed": case "disabled": return `cell ${cell.state}`;
    case "not_available": return "cell na";
    case "slower": case "faster": {
      const r = Math.abs(cell.change ?? 0) / (cell.band || 0.03);
      return `cell ${cell.state} ${r < 2 ? "s1" : r < 4 ? "s2" : "s3"}`;
    }
    default: return "cell empty";
  }
}

export function cellTitle(cell, c) {
  if (!cell) return `${caseLabel(c)}${DOT}no data`;
  const parts = [caseLabel(c), cell.state.replaceAll("_", " ")];
  if (cell.change !== undefined && cell.change !== null) {
    parts.push(`${formatChange(cell.change)}${cell.band != null ? ` (band ±${(cell.band * 100).toFixed(1)}%)` : ""}`);
  }
  if (cell.time !== undefined && cell.time !== null) parts.push(formatValue(cell.time, "time"));
  if (cell.pct !== undefined && cell.pct !== null) parts.push(`${formatValue(cell.pct, "pct")} of node bandwidth`);
  return parts.join(DOT);
}

export function heatmapModel(tabData, machine, params = {}) {
  const data = tabData.machines?.[machine];
  const cases = tabData.cases.filter((c) => rowFilter(tabData.tab, c, params));
  const columns = (data?.columns ?? []).map((col) => ({
    label: col.runs.length > 1 ? `${col.date} ·${col.runs.length}` : col.date,
    title: `${(col.commit || "").slice(0, 7)} ${col.ref || ""} ${col.runs.join(", ")}`.replace(/\s+/g, " "),
    setup: col.setup_changes?.length ? col.setup_changes.join("; ") : null,
  }));
  const rows = cases.map((c) => ({
    key: c.key,
    label: caseLabel(c),
    cells: (data?.cells?.[c.key] ?? []).map((cell, col) => ({
      cls: cellClass(cell), title: cellTitle(cell, c), col, dot: cell?.iterations_prev != null,
    })),
  }));
  const kinds = new Set(cases.map((c) => c.compare?.[machine] === "bandwidth"));
  const legend = kinds.size > 1 ? "bandwidth-limited rows: % of node bandwidth; others: time"
    : kinds.has(true) ? "% of node bandwidth" : "time";
  return { columns, rows, legend };
}

// ---- charts: pure models, coordinates in the svg's own units ----

// log10 mapping: d0 -> r0, d1 -> r1
export function logScale(d0, d1, r0, r1) {
  const a = Math.log10(d0);
  const b = Math.log10(d1);
  return (v) => (b === a ? (r0 + r1) / 2 : r0 + ((Math.log10(v) - a) / (b - a)) * (r1 - r0));
}

const MARGIN = { left: 60, right: 14, top: 12, bottom: 38 };
const SUPERSCRIPT = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const tenPower = (e) => `10${String(e).replace(/./g, (ch) => SUPERSCRIPT[ch])}`;

// whole decades covering [lo, hi]; at least one decade wide
function decadeDomain(lo, hi) {
  let a = Math.floor(Math.log10(lo) + 1e-9);
  let b = Math.ceil(Math.log10(hi) - 1e-9);
  if (b <= a) b = a + 1;
  return [a, b];
}

function decadeTicks(e0, e1, scale) {
  const ticks = [];
  for (let e = e0; e <= e1; e++) ticks.push({ value: 10 ** e, pos: scale(10 ** e), label: tenPower(e) });
  return ticks;
}

function linearTicks(lo, hi, scale, format) {
  const step0 = (hi - lo) / 5;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const ticks = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) {
    const value = Math.round(v / step) * step;
    ticks.push({ value, pos: scale(value), label: format(value) });
  }
  return ticks;
}

const Y_KINDS = {
  pct: { name: "% of node bandwidth", format: (v) => `${Math.round(v * 100)}%`, kind: "pct" },
  time: { name: "time", format: (v) => formatValue(v, "time"), kind: "time" },
  gbs: { name: "GB/s", format: (v) => formatValue(v, "gbs"), kind: "gbs" },
  gflops: { name: "GFLOP/s", format: (v) => formatValue(v, "gflops"), kind: "gflops" },
};

// The shown cases' props that can be plotted, and the default x.
export function scatterProps(cases) {
  const keys = new Set();
  for (const c of cases) for (const [k, v] of Object.entries(c.props ?? {})) if (Number.isFinite(v)) keys.add(k);
  const list = [...keys].sort();
  return { keys: list, default: ["row_length_cv", "nnz_per_row", "n"].find((k) => keys.has(k)) ?? list[0] };
}

// points: latest[machine].points. x is a case property (log axis), y is
// "pct" or "time" of the point; each prev adds a faint dot.
export function scatterModel(points, cases, { x = "row_length_cv", y = "pct", width = 640, height = 320 } = {}) {
  const byKey = new Map(cases.map((c) => [c.key, c]));
  const yk = Y_KINDS[y] ?? Y_KINDS.pct;
  const raw = [];
  for (const p of points ?? []) {
    const c = byKey.get(p.key);
    const xv = c?.props?.[x];
    if (!(xv > 0) || !Number.isFinite(xv)) continue; // log axis
    const label = c ? caseLabel(c) : p.key;
    const ok = (v) => Number.isFinite(v) && (y === "pct" || v > 0); // the other axes are logarithmic
    if (ok(p[y])) raw.push({ key: p.key, xv, yv: p[y], faint: false, title: `${label}${DOT}${x} ${xv}${DOT}${yk.format(p[y])}` });
    if (p.prev && ok(p.prev[y])) raw.push({ key: p.key, xv, yv: p.prev[y], faint: true, title: `${label}${DOT}previous column${DOT}${yk.format(p.prev[y])}` });
  }
  const plot = { x0: MARGIN.left, x1: width - MARGIN.right, y0: height - MARGIN.bottom, y1: MARGIN.top };
  if (!raw.length) return { dots: [], xTicks: [], yTicks: [], plot, xLabel: x, yLabel: yk.name };
  const [e0, e1] = decadeDomain(Math.min(...raw.map((d) => d.xv)), Math.max(...raw.map((d) => d.xv)));
  const sx = logScale(10 ** e0, 10 ** e1, plot.x0, plot.x1);
  const yHi = Math.max(...raw.map((d) => d.yv));
  let sy;
  let yTicks;
  if (y !== "pct") { // time, GB/s and GFLOP/s span decades: log axis
    const [f0, f1] = decadeDomain(Math.min(...raw.map((d) => d.yv)), yHi);
    sy = logScale(10 ** f0, 10 ** f1, plot.y0, plot.y1);
    yTicks = decadeTicks(f0, f1, sy).map((t) => ({ ...t, label: yk.format(t.value) }));
  } else { // pct: linear from 0
    const yMax = Math.max(1, yHi);
    sy = (v) => plot.y0 + (v / yMax) * (plot.y1 - plot.y0);
    yTicks = linearTicks(0, yMax, sy, yk.format);
  }
  return {
    dots: raw.map((d) => ({ cx: sx(d.xv), cy: sy(d.yv), key: d.key, title: d.title, faint: d.faint })),
    xTicks: decadeTicks(e0, e1, sx),
    yTicks,
    plot, xLabel: x, yLabel: yk.name,
  };
}

// machines: {id: {label, fp64_peak}}; latest: tabData.latest. x is the
// intensity (flops/byte), y GFLOP/s, both log. The ceiling of a machine is
// min(triad * I, peak); without a triad only the flat peak line is drawn.
export function rooflineModel(latest, machines, cases, { width = 640, height = 360 } = {}) {
  const byKey = new Map(cases.map((c) => [c.key, c]));
  const plot = { x0: MARGIN.left, x1: width - MARGIN.right, y0: height - MARGIN.bottom, y1: MARGIN.top };
  const raw = [];
  const ceil = [];
  for (const [id, info] of Object.entries(machines ?? {})) {
    const data = latest?.[id];
    if (!data) continue;
    for (const p of data.points ?? []) {
      if (!(p.intensity > 0) || !(p.gflops > 0)) continue;
      const c = byKey.get(p.key);
      raw.push({ machine: id, key: p.key, i: p.intensity, g: p.gflops / 1e9,
        title: `${c ? caseLabel(c) : p.key}${DOT}${info.label ?? id}${DOT}${p.intensity.toPrecision(2)} flop/byte${DOT}${formatValue(p.gflops, "gflops")}` });
    }
    const peak = info.fp64_peak > 0 ? info.fp64_peak / 1e9 : null;
    const triad = data.triad > 0 ? data.triad / 1e9 : null;
    if (peak) ceil.push({ machine: id, peak, triad, knee: triad ? peak / triad : null });
  }
  const xs = [...raw.map((d) => d.i), ...ceil.filter((c) => c.knee).map((c) => c.knee)];
  const ys = [...raw.map((d) => d.g), ...ceil.map((c) => c.peak)];
  if (!xs.length || !ys.length) return { dots: [], ceilings: [], xTicks: [], yTicks: [], plot };
  const [x0e, x1e] = decadeDomain(Math.min(...xs), Math.max(...xs));
  // the ceiling's lowest point is min(triad * xlo, peak): part of the y domain
  const lows = ceil.map((c) => (c.triad ? Math.min(c.triad * 10 ** x0e, c.peak) : c.peak));
  const [y0e, y1e] = decadeDomain(Math.min(...ys, ...lows), Math.max(...ys));
  const sx = logScale(10 ** x0e, 10 ** x1e, plot.x0, plot.x1);
  const sy = logScale(10 ** y0e, 10 ** y1e, plot.y0, plot.y1);
  const xlo = 10 ** x0e;
  const xhi = 10 ** x1e;
  const ceilings = ceil.map((c) => {
    const pts = [];
    if (c.triad) {
      pts.push([xlo, Math.min(c.triad * xlo, c.peak)]);
      if (c.knee > xlo && c.knee < xhi) pts.push([c.knee, c.peak]);
      pts.push([xhi, Math.min(c.triad * xhi, c.peak)]);
    } else {
      pts.push([xlo, c.peak], [xhi, c.peak]);
    }
    return { machine: c.machine, knee: c.knee, points: pts.map(([i, g]) => `${sx(i)},${sy(g)}`).join(" ") };
  });
  return {
    dots: raw.map((d) => ({ cx: sx(d.i), cy: sy(d.g), machine: d.machine, key: d.key, title: d.title })),
    ceilings,
    xTicks: decadeTicks(x0e, x1e, sx),
    yTicks: decadeTicks(y0e, y1e, sy),
    plot,
  };
}

const NUMERIC_STATES = ["history", "same", "slower", "faster"];
const numeric = (cell) => !!cell && NUMERIC_STATES.includes(cell.state);
const lastCell = (tabData, machine, key) => {
  const cells = tabData.machines?.[machine]?.cells?.[key] ?? [];
  return cells[cells.length - 1] ?? null;
};

// The "this week" solver table: the last column's cells, in case order.
export function solverRows(tabData, machine) {
  return tabData.cases.map((c) => {
    const cell = lastCell(tabData, machine, c.key);
    const has = numeric(cell);
    const pick = (field) => (has && cell[field] !== undefined ? cell[field] : null);
    return {
      solver: c.labels.solver, stencil: c.labels.stencil,
      iterations: pick("iterations"),
      msPerIteration: has && cell.time != null ? cell.time * 1e3 : null,
      gbs: pick("gbs"), pct: pick("pct"), residual: pick("residual"),
    };
  });
}

// One row per preconditioner on the matrix; non-numeric states show the state only.
export function preconditionerRows(tabData, machine, matrix) {
  const byName = new Map();
  for (const c of tabData.cases) {
    if (c.labels.matrix !== matrix) continue;
    const entry = byName.get(c.labels.preconditioner) ?? {};
    entry[c.labels.stage] = lastCell(tabData, machine, c.key);
    byName.set(c.labels.preconditioner, entry);
  }
  return [...byName].map(([preconditioner, { apply, generate }]) => {
    const state = (apply ?? generate)?.state ?? "not_available";
    const ok = numeric(apply);
    const ms = (cell) => (numeric(cell) && cell.time != null ? cell.time * 1e3 : null);
    return {
      preconditioner,
      generateMs: ms(generate),
      applyMs: ms(apply),
      storageMb: ok && apply.storage != null ? apply.storage / 1e6 : null,
      levels: ok && apply.levels != null ? apply.levels : null,
      state,
    };
  });
}
