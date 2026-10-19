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

const STATUS_FLAGS = { new_failure: "new failure", new_not_available: "now not available", new_missing: "now missing" };

// The badge before the flag text: the direction of a speed change, or
// "status" for a new failure, not available or missing, which the text names.
export function flagBadge(state) {
  return state === "slower" || state === "faster" ? state : "status";
}

export function flagText(flag, c, machineLabel) {
  const what = STATUS_FLAGS[flag.state]
    ?? `${formatChange(flag.change)}${flag.band === undefined || flag.band === null ? "" : ` (noise ±${(flag.band * 100).toFixed(1)}%)`}`;
  return [what, caseLabel(c), machineLabel].join(DOT);
}

// The heatmap view that shows the case: operation and format for basic,
// stage for preconditioners, as rowFilter reads them.
function viewParams(tab, c) {
  const l = c.labels || {};
  if (tab === "basic") return { operation: l.operation, format: l.operation === "spmv" ? l.format : undefined };
  if (tab === "preconditioners") return { stage: l.stage };
  return {};
}

// The link to a heatmap cell, in the view that shows its row.
export function cellLink(tab, c, machine, column) {
  return formatHash(tab, { ...viewParams(tab, c), machine, cell: `${c.key}@${column}` });
}

// Whether a render shows a newly selected cell, which is then scrolled into
// view; a filter change keeps the scroll position.
export function revealCell(prev, route) {
  const cell = route.params.cell;
  if (!cell) return false;
  return !prev || prev.tab !== route.tab || prev.params.cell !== cell || prev.params.machine !== route.params.machine;
}

// The children to append: nested arrays flattened, null, undefined and false
// dropped, so a condition or a map can be passed as a child.
export function flatChildren(children) {
  return children.flat(Infinity).filter((c) => c !== undefined && c !== null && c !== false);
}

export function formatSeconds(s) {
  if (s === undefined || s === null) return "";
  return s >= 3600 ? `${(s / 3600).toFixed(1)} h` : `${Math.round(s / 60)} min`;
}

// The header's line for a machine: meta.json's last_run.
export function runSummary(label, run) {
  if (!run) return `${label}: no run`;
  return `${label}: ${run.commit_date} at ${(run.commit || "").slice(0, 7)}, node ${run.node ?? "unknown"}, ${run.status}, ${formatSeconds(run.seconds)}`;
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

const COMPARED = ["same", "slower", "faster"];

// A bandwidth case compared in raw time, as the build marks it with raw:
// this run, or too many earlier ones, had no node bandwidth. A history cell
// is not compared at all, so raw says nothing there.
export function rawNote(cell) {
  if (!cell?.raw || !COMPARED.includes(cell.state)) return null;
  return cell.pct === undefined || cell.pct === null
    ? "compared in raw time: no node bandwidth for this run"
    : "compared in raw time: too few earlier runs with a node bandwidth";
}

// r = |change| / band: s1 below 2, s2 below 4, s3 above; raw marks rawNote
export function cellClass(cell) {
  if (!cell) return "cell empty";
  const raw = rawNote(cell) ? " raw" : "";
  switch (cell.state) {
    case "same": return `cell same${raw}`;
    case "history": case "failed": case "disabled": return `cell ${cell.state}`;
    case "not_available": return "cell na";
    case "slower": case "faster": {
      const r = Math.abs(cell.change ?? 0) / (cell.band || 0.03);
      return `cell ${cell.state} ${r < 2 ? "s1" : r < 4 ? "s2" : "s3"}${raw}`;
    }
    default: return "cell empty";
  }
}

export function cellTitle(cell, c) {
  if (!cell) return `${caseLabel(c)}${DOT}no data`;
  const parts = [caseLabel(c), cell.state === "history" ? "not enough history yet" : cell.state.replaceAll("_", " ")];
  if (cell.change !== undefined && cell.change !== null) {
    parts.push(`${formatChange(cell.change)}${cell.band != null ? ` (band ±${(cell.band * 100).toFixed(1)}%)` : ""}`);
  }
  const raw = rawNote(cell);
  if (raw) parts.push(raw);
  if (cell.time !== undefined && cell.time !== null) parts.push(formatValue(cell.time, "time"));
  if (cell.pct !== undefined && cell.pct !== null) parts.push(`${formatValue(cell.pct, "pct")} of node bandwidth`);
  return parts.join(DOT);
}

const MIN_HISTORY = 4; // as in dashboard/changes.py
const count = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

// What the click box says the cell is compared with. The baseline is the
// median of the previous baseline_n values (at most 8), not the previous
// column alone.
export function baselineText(cell, columns, col) {
  const prev = columns[col - 1];
  if (!prev) return "First column, nothing to compare with.";
  const previous = `${(prev.commit || "").slice(0, 7)} (${prev.date})`;
  if (cell.change !== undefined && cell.change !== null) {
    return `Change against the median of the previous ${count(cell.baseline_n, "column")}; previous column: ${previous}.`;
  }
  if (cell.state === "history" && cell.baseline_n != null) {
    return `Not enough history yet: ${count(cell.baseline_n, "earlier column")} with a value, ${MIN_HISTORY} needed; previous column: ${previous}.`;
  }
  return `Previous column: ${previous}.`;
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

function linearTicks(lo, hi, scale, format, count = 5) {
  const step0 = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const ticks = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) {
    const value = Math.round(v / step) * step;
    ticks.push({ value, pos: scale(value), label: format(value) });
  }
  return ticks;
}

// a tick label: plain from 10⁻³ to below 10⁵, else m·10ⁿ
export function tickNumber(v) {
  const e = v > 0 ? Math.floor(Math.log10(v) + 1e-9) : 0;
  if (e >= -3 && e < 5) return String(Number(v.toPrecision(6)));
  const m = Number((v / 10 ** e).toPrecision(6));
  return m === 1 ? tenPower(e) : `${m}·${tenPower(e)}`;
}

// The ticks of a log axis over [lo, hi]: whole decades when three fit, else
// 1, 2 and 5 of each decade, else about eight evenly spaced ones (deep in
// a zoom).
export function logTicks(lo, hi, scale) {
  const inside = (v) => v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9);
  for (const ms of [[1], [1, 2, 5]]) {
    const ticks = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      for (const m of ms) {
        const value = m * 10 ** e;
        if (inside(value)) ticks.push({ value, pos: scale(value), label: ms.length === 1 ? tenPower(e) : tickNumber(value) });
      }
    }
    if (ticks.length >= 3) return ticks;
  }
  return linearTicks(lo, hi, scale, tickNumber, 8);
}

// ---- zoom and pan of a log-log chart; a range is [lo, hi] in data units ----

const MIN_DECADES = 0.02; // the deepest zoom, in decades of an axis
const FIT_DECADES = 0.1; // the narrowest range "fit the results" picks

const logRange = ([lo, hi]) => [Math.log10(lo), Math.log10(hi)];

// the log range [a, a + span], moved and shrunk to lie inside the domain
function placeRange(a, span, [da, db]) {
  const s = Math.min(Math.max(span, MIN_DECADES), db - da);
  const start = Math.min(Math.max(a, da), db - s);
  return [10 ** start, 10 ** (start + s)];
}

// range scaled by factor (below 1 zooms in) around the point at fraction at
// of it (0 at lo, 1 at hi), which stays where it is
export function zoomRange(range, domain, factor, at = 0.5) {
  const [a, b] = logRange(range);
  const d = logRange(domain);
  const span = Math.min(Math.max((b - a) * factor, MIN_DECADES), d[1] - d[0]);
  return placeRange(a + at * (b - a) - at * span, span, d);
}

// range moved by frac of its width, towards hi when positive
export function panRange(range, domain, frac) {
  const [a, b] = logRange(range);
  return placeRange(a + frac * (b - a), b - a, logRange(domain));
}

// [lo, hi] with pad of its width on each side
export function fitRange([lo, hi], domain, pad = 0.08) {
  const [a, b] = logRange([lo, hi]);
  const span = Math.max(b - a, FIT_DECADES) * (1 + 2 * pad);
  return placeRange((a + b - span) / 2, span, logRange(domain));
}

// a view is {x: range, y: range}; fx and fy are fractions across the plot
export const zoomView = (view, domain, factor, fx = 0.5, fy = 0.5) =>
  ({ x: zoomRange(view.x, domain.x, factor, fx), y: zoomRange(view.y, domain.y, factor, fy) });
export const panView = (view, domain, fx, fy) =>
  ({ x: panRange(view.x, domain.x, fx), y: panRange(view.y, domain.y, fy) });

// Where a pointer at clientX, clientY is across the plot of an svg of
// viewBox size drawn in rect: fx 0 at the left, fy 0 at the bottom, 1 at
// the right and the top.
export function plotFraction(plot, { width, height }, rect, clientX, clientY) {
  const ux = ((clientX - rect.left) / rect.width) * width;
  const uy = ((clientY - rect.top) / rect.height) * height;
  return { fx: (ux - plot.x0) / (plot.x1 - plot.x0), fy: (plot.y0 - uy) / (plot.y0 - plot.y1) };
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

const SHAPES = ["circle", "square"]; // one marker shape per machine

// The marker of a machine in the charts: its colour class m<index> and shape,
// by its place in meta.json's machines.
export function markerSeries(order, id) {
  const index = Math.max(0, order.indexOf(id));
  return { index, shape: SHAPES[index] ?? "circle" };
}

// machines: {id: {label, fp64_peak}}; latest: tabData.latest. x is the
// intensity (flops/byte), y GFLOP/s, both log. The ceiling of a machine is
// min(triad * I, peak); without a triad only the flat peak line is drawn.
// domain is the whole roofline, view the part drawn (the domain unless a
// view is given), fit the view holding all results and the ceilings above
// them; total counts the results, dots only those in the view. Ceilings
// can leave a zoomed plot: the page clips them.
export function rooflineModel(latest, machines, cases, { width = 640, height = 360, view = null } = {}) {
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
  if (!raw.length || !xs.length || !ys.length) return { dots: [], ceilings: [], xTicks: [], yTicks: [], plot, total: 0 };
  // the domain: whole decades, widened to hold the fit, so that no result
  // of the fit sits on the plot's edge
  const along = (lo, hi) => fitRange([lo, hi], [1e-300, 1e300]);
  const union = ([a, b], [c, d]) => [Math.min(a, c), Math.max(b, d)];
  const fitX = along(Math.min(...raw.map((d) => d.i)), Math.max(...raw.map((d) => d.i)));
  const [x0e, x1e] = decadeDomain(Math.min(...xs), Math.max(...xs));
  const dx = union([10 ** x0e, 10 ** x1e], fitX);
  const ceilingAt = (c, i) => (c.triad ? Math.min(c.triad * i, c.peak) : c.peak);
  const above = ceil.flatMap((c) => fitX.map((i) => ceilingAt(c, i)));
  const fitY = along(Math.min(...raw.map((d) => d.g)), Math.max(...raw.map((d) => d.g), ...above));
  // the ceiling's lowest point is min(triad * xlo, peak): part of the y domain
  const lows = ceil.map((c) => ceilingAt(c, dx[0]));
  const [y0e, y1e] = decadeDomain(Math.min(...ys, ...lows), Math.max(...ys));
  const domain = { x: dx, y: union([10 ** y0e, 10 ** y1e], fitY) };
  const fit = { x: fitX, y: fitY };
  const shown = view ? panView(view, domain, 0, 0) : domain; // moved inside the domain
  const [xlo, xhi] = shown.x;
  const [ylo, yhi] = shown.y;
  const sx = logScale(xlo, xhi, plot.x0, plot.x1);
  const sy = logScale(ylo, yhi, plot.y0, plot.y1);
  const inView = (d) => d.i >= xlo * (1 - 1e-9) && d.i <= xhi * (1 + 1e-9) && d.g >= ylo * (1 - 1e-9) && d.g <= yhi * (1 + 1e-9);
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
    dots: raw.filter(inView).map((d) => ({ cx: sx(d.i), cy: sy(d.g), machine: d.machine, key: d.key, title: d.title })),
    ceilings,
    xTicks: logTicks(xlo, xhi, sx),
    yTicks: logTicks(ylo, yhi, sy),
    plot, domain, view: shown, fit, total: raw.length,
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

// The path of a child, as the build writes it: a.b for keys, a[0] for list items.
export function childPath(path, key, isArray) {
  if (isArray) return `${path}[${key}]`;
  return path ? `${path}.${key}` : String(key);
}

// The leaves of a parameter tree with their paths; empty dicts and lists are leaves.
export function parameterPaths(value, path = "") {
  const isArray = Array.isArray(value);
  if (value !== null && typeof value === "object" && Object.keys(value).length) {
    return Object.entries(value).flatMap(([k, v]) => parameterPaths(v, childPath(path, k, isArray)));
  }
  return [{ path, value }];
}

// ---- machines tab ----

// history: machines.json triad list, in run order. x is the run order, y the
// triad bandwidth, linear with 5% padding; cls cycles n0-n5 by first sight of the node.
export function triadChartModel(history, { width = 480, height = 160 } = {}) {
  const plot = { x0: MARGIN.left, x1: width - MARGIN.right, y0: height - MARGIN.bottom, y1: MARGIN.top };
  const runs = (history ?? []).filter((h) => Number.isFinite(h.triad));
  if (!runs.length) return { dots: [], xTicks: [], yTicks: [], plot };
  const lo = Math.min(...runs.map((h) => h.triad));
  const hi = Math.max(...runs.map((h) => h.triad));
  const pad = hi > lo ? (hi - lo) * 0.05 : hi * 0.05;
  const sy = (v) => plot.y0 + ((v - (lo - pad)) / (hi + pad - (lo - pad))) * (plot.y1 - plot.y0);
  const sx = (i) => (runs.length === 1 ? (plot.x0 + plot.x1) / 2 : plot.x0 + (i / (runs.length - 1)) * (plot.x1 - plot.x0));
  const seen = [];
  const dots = runs.map((h, i) => {
    if (!seen.includes(h.node)) seen.push(h.node);
    return {
      cx: sx(i), cy: sy(h.triad), node: h.node, cls: `n${seen.indexOf(h.node) % 6}`,
      title: `${h.node}${DOT}${h.date ?? h.dir}${DOT}${formatValue(h.triad, "gbs")}`,
    };
  });
  const every = Math.ceil(runs.length / 6);
  const xTicks = runs.map((h, i) => ({ value: i, pos: sx(i), label: h.date ?? h.dir })).filter((_, i) => i % every === 0);
  return { dots, xTicks, yTicks: linearTicks(lo - pad, hi + pad, sy, (v) => formatValue(v, "gbs")), plot };
}

// groups: machines.json groups; the rows are those of the latest run directory.
export function energyRows(groups) {
  if (!groups?.length) return [];
  const latest = groups[groups.length - 1].dir;
  const orNull = (v) => (v === undefined ? null : v);
  return groups.filter((g) => g.dir === latest).map((g) => ({
    group: g.group, seconds: g.seconds, gpuJ: orNull(g.gpu_j),
    gpuW: g.gpu_j !== undefined && g.seconds ? g.gpu_j / g.seconds : null,
    nodeJ: orNull(g.node_j), cpuJ: orNull(g.cpu_j),
  }));
}

// A version sha for display: 7 characters, the full value as the title.
export function shortVersion(v) {
  return typeof v === "string" && v ? { text: v.slice(0, 7), title: v } : { text: "unknown", title: "unknown" };
}
