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
