// The Kernels tab's logic, without the DOM: the bars of where a case's time goes,
// run by run, and the table that maps each Ginkgo range's time to its counts and
// counters. The data is site/data/kernels.json (dashboard/kernels.py).

export const OTHER = "other";
export const COLOURS = 8;            // named colours; the rest fold into "other"
export const OTHER_BELOW = 0.03;     // a range under 3 % of the total folds into "other"
export const CHANGE = 0.05;          // a time or counter that moves more than this is marked
export const COUNTS = ["calls", "launches", "allocations", "copies"];
export const MEASURES = ["time", "copy_bytes", "dram_read_bytes", "dram_write_bytes", "fp64_ops",
  "l2_hit_rate"];

const ofStage = (run, stage) => (run.operations || []).filter((o) => (o.stage || "") === (stage || ""));

// the colour of every range name across the runs: by its largest share of any run's total,
// so a name keeps its colour whatever else appears
function colourMap(runs, stage, otherBelow) {
  const share = new Map();
  for (const run of runs) {
    const ops = ofStage(run, stage).filter((o) => o.time);
    const total = ops.reduce((s, o) => s + o.time, 0);
    for (const o of ops) {
      if (total > 0) share.set(o.name, Math.max(share.get(o.name) ?? 0, o.time / total));
    }
  }
  const named = [...share].filter(([, s]) => s >= otherBelow).sort((a, b) => b[1] - a[1])
    .slice(0, COLOURS).map(([name]) => name);
  return new Map(named.map((name, i) => [name, `k${i}`]));
}

// one bar per run, oldest first: {dir, total, segments: [{name, time, colour}], reason}
export function barModel(runs, stage, { otherBelow = OTHER_BELOW } = {}) {
  const colours = colourMap(runs, stage, otherBelow);
  return runs.map((run) => {
    const ops = ofStage(run, stage).filter((o) => o.time);
    if (run.status === "failed" || run.status === "skipped" || run.status === "not_available"
        || !ops.length) {
      return { dir: run.dir, total: null, segments: [], reason: run.reason ?? run.status };
    }
    const total = ops.reduce((s, o) => s + o.time, 0);
    const segments = [];
    let other = 0;
    for (const o of [...ops].sort((a, b) => b.time - a.time)) {
      const colour = colours.get(o.name);
      if (colour && o.time / total >= otherBelow) segments.push({ name: o.name, time: o.time, colour });
      else other += o.time;
    }
    if (other > 0) segments.push({ name: OTHER, time: Math.round(other * 1e12) / 1e12, colour: "other" });
    return { dir: run.dir, total: Math.round(total * 1e12) / 1e12, segments, reason: null };
  });
}

function change(key, now, before) {
  if (now === undefined || now === null || before === undefined || before === null) return null;
  if (COUNTS.includes(key)) return now !== before ? "changed" : null;
  if (key === "l2_hit_rate") return Math.abs(now - before) > CHANGE ? (now > before ? "down" : "up") : null;
  if (before === 0) return now === 0 ? null : "up";
  const rel = now / before - 1;
  if (rel > CHANGE) return "up";
  if (rel < -CHANGE) return "down";
  return null;
}

// the rows of the newest run against the one before: by time, slowest first, then the
// ranges the newest run no longer has
export function tableModel(run, previous, stage) {
  const ops = ofStage(run, stage);
  const before = new Map((previous ? ofStage(previous, stage) : []).map((o) => [o.name, o]));
  const keys = [...COUNTS, ...MEASURES];
  const rows = ops.map((o) => {
    const p = before.get(o.name);
    const changes = {};
    for (const k of keys) changes[k] = p ? change(k, o[k], p[k]) : null;
    return { name: o.name, parent: o.parent, values: o, changes, isNew: Boolean(previous) && !p,
      isGone: false, kernels: o.kernels || [] };
  });
  rows.sort((a, b) => (b.values.time ?? -1) - (a.values.time ?? -1));
  const now = new Set(ops.map((o) => o.name));
  for (const [name, o] of before) {
    if (!now.has(name)) {
      rows.push({ name, parent: o.parent, values: o, changes: {}, isNew: false, isGone: true, kernels: [] });
    }
  }
  return rows;
}

// "void gko::kernels::cuda::csr::kernel::abstract_classical_spmv<4ul, ...>(...)" -> "csr::kernel::abstract_classical_spmv"
export function shortKernel(name) {
  let s = name.replace(/^void /, "");
  const generic = s.match(/generic_kernel_\w+<void (?:gko::kernels::\w+::)?([\w:]+)</);
  if (generic) return `${generic[1]} (generic kernel)`;
  // cut at the first "<" or "(" outside any brackets
  let depth = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === "<" || c === "(") {
      if (depth === 0 && !(c === "(" && s.startsWith("(anonymous namespace)", i))) {
        s = s.slice(0, i);
        break;
      }
      depth++;
    } else if (c === ">" || c === ")") depth--;
  }
  return s.replace(/^gko::kernels::\w+::/, "").replace(/\(anonymous namespace\)::/g, "").replace(/^gko::/, "")
    .replace(/CUB_\w+_NS::/g, "");
}

// a range's name for the table: an operation as it is; a linear operator's without "gko::"
// and with its types' template arguments folded ("generate(multigrid::Pgm<…>::Factory)")
export function shortRange(name) {
  if (!name.includes("(")) return name;
  let out = "";
  let depth = 0;
  for (const c of name.replace(/gko::/g, "")) {
    if (c === "<") {
      if (depth === 0) out += "<…>";
      depth++;
    } else if (c === ">") depth--;
    else if (depth === 0) out += c;
  }
  return out;
}

// measured DRAM bytes against the dashboard's model of the case, or null without either
export function modelRatio(values, modelBytes) {
  if (!modelBytes) return null;
  const measured = (values.dram_read_bytes ?? 0) + (values.dram_write_bytes ?? 0);
  return measured ? measured / modelBytes : null;
}
