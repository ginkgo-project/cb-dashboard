// Renders the Kernels tab from site/data/kernels.json: for one case, where its time goes
// run by run (one bar per run, split by Ginkgo range), and a table that maps each range's
// time in the newest run to its counts and hardware counters, against the run before.
// A row opens into its GPU kernels. The DOM helpers come from app.js.
import { barModel, tableModel, shortKernel, shortRange, modelRatio, OTHER } from "./kernels-logic.js";
import { formatValue } from "./logic.js";
import { formatBytes } from "./distributed-logic.js";

const cache = { data: null };
const GROUPS = [["multigrid", "Multigrid"], ["spmv", "SpMV"], ["spgemm", "SpGEMM"], ["blas", "BLAS"],
  ["gemv", "GEMV"]];
const STAGES = [["generate", "setup"], ["apply", "apply"]];
const BARS = 8;  // the newest runs shown as bars
const shortSha = (sha) => (sha || "").slice(0, 7);
const dash = "–";
const fmtTime = (v) => (v === null || v === undefined ? dash : formatValue(v, "time"));
const fmtCount = (v) => (v === null || v === undefined ? dash : String(v));
const fmtBytes = (v) => (v === null || v === undefined ? dash : formatBytes(v));
const fmtOps = (v) => {
  if (v === null || v === undefined) return dash;
  for (const [d, u] of [[1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "k"]]) {
    if (v >= d) return `${(v / d).toFixed(v / d >= 100 ? 0 : 1)} ${u}`;
  }
  return String(Math.round(v));
};
const fmtRate = (v) => (v === null || v === undefined ? dash : `${Math.round(v * 100)} %`);

// the table's columns: [key, heading, format]
const COLUMNS = [["time", "time", fmtTime], ["calls", "calls", fmtCount], ["launches", "kernels", fmtCount],
  ["allocations", "allocs", fmtCount], ["copies", "copies", fmtCount], ["copy_bytes", "copied", fmtBytes],
  ["dram_read_bytes", "DRAM read", fmtBytes], ["dram_write_bytes", "DRAM written", fmtBytes],
  ["fp64_ops", "FP64 ops", fmtOps], ["l2_hit_rate", "L2 hits", fmtRate]];

export async function renderKernels(root, ui) {
  const { el, put, formatHash, params } = ui;
  if (!cache.data) cache.data = await ui.loadJson("data/kernels.json");
  if (!ui.isLatest()) return;
  const machines = cache.data.machines;
  const ids = Object.keys(machines);
  put(root, el("h2", {}, "Where the time goes, range by range"));
  if (!ids.length) {
    put(root, el("p", { class: "none kernels-none" }, "No kernel counters yet."));
    return;
  }
  const firstWith = ids.find((id) => machines[id].runs.length) ?? ids[0];
  const machine = ids.includes(params.machine) ? params.machine : firstWith;
  const m = machines[machine];
  const link = (extra) => formatHash("kernels", { ...params, machine, ...extra });
  if (ids.length > 1) {
    put(root, el("p", { class: "machines" }, "Machine: ", ids.map((id) => el("a", {
      href: formatHash("kernels", { machine: id }), "aria-current": id === machine ? "true" : undefined,
    }, ui.machineLabel(id)))));
  }
  if (!m.runs.length) {
    put(root, el("p", { class: "none kernels-none" },
      `No kernel counters yet on ${ui.machineLabel(machine)}. They are read after the timing groups `
      + "of each single-node run, for a fixed set of cases."));
    return;
  }
  const keys = Object.keys(m.cases);
  const key = keys.includes(params.case) ? params.case
    : (keys.find((k) => m.cases[k].group === "multigrid") ?? keys[0]);
  const c = m.cases[key];
  const staged = c.runs.some((r) => (r.operations || []).some((o) => o.stage));
  const stage = staged ? (STAGES.some(([s]) => s === params.stage) ? params.stage : "generate") : "";

  // the controls: the case, grouped, and the stage of a multigrid case
  const caseSelect = el("select", { "aria-label": "case",
    onchange: (e) => { location.hash = link({ case: e.target.value, row: undefined }); } },
  GROUPS.filter(([g]) => keys.some((k) => m.cases[k].group === g)).map(([g, label]) =>
    el("optgroup", { label }, keys.filter((k) => m.cases[k].group === g).map((k) =>
      el("option", { value: k, selected: k === key }, m.cases[k].label)))));
  const controls = [el("label", {}, "case ", caseSelect)];
  if (staged) {
    controls.push(el("label", {}, "stage ", el("select", { "aria-label": "stage",
      onchange: (e) => { location.hash = link({ stage: e.target.value, row: undefined }); } },
    STAGES.map(([s, label]) => el("option", { value: s, selected: s === stage }, label)))));
  }
  put(root, el("div", { class: "filters" }, controls));

  // where the time goes: one bar per run, newest at the bottom, on one scale
  const runs = c.runs.slice(-BARS);
  const byDir = new Map(m.runs.map((r) => [r.dir, r]));
  const bars = barModel(runs, stage);
  const scale = Math.max(...bars.map((b) => b.total ?? 0)) || 1;
  const names = new Map();
  const barRows = bars.map((b) => {
    const r = byDir.get(b.dir) || {};
    const label = el("span", { class: "kbar-label" }, `${r.date ?? b.dir} ${shortSha(r.commit)}`);
    if (b.total === null) {
      return el("div", { class: "kbar-row" }, label, el("span", { class: "kbar-none" }, b.reason || "no data"));
    }
    const segments = b.segments.map((s) => {
      names.set(s.name, s.colour);
      return el("span", { class: `kseg ${s.colour}`, style: `width: ${(100 * s.time / scale).toFixed(2)}%`,
        title: `${shortRange(s.name)}: ${fmtTime(s.time)}` });
    });
    return el("div", { class: "kbar-row" }, label, el("span", { class: "kbar" }, segments),
      el("span", { class: "kbar-total" }, fmtTime(b.total)));
  });
  put(root, el("div", { class: "kbars", role: "img",
    "aria-label": `Time of ${c.label}${stage ? ` (${stage})` : ""} by range, per run` }, barRows));
  put(root, el("p", { class: "legend klegend" }, [...names].map(([name, colour]) =>
    el("span", { class: "kkey", title: name }, el("span", { class: `kswatch ${colour}` }),
      name === OTHER ? "other" : shortRange(name)))));
  put(root, el("p", { class: "legend" },
    "Times from Ginkgo's own timing of each range (-detailed), exclusive of the ranges inside it; "
    + "the counters from a separate profiled run."));

  // the table of the newest run with data, against the one before it
  const withData = c.runs.filter((r) => (r.operations || []).length);
  const newest = withData[withData.length - 1];
  if (!newest) {
    put(root, el("p", { class: "none" }, "No run of this case has counters yet."));
    return;
  }
  const previous = withData.length > 1 ? withData[withData.length - 2] : null;
  const rows = tableModel(newest, previous, stage);
  const model = newest.model_bytes;
  const head = el("tr", {}, el("th", { scope: "col" }, "range"),
    COLUMNS.map(([, h]) => el("th", { scope: "col" }, h)),
    model ? el("th", { scope: "col", title: "measured DRAM bytes against the dashboard's model" }, "bytes / model") : null);
  const body = el("tbody");
  for (const row of rows) {
    const opened = params.row === row.name;
    const cls = [row.isNew ? "new" : "", row.isGone ? "gone" : "", opened ? "opened" : ""].filter(Boolean).join(" ");
    const name = row.kernels.length && !row.isGone
      ? el("a", { href: link({ row: opened ? undefined : row.name }) }, `${opened ? "▾" : "▸"} ${shortRange(row.name)}`)
      : shortRange(row.name);
    put(body, el("tr", { class: cls || undefined },
      el("th", { scope: "row", title: row.parent ? `${row.name}, in ${row.parent}` : row.name }, name, row.isNew ? el("span", { class: "tag" }, "new") : null,
        row.isGone ? el("span", { class: "tag" }, "gone") : null),
      COLUMNS.map(([k, , f]) => el("td", { class: row.changes[k] ? `chg-${row.changes[k]}` : undefined },
        row.isGone ? dash : f(row.values[k]))),
      model ? el("td", {}, (() => { const r = modelRatio(row.values, model); return r === null ? dash : r.toFixed(2); })())
        : null));
    if (opened) {
      for (const k of row.kernels) {
        put(body, el("tr", { class: "kernel-of" },
          el("th", { scope: "row", title: k.name }, shortKernel(k.name)),
          COLUMNS.map(([key, , f]) => el("td", {}, key === "time" || key === "calls"
            || key === "allocations" || key === "copies" || key === "copy_bytes" ? "" : f(k[key]))),
          model ? el("td", {}) : null));
      }
    }
  }
  const r = byDir.get(newest.dir) || {};
  const p = previous ? byDir.get(previous.dir) || {} : null;
  put(root, el("h3", {}, `The ranges of ${r.date ?? newest.dir} ${shortSha(r.commit)}`
    + (p ? `, against ${p.date ?? previous.dir} ${shortSha(p.commit)}` : "")));
  put(root, el("table", { class: "kernels" }, el("thead", {}, head), body));
  put(root, el("p", { class: "legend" },
    "Red: more time, bytes or operations than the run before (beyond 5 %), or a lower L2 hit rate; "
    + "green: less; yellow: a count that changed (counts are exact). A range opens into its GPU kernels."));
  if (newest.status !== "ok" && newest.reason) {
    put(root, el("p", { class: "legend" }, `This run: ${newest.status}: ${newest.reason}.`));
  }
  if (m.tool === "rocprofv3") {
    put(root, el("p", { class: "legend" },
      "On the MI300A: \"DRAM\" bytes are everything fetched past the L2, including what the 256 MB "
      + "Infinity Cache serves; FP64 operations are counted per wavefront of 64 lanes, so partly "
      + "filled wavefronts count as full; copies have no sizes (ROCm 6.3). Compare hit rates, not "
      + "counts, across the two machines."));
  }
}
