// Renders the benchmark dashboard from site/data/*.json. All text goes
// through textContent (the el helper), so node names, errors and reasons
// cannot inject markup.
import { TABS, parseHash, formatHash, caseLabel, flagText, renderTokens, heatmapModel, formatValue, formatChange } from "./logic.js";

// tabs: the tab files, loaded on the first visit of the tab
const state = { meta: null, tabs: {}, runs: {}, route: { tab: "basic", params: {} } };
const tokens = renderTokens();
const boxTokens = renderTokens(); // one click box fill at a time
const TAB_FILES = ["basic", "solvers", "preconditioners"];

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === undefined || child === null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// used by the charts of the later tasks
export function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== undefined && value !== null) node.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === undefined || child === null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function showBanner(level, text) {
  document.getElementById("banners").append(el("p", { class: `banner ${level}`, role: "status" }, text));
}

async function loadJson(path) {
  const response = await fetch(path, { cache: "no-cache" });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

const machineLabel = (id) => state.meta?.machines[id]?.label ?? id;
const shortSha = (sha) => (sha || "").slice(0, 7);

function formatSeconds(s) {
  if (s === undefined || s === null) return "";
  return s >= 3600 ? `${(s / 3600).toFixed(1)} h` : `${Math.round(s / 60)} min`;
}

function renderMeta() {
  const meta = document.getElementById("meta");
  meta.replaceChildren();
  for (const [id, machine] of Object.entries(state.meta.machines)) {
    const run = machine.last_run;
    meta.append(el("span", { class: "run" }, run
      ? `${machine.label}: ${run.commit_date} at ${shortSha(run.commit)}, ${run.status}, ${formatSeconds(run.seconds)}`
      : `${machine.label}: no run`));
  }
  const counts = state.meta.flag_counts;
  const total = TAB_FILES.reduce((sum, tab) => sum + (counts[tab] ?? 0), 0);
  meta.append(el("span", { class: "flag-counts" },
    `${total} flagged: ${TAB_FILES.map((tab) => `${tab} ${counts[tab] ?? 0}`).join(", ")}`));
}

function renderTabs() {
  for (const tab of TABS) {
    const link = document.getElementById(tab);
    if (tab === state.route.tab) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
}

// the flagged list: each entry links to its cell in the heatmap
function renderFlags(root, data, tab) {
  const byKey = new Map(data.cases.map((c) => [c.key, c]));
  root.append(el("h2", {}, "Flagged"));
  if (!data.flags.length) {
    root.append(el("p", { class: "none" }, "Nothing flagged in the latest run."));
    return;
  }
  root.append(el("ul", { class: "flags" }, data.flags.map((flag) => {
    const c = byKey.get(flag.key) ?? { key: flag.key, group: "", labels: {} };
    return el("li", {},
      el("span", { class: `flag ${flag.state}` }, flag.state.replace("_", " ")),
      el("a", { href: formatHash(tab, { machine: flag.machine, cell: `${flag.key}@${flag.column}` }) },
        flagText(flag, c, machineLabel(flag.machine))));
  })));
}

// "10 not available (mg-pmis) on Beverin MI300A, Daint GH200"
function renderUnchanged(root, data) {
  const parts = data.unchanged.map((u) => {
    const names = u.names?.length ? ` (${u.names.join(", ")})` : "";
    const machines = u.machines?.length ? ` on ${u.machines.map(machineLabel).join(", ")}` : "";
    return `${u.count} ${u.state.replaceAll("_", " ")}${names}${machines}`;
  });
  root.append(el("p", { class: "summary" }, parts.length ? `Not flagged: ${parts.join("; ")}` : "Not flagged: nothing else to report."));
}

function table(columns, rows, className) {
  return el("div", { class: "scroll" }, el("table", { class: className },
    el("thead", {}, el("tr", {}, columns.map((c) => el("th", { scope: "col" }, c)))),
    el("tbody", {}, rows.map((row) => el("tr", {}, row.map((cell) =>
      el("td", { class: typeof cell === "number" ? "number" : undefined }, cell === true ? "yes" : cell === false ? "no" : cell)))))));
}

function renderDataset(root, data) {
  if (!data.dataset) return;
  root.append(el("details", { class: "detail" },
    el("summary", {}, data.dataset.title),
    table(data.dataset.columns, data.dataset.rows, "dataset")));
}

function renderModels(root, data) {
  if (!data.models?.length) return;
  root.append(el("details", { class: "detail" },
    el("summary", {}, "Cost models"),
    table(["model", "bytes", "flops", "source", "bandwidth bound"],
      data.models.map((m) => [m.name, m.bytes, m.flops, m.source, m.bandwidth_bound]), "models")));
}

// Tasks 11 to 14 fill these in
async function loadRun(machine, dir) {
  const path = `data/runs/${machine}/${dir}.json`;
  if (!state.runs[path]) state.runs[path] = await loadJson(path);
  return state.runs[path];
}

const watts = (joules, seconds) => (joules != null && seconds ? ` · ${(joules / seconds).toFixed(0)} W` : "");

function runBlock(machine, dir, run, key) {
  const nodeStats = state.meta.machines[machine]?.nodes?.[run.node];
  const info = run.cases?.[key];
  const inv = info ? run.invocations?.[info.invocation] : null;
  const joules = inv?.energy_j?.accel0_energy;
  const triad = run.triad == null ? "triad not measured"
    : `triad ${(run.triad / 1e9).toFixed(0)} GB/s${nodeStats
      ? ` (node range ${(nodeStats.triad_min / 1e9).toFixed(0)} to ${(nodeStats.triad_max / 1e9).toFixed(0)} GB/s over ${nodeStats.runs} runs)` : ""}`;
  return el("li", {},
    el("strong", {}, dir), ` · node ${run.node ?? "unknown"} · ${triad}`,
    el("br"), `commit ${shortSha(run.ginkgo?.commit)} on ${run.ginkgo?.ref ?? "unknown ref"}`,
    el("br"), joules == null ? "no energy recorded"
      : `energy ${Math.round(joules)} J${watts(joules, inv.seconds)} (whole benchmark run, including loading)`);
}

function levelsTable(levels) {
  const short = (name) => String(name ?? "").replace(/^gko::(solver|preconditioner)::/, "").replace(/<.*$/, "");
  return el("div", { class: "scroll" }, el("table", { class: "levels" },
    el("thead", {}, el("tr", {}, ["level", "rows", "coarse rows", "coarse nonzeros", "smoother", "smoother solver"].map((h) => el("th", { scope: "col" }, h)))),
    el("tbody", {}, levels.map((lv, i) => el("tr", {},
      el("td", { class: "number" }, i), el("td", { class: "number" }, lv.fine_rows), el("td", { class: "number" }, lv.coarse_rows),
      el("td", { class: "number" }, lv.coarse_nonzeros), el("td", {}, short(lv.pre_smoother)), el("td", {}, short(lv.pre_smoother_solver)))))));
}

async function fillBox(box, data, tab, machine, key, col) {
  const token = boxTokens.next();
  const c = data.cases.find((x) => x.key === key);
  const columns = data.machines[machine]?.columns ?? [];
  const cell = data.machines[machine]?.cells?.[key]?.[col];
  box.replaceChildren();
  if (!c || !cell) {
    box.append(el("p", { class: "none" }, "No data for the selected cell."));
    return;
  }
  const values = [
    ["state", cell.state.replaceAll("_", " ")],
    ["time", cell.time == null ? null : `${formatValue(cell.time, "time")}${tab === "solvers" ? " per iteration" : ""}`],
    ["bandwidth", cell.gbs == null ? null : formatValue(cell.gbs, "gbs")],
    ["flop rate", cell.gflops == null ? null : formatValue(cell.gflops, "gflops")],
    ["of node bandwidth", cell.pct == null ? null : formatValue(cell.pct, "pct")],
    ["change", cell.change == null ? null : `${formatChange(cell.change)}, band ±${((cell.band ?? 0) * 100).toFixed(1)}%`],
    ["iterations", cell.iterations == null ? null : `${cell.iterations}${cell.iterations_prev == null ? "" : ` (previously ${cell.iterations_prev})`}`],
    ["residual", cell.residual == null ? null : cell.residual.toExponential(2)],
    ["error", cell.error],
  ].filter(([, v]) => v != null);
  const prev = columns[col - 1];
  box.append(
    el("h3", {}, `${caseLabel(c)} · ${machineLabel(machine)} · ${columns[col]?.date ?? ""} at ${shortSha(columns[col]?.commit)}`),
    el("dl", { class: "values" }, values.map(([k, v]) => [el("dt", {}, k), el("dd", {}, v)])),
    cell.params_changed ? el("p", { class: "banner warning" }, "Parameters changed since the previous column.") : null,
    el("p", {}, prev ? `Compared with the previous column at commit ${shortSha(prev.commit)} (${prev.date}).` : "First column, nothing to compare with."));
  const runsList = el("ul", { class: "runs-list" }, el("li", { class: "none" }, "Loading the runs ..."));
  const command = el("p", { class: "none" }, "Loading the command ...");
  box.append(el("h3", {}, "Runs"), runsList, el("h3", {}, "Command to reproduce"), command);
  const dirs = cell.runs ?? columns[col]?.runs ?? [];
  const runs = await Promise.allSettled(dirs.map((dir) => loadRun(machine, dir)));
  if (!boxTokens.isLatest(token)) return; // another cell was selected meanwhile
  runsList.replaceChildren(...runs.map((r, i) => r.status === "fulfilled"
    ? runBlock(machine, dirs[i], r.value, key)
    : el("li", { class: "none" }, `${dirs[i]}: could not be loaded (${r.reason.message})`)));
  const loaded = runs.filter((r) => r.status === "fulfilled").map((r) => r.value);
  const withCommand = loaded.find((run) => run.cases?.[key]?.command);
  command.replaceWith(withCommand ? el("pre", { class: "command" }, el("code", {}, withCommand.cases[key].command))
    : el("p", { class: "none" }, "No command recorded."));
  const levels = loaded.map((run) => run.cases?.[key]?.levels).find((lv) => lv?.length);
  if (levels) box.append(el("h3", {}, "AMG levels"), levelsTable(levels));
}

function renderHeatmap(root, data, tab) {
  const params = state.route.params;
  const ids = Object.keys(data.machines);
  const machine = ids.includes(params.machine) ? params.machine : ids[0];
  root.append(el("h2", {}, "Changes by commit"));
  if (ids.length > 1) {
    root.append(el("p", { class: "machines" }, "Machine: ", ids.map((id) => el("a", {
      href: formatHash(tab, { ...params, machine: id, cell: undefined }),
      "aria-current": id === machine ? "true" : undefined,
    }, machineLabel(id)))));
  }
  const model = heatmapModel(data, machine, params);
  if (!model.rows.length || !model.columns.length) {
    root.append(el("p", { class: "none" }, "No cases to show for this selection."));
    return;
  }
  const [selKey, selCol] = (params.cell ?? "").split(/@(?=[^@]*$)/);
  const box = el("section", { class: "clickbox", "aria-live": "polite" });
  root.append(
    el("p", { class: "legend" }, `Colour: change against the case's own history; ${model.legend}. A dot marks a changed iteration count.`),
    el("div", { class: "scroll" }, el("table", { class: "heatmap" },
      el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "case"),
        model.columns.map((col) => el("th", { scope: "col", class: col.setup ? "setup" : undefined,
          title: col.setup ? `Setup changed: ${col.setup}\n${col.title}` : col.title }, col.label)))),
      el("tbody", {}, model.rows.map((row) => el("tr", {}, el("th", { scope: "row" }, row.label),
        row.cells.map((cell) => el("td", {
          class: `${cell.cls}${model.columns[cell.col].setup ? " setup" : ""}${row.key === selKey && String(cell.col) === selCol ? " selected" : ""}`,
          title: cell.title, tabindex: cell.cls === "cell empty" ? undefined : "0",
          onclick: () => { location.hash = formatHash(tab, { ...params, machine, cell: `${row.key}@${cell.col}` }); },
          onkeydown: (e) => { if (e.key === "Enter") e.target.click(); },
        }, cell.dot ? "•" : ""))))))),
    box);
  if (selKey && selCol !== undefined) fillBox(box, data, tab, machine, selKey, Number(selCol));
  else box.append(el("p", { class: "none" }, "Click a cell for the numbers, the node, the energy and the command."));
}

function renderCharts(root, data, tab) {}
function renderTabTables(root, data, tab) {}

function renderMachines(root) {
  root.append(el("p", { class: "none" }, "The machines tab is not built yet."));
}

function renderTab(root, tab, data) {
  renderFlags(root, data, tab);
  renderUnchanged(root, data);
  renderHeatmap(root, data, tab);
  renderCharts(root, data, tab);
  renderTabTables(root, data, tab);
  renderDataset(root, data);
  renderModels(root, data);
}

async function render() {
  const token = tokens.next();
  const { tab } = state.route;
  renderTabs();
  const scrollY = window.scrollY;
  const root = document.getElementById("view");
  root.replaceChildren();
  if (tab === "machines") {
    renderMachines(root);
    return;
  }
  try {
    if (!state.tabs[tab]) state.tabs[tab] = await loadJson(`data/${tab}.json`);
  } catch (error) {
    showBanner("error", `Could not load the ${tab} data: ${error.message}`);
    return;
  }
  if (!tokens.isLatest(token)) return; // a newer render started while loading
  renderTab(root, tab, state.tabs[tab]);
  window.scrollTo(0, scrollY);
}

function onHashChange() {
  state.route = parseHash(location.hash);
  render();
}

async function main() {
  try {
    state.meta = await loadJson("data/meta.json");
  } catch (error) {
    showBanner("error", `Could not load the data: ${error.message}`);
    return;
  }
  for (const warning of state.meta.warnings ?? []) showBanner("warning", warning);
  renderMeta();
  window.addEventListener("hashchange", onHashChange);
  onHashChange();
}

main();
