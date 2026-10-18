// Renders the benchmark dashboard from site/data/*.json. All text goes
// through textContent (the el helper), so node names, errors and reasons
// cannot inject markup.
import { TABS, parseHash, formatHash, caseLabel, flagText } from "./logic.js";

// tabs: the tab files, loaded on the first visit of the tab
const state = { meta: null, tabs: {}, route: { tab: "basic", params: {} } };
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
function renderHeatmap(root, data, tab) {}
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
  const { tab } = state.route;
  renderTabs();
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
  if (state.route.tab !== tab) return; // another tab was picked while loading
  renderTab(root, tab, state.tabs[tab]);
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
