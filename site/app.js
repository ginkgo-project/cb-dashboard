// Renders the benchmark dashboard from site/data/*.json. All text goes
// through text nodes (the put helper), so node names, errors and reasons
// cannot inject markup.
import { TABS, parseHash, formatHash, caseLabel, flagText, renderTokens, heatmapModel, formatValue, formatChange,
  flatChildren, cellLink, revealCell, rawNote, baselineText, flagBadge, runSummary, formatSeconds, markerSeries, rowFilter, scatterProps, scatterModel, rooflineModel, zoomView, panView, plotFraction, childPath, solverRows, preconditionerRows, triadChartModel, energyRows, shortVersion } from "./logic.js";

// tabs: the tab files, loaded on the first visit of the tab; shown: the route
// of the last rendered tab; views: the roofline view of each tab
const state = { meta: null, tabs: {}, runs: {}, route: { tab: "basic", params: {} }, shown: null, views: {} };
const tokens = renderTokens();
const boxTokens = renderTokens(); // one click box fill at a time
const TAB_FILES = ["basic", "solvers", "preconditioners"];

// Every child is appended here: arrays are flattened, null, undefined and
// false dropped, and anything that is not a node becomes a text node. The
// native append would write an array or null as text.
function put(node, ...children) {
  for (const child of flatChildren(children)) {
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function replace(node, ...children) {
  node.replaceChildren();
  return put(node, ...children);
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  return put(node, ...children);
}

// used by the charts of the later tasks
export function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== undefined && value !== null) node.setAttribute(key, value);
  }
  return put(node, ...children);
}

function showBanner(level, text) {
  put(document.getElementById("banners"), el("p", { class: `banner ${level}`, role: "status" }, text));
}

async function loadJson(path) {
  const response = await fetch(path, { cache: "no-cache" });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

const machineLabel = (id) => state.meta?.machines[id]?.label ?? id;
const shortSha = (sha) => (sha || "").slice(0, 7);

function renderMeta() {
  const meta = document.getElementById("meta");
  replace(meta);
  for (const machine of Object.values(state.meta.machines)) {
    put(meta, el("span", { class: "run" }, runSummary(machine.label, machine.last_run)));
  }
  const counts = state.meta.flag_counts;
  const total = TAB_FILES.reduce((sum, tab) => sum + (counts[tab] ?? 0), 0);
  put(meta, el("span", { class: "flag-counts" },
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
  put(root, el("h2", {}, "Flagged"));
  if (!data.flags.length) {
    put(root, el("p", { class: "none" }, "Nothing flagged in the latest run."));
    return;
  }
  put(root, el("ul", { class: "flags" }, data.flags.map((flag) => {
    const c = byKey.get(flag.key) ?? { key: flag.key, group: "", labels: {} };
    return el("li", {},
      el("span", { class: `flag ${flag.state}` }, flagBadge(flag.state)),
      el("a", { href: cellLink(tab, c, flag.machine, flag.column) }, flagText(flag, c, machineLabel(flag.machine))));
  })));
}

// "10 not available (mg-pmis) on Beverin MI300A, Daint GH200"
function renderUnchanged(root, data) {
  const parts = data.unchanged.map((u) => {
    const names = u.names?.length ? ` (${u.names.join(", ")})` : "";
    const machines = u.machines?.length ? ` on ${u.machines.map(machineLabel).join(", ")}` : "";
    return `${u.count} ${u.state.replaceAll("_", " ")}${names}${machines}`;
  });
  put(root, el("p", { class: "summary" }, parts.length ? `Not flagged: ${parts.join("; ")}` : "Not flagged: nothing else to report."));
}

function table(columns, rows, className) {
  return el("div", { class: "scroll" }, el("table", { class: className },
    el("thead", {}, el("tr", {}, columns.map((c) => el("th", { scope: "col" }, c)))),
    el("tbody", {}, rows.map((row) => el("tr", {}, row.map((cell) =>
      el("td", { class: typeof cell === "number" ? "number" : undefined }, cell === true ? "yes" : cell === false ? "no" : cell)))))));
}

function renderDataset(root, data) {
  if (!data.dataset) return;
  put(root, el("details", { class: "detail" },
    el("summary", {}, data.dataset.title),
    table(data.dataset.columns, data.dataset.rows, "dataset")));
}

function renderModels(root, data) {
  if (!data.models?.length) return;
  put(root, el("details", { class: "detail" },
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
  replace(box);
  if (!c || !cell) {
    put(box, el("p", { class: "none" }, "No data for the selected cell."));
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
  const raw = rawNote(cell);
  put(box,
    el("h3", {}, `${caseLabel(c)} · ${machineLabel(machine)} · ${columns[col]?.date ?? ""} at ${shortSha(columns[col]?.commit)}`),
    el("dl", { class: "values" }, values.map(([k, v]) => [el("dt", {}, k), el("dd", {}, v)])),
    cell.params_changed ? el("p", { class: "banner warning" }, "Parameters changed since the previous column.") : null,
    raw ? el("p", { class: "banner warning" }, `${raw[0].toUpperCase()}${raw.slice(1)}.`) : null,
    el("p", {}, baselineText(cell, columns, col)));
  const runsList = el("ul", { class: "runs-list" }, el("li", { class: "none" }, "Loading the runs ..."));
  const command = el("p", { class: "none" }, "Loading the command ...");
  put(box, el("h3", {}, "Runs"), runsList, el("h3", {}, "Command to reproduce"), command);
  const dirs = cell.runs ?? columns[col]?.runs ?? [];
  const runs = await Promise.allSettled(dirs.map((dir) => loadRun(machine, dir)));
  if (!boxTokens.isLatest(token)) return; // another cell was selected meanwhile
  replace(runsList, runs.map((r, i) => r.status === "fulfilled"
    ? runBlock(machine, dirs[i], r.value, key)
    : el("li", { class: "none" }, `${dirs[i]}: could not be loaded (${r.reason.message})`)));
  const loaded = runs.filter((r) => r.status === "fulfilled").map((r) => r.value);
  const withCommand = loaded.find((run) => run.cases?.[key]?.command);
  command.replaceWith(withCommand ? el("pre", { class: "command" }, el("code", {}, withCommand.cases[key].command))
    : el("p", { class: "none" }, "No command recorded."));
  const levels = loaded.map((run) => run.cases?.[key]?.levels).find((lv) => lv?.length);
  if (levels) put(box, el("h3", {}, "AMG levels"), levelsTable(levels));
}

function renderHeatmap(root, data, tab) {
  const params = state.route.params;
  const ids = Object.keys(data.machines);
  const machine = ids.includes(params.machine) ? params.machine : ids[0];
  put(root, el("h2", {}, "Changes by commit"));
  if (ids.length > 1) {
    put(root, el("p", { class: "machines" }, "Machine: ", ids.map((id) => el("a", {
      href: formatHash(tab, { ...params, machine: id, cell: undefined }),
      "aria-current": id === machine ? "true" : undefined,
    }, machineLabel(id)))));
  }
  const model = heatmapModel(data, machine, params);
  if (!model.rows.length || !model.columns.length) {
    put(root, el("p", { class: "none" }, "No cases to show for this selection."));
    return;
  }
  const [selKey, selCol] = (params.cell ?? "").split(/@(?=[^@]*$)/);
  const box = el("section", { class: "clickbox", "aria-live": "polite" });
  put(root,
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
    el("p", { class: "legend" }, "A dotted outline: not enough history yet, fewer than 4 earlier columns with a value, so no colour. "
      + "A corner mark: compared in raw time, because a run had no node bandwidth."),
    box);
  if (selKey && selCol !== undefined) fillBox(box, data, tab, machine, selKey, Number(selCol));
  else put(box, el("p", { class: "none" }, "Click a cell for the numbers, the node, the energy and the command."));
}

function select(label, name, options, current, tab, params) {
  return el("label", {}, label, el("select", {
    onchange: (e) => { location.hash = formatHash(tab, { ...params, [name]: e.target.value }); },
  }, options.map(([value, text]) => el("option", { value, selected: value === current }, text))));
}

function chartFrame(model, title, xLabel, yLabel) {
  const { plot, size: { width, height } } = model;
  return svg("svg", { class: "chart", viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": title },
    svg("title", {}, title),
    model.yMinor.map((pos) => svg("line", { class: "grid minor", x1: plot.x0, x2: plot.x1, y1: pos, y2: pos })),
    model.xMinor.map((pos) => svg("line", { class: "grid minor", x1: pos, x2: pos, y1: plot.y1, y2: plot.y0 })),
    model.yTicks.map((t) => [
      svg("line", { class: "grid", x1: plot.x0, x2: plot.x1, y1: t.pos, y2: t.pos }),
      svg("text", { class: "tick", x: plot.x0 - 6, y: t.pos + 5, "text-anchor": "end" }, t.label)]),
    model.xTicks.map((t) => [
      svg("line", { class: "grid", x1: t.pos, x2: t.pos, y1: plot.y1, y2: plot.y0 }),
      svg("text", { class: "tick", x: t.pos, y: plot.y0 + 18, "text-anchor": "middle" }, t.label)]),
    svg("text", { class: "axis", x: (plot.x0 + plot.x1) / 2, y: height - 6, "text-anchor": "middle" }, xLabel),
    svg("text", { class: "axis", x: 14, y: (plot.y0 + plot.y1) / 2, "text-anchor": "middle",
      transform: `rotate(-90 14 ${(plot.y0 + plot.y1) / 2})` }, yLabel));
}

// Each chart is drawn at its host's width in css pixels, so that its text
// keeps one size at any width, and its height follows the width: [ratio,
// min, max]. A host is drawn again when its width changes by more than a
// few pixels; a scroll bar coming and going does not redraw it back and forth.
const SIZES = { scatter: [0.36, 280, 460], roofline: [0.42, 320, 540], triad: [0.36, 200, 280] };
const redraws = new WeakMap();
const resizer = new ResizeObserver((entries) => {
  for (const { target } of entries) {
    if (target.isConnected) redraws.get(target)?.();
    else resizer.unobserve(target);
  }
});

// draw(size) returns the chart's svg; redraw draws it again at once, as
// after a zoom. Until the host is laid out its width is the view's.
function chartHost(kind, draw, className = "") {
  const host = el("div", { class: `chart-host ${className}` });
  let drawn = 0;
  const redraw = () => {
    const width = host.clientWidth || document.getElementById("view").clientWidth || 640;
    const [ratio, lo, hi] = SIZES[kind];
    drawn = width;
    replace(host, draw({ width, height: Math.round(Math.min(hi, Math.max(lo, width * ratio))) }));
  };
  redraws.set(host, () => { if (Math.abs(host.clientWidth - drawn) > 8) redraw(); });
  resizer.observe(host);
  redraw();
  return { host, redraw };
}

function marker(shape, cx, cy, machineIndex, title, extra = {}) {
  const cls = `dot m${machineIndex}${extra.faint ? " faint" : ""}`;
  return shape === "square"
    ? svg("rect", { class: cls, x: cx - 4, y: cy - 4, width: 8, height: 8 }, svg("title", {}, title))
    : svg("circle", { class: cls, cx, cy, r: 4.5 }, svg("title", {}, title));
}

function renderCharts(root, data, tab) {
  const params = state.route.params;
  const ids = Object.keys(data.machines);
  const machine = ids.includes(params.machine) ? params.machine : ids[0];
  const shown = data.cases.filter((c) => rowFilter(tab, c, params));
  const props = scatterProps(shown);
  const x = props.keys.includes(params.x) ? params.x : props.default;
  const hasPct = shown.some((c) => c.compare?.[machine] === "bandwidth");
  const y = params.values === "pct" && hasPct ? "pct" : ["time", "gbs", "gflops"].includes(params.values) ? params.values : hasPct ? "pct" : "time";
  const yOptions = [...(hasPct ? [["pct", "% of node bandwidth"]] : []), ["time", "time"], ["gbs", "GB/s"], ["gflops", "GFLOP/s"]];

  put(root, el("h2", {}, "This week, against a matrix property"));
  const points = (data.latest?.[machine]?.points ?? []).filter((p) => shown.some((c) => c.key === p.key));
  const model = x ? scatterModel(points, shown, { x, y }) : null;
  if (!model || !model.dots.length) {
    put(root, el("p", { class: "none" }, "No cases with a matrix property and a result this week for this selection."));
  } else {
    const series = markerSeries(Object.keys(state.meta.machines), machine); // as in the roofline
    const { host } = chartHost("scatter", (size) => {
      const sized = scatterModel(points, shown, { x, y, ...size });
      return put(chartFrame(sized, "This week, against a matrix property", x, sized.yLabel),
        svg("g", {}, sized.dots.map((d) => marker(series.shape, d.cx, d.cy, series.index, d.title, { faint: d.faint }))));
    });
    put(root,
      el("div", { class: "filters" }, select("x axis", "x", props.keys.map((k) => [k, k]), x, tab, params),
        select("y axis", "values", yOptions, y, tab, params)),
      el("p", { class: "legend" }, `${machineLabel(machine)}, this week; faint dots: the previous column. The x axis is logarithmic.`),
      host);
  }

  renderRoofline(root, data, tab);
}

function rooflineFrame(rf) {
  const order = Object.keys(state.meta.machines);
  const { plot } = rf;
  const frame = chartFrame(rf, "Roofline, this week", "intensity (flop per byte)", "GFLOP/s");
  return put(frame,
    svg("clipPath", { id: "roofline-plot" },
      svg("rect", { x: plot.x0, y: plot.y1, width: plot.x1 - plot.x0, height: plot.y0 - plot.y1 })),
    svg("g", { "clip-path": "url(#roofline-plot)" },
      rf.ceilings.map((c) => svg("polyline", { class: `ceiling m${order.indexOf(c.machine)}`, points: c.points, fill: "none" },
        svg("title", {}, `${machineLabel(c.machine)} ceiling`))),
      rf.dots.map((d) => {
        const series = markerSeries(order, d.machine);
        return marker(series.shape, d.cx, d.cy, series.index, d.title);
      })));
}

// The roofline, zoomed and panned in place: the view of each tab is kept in
// state.views while the page is open, null for the whole roofline. The host
// stays while the svg inside it is redrawn, so a drag keeps its pointer.
function renderRoofline(root, data, tab) {
  put(root, el("h2", {}, "Roofline, this week"));
  const model = (view, size) => rooflineModel(data.latest, state.meta.machines, data.cases, { ...size, view });
  let rf = model(state.views[tab]);
  if (!rf.total) {
    put(root, el("p", { class: "none" }, "No results with a known flops-per-byte this week."));
    return;
  }
  const count = el("span", { class: "legend" });
  const { host, redraw } = chartHost("roofline", (size) => {
    rf = model(state.views[tab], size);
    replace(count, rf.dots.length < rf.total ? `${rf.dots.length} of ${rf.total} results in view` : "");
    return rooflineFrame(rf);
  }, "zoomable");
  const draw = (view) => {
    state.views[tab] = view;
    redraw();
  };
  let next = null; // the view of the next animation frame
  const later = (view) => {
    if (!next) requestAnimationFrame(() => { const v = next; next = null; draw(v); });
    next = view;
  };
  const at = (e) => plotFraction(rf.plot, rf.size, host.firstChild.getBoundingClientRect(), e.clientX, e.clientY);
  const clamp = (f) => Math.min(1, Math.max(0, f));

  host.addEventListener("wheel", (e) => {
    if (!e.ctrlKey && !e.metaKey) return; // a plain wheel scrolls the page; a pinch comes with ctrlKey
    e.preventDefault();
    const { fx, fy } = at(e);
    const pixels = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
    later(zoomView(next ?? rf.view, rf.domain, Math.exp(pixels * 0.002), clamp(fx), clamp(fy)));
  }, { passive: false });
  let drag = null;
  host.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    drag = { id: e.pointerId, from: at(e), view: rf.view };
    host.setPointerCapture(e.pointerId);
    host.classList.add("dragging");
  });
  host.addEventListener("pointermove", (e) => {
    if (drag?.id !== e.pointerId) return;
    const p = at(e);
    later(panView(drag.view, rf.domain, drag.from.fx - p.fx, drag.from.fy - p.fy));
  });
  const drop = (e) => {
    if (drag?.id !== e.pointerId) return;
    drag = null;
    host.classList.remove("dragging");
  };
  host.addEventListener("pointerup", drop);
  host.addEventListener("pointercancel", drop);
  host.addEventListener("dblclick", () => draw(null));

  const button = (text, title, onclick) => el("button", { type: "button", title, onclick }, text);
  const order = Object.keys(state.meta.machines);
  put(root,
    el("p", { class: "legend" }, order.map((id, i) => `${i === 0 ? "circles" : "squares"}: ${machineLabel(id)}`).join("; ")
      + ". Lines: measured triad bandwidth times intensity, up to the fp64 peak."
      + " Drag to pan; ctrl or ⌘ and the wheel, or a pinch, zoom; a double click shows the whole roofline."),
    el("div", { class: "filters zoom" },
      button("+", "zoom in", () => draw(zoomView(rf.view, rf.domain, 0.5))),
      button("−", "zoom out", () => draw(zoomView(rf.view, rf.domain, 2))),
      button("fit the results", "zoom to the results and the ceilings above them", () => draw(rf.fit)),
      button("whole roofline", "zoom out to the whole roofline", () => draw(null)),
      count),
    host);
}
const STATE_TEXT = (state) => String(state).replaceAll("_", " ");
const num = (x, render) => (x === null || x === undefined ? "–" : render(x));
const ms = (x) => formatValue(x / 1e3, "time");

function machineOf(data) {
  const ids = Object.keys(data.machines);
  return ids.includes(state.route.params.machine) ? state.route.params.machine : ids[0];
}

// the operation select, and the format select for spmv only
function renderBasicFilters(root, data) {
  const params = state.route.params;
  const operations = [...new Set(data.cases.map((c) => c.labels.operation))];
  const operation = operations.includes(params.operation) ? params.operation : "spmv";
  const formats = [...new Set(data.cases.filter((c) => c.labels.operation === "spmv").map((c) => c.labels.format))];
  const format = formats.includes(params.format) ? params.format : "csr";
  put(root, el("div", { class: "filters" },
    select("operation", "operation", operations.map((o) => [o, o]), operation, "basic", params),
    operation === "spmv" ? select("format", "format", formats.map((f) => [f, f]), format, "basic", params) : null));
}

// apply or generate: the spec shows the two stages separately
function renderStageFilter(root) {
  const params = state.route.params;
  const stage = params.stage === "generate" ? "generate" : "apply";
  put(root, el("div", { class: "filters" },
    select("stage", "stage", [["apply", "apply"], ["generate", "generate"]], stage, "preconditioners", params)));
}

function renderSolverTable(root, data) {
  const machine = machineOf(data);
  put(root, el("h2", {}, "This week"),
    el("p", { class: "legend" }, `${machineLabel(machine)}, the latest column; times are per iteration.`),
    table(["solver", "stencil", "iterations", "time per iteration", "bandwidth", "of node bandwidth", "residual"],
      solverRows(data, machine).map((r) => [r.solver, r.stencil, num(r.iterations, String),
        num(r.msPerIteration, ms), num(r.gbs, (v) => formatValue(v, "gbs")), num(r.pct, (v) => formatValue(v, "pct")),
        num(r.residual, (v) => v.toExponential(2))]), "this-week"));
}

// parameters as a nested list; a changed path (a.b for keys, a[0] for list items, as the build writes it) is marked; seen collects the paths shown
function parameterList(value, path, changed, seen) {
  if (value === null || typeof value !== "object") return null;
  const entries = Array.isArray(value) ? value.map((v, i) => [String(i), v]) : Object.entries(value);
  return el("ul", { class: "params" }, entries.map(([name, v]) => {
    const here = childPath(path, name, Array.isArray(value));
    seen.add(here);
    const mark = changed.get(here);
    const nested = parameterList(v, here, changed, seen);
    return el("li", { class: mark ? "changed" : undefined },
      el("span", { class: "param" }, name), nested ? "" : `: ${JSON.stringify(v)}`,
      mark ? el("span", { class: "was" }, ` (changed from ${JSON.stringify(mark.before)} to ${JSON.stringify(mark.after)})`) : null,
      nested);
  }));
}

function renderSettings(root, data, names) {
  const known = names.filter((n) => data.settings?.[n]);
  if (!known.length) return;
  put(root, el("h2", {}, "Settings"),
    el("p", { class: "legend" }, "The parameters of the latest run. Changed values are marked."),
    known.map((name) => {
      const { parameters, changed = [] } = data.settings[name];
      const paths = new Map(changed.map((c) => [c.path, c]));
      const seen = new Set();
      const list = Object.keys(parameters ?? {}).length ? parameterList(parameters, "", paths, seen)
        : el("p", { class: "none" }, "No parameters recorded.");
      return el("details", { class: "detail settings", open: changed.length > 0 },
        el("summary", {}, changed.length ? `${name} (${changed.length} changed)` : name),
        list,
        changed.filter((c) => !seen.has(c.path)).map((c) =>
          el("p", { class: "changed" }, `${c.path}: changed from ${JSON.stringify(c.before)} to ${JSON.stringify(c.after)}`)));
    }));
}

function renderPreconditionerTable(root, data, tab) {
  const params = state.route.params;
  const machine = machineOf(data);
  const matrices = [...new Set(data.cases.map((c) => c.labels.matrix))];
  const matrix = matrices.includes(params.matrix) ? params.matrix : matrices[0];
  put(root, el("h2", {}, "This week"),
    el("div", { class: "filters" }, select("matrix", "matrix", matrices.map((m) => [m, m]), matrix, tab, params)),
    el("p", { class: "legend" }, `${machineLabel(machine)}, the latest column.`),
    table(["preconditioner", "generate", "apply", "storage", "levels", "state"],
      preconditionerRows(data, machine, matrix).map((r) => [r.preconditioner,
        num(r.generateMs, ms), num(r.applyMs, ms), num(r.storageMb, (v) => `${v.toFixed(1)} MB`),
        num(r.levels, String), ["history", "same", "slower", "faster"].includes(r.state) ? "" : STATE_TEXT(r.state)]), "this-week"));
  renderSettings(root, data, [...new Set(data.cases.map((c) => c.labels.preconditioner))]);
}

function renderTabTables(root, data, tab) {
  if (tab === "solvers") {
    renderSolverTable(root, data);
    renderSettings(root, data, [...new Set(data.cases.map((c) => c.labels.solver))]);
  } else if (tab === "preconditioners") renderPreconditionerTable(root, data, tab);
}

const fmt = (x, kind) => (x === null || x === undefined ? "–" : formatValue(x, kind));
const joules = (x) => (x === null || x === undefined ? "–" : Math.round(x).toLocaleString("en"));

// nested object -> nested list; keys and values are text nodes
function nestedList(value) {
  if (value === null || typeof value !== "object") return String(value);
  return el("ul", { class: "nested" }, Object.entries(value).map(([k, v]) =>
    el("li", {}, typeof v === "object" && v !== null ? [el("strong", {}, k), nestedList(v)] : `${k}: ${v}`)));
}

function renderSetup(setup) {
  const rows = [
    ["GPU", setup.gpu], ["Driver", setup.driver],
    ["Power limit", setup.power_limit_w != null ? `${setup.power_limit_w} W` : null],
    ["Partition", setup.partition], ["Binding", setup.binding ? `${(setup.binding.srun ?? []).join(" ")} (cpus ${setup.binding.cpus ?? "?"})` : null],
    ["uenv", setup.uenv], ["Compilers", Object.values(setup.toolchain ?? {}).join("; ")],
    ["Dataset version", setup.dataset_version],
    ["Script version", shortVersion(setup.script_version)],
  ].filter(([, v]) => v !== null && v !== undefined && v !== "");
  return [
    el("dl", { class: "values setup" }, rows.map(([k, v]) => [el("dt", {}, k),
      typeof v === "object" ? el("dd", { title: v.title }, v.text) : el("dd", {}, v)])),
    el("details", { class: "detail" }, el("summary", {}, "CMake flags"),
      el("pre", {}, Object.entries(setup.cmake_cache ?? {}).filter(([, v]) => v !== "").map(([k, v]) => `${k}=${v}`).join("\n"))),
    el("details", { class: "detail" }, el("summary", {}, "Benchmark settings"),
      el("dl", { class: "values" }, Object.entries(setup.settings ?? {}).map(([k, v]) => [el("dt", {}, k), el("dd", {}, String(v))]))),
  ];
}

function renderTriad(m) {
  const model = triadChartModel(m.triad);
  const out = [];
  if (!model.dots.length) out.push(el("p", { class: "none" }, "No run has a node bandwidth yet."));
  else {
    const nodes = [...new Set(m.triad.map((t) => t.node))];
    const { host } = chartHost("triad", (size) => {
      const sized = triadChartModel(m.triad, size);
      return put(chartFrame(sized, "Node bandwidth over time", "run", "triad GB/s"),
        svg("g", {}, sized.dots.map((d) => svg("circle", { class: `dot ${d.cls}`, cx: d.cx, cy: d.cy, r: 4.5 }, svg("title", {}, d.title)))));
    });
    out.push(host, el("p", { class: "legend" }, "Colour is the node: ",
      nodes.map((n, i) => el("span", { class: "key" }, el("span", { class: `swatch n${i % 6}` }), n)), ". Only runs with a measured triad are shown."));
  }
  out.push(el("details", { class: "detail" }, el("summary", {}, "Nodes seen"),
    table(["node", "runs", "min", "median", "max"], m.nodes.map((n) => [n.node, n.runs, fmt(n.triad_min, "gbs"), fmt(n.triad_median, "gbs"), fmt(n.triad_max, "gbs")])),
    m.nodes.filter((n) => n.gpu_after).map((n) => el("details", { class: "detail" }, el("summary", {}, `${n.node}: last gpu_after`), nestedList(n.gpu_after)))));
  return out;
}

function renderEnergy(id, m) {
  const rows = energyRows(m.groups);
  const out = [];
  if (!rows.some((r) => r.gpuJ !== null || r.nodeJ !== null)) out.push(el("p", { class: "none" }, "The latest run has no energy readings."));
  else {
    out.push(table(["group", "seconds", "GPU J", "GPU W", "node J"], rows.map((r) =>
      [r.group, r.seconds != null ? r.seconds.toFixed(0) : "–", joules(r.gpuJ), r.gpuW !== null ? r.gpuW.toFixed(0) : "–", joules(r.nodeJ)]), "energy"));
    if (id.startsWith("beverin")) out.push(el("p", { class: "legend" }, "On the MI300A, accel0 is the whole APU, CPU included, so GPU J includes the CPU."));
  }
  if (rows.some((r) => r.cpuJ !== null)) {
    out.push(el("details", { class: "detail" }, el("summary", {}, "CPU energy"),
      table(["group", "CPU J"], rows.map((r) => [r.group, joules(r.cpuJ)]))));
  }
  out.push(el("h4", {}, "Run durations"), el("ul", { class: "runs" }, m.runs.map((r) =>
    el("li", {}, `${r.dir}: ${r.status}, ${formatSeconds(r.seconds)}${r.failed ? `, ${r.failed} failed` : ""}${r.killed ? ", killed" : ""}`))));
  return out;
}

async function renderMachines(root, token) {
  let data;
  try {
    if (!state.machines) state.machines = await loadJson("data/machines.json");
    data = state.machines;
  } catch (error) {
    if (tokens.isLatest(token)) showBanner("error", `Could not load the machines data: ${error.message}`);
    return;
  }
  if (!tokens.isLatest(token)) return;
  const ids = Object.keys(data.machines);
  const column = (title, build) => [el("h2", {}, title),
    el("div", { class: "machine-grid" }, ids.map((id) => el("section", { class: "machine" }, el("h3", {}, machineLabel(id)), build(id, data.machines[id]))))];
  put(root,
    column("Setup", (id, m) => renderSetup(m.setup)),
    column("Node bandwidth over time", (id, m) => renderTriad(m)),
    column("Runs and energy", (id, m) => renderEnergy(id, m)));
}

function renderTab(root, tab, data) {
  renderFlags(root, data, tab);
  renderUnchanged(root, data);
  if (tab === "basic") renderBasicFilters(root, data);
  if (tab === "preconditioners") renderStageFilter(root);
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
  replace(root);
  if (tab === "machines") {
    await renderMachines(root, token);
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
  // a cell selected from a flag link or a pasted link: show its click box
  if (revealCell(state.shown, state.route)) document.querySelector(".clickbox")?.scrollIntoView({ block: "nearest" });
  state.shown = state.route;
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
