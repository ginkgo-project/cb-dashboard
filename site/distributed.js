// Renders the Distributed tab from site/data/distributed.json: the change table
// of two weekly runs of a machine, and an opened row's plots and details (the
// run details are site/data/distributed/<machine>/<dir>.json, loaded when a
// row opens). The DOM helpers come from app.js, so all text goes through text
// nodes there too.
import { GPU_COUNTS, GPUS_PER_NODE, changeTable, defaultPair, efficiency, efficiencyAxis, formatBytes, formatRate,
  logPlot, mpiMap, mpiParts, pickRun, rowLabel } from "./distributed-logic.js";
import { formatValue } from "./logic.js";

const cache = { data: null, runs: new Map() };
const N = GPU_COUNTS.map(String);
const fmtTime = (v) => (v === null || v === undefined ? "–" : formatValue(v, "time"));
const fmtNum = (v, digits = 0) => (v === null || v === undefined ? "–"
  : Number(v).toLocaleString("en", { maximumFractionDigits: digits, minimumFractionDigits: digits }));
const fmtEff = (v) => (v === null || v === undefined ? "–" : `${Math.round(v * 100)}%`);
const shortSha = (sha) => (sha || "").slice(0, 7);

async function runDetail(ui, machine, dir) {
  const key = `${machine}/${dir}`;
  if (!cache.runs.has(key)) {
    cache.runs.set(key, ui.loadJson(`data/distributed/${encodeURIComponent(machine)}/${encodeURIComponent(dir)}.json`)
      .catch(() => null));
  }
  return cache.runs.get(key);
}

export async function renderDistributed(root, ui) {
  const { el, put, formatHash, params } = ui;
  if (!cache.data) cache.data = await ui.loadJson("data/distributed.json");
  if (!ui.isLatest()) return;
  const data = cache.data;
  const ids = Object.keys(data.machines);
  put(root, el("h2", {}, "Weak scaling, 1 to 64 GPUs"));
  if (!ids.length) {
    put(root, el("p", { class: "none dist-none" },
      "No distributed runs yet. They run on Saturday nights, on develop's head, from 1 to 64 GPUs in one job."));
    return;
  }
  const machine = ids.includes(params.machine) ? params.machine : ids[0];
  const m = data.machines[machine];
  const [da, db] = defaultPair(m.runs);
  const ib = pickRun(m.runs, params.b, db);
  const ia = pickRun(m.runs, params.a, da);
  const link = (extra) => formatHash("distributed", { ...params, machine, ...extra });
  if (ids.length > 1) {
    put(root, el("p", { class: "machines" }, "Machine: ", ids.map((id) => el("a", {
      href: formatHash("distributed", { machine: id }), "aria-current": id === machine ? "true" : undefined,
    }, ui.machineLabel(id)))));
  }
  const runText = (r) => `${r.date} ${shortSha(r.commit)}`;
  const runSelect = (name, current, allowNone) => el("select", {
    "aria-label": name === "a" ? "earlier run" : "later run",
    onchange: (e) => { location.hash = link({ [name]: e.target.value || undefined }); },
  }, allowNone && current === null ? el("option", { value: "", selected: true }, "no earlier run") : null,
  m.runs.map((r, i) => el("option", { value: r.dir, selected: i === current }, runText(r))));
  put(root, el("p", { class: "dist-controls" }, "Compare ", runSelect("a", ia, true), " with ", runSelect("b", ib, false)));
  put(root, el("p", { class: "legend" },
    "Each cell: the change in time at that GPU count, from the first run to the second; + is slower. "
    + "Red: slower, green: faster, beyond the noise band of the earlier weekly runs (at least ±3%); grey: within it. "
    + "Hatched: failed or skipped. Solver apply rows are per iteration. Click a row for its curves."));

  const rows = changeTable(m, ia, ib);
  const body = el("tbody");
  for (const row of rows) {
    const opened = params.row === row.key;
    const toggle = () => { location.hash = link({ row: opened ? undefined : row.key, n: undefined }); };
    put(body, el("tr", { class: opened ? "opened" : undefined },
      el("th", { scope: "row" }, el("a", { href: link({ row: opened ? undefined : row.key, n: undefined }) },
        `${opened ? "▾" : "▸"} ${row.label}`)),
      row.cells.map((cell) => el("td", {
        class: cell.cls, title: cell.title, tabindex: "0", onclick: toggle,
        onkeydown: (e) => { if (e.key === "Enter") toggle(); },
      }, cell.text))));
    if (opened) {
      const box = el("section", { class: "clickbox dist-open", "aria-live": "polite" });
      put(body, el("tr", { class: "dist-open-row" }, el("td", { colspan: String(GPU_COUNTS.length + 1) }, box)));
      const fill = row.mpi ? openMpi : openBenchmark;
      fill(box, ui, { m, machine, ia, ib, row, rows, link });
    }
  }
  put(root, el("div", { class: "scroll" }, el("table", { class: "heatmap dist" },
    el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "GPUs"),
      GPU_COUNTS.map((n) => el("th", { scope: "col", title: `${Math.ceil(n / GPUS_PER_NODE)} node(s)` }, String(n))))),
    body)));
}

function runLine(ui, m, ia, ib) {
  const { el } = ui;
  const one = (i) => {
    const r = m.runs[i];
    return [`${r.date} `, el("code", {}, shortSha(r.commit)),
      r.ci?.job_url ? [" (", el("a", { href: r.ci.job_url }, "CI job"), ")"] : null];
  };
  return el("p", { class: "dist-runs" }, one(ib), ia === null ? " · no earlier run" : [" against ", one(ia)]);
}

function legendLine(ui, m, ia, ib) {
  const { el, svg } = ui;
  const sw = (cls) => svg("svg", { class: "dist-swatch", width: 22, height: 8, "aria-hidden": "true" },
    svg("line", { class: `line ${cls}`, x1: 0, x2: 22, y1: 4, y2: 4 }));
  return el("p", { class: "legend" }, sw("r1"), ` ${m.runs[ib].date}`,
    ia === null ? null : ["  ", sw("r0"), ` ${m.runs[ia].date}`]);
}

// one plot: `model` from logPlot; the series are [earlier run, later run]
function plot(ui, model, title, yLabel, xLabel, pointLabel, nodeLine) {
  const { svg } = ui;
  const { plot: p, size } = model;
  const nodes = nodeLine ? (model.x(2) + model.x(3)) / 2 : null;
  return svg("svg", { class: "chart dist-chart", viewBox: `0 0 ${size.width} ${size.height}`, role: "img", "aria-label": title },
    svg("title", {}, title),
    model.yTicks.map((t) => [
      svg("line", { class: "grid", x1: p.x0, x2: p.x1, y1: t.pos, y2: t.pos }),
      svg("text", { class: "tick", x: p.x0 - 6, y: t.pos + 4, "text-anchor": "end" }, t.label)]),
    model.xTicks.map((t) => svg("text", { class: "tick", x: t.pos, y: p.y0 + 15, "text-anchor": "middle" }, t.label)),
    svg("text", { class: "axis", x: (p.x0 + p.x1) / 2, y: size.height - 3, "text-anchor": "middle" }, xLabel),
    svg("text", { class: "axis", x: p.x0, y: 14 }, yLabel),
    nodes === null ? null : [
      svg("line", { class: "nodes", x1: nodes, x2: nodes, y1: p.y1, y2: p.y0 }),
      svg("text", { class: "tick", x: nodes + 4, y: p.y1 + 10 }, "2+ nodes")],
    model.paths.map((path) => {
      const cls = path.index === 1 ? "r1" : "r0";
      return [
        svg("polyline", { class: `line ${cls}`, points: path.points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ") }),
        path.points.map(([x, y, v, i]) => svg("circle", { class: `pt ${cls}`, cx: x.toFixed(1), cy: y.toFixed(1), r: 4 },
          svg("title", {}, pointLabel(v, i, path.index))))];
    }));
}

const series = (row, i) => N.map((n) => (i === null ? null : row[n]?.time[i] ?? null));

async function openBenchmark(box, ui, { m, machine, ia, ib, row, rows }) {
  const { el, put } = ui;
  const [detB, detA] = await Promise.all([runDetail(ui, machine, m.runs[ib].dir),
    ia === null ? null : runDetail(ui, machine, m.runs[ia].dir)]);
  if (!box.isConnected) return;
  const perGpu = (det) => N.map((n, i) => (det?.gpus?.[n]?.rows ? det.gpus[n].rows / GPU_COUNTS[i] : null));
  const g1 = detB?.gpus?.["1"] ?? {};
  const tB = series(m.rows[row.key], ib);
  const tA = series(m.rows[row.key], ia);
  const eB = efficiency(tB, perGpu(detB));
  const eA = efficiency(tA, perGpu(detA));
  const label = (v, i, s) => `${GPU_COUNTS[i]} GPUs, ${m.runs[s === 1 ? ib : ia].date}: ${fmtTime(v)}`;
  const effLabel = (v, i, s) => `${GPU_COUNTS[i]} GPUs, ${m.runs[s === 1 ? ib : ia].date}: ${fmtEff(v)}`;
  const size = { width: 440, height: 250 };
  const timeModel = logPlot([tA, tB], { ...size, xLabels: N });
  const effModel = logPlot([eA, eB], { ...size, xLabels: N, linear: efficiencyAxis([...eA, ...eB]), format: fmtEff });
  put(box,
    el("p", {}, el("strong", {}, row.label), ` on ${ui.machineLabel(machine)}: 3D 7-point stencil, `
      + `${fmtNum(g1.rows)} rows and ${fmtNum(g1.nonzeros)} nonzeros per GPU · ${GPUS_PER_NODE} GPUs per node`),
    runLine(ui, m, ia, ib), legendLine(ui, m, ia, ib),
    el("div", { class: "dist-plots" },
      el("div", {}, plot(ui, timeModel, `${row.label}: time`, "time, log", "GPUs", label, true),
        el("table", { class: "heatmap dist-strip" }, el("tbody", {}, el("tr", {}, el("th", { scope: "row" }, "change"),
          row.cells.map((c) => el("td", { class: c.cls, title: c.title }, c.text)))))),
      el("div", {}, plot(ui, effModel, `${row.label}: weak-scaling efficiency`, "efficiency, per row", "GPUs", effLabel, true))),
    el("div", { class: "dist-blocks" },
      block(ui, "The numbers", numbersTable(ui, m, ia, ib, tA, tB, eA, eB), "wide"),
      block(ui, "MPI at the same GPU counts", mpiTable(ui, rows), "wide"),
      row.key.startsWith("mg-") ? block(ui, "Hierarchy", hierarchyTable(ui, row.key.split("/")[0], detA, detB, ia, m, ib)) : null,
      /^(cg|gmres)\//.test(row.key) ? block(ui, "Iterations", iterationsTable(ui, row.key.split("/")[0], detA, detB, ia, m, ib)) : null));
}

function block(ui, title, content, cls) {
  const { el } = ui;
  return el("section", { class: cls }, el("h4", {}, title), el("div", { class: "scroll" }, content));
}

function simpleTable(ui, head, body) {
  const { el } = ui;
  return el("table", { class: "dist-numbers" },
    el("thead", {}, el("tr", {}, head.map((h) => el("th", { scope: "col" }, h)))),
    el("tbody", {}, body.map((cells) => el("tr", {}, cells.map((c, i) => (i === 0 ? el("th", { scope: "row" }, c) : el("td", {}, c)))))));
}

function numbersTable(ui, m, ia, ib, tA, tB, eA, eB) {
  const date = (i) => (i === null ? "–" : m.runs[i].date);
  const change = (a, b) => (a && b ? `${b / a >= 1 ? "+" : "−"}${Math.round(Math.abs(b / a - 1) * 100)}%` : "–");
  return simpleTable(ui, ["GPUs", date(ib), date(ia), "change", `efficiency ${date(ib)}`, `efficiency ${date(ia)}`],
    N.map((n, i) => [n, fmtTime(tB[i]), fmtTime(tA[i]), change(tA[i], tB[i]), fmtEff(eB[i]), fmtEff(eA[i])]));
}

function mpiTable(ui, rows) {
  const { el } = ui;
  const mpi = rows.filter((r) => r.mpi);
  if (!mpi.length) return el("p", { class: "none" }, "No MPI measurements in these runs.");
  return el("table", { class: "heatmap dist-small" },
    el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "GPUs"), N.map((n) => el("th", { scope: "col" }, n)))),
    el("tbody", {}, mpi.map((r) => el("tr", {}, el("th", { scope: "row" }, r.label),
      r.cells.map((c) => el("td", { class: c.cls, title: c.title }, c.text))))));
}

function hierarchyTable(ui, name, detA, detB, ia, m, ib) {
  const lv = (det, n) => det?.gpus?.[n]?.levels?.[name] ?? {};
  const both = (b, a) => (ia === null || a === b ? b : `${b} (${a})`);
  return [simpleTable(ui, ["GPUs", "levels", "operator complexity", "coarsest rows"], N.map((n) => {
    const b = lv(detB, n);
    const a = lv(detA, n);
    return [n, both(fmtNum(b.levels), fmtNum(a.levels)), both(fmtNum(b.operator_complexity, 3), fmtNum(a.operator_complexity, 3)),
      both(fmtNum(b.coarsest_rows), fmtNum(a.coarsest_rows))];
  })), ui.el("p", { class: "legend" }, ia === null ? `Run ${m.runs[ib].date}.`
    : `Run ${m.runs[ib].date}; in brackets ${m.runs[ia].date}, where it differs.`)];
}

function iterationsTable(ui, name, detA, detB, ia, m, ib) {
  const date = (i) => (i === null ? "–" : m.runs[i].date);
  const its = (det, n) => det?.gpus?.[n]?.iterations?.[name] ?? null;
  const per = (det, n) => {
    const t = det?.gpus?.[n]?.solver_time?.[name];
    const i = its(det, n);
    return t && i ? t / i : null;
  };
  return simpleTable(ui, ["GPUs", `iterations ${date(ib)}`, `iterations ${date(ia)}`, `per iteration ${date(ib)}`, `per iteration ${date(ia)}`],
    N.map((n) => [n, fmtNum(its(detB, n)), fmtNum(its(detA, n)), fmtTime(per(detB, n)), fmtTime(per(detA, n))]));
}

function openMpi(box, ui, { m, ia, ib, row, link }) {
  const { el, put } = ui;
  const { op } = mpiParts(row.key);
  const map = mpiMap(m, op, ia, ib);
  const withData = GPU_COUNTS.filter((n) => map.rows.some((r) => r.cells[GPU_COUNTS.indexOf(n)].state !== "missing"));
  const chosen = Number(ui.params.n);
  const n = withData.includes(chosen) ? chosen : withData[withData.length - 1] ?? 64;
  const at = GPU_COUNTS.indexOf(n);
  const tB = map.sizes.map((s) => m.rows[`mpi/${op}/${s}`][String(n)]?.time[ib] ?? null);
  const tA = map.sizes.map((s) => (ia === null ? null : m.rows[`mpi/${op}/${s}`][String(n)]?.time[ia] ?? null));
  const bw = (ts) => ts.map((t, i) => (t ? map.sizes[i] / t : null));
  const xLabels = map.sizes.map(formatBytes);
  const size = { width: 440, height: 250 };
  const date = (s) => m.runs[s === 1 ? ib : ia].date;
  const opName = rowLabel(row.key).split(",")[0];
  put(box,
    el("p", {}, el("strong", {}, opName), `, on GPU buffers: the change of each message size (per rank) at each GPU count.`),
    runLine(ui, m, ia, ib),
    el("div", { class: "dist-plots" },
      el("div", { class: "scroll" }, el("table", { class: "heatmap dist-small" },
        el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "size \\ GPUs"), N.map((g) => el("th", { scope: "col" }, g)))),
        el("tbody", {}, map.rows.map((r) => el("tr", {}, el("th", { scope: "row" }, formatBytes(r.bytes)),
          r.cells.map((c, i) => el("td", {
            class: `${c.cls}${i === at ? " selected" : ""}`, title: c.title,
            onclick: () => { location.hash = link({ n: String(GPU_COUNTS[i]) }); },
          }, c.text))))))),
      el("div", {},
        el("p", { class: "dist-chips" }, "GPUs: ", withData.map((g) => el("a", {
          href: link({ n: String(g) }), "aria-current": g === n ? "true" : undefined }, String(g)))),
        legendLine(ui, m, ia, ib),
        plot(ui, logPlot([tA, tB], { ...size, xLabels, labelEvery: 3 }), `${opName} at ${n} GPUs: time`, "time, log",
          "message size", (v, i, s) => `${xLabels[i]}, ${date(s)}: ${fmtTime(v)}`, false),
        plot(ui, logPlot([bw(tA), bw(tB)], { ...size, xLabels, labelEvery: 3, format: formatRate }),
          `${opName} at ${n} GPUs: bandwidth`, "size / time, log", "message size",
          (v, i, s) => `${xLabels[i]}, ${date(s)}: ${formatRate(v)}`, false))));
}
