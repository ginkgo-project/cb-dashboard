import json
import shutil
import subprocess
from pathlib import Path

import pytest
from dash_fakes import write_manifest, write_run

from dashboard.__main__ import main
from dashboard.models import load_machines
from dashboard.site_data import TABS, build_site

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DAINT, BEV = "daint-gh200", "beverin-mi300a"
COMMITS = [("c1" * 20, "2026-10-01"), ("c2" * 20, "2026-10-08"),
           ("c3" * 20, "2026-10-15")]
SPMV_KEY = "spmv/csr/cage15"
BLAS_ARGS = ["-warmup", "1", "-repetitions", "5", "-gpu_timer",
             "-timer_method", "median", "-seed", "42", "-operations", "dot"]
CRITERIA = [{"type": "ResidualNorm", "baseline": "rhs_norm",
             "reduction_factor": 1e-10},
            {"type": "Iteration", "max_iters": 1000}]


def _inv(label, args, status="ok", energy=None):
    inv = {"label": label, "args": args, "status": status, "reason": None,
           "seconds": 2.0}
    if energy:
        inv["energy_j"] = energy
    return inv


def _groups(solver_ok):
    def g(status, *invs):
        return {"status": status, "seconds": 1.0, "invocations": list(invs)}
    return {
        "blas": g("ok", _inv("vectors", BLAS_ARGS,
                             energy={"accel0_energy": 10, "energy": 50})),
        "spmv": g("ok", _inv("cage15", ["-formats", "csr", "-input_matrix",
                                        "/m/cage15.bin"])),
        "spgemm": {"status": "skipped", "reason": "not requested"},
        "solver": g("ok" if solver_ok else "partial",
                    _inv("mgs", ["-solvers", "cg"],
                         "ok" if solver_ok else "partial")),
        "preconditioner": g("ok", _inv("pp/jacobi", [
            "-preconditioners", "jacobi", "-input_matrix", "/m/pp.bin"])),
    }


def _data(i, solver_ok):
    cg = {"completed": True, "apply": {"time": 0.1, "iterations": 10},
          "generate": {"time": 0.01}, "rhs_norm": 1.0,
          "residual_norm": 1e-10, "parameters": {"criteria": CRITERIA}}
    if not solver_ok:
        cg = {"completed": False, "error_type": "gko::Error", "error": "boom",
              "parameters": {"criteria": CRITERIA}}
    later = i == 2
    jacobi = {"completed": True, "storage": 8000, "apply": {"time": 1e-4},
              "generate": {"time": 1e-3},
              "parameters": {"max_block_size": 32,
                             "relax": [0.9, 0.5 if later else 0.0],
                             "coarse": {"iters": 2 if later else 1}}}
    return {
        "blas": [{"n": 1048576, "blas": {"dot": {
            "time": 1e-5, "repetition_times": [1e-5], "flops": 2e11,
            "bandwidth": 1.6e12, "completed": True}}}],
        "spmv": [{"filename": "/m/cage15.bin", "rows": 1000, "cols": 1000,
                  "nonzeros": 5000, "spmv": {"csr": {
                      "time": 1e-4 * (1 + i / 10), "repetition_times": [1e-4],
                      "completed": True}}}],
        "solver": [{"size": 1000, "stencil": "7pt",
                    "optimal": {"spmv": "csr"}, "rows": 1000,
                    "solver": {"cg-jacobi": cg}}],
        "preconditioner": [{"filename": "/m/pp.bin", "rows": 1000,
                            "cols": 1000, "nonzeros": 5000,
                            "preconditioner": {"jacobi-32-0,0": jacobi}}],
    }


def _matrix(name, set_, roles):
    return {"name": name, "set": set_, "roles": roles, "source": "test",
            "rows": 1000, "nonzeros": 5000,
            "row_length": {"min": 1, "max": 9, "mean": 5.0, "std": 1.0,
                           "cv": 0.2},
            "spd": True if set_ == "fem" else None, "run_sellp": True}


def _fake(tmp):
    """Two machines, three columns; the Beverin solver fails in the last."""
    results = tmp / "results"
    for machine in (DAINT, BEV):
        for i, (commit, date) in enumerate(COMMITS):
            solver_ok = not (machine == BEV and i == 2)
            write_run(results, machine, f"{date}_{commit[:10]}",
                      commit=commit, date=date,
                      started=f"{date}T10:00:00+00:00",
                      groups=_groups(solver_ok), data=_data(i, solver_ok),
                      extra={"finished": f"{date}T10:10:00+00:00"})
    manifest = write_manifest(tmp / "manifest.json", [
        _matrix("cage15", "spmv", []), _matrix("pp", "fem", ["precond"]),
        _matrix("poisson_p1", "fem", ["spgemm"])])
    catalog = tmp / "catalog.toml"
    catalog.write_text('[[entry]]\nname = "pp"\n'
                       'fem = {problem = "porous_pressure", size = 8}\n')
    out = tmp / "out"
    build_site(results, manifest, catalog, out, machines=load_machines(),
               frozen={}, script_changed=lambda a, b: False, warnings=[])
    return out


@pytest.fixture
def built(tmp_path):
    return _fake(tmp_path)


def _read(out, name):
    return json.loads((out / name).read_text())


def test_build_writes_every_file(built):
    for name in ("meta.json", "basic.json", "solvers.json",
                 "preconditioners.json", "machines.json"):
        assert (built / name).is_file(), name
    runs = sorted(p.relative_to(built / "runs").as_posix()
                  for p in (built / "runs").glob("*/*.json"))
    assert runs == sorted(f"{m}/{date}_{commit[:10]}.json"
                          for m in (DAINT, BEV) for commit, date in COMMITS)


def test_cells_align_with_columns(built):
    for tab in TABS:
        data = _read(built, f"{tab}.json")
        assert data["tab"] == tab and data["cases"]
        for machine in (DAINT, BEV):
            mach = data["machines"][machine]
            assert len(mach["columns"]) == 3
            for case in data["cases"]:
                assert len(mach["cells"][case["key"]]) == len(mach["columns"])


def test_flag_counts_match_flags(built):
    meta = _read(built, "meta.json")
    for tab in TABS:
        assert meta["flag_counts"][tab] == len(_read(built, f"{tab}.json")["flags"])
    assert meta["flag_counts"]["solvers"] == 1
    assert _read(built, "solvers.json")["flags"] == [
        {"key": "solver/cg/7pt", "machine": BEV, "column": 2,
         "state": "new_failure"}]


def test_run_detail_has_commands(built):
    run = _read(built, f"runs/{DAINT}/2026-10-15_c3c3c3c3c3.json")
    case = run["cases"][SPMV_KEY]
    assert case["command"].startswith("srun")
    assert case["invocation"] == "spmv:cage15"
    assert run["invocations"]["blas:vectors"]["energy_j"]["accel0_energy"] == 10


def test_cases_props_and_compare(built):
    basic = _read(built, "basic.json")
    assert [c["key"] for c in basic["cases"]] == ["blas/dot/1048576", SPMV_KEY]
    spmv = basic["cases"][1]
    assert spmv["model"] == "spmv-csr"
    assert spmv["props"] == {"rows": 1000, "nonzeros": 5000, "nnz_per_row": 5.0,
                             "max_row": 9, "row_length_cv": 0.2}
    assert spmv["compare"] == {DAINT: "time", BEV: "time"}
    assert basic["cases"][0]["props"] == {"n": 1048576}
    solver = _read(built, "solvers.json")["cases"][0]
    assert solver["props"] == {"rows": 1000, "nnz_per_row": 6.4}
    assert solver["compare"] == {DAINT: "bandwidth", BEV: "bandwidth"}
    pre = _read(built, "preconditioners.json")
    assert pre["dataset"]["rows"] == [
        ["pp", "porous_pressure", 1000, 5000, 5.0, 0.2, True]]


def test_latest_points_and_prev(built):
    latest = _read(built, "basic.json")["latest"][DAINT]
    assert latest["column"] == 2 and latest["triad"] == 3.0e12
    point = next(p for p in latest["points"] if p["key"] == SPMV_KEY)
    assert point["time"] == pytest.approx(1.2e-4)
    assert point["prev"]["time"] == pytest.approx(1.1e-4)
    assert point["intensity"] == pytest.approx(10000 / 80004)


def test_settings_diff_paths(built):
    settings = _read(built, "preconditioners.json")["settings"]
    assert settings["jacobi"]["parameters"]["relax"] == [0.9, 0.5]
    assert settings["jacobi"]["changed"] == [
        {"path": "coarse.iters", "before": 1, "after": 2},
        {"path": "relax[1]", "before": 0.0, "after": 0.5}]
    assert _read(built, "solvers.json")["settings"]["cg"]["changed"] == []


def test_meta_last_run_status_and_seconds(built):
    meta = _read(built, "meta.json")
    bev = meta["machines"][BEV]
    assert bev["label"] == "Beverin MI300A"
    assert bev["last_run"]["dir"] == "2026-10-15_c3c3c3c3c3"
    # the spgemm group was not requested: ignored, not counted as failed
    assert bev["last_run"]["status"] == "partial"
    assert bev["last_run"]["seconds"] == 600.0
    assert meta["machines"][DAINT]["last_run"]["status"] == "ok"
    assert meta["machines"][DAINT]["nodes"]["nid000001"]["runs"] == 3
    setup = _read(built, "machines.json")["machines"][DAINT]["setup"]
    assert setup["settings"] == {"warmup": "1", "repetitions": "5",
                                 "timer_method": "median", "gpu_timer": True,
                                 "seed": "42"}


def test_cli_build_without_runs(tmp_path):
    assert main(["build", "--results", str(tmp_path / "none"),
                 "--out", str(tmp_path / "out")]) == 1


def _committed_results(tmp):
    """A copy of the runs git tracks; local debugging runs stay out."""
    try:
        listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z",
                                 "results"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return RESULTS
    files = [f for f in listed.stdout.decode().split("\0")
             if f and "/logs/" not in f]
    if not files:
        return RESULTS
    for f in files:
        (tmp / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / f, tmp / f)
    return tmp / "results"


@pytest.mark.skipif(not RESULTS.is_dir(), reason="no results/")
def test_real_results_build(tmp_path):
    out = tmp_path / "out"
    assert main(["build", "--results", str(_committed_results(tmp_path)),
                 "--manifest", str(ROOT / "datasets" / "manifest.json"),
                 "--catalog", str(ROOT / "datasets" / "catalog.toml"),
                 "--frozen", str(ROOT / "dashboard" / "frozen.json"),
                 "--out", str(out)]) == 0
    pre = _read(out, "preconditioners.json")
    assert len(pre["cases"]) == 86
    for machine in (DAINT, BEV):
        cols = pre["machines"][machine]["columns"]
        assert len(cols) == 1 and len(cols[0]["runs"]) == 3
    cells = {m: pre["machines"][m]["cells"] for m in (DAINT, BEV)}
    # Disabled on Beverin from the _3 run on; the two runs before it ran
    # ilu-syncfree, so the column has their values (median of the ok runs).
    run3 = _read(out, f"runs/{BEV}/2026-10-18_9cb13ee6ab_3.json")
    assert run3["invocations"]["preconditioner:porous_pressure/ilu-syncfree"][
        "status"] == "disabled"
    ilu = cells[BEV]["preconditioner/apply/ilu-syncfree/porous_pressure"][-1]
    assert ilu["state"] == "history" and len(ilu["runs"]) == 3
    for machine in (DAINT, BEV):
        pmis = cells[machine]["preconditioner/apply/mg-pmis/curl_curl"][-1]
        assert pmis["state"] == "not_available"
    meta = _read(out, "meta.json")
    assert set(meta["machines"]) == {DAINT, BEV}
    assert all(v["last_run"]["dir"].endswith("_3")
               for v in meta["machines"].values())
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    assert total < 2_000_000
