import json
import subprocess

from dash_fakes import write_run

from dashboard.kernels import build_kernels, case_group

ROWS = [{"stage": "", "name": "csr::spmv", "parent": None, "time": 1e-3, "calls": 1, "launches": 1,
         "allocations": 0, "copies": 0, "copy_bytes": 0, "dram_read_bytes": 1e6, "kernels": []}]


def kernels_file(d, cases, tool="ncu"):
    (d / "kernels.json").write_text(json.dumps({"tool": tool, "cases": cases}))


def run(results, name, date, **kw):
    return write_run(results, "daint-gh200", name, commit=name[-1] * 40, date=date,
                     started=f"{date}T00:00:00+00:00", **kw)


def build(results, out, **kw):
    warnings = []
    build_kernels(results, out, warnings, **kw)
    return json.loads((out / "kernels.json").read_text()), warnings


def test_runs_without_kernels_skipped(tmp_path):
    results = tmp_path / "results"
    run(results, "2026-11-01_a", "2026-11-01")
    run(results, "2026-11-05_b", "2026-11-05")
    d = run(results, "2026-11-11_c", "2026-11-11")
    kernels_file(d, {"spmv/csr/audikw1": {"status": "ok", "operations": ROWS}})
    data, _ = build(results, tmp_path / "out")
    m = data["machines"]["daint-gh200"]
    assert m["tool"] == "ncu"
    assert [r["dir"] for r in m["runs"]] == ["2026-11-11_c"]
    case = m["cases"]["spmv/csr/audikw1"]
    assert case["group"] == "spmv" and case["label"] == "CSR · audikw1"
    assert case["runs"] == [{"dir": "2026-11-11_c", "status": "ok", "operations": ROWS}]


def test_machine_without_any(tmp_path):
    results = tmp_path / "results"
    run(results, "2026-11-01_a", "2026-11-01")
    data, _ = build(results, tmp_path / "out")
    assert data["machines"]["daint-gh200"] == {"tool": None, "runs": [], "cases": {}}


def test_model_bytes_for_spmv(tmp_path):
    results = tmp_path / "results"
    d = run(results, "2026-11-11_c", "2026-11-11")
    kernels_file(d, {"spmv/csr/audikw1": {"status": "ok", "operations": ROWS},
                     "blas/gemv/2048": {"status": "ok", "operations": ROWS}})
    data, _ = build(results, tmp_path / "out",
                    model_bytes={"daint-gh200": {"2026-11-11_c": {"spmv/csr/audikw1": 2.5e6}}})
    cases = data["machines"]["daint-gh200"]["cases"]
    assert cases["spmv/csr/audikw1"]["runs"][0]["model_bytes"] == 2.5e6
    assert "model_bytes" not in cases["blas/gemv/2048"]["runs"][0]


def test_failed_case_keeps_reason(tmp_path):
    results = tmp_path / "results"
    d = run(results, "2026-11-11_c", "2026-11-11")
    kernels_file(d, {"spgemm/poisson_p1": {"status": "failed", "reason": "time: exit 1",
                                           "operations": []}})
    data, _ = build(results, tmp_path / "out")
    r = data["machines"]["daint-gh200"]["cases"]["spgemm/poisson_p1"]["runs"][0]
    assert r["status"] == "failed" and r["reason"] == "time: exit 1"


def test_unreadable_kernels_json_warns(tmp_path):
    results = tmp_path / "results"
    d = run(results, "2026-11-11_c", "2026-11-11")
    (d / "kernels.json").write_text("{not json")
    data, warnings = build(results, tmp_path / "out")
    assert data["machines"]["daint-gh200"]["runs"] == []
    assert any("kernels.json" in w for w in warnings)


def test_untracked_run_ignored(tmp_path):
    results = tmp_path / "results"
    a = run(results, "2026-11-05_b", "2026-11-05")
    kernels_file(a, {"spmv/csr/audikw1": {"status": "ok", "operations": ROWS}})
    git = ["git", "-C", str(results), "-c", "user.name=t", "-c", "user.email=t@t", "-c",
           "commit.gpgsign=false"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "x"], check=True)
    b = run(results, "2026-11-11_c", "2026-11-11")  # not committed
    kernels_file(b, {"spmv/csr/audikw1": {"status": "ok", "operations": ROWS}})
    data, _ = build(results, tmp_path / "out")
    assert [r["dir"] for r in data["machines"]["daint-gh200"]["runs"]] == ["2026-11-05_b"]


def test_case_groups():
    assert case_group("preconditioner/mg-pgm/curl_curl") == ("multigrid", "mg-pgm · curl_curl")
    assert case_group("spmv/sellp/ecology2") == ("spmv", "SELL-P · ecology2")
    assert case_group("spgemm/poisson_p1") == ("spgemm", "poisson_p1")
    assert case_group("blas/dot/1048576") == ("blas", "dot · 1048576")
    assert case_group("blas/gemv/8192") == ("gemv", "gemv · 8192")
