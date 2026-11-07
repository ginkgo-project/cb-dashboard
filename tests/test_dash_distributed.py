import json
import subprocess

import pytest

from bench.distributed import GPU_COUNTS
from dashboard import distributed as D
from dashboard.changes import BAND_FLOOR, noise_band

M = "daint-gh200"
SIZES = {"allreduce": 15, "alltoallv": 12, "neighbor_alltoallv": 12}


def mpi_cases(n, scale=1.0):
    out = []
    for op, count in SIZES.items():
        for k in range(count):
            c = 4 ** k
            out.append({"operation": op, "count": c, "bytes": 8 * c, "ranks": n,
                        "mpi": {op: {"completed": True, "time": scale * 1e-6 * (k + 1),
                                     "min_time": 1e-6, "repetitions": 3}}})
    if n == 8:
        for op in ("pingpong_intra", "pingpong_inter"):
            for k in range(14):
                out.append({"operation": op, "count": 4 ** k, "bytes": 8 * 4 ** k, "ranks": n,
                            "mpi": {op: {"completed": True, "time": 2e-6, "min_time": 1e-6,
                                         "repetitions": 3}}})
    return out


def ginkgo_cases(n, t):
    rows, nnz = 16777216 * n, 117440512 * n
    case = {"size": rows, "stencil": "7pt", "rows": rows, "cols": rows, "nonzeros": nnz}
    levels = [{"fine_rows": rows, "coarse_rows": rows // 8, "coarse_nonzeros": nnz // 4},
              {"fine_rows": rows // 8, "coarse_rows": rows // 64, "coarse_nonzeros": nnz // 32}]
    return {
        "spmv": [{**case, "spmv": {"csr-csr": {"completed": True, "time": t}}}],
        "spgemm": [{**case, "sparse_blas": {"spgemm": {"completed": True, "time": 10 * t,
                                                       "nonzeros_out": 3 * nnz}}}],
        "solver": [{**case, "solver": {s: {"completed": True, "generate": {"time": 2 * t},
                                           "apply": {"time": 100 * t, "iterations": 50 + n}}}}
                   for s in ("cg", "gmres")],
        "preconditioner": [{**case, "preconditioner": {p: {
            "completed": True, "generate": {"time": 5 * t}, "apply": {"time": t},
            "levels": levels}}} for p in ("mg-pgm", "mg-uniform", "mg-pmis")],
    }


def make_run(results, name, date, started, t=1e-3, failed=(), job=1):
    d = results / "distributed" / M / name
    gpus = {}
    for n in GPU_COUNTS:
        g = d / f"gpus-{n}"
        g.mkdir(parents=True)
        files = {"mpi": mpi_cases(n), **ginkgo_cases(n, t)}
        gpus[str(n)] = {}
        for group, data in files.items():
            if (n, group) in failed:
                gpus[str(n)][group] = {"status": "failed", "reason": "exit code 1"}
                (g / f"{group}.json").write_text("[]")
                continue
            gpus[str(n)][group] = {"status": "ok"}
            (g / f"{group}.json").write_text(json.dumps(data))
    rec = {"machine": M, "kind": "distributed",
           "ginkgo": {"commit": name[-10:] + "0" * 30, "commit_date": date,
                      "ref": "origin/develop"},
           "started": started, "finished": started, "groups": {}, "gpus": gpus,
           "distributed": {"gpu_counts": GPU_COUNTS, "gpus_per_node": 4, "nodes": "nid[1-16]",
                           "rows_per_gpu": 16777216},
           "ci": {"job_id": job, "job_url": f"https://ci/{job}"}}
    (d / "run.json").write_text(json.dumps(rec))
    return d


@pytest.fixture
def three(tmp_path):
    results = tmp_path / "results"
    for i, t in enumerate([1e-3, 1.1e-3, 1.2e-3]):
        make_run(results, f"2026-11-{14 + 7 * i:02d}_{i}aaaaaaaaa", f"2026-11-{14 + 7 * i:02d}",
                 f"2026-11-{14 + 7 * i:02d}T22:00:00+00:00", t=t,
                 failed={(64, "spgemm")} if i == 2 else ())
    return results


def build(results, tmp_path):
    out = tmp_path / "site"
    D.build_distributed(results, out, warnings=[])
    return json.loads((out / "distributed.json").read_text()), out


def test_runs_listed_oldest_first(three, tmp_path):
    data, _ = build(three, tmp_path)
    runs = data["machines"][M]["runs"]
    assert [r["date"] for r in runs] == ["2026-11-14", "2026-11-21", "2026-11-28"]
    assert runs[0]["ci"]["job_url"] == "https://ci/1"
    assert runs[0]["nodes"] == "nid[1-16]"


def test_rows_and_gpu_counts(three, tmp_path):
    rows = build(three, tmp_path)[0]["machines"][M]["rows"]
    bench = [k for k in rows if not k.startswith("mpi/")]
    assert bench == ["spmv", "spgemm", "cg/generate", "cg/apply", "gmres/generate",
                     "gmres/apply", "mg-pgm/generate", "mg-pgm/apply", "mg-uniform/generate",
                     "mg-uniform/apply", "mg-pmis/generate", "mg-pmis/apply"]
    assert all(list(rows[k]) == [str(n) for n in GPU_COUNTS] for k in bench)


def test_time_parallel_to_runs(three, tmp_path):
    rows = build(three, tmp_path)[0]["machines"][M]["rows"]
    assert rows["spmv"]["1"]["time"] == pytest.approx([1e-3, 1.1e-3, 1.2e-3])
    assert rows["spmv"]["1"]["status"] == ["ok", "ok", "ok"]
    assert rows["spgemm"]["64"]["time"][2] is None
    assert rows["spgemm"]["64"]["status"] == ["ok", "ok", "failed"]


def test_solver_apply_is_per_iteration(three, tmp_path):
    rows = build(three, tmp_path)[0]["machines"][M]["rows"]
    assert rows["cg/apply"]["8"]["time"][0] == pytest.approx(100e-3 / 58)
    assert rows["cg/generate"]["8"]["time"][0] == pytest.approx(2e-3)


def test_band_floor_until_history(three, tmp_path):
    rows = build(three, tmp_path)[0]["machines"][M]["rows"]
    assert rows["spmv"]["1"]["band"] == [BAND_FLOOR] * 3


def test_band_from_history(tmp_path):
    results = tmp_path / "results"
    times = [1.0e-3, 1.1e-3, 0.9e-3, 1.0e-3, 1.2e-3]
    for i, t in enumerate(times):
        make_run(results, f"2026-12-{1 + i:02d}_{i}bbbbbbbbb", f"2026-12-{1 + i:02d}",
                 f"2026-12-{1 + i:02d}T22:00:00+00:00", t=t)
    band = build(results, tmp_path)[0]["machines"][M]["rows"]["spmv"]["1"]["band"]
    assert band[:4] == [BAND_FLOOR] * 4
    assert band[4] == pytest.approx(noise_band([1 / t for t in times[:4]]))
    assert band[4] > BAND_FLOOR


def test_mpi_rows_only_for_swept_sizes(three, tmp_path):
    rows = build(three, tmp_path)[0]["machines"][M]["rows"]
    sizes = {op: sorted(int(k.rsplit("/", 1)[1]) for k in rows if k.startswith(f"mpi/{op}/"))
             for op in (*SIZES, "pingpong_intra", "pingpong_inter")}
    assert sizes["allreduce"] == [8 * 4 ** k for k in range(15)]
    assert sizes["alltoallv"] == [8 * 4 ** k for k in range(12)]
    assert len(sizes["pingpong_intra"]) == 14
    pp = rows["mpi/pingpong_intra/8"]
    assert pp["8"]["status"] == ["ok"] * 3
    assert pp["1"]["status"] == ["missing"] * 3 and pp["1"]["time"] == [None] * 3


def test_run_details(three, tmp_path):
    _, out = build(three, tmp_path)
    det = json.loads((out / "distributed" / M / "2026-11-14_0aaaaaaaaa.json").read_text())
    g8 = det["gpus"]["8"]
    assert g8["rows"] == 16777216 * 8 and g8["nonzeros"] == 117440512 * 8
    assert g8["iterations"] == {"cg": 58, "gmres": 58}
    assert g8["solver_time"]["cg"] == pytest.approx(100e-3)
    assert g8["levels"]["mg-pgm"] == {"levels": 2, "coarsest_rows": 16777216 * 8 // 64,
                                      "operator_complexity": pytest.approx(1 + 1 / 4 + 1 / 32)}


def test_untracked_runs_ignored(tmp_path):
    results = tmp_path / "results"
    make_run(results, "2026-11-14_0aaaaaaaaa", "2026-11-14", "2026-11-14T22:00:00+00:00")
    make_run(results, "2026-11-21_1aaaaaaaaa", "2026-11-21", "2026-11-21T22:00:00+00:00")
    subprocess.run(["git", "init", "-q", str(results)], check=True)
    subprocess.run(["git", "-C", str(results), "add",
                    f"distributed/{M}/2026-11-14_0aaaaaaaaa/run.json"], check=True)
    runs = D.load_distributed(results)
    assert [r.dir for r in runs[M]] == ["2026-11-14_0aaaaaaaaa"]


def test_no_distributed_runs(tmp_path):
    (tmp_path / "results").mkdir()
    data, _ = build(tmp_path / "results", tmp_path)
    assert data == {"machines": {}}
