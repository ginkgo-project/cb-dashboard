import json
import shutil
import subprocess
from pathlib import Path

import pytest

from bench import kernel_cases as K
from bench import kernels as S
from bench.machine import load_machine

DATA = Path(__file__).parent / "data" / "profiles" / "daint"


@pytest.fixture
def m(tmp_path):
    mach = load_machine("daint-gh200", env={"SCRATCH": str(tmp_path / "s")})
    mach.root = tmp_path / "root"
    return mach


def blas(m, *keys):
    cases = K.kernel_cases(m, {"matrices": []})
    return [next(c for c in cases if c.key == k) for k in keys]


def fake_run(fail=None, hang=None):
    """A stand-in for subprocess.run that writes each Daint pass's fixture where the pass
    puts its output. `fail`/`hang`: (case label, pass name) pairs."""
    calls = []

    def run(argv, stdin=None, stdout=None, stderr=None, timeout=None, input=None, check=False):
        calls.append(argv)
        label = next((p for p in Path(stdout.name).parts if p.startswith("blas_")), None) \
            if stdout is not None else None
        name = Path(stdout.name).stem if stdout is not None else "post"
        if hang and (label, name) in hang:
            raise subprocess.TimeoutExpired(argv, timeout)
        if fail and (label, name) in fail:
            return subprocess.CompletedProcess(argv, 1)
        if name == "time":
            stdout.write((DATA / "dot-detailed.json").read_text())
        elif name == "trace":
            out = Path(argv[argv.index("-o") + 1]).parent
            shutil.copy(DATA / "dot-nsys.sqlite", out / "trace.sqlite")
        elif name.startswith("counters-"):
            shutil.copy(DATA / "dot-ncu.csv", argv[argv.index("--log-file") + 1])
        return subprocess.CompletedProcess(argv, 0)

    run.calls = calls
    return run


def test_ok_case_writes_kernels_json(m, tmp_path):
    out = tmp_path / "run"
    run = fake_run()
    rec = S.run_kernels(m, blas(m, "blas/dot/1048576"), out, deadline=None, run=run)
    assert rec["status"] == "ok"
    [c] = rec["cases"]
    assert c["key"] == "blas/dot/1048576" and c["status"] == "ok"
    assert [p["name"] for p in c["passes"]] == ["time", "trace", "counters-0", "counters-1",
                                               "counters-2", "counters-3"]
    assert all(p["exit_code"] == 0 for p in c["passes"])
    data = json.loads((out / "kernels.json").read_text())
    assert data["tool"] == "ncu"
    rows = data["cases"]["blas/dot/1048576"]["operations"]
    dot = next(r for r in rows if r["name"] == "dense::compute_dot_dispatch")
    assert dot["launches"] == 2 and dot["dram_read_bytes"] > 0
    # every pass ran under srun with the machine's options
    assert all(a[0] == "srun" for a in run.calls if a[1:2] != ["export"] and "export" not in a)
    # raw outputs stay out of the run directory, which goes into the CI artifacts
    assert (out / "logs" / "kernels" / "blas_dot_1048576" / "time.json").is_file()
    assert not list(out.rglob("*.csv")) and not list(out.rglob("*.sqlite"))
    assert list(S.raw_dir(m, out).rglob("counters-0.csv"))


def test_case_failure_and_hang(m, tmp_path):
    a, b, c = blas(m, "blas/dot/1048576", "blas/dot/16777216", "blas/axpy/1048576")
    run = fake_run(fail={(a.label, "time")}, hang={(b.label, "counters-1")})
    rec = S.run_kernels(m, [a, b, c], tmp_path / "run", deadline=None, run=run)
    st = {x["key"]: x for x in rec["cases"]}
    assert st[a.key]["status"] == "failed" and "time" in st[a.key]["reason"]
    assert st[b.key]["status"] == "partial" and "counters-1" in st[b.key]["reason"] \
        and "timed out" in st[b.key]["reason"]
    assert st[c.key]["status"] == "ok"
    assert rec["status"] == "partial"
    # a failed case's time pass stops it: no profiler pass ran for it
    assert [p["name"] for p in st[a.key]["passes"]] == ["time"]


def test_budget_skips_rest(m, tmp_path):
    m.kernels.budget_s = 0
    rec = S.run_kernels(m, blas(m, "blas/dot/1048576", "blas/axpy/1048576"), tmp_path / "run",
                        deadline=None, run=fake_run())
    assert [x["status"] for x in rec["cases"]] == ["skipped", "skipped"]
    assert "budget" in rec["cases"][0]["reason"] and rec["status"] == "skipped"


def test_deadline_skips(m, tmp_path):
    import time
    rec = S.run_kernels(m, blas(m, "blas/dot/1048576"), tmp_path / "run",
                        deadline=time.time() + 10, run=fake_run())
    assert rec["cases"][0]["status"] == "skipped" and "job time" in rec["cases"][0]["reason"]


def test_unavailable_and_missing(m, tmp_path):
    [a, b] = blas(m, "blas/dot/1048576", "blas/axpy/1048576")
    a.unavailable = "not in this Ginkgo commit"
    b.missing_input = tmp_path / "nope.bin"
    rec = S.run_kernels(m, [a, b], tmp_path / "run", deadline=None, run=fake_run())
    assert [x["status"] for x in rec["cases"]] == ["not_available", "failed"]
    assert rec["cases"][1]["reason"] == "missing input"
    data = json.loads((tmp_path / "run" / "kernels.json").read_text())
    assert data["cases"][a.key] == {"status": "not_available", "reason": "not in this Ginkgo commit",
                                   "operations": []}


def test_unreadable_output_is_partial(m, tmp_path):
    [a] = blas(m, "blas/dot/1048576")

    def run(argv, stdin=None, stdout=None, stderr=None, timeout=None, input=None, check=False):
        if Path(stdout.name).stem == "time":
            stdout.write((DATA / "dot-detailed.json").read_text())
        return subprocess.CompletedProcess(argv, 0)  # but writes no profiler output

    rec = S.run_kernels(m, [a], tmp_path / "run", deadline=None, run=run)
    c = rec["cases"][0]
    assert c["status"] == "partial" and "cannot read" in c["reason"]


def test_imports_without_sqlite3():
    """Beverin's venv Python has no sqlite3; only Daint's nsys reader needs it."""
    import sys
    code = ("import sys; sys.modules['sqlite3'] = None; sys.modules['_sqlite3'] = None\n"
            "import bench.__main__, bench.kernels, bench.profilers\n"
            "from bench.profilers import read_rocprof\n"
            f"print(len(read_rocprof({str(DATA.parent / 'beverin' / 'dot-0')!r})[1]))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=Path(__file__).parent.parent)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "9"
