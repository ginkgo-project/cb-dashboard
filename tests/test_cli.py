import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from bench import __main__ as cli
from bench.checkout import Commit
from bench.machine import Machine

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


def sh_env(**extra):
    return {**os.environ, "SCRATCH": "/scratch", "CB_PYTHON": str(REPO / ".venv/bin/python"), **extra}


def test_sbatch_args_daint():
    r = subprocess.run([PY, "-m", "bench", "sbatch-args", "daint-gh200"], cwd=REPO,
                       capture_output=True, text=True, env=sh_env(), check=True)
    assert r.stdout.strip() == ("--partition=normal --nodes=1 --exclusive --time=04:00:00 "
                                "--account=d131 --uenv=prgenv-gnu/26.3:v1")


def test_submit_dry_run():
    r = subprocess.run(["bench/submit.sh", "daint-gh200", "--force"], cwd=REPO, capture_output=True,
                       text=True, env=sh_env(SUBMIT_DRY_RUN="1"), check=True)
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("sbatch --partition=normal")
    assert "--uenv=prgenv-gnu/26.3:v1 --view=default --job-name=cb-daint-gh200" in lines[0]
    assert "--output=/scratch/cb/logs/%j.out" in lines[0]
    assert lines[0].endswith("bench/job.sh daint-gh200 --force")


def test_job_sh_text():
    t = (REPO / "bench/job.sh").read_text()
    assert "set -euo pipefail" in t
    assert 'exec "$CB_ROOT/venv/$1/bin/python" -m bench run "$@"' in t


MANIFEST = {"version": "1", "matrices": [
    {"name": "m1", "path": "matrices/m1.mtx", "set": "spmv", "roles": ["spmv"], "run_sellp": False},
]}


def fake_machine(tmp_path, arch=None):
    import platform
    return Machine("daint-gh200", "cuda", "u/1", [], [], [], False, arch or platform.machine(),
                   tmp_path / "root", tmp_path / "build", tmp_path / "results")


@pytest.fixture
def manifest(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(MANIFEST))
    return p


def patch_common(monkeypatch, m, commit):
    monkeypatch.setattr(cli, "load_machine", lambda name, **kw: m)
    monkeypatch.setattr(cli, "ensure_clone", lambda url, clone: None)
    monkeypatch.setattr(cli, "resolve", lambda clone, ref: commit)
    monkeypatch.setattr(cli, "prepare_worktree", lambda *a: None)
    monkeypatch.setattr(cli, "gpu_info", lambda m: {"gpu": "g"})
    monkeypatch.setattr(cli, "toolchain_info", lambda m: {"g++": "x"})
    monkeypatch.setattr(cli, "job_deadline", lambda: None)


COMMIT = Commit("a" * 40, "2026-10-16", "b" * 40)


def test_nothing_to_do_exits_zero(tmp_path, monkeypatch, manifest, capsys):
    m = fake_machine(tmp_path)
    patch_common(monkeypatch, m, COMMIT)
    (m.root / "state" / m.name).mkdir(parents=True)
    (m.root / "state" / m.name / "last_commit").write_text(COMMIT.sha + "\n")
    monkeypatch.setattr(cli, "build", lambda *a, **k: pytest.fail("built"))
    assert cli.main(["run", "daint-gh200", "--manifest", str(manifest)]) == 0
    assert f"nothing to do: {COMMIT.sha} already benchmarked" in capsys.readouterr().out
    assert not m.results_dir.exists()


def test_dry_run_lists_invocations(tmp_path, monkeypatch, manifest, capsys):
    m = fake_machine(tmp_path, arch="nonsense")
    monkeypatch.setattr(cli, "load_machine", lambda name, **kw: m)
    for n in ("ensure_clone", "resolve", "build", "run_group", "prepare_worktree"):
        monkeypatch.setattr(cli, n, lambda *a, **k: pytest.fail("ran"))
    assert cli.main(["run", "daint-gh200", "--dry-run", "--manifest", str(manifest),
                     "--groups", "blas,spmv"]) == 0
    out = capsys.readouterr().out
    assert "daint-gh200" in out and "origin/develop" in out
    lines = [l for l in out.splitlines() if l.startswith(("blas", "spmv"))]
    assert len(lines) == 2
    assert "vectors" in lines[0] and "m1" in lines[1] and "-formats" in lines[1]


def test_arch_mismatch_exits_two(tmp_path, monkeypatch, manifest):
    m = fake_machine(tmp_path, arch="nonsense")
    patch_common(monkeypatch, m, COMMIT)
    assert cli.main(["run", "daint-gh200", "--manifest", str(manifest)]) == 2


def full_run(tmp_path, monkeypatch, manifest, argv=(), status="ok", during=None):
    """Run the CLI with fakes; `during(group, out)` runs inside each fake group."""
    m = fake_machine(tmp_path)
    patch_common(monkeypatch, m, COMMIT)
    order = []
    monkeypatch.setattr(cli, "build", lambda m, src: order.append("build") or
                        {"configured": True, "seconds": 1.0, "cmake_cache": {}})
    def rg(group, invs, srun, out, deadline=None):
        order.append(group)
        out.mkdir(parents=True, exist_ok=True)
        if during:
            during(group, out)
        return {"status": status, "seconds": 0.0, "invocations": []}
    monkeypatch.setattr(cli, "run_group", rg)
    rc = cli.main(["run", "daint-gh200", "--manifest", str(manifest), *argv])
    return m, rc, order


def run_json(m):
    return json.loads(next(m.results_dir.glob("*/run.json")).read_text())


def test_full_run_writes_record_and_last(tmp_path, monkeypatch, manifest):
    m, rc, order = full_run(tmp_path, monkeypatch, manifest, status="partial")
    assert rc == 0
    assert order == ["build", "blas", "spmv", "spgemm", "solver", "preconditioner"]
    d = m.results_dir / "2026-10-16_aaaaaaaaaa"
    rec = json.loads((d / "run.json").read_text())
    assert rec["gpu"] == {"gpu": "g"} and rec["toolchain"] == {"g++": "x"}
    assert set(rec["groups"]) == set(order[1:])
    assert (m.root / "state" / m.name / "last_commit").read_text().strip() == COMMIT.sha


def test_other_ref_does_not_write_last(tmp_path, monkeypatch, manifest):
    m, rc, _ = full_run(tmp_path, monkeypatch, manifest, ["--ref", "origin/other", "--groups", "blas"])
    assert rc == 0
    assert not (m.root / "state" / m.name / "last_commit").exists()


def test_existing_dir_gets_suffix(tmp_path, monkeypatch, manifest):
    m = fake_machine(tmp_path)
    (m.results_dir / "2026-10-16_aaaaaaaaaa").mkdir(parents=True)
    m2, rc, _ = full_run(tmp_path, monkeypatch, manifest, ["--force", "--groups", "blas"])
    assert (m.results_dir / "2026-10-16_aaaaaaaaaa_2" / "run.json").is_file()


def test_build_failure(tmp_path, monkeypatch, manifest):
    m = fake_machine(tmp_path)
    patch_common(monkeypatch, m, COMMIT)
    def boom(m, src):
        raise subprocess.CalledProcessError(2, ["cmake"])
    monkeypatch.setattr(cli, "build", boom)
    monkeypatch.setattr(cli, "run_group", lambda *a, **k: pytest.fail("ran"))
    assert cli.main(["run", "daint-gh200", "--manifest", str(manifest)]) == 1
    rec = run_json(m)
    assert "error" in rec["build"] and rec["finished"] is not None
    assert rec["groups"] == {g: {"status": "skipped", "reason": "build failed"}
                             for g in cli.GROUPS}
    assert not (m.root / "state" / m.name / "last_commit").exists()


def test_subset_groups_does_not_write_last(tmp_path, monkeypatch, manifest):
    m, rc, _ = full_run(tmp_path, monkeypatch, manifest, ["--groups", "blas"])
    assert rc == 0
    assert not (m.root / "state" / m.name / "last_commit").exists()


def test_record_written_after_build_and_each_group(tmp_path, monkeypatch, manifest):
    seen = {}

    def during(group, out):
        seen[group] = json.loads((out / "run.json").read_text())
    m, rc, order = full_run(tmp_path, monkeypatch, manifest, during=during)
    assert rc == 0
    first = seen["blas"]
    assert first["finished"] is None and first["build"]["configured"] is True
    assert first["groups"]["blas"] == {"status": "running"}
    assert all(first["groups"][g] == {"status": "pending"} for g in cli.GROUPS[1:])
    later = seen["solver"]
    assert [later["groups"][g]["status"] for g in cli.GROUPS] == [
        "ok", "ok", "ok", "running", "pending"]
    rec = run_json(m)
    assert rec["finished"] is not None
    assert all(rec["groups"][g]["status"] == "ok" for g in cli.GROUPS)
    assert not list(m.results_dir.glob("*/run.json.tmp"))


def test_exception_in_group_recorded_next_runs(tmp_path, monkeypatch, manifest):
    def during(group, out):
        if group == "spgemm":
            raise RuntimeError("no such matrix field")
    m, rc, order = full_run(tmp_path, monkeypatch, manifest, during=during)
    assert rc == 0
    assert order == ["build", *cli.GROUPS]
    g = run_json(m)["groups"]
    assert g["spgemm"]["status"] == "failed"
    assert g["spgemm"]["reason"] == "RuntimeError: no such matrix field"
    assert g["solver"]["status"] == "ok" and g["preconditioner"]["status"] == "ok"


def test_sigterm_writes_record_and_exits(tmp_path, monkeypatch, manifest):
    before = signal.getsignal(signal.SIGTERM)

    def during(group, out):
        if group == "spgemm":
            os.kill(os.getpid(), signal.SIGTERM)
            raise AssertionError("SIGTERM did not stop the group")
    m, rc, order = full_run(tmp_path, monkeypatch, manifest, during=during)
    assert rc == 128 + signal.SIGTERM
    assert order == ["build", "blas", "spmv", "spgemm"]
    rec = run_json(m)
    assert rec["finished"] is not None
    assert [rec["groups"][g]["status"] for g in cli.GROUPS] == [
        "ok", "ok", "killed", "skipped", "skipped"]
    assert rec["groups"]["solver"]["reason"] == "job killed"
    assert not (m.root / "state" / m.name / "last_commit").exists()
    assert signal.getsignal(signal.SIGTERM) is before


def test_sigterm_during_build(tmp_path, monkeypatch, manifest):
    m = fake_machine(tmp_path)
    patch_common(monkeypatch, m, COMMIT)

    def build(m, src):
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM did not stop the build")
    monkeypatch.setattr(cli, "build", build)
    monkeypatch.setattr(cli, "run_group", lambda *a, **k: pytest.fail("ran"))
    assert cli.main(["run", "daint-gh200", "--manifest", str(manifest)]) == 143
    rec = run_json(m)
    assert "SIGTERM" in rec["build"]["error"]
    assert all(rec["groups"][g] == {"status": "skipped", "reason": "job killed"}
               for g in cli.GROUPS)


def test_children_start_with_sigchld_unblocked(tmp_path, monkeypatch, manifest):
    """A job can start with SIGCHLD blocked; CMake inherits the mask and hangs."""
    m = fake_machine(tmp_path)
    patch_common(monkeypatch, m, COMMIT)
    seen = []

    def build(m, src):
        status = subprocess.run(["grep", "SigBlk", "/proc/self/status"],
                                capture_output=True, text=True, check=True).stdout
        seen.append(int(status.split()[1], 16) & (1 << (signal.SIGCHLD - 1)))
        return {"configured": True, "seconds": 1.0, "cmake_cache": {}}
    monkeypatch.setattr(cli, "build", build)
    monkeypatch.setattr(cli, "run_group", lambda *a, **k:
                        {"status": "ok", "seconds": 0.0, "invocations": []})
    before = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGCHLD})
    try:
        assert cli.main(["run", "daint-gh200", "--manifest", str(manifest),
                         "--groups", "blas"]) == 0
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, before)
    assert seen == [0]


def test_groups_in_any_order_are_a_full_run(tmp_path, monkeypatch, manifest):
    m, rc, order = full_run(tmp_path, monkeypatch, manifest,
                            ["--groups", "preconditioner,solver,spgemm,spmv,blas,spmv"])
    assert rc == 0
    assert order == ["build", *cli.GROUPS]
    assert (m.root / "state" / m.name / "last_commit").read_text().strip() == COMMIT.sha


def test_unrequested_groups_recorded_as_skipped(tmp_path, monkeypatch, manifest):
    m, rc, order = full_run(tmp_path, monkeypatch, manifest, ["--groups", "solver,blas"])
    assert order == ["build", "blas", "solver"]
    g = run_json(m)["groups"]
    assert list(g) == cli.GROUPS
    assert g["blas"]["status"] == "ok" and g["solver"]["status"] == "ok"
    for name in ("spmv", "spgemm", "preconditioner"):
        assert g[name] == {"status": "skipped", "reason": "not requested"}


def test_deadline_passed_to_groups(tmp_path, monkeypatch, manifest):
    m = fake_machine(tmp_path)
    patch_common(monkeypatch, m, COMMIT)
    monkeypatch.setattr(cli, "job_deadline", lambda: 12345.0)
    monkeypatch.setattr(cli, "build", lambda m, src: {"cmake_cache": {}})
    seen = []
    monkeypatch.setattr(cli, "run_group", lambda g, invs, srun, out, deadline=None:
                        seen.append(deadline) or {"status": "ok", "seconds": 0.0, "invocations": []})
    assert cli.main(["run", "daint-gh200", "--manifest", str(manifest), "--groups", "blas"]) == 0
    assert seen == [12345.0]
