import json
import os
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


def full_run(tmp_path, monkeypatch, manifest, argv=(), status="ok"):
    m = fake_machine(tmp_path)
    patch_common(monkeypatch, m, COMMIT)
    order = []
    monkeypatch.setattr(cli, "build", lambda m, src: order.append("build") or
                        {"configured": True, "seconds": 1.0, "cmake_cache": {}})
    def rg(group, invs, srun, out):
        order.append(group)
        out.mkdir(parents=True, exist_ok=True)
        return {"status": status, "seconds": 0.0, "invocations": []}
    monkeypatch.setattr(cli, "run_group", rg)
    rc = cli.main(["run", "daint-gh200", "--manifest", str(manifest), *argv])
    return m, rc, order


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
    rec = json.loads(next(m.results_dir.glob("*/run.json")).read_text())
    assert "error" in rec["build"] and rec["groups"] == {}
    assert not (m.root / "state" / m.name / "last_commit").exists()
