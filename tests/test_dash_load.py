import shutil
import subprocess

import pytest
from dash_fakes import write_run

from dashboard.load import columns, load_runs


def test_load_runs_reads_records_and_groups(tmp_path):
    write_run(tmp_path, "m1", "A", commit="c1", date="2026-10-11",
              started="2026-10-11T10:00", data={"blas": [{"n": 1}]})
    write_run(tmp_path, "m2", "B", commit="c2", date="2026-10-18",
              started="2026-10-18T10:00",
              data={"blas": [{"n": 2}]})
    (tmp_path / "m1" / "empty").mkdir()
    bad = write_run(tmp_path, "m1", "C", commit="c3", date="2026-10-19",
                    started="2026-10-19T10:00")
    (bad / "spmv.json").write_text('[{"a"')
    runs = load_runs(tmp_path)
    assert [(r.machine, r.dir) for r in runs] == [
        ("m1", "A"), ("m1", "C"), ("m2", "B")]
    assert runs[0].data["blas"] == [{"n": 1}]
    assert runs[0].data["spgemm"] is None
    assert runs[1].data["spmv"] is None


def test_columns_merge_repeats_and_order_by_date(tmp_path):
    write_run(tmp_path, "m", "A", commit="c1", date="2026-10-11",
              started="2026-10-11T10:00")
    write_run(tmp_path, "m", "B2", commit="c2", date="2026-10-18",
              started="2026-10-18T12:00")
    write_run(tmp_path, "m", "B", commit="c2", date="2026-10-18",
              started="2026-10-18T10:00")
    cols = columns(load_runs(tmp_path), "m")
    assert [c.commit for c in cols] == ["c1", "c2"]
    assert [r.dir for r in cols[1].runs] == ["B", "B2"]


def test_triad_absent_or_failed(tmp_path):
    kw = dict(commit="c", date="2026-10-11", started="2026-10-11T10:00")
    write_run(tmp_path, "m", "none", triad=None, **kw)
    write_run(tmp_path, "m", "err", extra={"stream": {"error": "x"}}, **kw)
    write_run(tmp_path, "m", "ok", **kw)
    by_dir = {r.dir: r for r in load_runs(tmp_path)}
    assert by_dir["none"].triad is None
    assert by_dir["err"].triad is None
    assert by_dir["ok"].triad == 3.0e12


@pytest.mark.skipif(shutil.which("git") is None, reason="no git")
def test_load_runs_keeps_only_tracked_runs_in_a_repository(tmp_path):
    results = tmp_path / "results"
    kw = dict(commit="c", date="2026-10-11", started="2026-10-11T10:00")
    write_run(results, "m", "tracked", **kw)
    write_run(results, "m", "local", **kw)
    # outside a repository every run is kept
    assert sorted(r.dir for r in load_runs(results)) == ["local", "tracked"]
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t",
           "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "results/m/tracked/run.json"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "runs"], check=True)
    assert [r.dir for r in load_runs(results)] == ["tracked"]
