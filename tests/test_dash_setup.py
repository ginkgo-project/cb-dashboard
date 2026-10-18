import subprocess

from dash_fakes import write_run

from dashboard.load import load_runs
from dashboard.setup import git_script_changed, setup_changes


def _run(tmp_path, name, **extra):
    write_run(tmp_path, "m", name, commit="c" + name, date="2026-01-01",
              started="2026-01-01T00:00:00", extra=extra)
    return [r for r in load_runs(tmp_path) if r.dir == name][0]


def test_setup_changes_lists_what_changed(tmp_path):
    a = _run(tmp_path, "a", uenv="u/1")
    b = _run(tmp_path, "b", uenv="u/2")
    c = _run(tmp_path, "c", uenv="u/1")
    assert setup_changes(a, b, lambda x, y: False) == ["uenv: u/1 → u/2"]
    assert setup_changes(a, c, lambda x, y: False) == []


def test_git_script_changed(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t",
                        "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                        *args], check=True, capture_output=True)

    def sha():
        return subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                              check=True, capture_output=True,
                              text=True).stdout.strip()

    git("init")
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench/x.py").write_text("1")
    git("add", "-A")
    git("commit", "-m", "1")
    s1 = sha()
    (tmp_path / "README").write_text("r")
    git("add", "-A")
    git("commit", "-m", "2")
    s2 = sha()
    (tmp_path / "bench/x.py").write_text("2")
    git("add", "-A")
    git("commit", "-m", "3")
    s3 = sha()
    changed = git_script_changed(tmp_path)
    assert changed(s1, s2) is False
    assert changed(s2, s3) is True
    assert changed("0" * 40, s3) is True
