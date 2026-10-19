import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bench.checkout import (
    Commit,
    clone_lock,
    ensure_clone,
    lacks,
    prepare_worktree,
    read_last,
    resolve,
    should_run,
    write_last,
)

ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "T",
    "GIT_AUTHOR_EMAIL": "t@example.org",
    "GIT_COMMITTER_NAME": "T",
    "GIT_COMMITTER_EMAIL": "t@example.org",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
}


def git(cwd, *args, date="2026-01-01T12:00:00+00:00"):
    env = {**ENV, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    r = subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    )
    return r.stdout.strip()


def commit(repo, name, date):
    (repo / name).write_text(name)
    git(repo, "add", name)
    git(repo, "commit", "-m", name, date=date)
    return git(repo, "rev-parse", "HEAD")


def make_upstream(tmp_path):
    up = tmp_path / "upstream"
    up.mkdir()
    git(up, "init", "-b", "develop")
    c1 = commit(up, "a", "2026-01-01T12:00:00+00:00")
    c2 = commit(up, "b", "2026-02-03T12:00:00+00:00")
    return up, c1, c2


def make_clone(tmp_path, up):
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", str(up), str(clone)], check=True, env=ENV, capture_output=True
    )
    return clone


def test_resolve_develop(tmp_path, monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    up, c1, c2 = make_upstream(tmp_path)
    clone = tmp_path / "clone"
    ensure_clone(str(up), clone)
    c = resolve(clone)
    assert c == Commit(sha=c2, date="2026-02-03", develop_base=c2)


def test_ensure_clone_fetches(tmp_path, monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    up, c1, c2 = make_upstream(tmp_path)
    clone = tmp_path / "clone"
    ensure_clone(str(up), clone)
    c3 = commit(up, "c", "2026-03-01T12:00:00+00:00")
    ensure_clone(str(up), clone)
    assert resolve(clone).sha == c3


def test_resolve_branch(tmp_path, monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    up, c1, c2 = make_upstream(tmp_path)
    git(up, "checkout", "-b", "feature")
    c3 = commit(up, "f", "2026-03-01T12:00:00+00:00")
    git(up, "checkout", "develop")
    clone = tmp_path / "clone"
    ensure_clone(str(up), clone)
    c = resolve(clone, "origin/feature")
    assert c.sha == c3
    assert c.develop_base == c2
    assert c.date == "2026-03-01"


def test_lacks_names_commits_not_merged_yet(tmp_path, monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    up, c1, c2 = make_upstream(tmp_path)
    git(up, "checkout", "-b", "feature", c1)
    c3 = commit(up, "f", "2026-03-01T12:00:00+00:00")
    clone = tmp_path / "clone"
    ensure_clone(str(up), clone)
    added = {"old": c1, "new": c2, "side": c3}
    assert lacks(clone, c2, added) == {"side"}
    assert lacks(clone, c1, added) == {"new", "side"}
    with pytest.raises(subprocess.CalledProcessError):
        lacks(clone, c2, {"unknown": "f" * 40})


def test_worktree_switches_commit(tmp_path, monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    up, c1, c2 = make_upstream(tmp_path)
    clone = make_clone(tmp_path, up)
    before = git(clone, "rev-parse", "HEAD")
    wt = tmp_path / "worktrees" / "m"
    prepare_worktree(clone, wt, c1)
    assert git(wt, "rev-parse", "HEAD") == c1
    prepare_worktree(clone, wt, c2)
    assert git(wt, "rev-parse", "HEAD") == c2
    assert git(clone, "rev-parse", "HEAD") == before


def test_state_roundtrip(tmp_path):
    d = tmp_path / "state" / "m"
    assert read_last(d) is None
    write_last(d, "abc")
    assert read_last(d) == "abc"
    assert not (d / "last_commit.tmp").exists()
    write_last(d, "def")
    assert read_last(d) == "def"


def test_should_run():
    c = Commit(sha="a", date="2026-01-01", develop_base="a")
    assert should_run(c, "a", "origin/develop", False) is False
    assert should_run(c, "a", "origin/develop", True) is True
    assert should_run(c, "a", "origin/feature", False) is True
    assert should_run(c, "b", "origin/develop", False) is True
    assert should_run(c, None, "origin/develop", False) is True


def test_worktree_recreated_after_rmtree(tmp_path, monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    up, c1, c2 = make_upstream(tmp_path)
    clone = make_clone(tmp_path, up)
    wt = tmp_path / "worktrees" / "m"
    prepare_worktree(clone, wt, c1)
    shutil.rmtree(wt)
    prepare_worktree(clone, wt, c2)
    assert git(wt, "rev-parse", "HEAD") == c2


def test_worktree_with_dangling_gitdir(tmp_path, monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    up, c1, c2 = make_upstream(tmp_path)
    clone = make_clone(tmp_path, up)
    wt = tmp_path / "worktrees" / "m"
    prepare_worktree(clone, wt, c1)
    shutil.rmtree(clone / ".git" / "worktrees")
    prepare_worktree(clone, wt, c2)
    assert git(wt, "rev-parse", "HEAD") == c2


def test_ensure_clone_waits_for_lock(tmp_path, monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    up, c1, c2 = make_upstream(tmp_path)
    clone = tmp_path / "clone"
    code = (
        "import fcntl,sys,time\n"
        f"f=open({str(clone) + '.lock'!r},'a')\n"
        "fcntl.flock(f,fcntl.LOCK_EX)\n"
        "print('ready',flush=True)\n"
        "time.sleep(1.0)\n"
    )
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready"
    t0 = time.monotonic()
    ensure_clone(str(up), clone)
    assert time.monotonic() - t0 >= 0.8
    p.wait()
    with clone_lock(clone):
        pass
