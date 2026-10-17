"""The shared Ginkgo clone, per-machine worktrees and the benchmarked-commit state."""
from __future__ import annotations

import fcntl
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

DEVELOP = "origin/develop"


@dataclass
class Commit:
    sha: str
    date: str  # YYYY-MM-DD, committer date
    develop_base: str


def _git(*args: str, cwd: Path | None = None) -> str:
    r = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return r.stdout.strip()


@contextmanager
def clone_lock(clone: Path):
    """Exclusive lock on `<clone>.lock`, serializing mutations of the shared clone."""
    clone.parent.mkdir(parents=True, exist_ok=True)
    with open(clone.with_name(clone.name + ".lock"), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def ensure_clone(url: str, clone: Path) -> None:
    with clone_lock(clone):
        if (clone / ".git").exists():
            _git("fetch", "--prune", "origin", cwd=clone)
        else:
            if clone.exists():  # half-finished clone
                shutil.rmtree(clone)
            _git("clone", url, str(clone))


def resolve(clone: Path, ref: str = DEVELOP) -> Commit:
    sha = _git("rev-parse", f"{ref}^{{commit}}", cwd=clone)
    date = _git("show", "-s", "--format=%cs", sha, cwd=clone)
    base = _git("merge-base", ref, DEVELOP, cwd=clone)
    return Commit(sha=sha, date=date, develop_base=base)


def _worktree_valid(worktree: Path) -> bool:
    if not (worktree / ".git").exists():
        return False
    r = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "--git-dir"],
        capture_output=True,
    )
    return r.returncode == 0


def prepare_worktree(clone: Path, worktree: Path, sha: str) -> None:
    with clone_lock(clone):
        if _worktree_valid(worktree):
            _git("-C", str(worktree), "checkout", "--detach", sha)
            return
        if (worktree / ".git").exists():  # .git points at a missing gitdir
            shutil.rmtree(worktree)
        _git("worktree", "prune", cwd=clone)
        worktree.parent.mkdir(parents=True, exist_ok=True)
        _git("worktree", "add", "--detach", str(worktree), sha, cwd=clone)


def read_last(state_dir: Path) -> str | None:
    f = state_dir / "last_commit"
    if not f.exists():
        return None
    return f.read_text().strip() or None


def write_last(state_dir: Path, sha: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    tmp = state_dir / "last_commit.tmp"
    tmp.write_text(sha + "\n")
    tmp.replace(state_dir / "last_commit")


def should_run(commit: Commit, last: str | None, ref: str, force: bool) -> bool:
    return force or ref != DEVELOP or commit.sha != last
