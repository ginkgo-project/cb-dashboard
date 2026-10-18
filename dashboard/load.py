"""Load the committed runs and group them by Ginkgo commit."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

GROUPS = ("blas", "spmv", "spgemm", "solver", "preconditioner")


@dataclass
class Run:
    machine: str
    dir: str
    record: dict
    data: dict[str, list | None]

    @property
    def commit(self) -> str:
        return self.record["ginkgo"]["commit"]

    @property
    def commit_date(self) -> str:
        return self.record["ginkgo"]["commit_date"]

    @property
    def ref(self) -> str:
        return self.record["ginkgo"]["ref"]

    @property
    def started(self) -> str:
        return self.record["started"]

    @property
    def node(self) -> str:
        return self.record["node"]

    @property
    def triad(self) -> float | None:
        stream = self.record.get("stream")
        if not isinstance(stream, dict) or "error" in stream:
            return None
        triad = stream.get("results", {}).get("triad")
        return triad["bandwidth"] if triad else None


@dataclass
class Column:
    commit: str
    date: str
    ref: str
    runs: list[Run]


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _tracked(results: Path) -> set[str] | None:
    """<machine>/<dir> of every run.json git tracks; None outside a work tree."""
    try:
        listed = subprocess.run(
            ["git", "-C", str(results), "ls-files", "-z", "--", "*/run.json"],
            capture_output=True)
    except OSError:
        return None
    if listed.returncode != 0:
        return None
    return {f.removesuffix("/run.json")
            for f in listed.stdout.decode().split("\0") if f}


def load_runs(results: Path) -> list[Run]:
    """Every results/<machine>/<dir>/run.json, sorted for display.

    What's committed is what's shown: inside a git work tree, only the runs
    whose run.json git tracks are read.
    """
    results = Path(results)
    tracked = _tracked(results) if results.is_dir() else None
    runs = []
    for run_json in results.glob("*/*/run.json"):
        d = run_json.parent
        if tracked is not None and f"{d.parent.name}/{d.name}" not in tracked:
            continue
        record = _read_json(run_json)
        if not isinstance(record, dict):
            continue
        data = {g: _read_json(d / f"{g}.json") for g in GROUPS}
        runs.append(Run(d.parent.name, d.name, record, data))
    runs.sort(key=lambda r: (r.machine, r.commit_date, r.started, r.dir))
    return runs


def columns(runs: list[Run], machine: str) -> list[Column]:
    """One column per Ginkgo commit on the machine, oldest first."""
    by_commit: dict[str, list[Run]] = {}
    for run in runs:
        if run.machine == machine:
            by_commit.setdefault(run.commit, []).append(run)
    cols = []
    for commit, group in by_commit.items():
        group.sort(key=lambda r: r.started)
        cols.append(Column(commit, group[0].commit_date, group[0].ref, group))
    cols.sort(key=lambda c: (c.date, c.runs[0].started))
    return cols
