"""Writers of fake results for the dashboard tests."""
from __future__ import annotations

import json
from pathlib import Path

GROUP_NAMES = ("blas", "spmv", "spgemm", "solver", "preconditioner")


def write_run(results: Path, machine: str, name: str, *, commit: str, date: str,
              started: str, ref="origin/develop", node="nid000001",
              triad: float | None = 3.0e12, groups: dict | None = None,
              data: dict[str, list] | None = None,
              extra: dict | None = None) -> Path:
    """Write results/<machine>/<name>/run.json and the group data files."""
    d = results / machine / name
    d.mkdir(parents=True, exist_ok=True)
    record = {
        "machine": machine, "node": node, "uenv": "u/1",
        "script_version": "s" * 40, "dataset_version": "2",
        "started": started, "finished": started,
        "gpu": {"gpu": "G", "driver": "1",
                "binding": {"srun": ["srun", "--ntasks=1"],
                            "visible_devices": "0", "cpus": "0-7"}},
        "toolchain": {}, "cmake_cache": {},
        "ginkgo": {"commit": commit, "commit_date": date,
                   "develop_base": commit, "ref": ref},
        "groups": groups if groups is not None else {
            g: {"status": "ok", "seconds": 1.0, "invocations": []}
            for g in GROUP_NAMES},
    }
    if triad is not None:
        record["stream"] = {"results": {"triad": {
            "bandwidth": triad, "avg_bandwidth": triad}}}
    record.update(extra or {})
    (d / "run.json").write_text(json.dumps(record))
    for group, rows in (data or {}).items():
        (d / f"{group}.json").write_text(json.dumps(rows))
    return d


def write_manifest(path: Path, matrices: list[dict], version="2") -> Path:
    """Write a dataset manifest with the given matrix entries."""
    path.write_text(json.dumps({"version": version, "matrices": matrices}))
    return path
