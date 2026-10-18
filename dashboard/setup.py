"""What changed in the setup between two runs."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

from .load import Run


def fingerprint(run: Run) -> dict[str, str]:
    """The parts of the setup that can move a result."""
    rec = run.record
    toolchain = rec.get("toolchain")
    if isinstance(toolchain, dict):
        toolchain = {k: v for k, v in toolchain.items() if k != "error"}
    script = rec.get("script_version")
    gpu = rec.get("gpu")
    return {
        "uenv": str(rec.get("uenv", "")),
        "driver": str(gpu.get("driver", "")) if isinstance(gpu, dict) else "",
        "compilers": json.dumps(toolchain, sort_keys=True),
        "cmake": json.dumps(rec.get("cmake_cache"), sort_keys=True),
        "dataset": str(rec.get("dataset_version", "")),
        "script": script if isinstance(script, str) else "",
    }


def setup_changes(prev: Run, cur: Run,
                  script_changed: Callable[[str, str], bool]) -> list[str]:
    a, b = fingerprint(prev), fingerprint(cur)
    out = []
    for key in ("uenv", "driver", "dataset"):
        if a[key] != b[key]:
            out.append(f"{key}: {a[key]} → {b[key]}")
    if a["compilers"] != b["compilers"]:
        out.append("compilers changed")
    if a["cmake"] != b["cmake"]:
        out.append("CMake flags changed")
    if a["script"] != b["script"] and script_changed(a["script"], b["script"]):
        out.append(f"benchmark scripts: {a['script'][:7]} → {b['script'][:7]}")
    return out


def git_script_changed(repo: Path) -> Callable[[str, str], bool]:
    """True when bench/ or datasets/ differ between two commits (cached)."""
    cache: dict[tuple[str, str], bool] = {}

    def changed(a: str, b: str) -> bool:
        key = (a, b)
        if key not in cache:
            try:
                rc = subprocess.run(
                    ["git", "-C", str(repo), "diff", "--quiet", a, b, "--",
                     "bench", "datasets", ":!datasets/manifest.json"],
                    capture_output=True).returncode
            except OSError:
                rc = 1
            cache[key] = rc != 0
        return cache[key]

    return changed
