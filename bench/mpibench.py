"""Build mpibench/ (the MPI benchmark of the distributed run) for a machine."""
from __future__ import annotations

import subprocess
from pathlib import Path

from bench.build import _cmake_env, detect_gcc_root
from bench.machine import Machine

SOURCE = Path(__file__).resolve().parent.parent / "mpibench"


def build(m: Machine, jobs: int = 16) -> Path:
    """Configure and build mpibench in the machine's mpibench build directory,
    with its mpibench_cmake_flags ({gcc_root} filled in as for Ginkgo)."""
    d = m.distributed
    if d is None:
        raise ValueError(f"{m.name} has no [distributed] settings")
    gcc_root = detect_gcc_root() if m.needs_gcc_root else None
    flags = [f.replace("{gcc_root}", str(gcc_root)) if gcc_root else f
             for f in d.mpibench_cmake_flags]
    # --fresh: each CI job may clone this repository into another runner
    # directory, and CMake refuses a cache made from another source path
    subprocess.run(["cmake", "--fresh", "-S", str(SOURCE), "-B", str(d.mpibench_build_dir),
                    *flags], check=True, env=_cmake_env())
    subprocess.run(["cmake", "--build", str(d.mpibench_build_dir), "--parallel", str(jobs)],
                   check=True, env=_cmake_env())
    return d.mpibench_build_dir / "mpibench"
