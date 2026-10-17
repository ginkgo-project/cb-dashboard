"""Configure and build Ginkgo's benchmark executables."""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable

from bench.machine import Machine

# mtx_to_binary is built from benchmark/tools, so exe() doesn't name it.
TARGETS = ["blas", "spmv", "sparse_blas", "solver", "preconditioner", "matrix_statistics",
           "mtx_to_binary"]
FINGERPRINT = "cb-config.json"

UENV_PREFIX = Path("/user-environment")
_CACHE_KEY = re.compile(r"^(CMAKE_BUILD_TYPE|CMAKE_\w*_COMPILER|CMAKE_\w*_FLAGS\w*|GINKGO_\w+)$")


def _default_candidates() -> list[Path]:
    return [Path("/user-environment/env/default/bin/g++")] + [
        Path(p) for p in sorted(glob.glob("/user-environment/linux-*/gcc-*/bin/g++"))
    ]


def _dumpversion(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def detect_gcc_root(
    candidates: Iterable[Path] | None = None,
    run: Callable[[list[str]], str] = _dumpversion,
    prefix: Path = UENV_PREFIX,
) -> Path:
    """GCC installation root of the uenv (as in 2ginkgo/build-mi300.sh)."""
    cands = list(_default_candidates() if candidates is None else candidates)
    gxx = next((Path(c) for c in cands if Path(c).is_file() and _executable(Path(c))), None)
    if gxx is None:
        raise RuntimeError("No GCC found in the uenv")
    gcc_bin = gxx.resolve().parent
    root = gcc_bin.parent
    major = int(run([str(gcc_bin / "g++"), "-dumpversion"]).strip().split(".")[0])
    if not gcc_bin.is_relative_to(Path(prefix).resolve()) or major < 9:
        raise RuntimeError(f"Need GCC 9+ from the uenv, found {gcc_bin}/g++ (major {major})")
    return root


def _executable(p: Path) -> bool:
    return os.access(p, os.X_OK)


def configure_args(m: Machine, src: Path, gcc_root: Path | None) -> list[str]:
    if m.needs_gcc_root and gcc_root is None:
        raise ValueError(f"{m.name} needs a gcc_root")
    flags = [f.replace("{gcc_root}", str(gcc_root)) if gcc_root else f for f in m.cmake_flags]
    return ["cmake", "-S", str(src), "-B", str(m.build_dir), *flags]


def read_cmake_cache(build_dir: Path) -> dict[str, str]:
    path = Path(build_dir) / "CMakeCache.txt"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith(("#", "//")) or "=" not in line:
            continue
        key, _, value = line.partition("=")
        name = key.split(":", 1)[0]
        if _CACHE_KEY.match(name):
            out[name] = value
    return out


def compiler_versions(run: Callable = subprocess.run) -> dict[str, str | None]:
    """The version line of each compiler on PATH, None where it isn't there."""
    out: dict[str, str | None] = {}
    for name, line in (("g++", 0), ("nvcc", -1), ("clang++", 0)):
        try:
            lines = run([name, "--version"], capture_output=True, text=True, check=True,
                        timeout=60).stdout.strip().splitlines()
            out[name] = lines[line] if lines else None
        except (OSError, subprocess.SubprocessError):
            out[name] = None
    return out


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def build(m: Machine, src: Path, jobs: int = 64) -> dict:
    """Configure if needed, then build TARGETS.

    It configures when there is no build.ninja (missing or half configured
    directory), or when the fingerprint in build_dir/cb-config.json (uenv,
    configure arguments, compiler versions) differs from this run's; then with
    `cmake --fresh` if a CMakeCache.txt is there to discard.
    """
    start = time.monotonic()
    gcc_root = detect_gcc_root() if m.needs_gcc_root else None
    args = configure_args(m, src, gcc_root)
    fingerprint = {"uenv": m.uenv, "configure_args": args, "compilers": compiler_versions()}
    fp_path = m.build_dir / FINGERPRINT
    changed = _read_json(fp_path) != fingerprint
    configured = changed or not (m.build_dir / "build.ninja").is_file()
    fresh = changed and (m.build_dir / "CMakeCache.txt").is_file()
    if configured:
        m.build_dir.mkdir(parents=True, exist_ok=True)
        fp_path.unlink(missing_ok=True)  # a failed configure must not leave a match
        subprocess.run([args[0], "--fresh", *args[1:]] if fresh else args, check=True)
        tmp = fp_path.with_name(fp_path.name + ".tmp")
        tmp.write_text(json.dumps(fingerprint, indent=2))
        os.replace(tmp, fp_path)
    subprocess.run(
        ["cmake", "--build", str(m.build_dir), "--parallel", str(jobs), "--target", *TARGETS],
        check=True,
    )
    return {
        "configured": configured,
        "fresh": fresh,
        "seconds": time.monotonic() - start,
        "cmake_cache": read_cmake_cache(m.build_dir),
    }


def exe(m: Machine, target: str) -> Path:
    return m.build_dir / "benchmark" / target / target
