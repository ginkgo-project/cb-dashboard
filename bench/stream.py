"""BabelStream: the node's memory bandwidth, measured with code that isn't Ginkgo's.

The dashboard divides the bandwidth of Ginkgo's memory-bound kernels by it, so a
slower node isn't taken for a regression. It can't be Ginkgo's own blas group: a
Ginkgo change to its vector kernels would then cancel out.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

from bench.build import _cmake_env, detect_gcc_root
from bench.checkout import ensure_clone, prepare_worktree
from bench.machine import Machine
from bench.record import gpu_state
from bench.runner import INTERRUPT_GRACE_S, KILL_GRACE_S, _killpg, _stop, srun_time_minutes

URL = "https://github.com/UoB-HPC/BabelStream.git"
# main after v6.0-rc, whose main.cpp doesn't build (fixed in 7136ab8), and before
# the rework of 2026-10. Its HIP dot is slow on large arrays (fixed in ea1ccc1),
# so the reference is triad.
COMMIT = "17ab377b0e919e14fd3df2b67268761fdac8abb3"
# Doubles per array: 1 GiB, as the largest blas vector, and far above the caches
# (the MI300A's Infinity Cache is 256 MB).
ARRAY_SIZE = 2**27
NUM_TIMES = 100
TIMEOUT_S = 10 * 60
FINGERPRINT = "cb-config.json"
UNITS = {"MB": 1e6, "GB": 1e9, "TB": 1e12, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40}


def paths(m: Machine) -> tuple[Path, Path, Path]:
    """The shared clone, this machine's worktree and its build directory."""
    base = m.root / "babelstream"
    return (base / "clone", base / "worktrees" / m.name,
            m.build_dir.with_name(m.build_dir.name + "-babelstream"))


def configure_args(m: Machine, src: Path, build_dir: Path, gcc_root: Path | None) -> list[str]:
    if not m.stream_cmake_flags:
        raise ValueError(f"{m.name} has no stream_cmake_flags")
    if m.needs_gcc_root and gcc_root is None:
        raise ValueError(f"{m.name} needs a gcc_root")
    flags = [f.replace("{gcc_root}", str(gcc_root)) if gcc_root else f
             for f in m.stream_cmake_flags]
    return ["cmake", "-S", str(src), "-B", str(build_dir), "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", *flags]


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def build(m: Machine, src: Path, build_dir: Path) -> Path:
    """Configure from scratch when the commit, uenv or flags changed; then build.

    Returns the executable, `<model>-stream`.
    """
    gcc_root = detect_gcc_root() if m.needs_gcc_root else None
    args = configure_args(m, src, build_dir, gcc_root)
    fingerprint = {"commit": COMMIT, "uenv": m.uenv, "configure_args": args}
    fp_path = build_dir / FINGERPRINT
    if _read_json(fp_path) != fingerprint or not (build_dir / "build.ninja").is_file():
        shutil.rmtree(build_dir, ignore_errors=True)
        build_dir.mkdir(parents=True)
        subprocess.run(args, check=True, env=_cmake_env())
        fp_path.write_text(json.dumps(fingerprint, indent=2))
    subprocess.run(["cmake", "--build", str(build_dir)], check=True, env=_cmake_env())
    model = next(f.split("=", 1)[1] for f in args if f.startswith("-DMODEL="))
    return build_dir / f"{model}-stream"


def parse_csv(text: str) -> dict:
    """BabelStream's --csv table, per kernel, after the device lines it may print first.

    "bandwidth" is in bytes/s, from the fastest repetition, as BabelStream reports
    it; "avg_bandwidth" is from the average time. Neither counts the first repetition.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("function,")), None)
    if start is None:
        raise ValueError("no CSV header in BabelStream's output")
    reader = csv.DictReader(lines[start:])
    unit = next((u.group(1) for c in reader.fieldnames
                 if (u := re.fullmatch(r"max_(\w+?)_per_sec", c))), None)
    if unit not in UNITS:
        raise ValueError(f"no known bandwidth column in {reader.fieldnames}")
    out = {}
    for row in reader:
        bw = float(row[f"max_{unit}_per_sec"]) * UNITS[unit]
        tmin, tavg = float(row["min_runtime"]), float(row["avg_runtime"])
        out[row["function"].lower()] = {
            "bandwidth": bw, "avg_bandwidth": bw * tmin / tavg, "min_runtime": tmin,
            "max_runtime": float(row["max_runtime"]), "avg_runtime": tavg}
    if "triad" not in out:
        raise ValueError(f"no Triad in BabelStream's output, only {sorted(out)}")
    return out


def run(m: Machine, exe: Path, out_dir: Path) -> dict:
    """Run it as a job step; its stdout goes to stream.txt and stderr to logs/stream.log."""
    prefix = ["srun", *m.srun, f"--time={srun_time_minutes(TIMEOUT_S)}"] if m.srun else []
    cmd = [*prefix, str(exe), "--csv", "--arraysize", str(ARRAY_SIZE),
           "--numtimes", str(NUM_TIMES)]
    log = out_dir / "logs" / "stream.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "wb") as errf:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=errf, start_new_session=True)
        try:
            stdout, _ = proc.communicate(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _stop(proc, KILL_GRACE_S)
            raise RuntimeError(f"timeout after {TIMEOUT_S} s") from None
        except BaseException:
            _stop(proc, INTERRUPT_GRACE_S)
            raise
        _killpg(proc, signal.SIGKILL)  # stragglers in the group, e.g. srun children
    text = stdout.decode(errors="replace")
    (out_dir / "stream.txt").write_text(text)
    if proc.returncode != 0:
        raise RuntimeError(f"exit code {proc.returncode}")
    return parse_csv(text)


def measure(m: Machine, out_dir: Path) -> dict:
    """Build and run BabelStream, then read the GPU's state, for run.json's "stream".

    An error is recorded, not raised, so the benchmarks still run; only Killed
    (the job's SIGTERM) gets through.
    """
    rec: dict = {"url": URL, "commit": COMMIT, "array_size": ARRAY_SIZE,
                 "num_times": NUM_TIMES}
    t = time.monotonic()
    try:
        clone, worktree, build_dir = paths(m)
        ensure_clone(URL, clone)
        prepare_worktree(clone, worktree, COMMIT)
        rec["results"] = run(m, build(m, worktree, build_dir), out_dir)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["seconds"] = time.monotonic() - t
    rec["gpu_after"] = gpu_state(m)
    return rec
