"""The run record (run.json) and the results directory naming."""
from __future__ import annotations

import csv
import io
import json
import re
import socket
import subprocess
from pathlib import Path
from typing import Callable

from bench.checkout import Commit
from bench.machine import Machine

REPO = Path(__file__).resolve().parent.parent


def results_dir_name(base: Path, commit: Commit) -> Path:
    """`<date>_<sha10>`, or the first free `_2`, `_3`, ... suffix; never an existing one."""
    name = f"{commit.date}_{commit.sha[:10]}"
    path = base / name
    n = 2
    while path.exists():
        path = base / f"{name}_{n}"
        n += 1
    return path


ERRORS = (ValueError, KeyError, StopIteration, IndexError,
          subprocess.CalledProcessError, OSError)


def _num(s) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group()) if m else None


def _put(d: dict, key: str, val, as_int: bool = True) -> None:
    if val is not None:
        d[key] = int(val) if as_int else val


def parse_nvidia_smi(csv_text: str) -> dict:
    rows = [r for r in csv.reader(io.StringIO(csv_text.strip()), skipinitialspace=True) if r]
    row = [c.strip() for c in rows[1]] if len(rows) > 1 else []
    row += [""] * (7 - len(row))
    clocks: dict = {}
    for key, i in (("sm_mhz", 2), ("mem_mhz", 3), ("max_sm_mhz", 4), ("max_mem_mhz", 5)):
        _put(clocks, key, _num(row[i]))
    _put(clocks, "power_limit_w", _num(row[6]), as_int=False)
    return {"gpu": row[0], "driver": row[1], "clocks": clocks}


def parse_rocm_smi(json_text: str) -> dict:
    data = json.loads(json_text)
    if not isinstance(data, dict):
        raise ValueError("unexpected rocm-smi output")
    card = next((v for k, v in data.items() if k.startswith("card") and isinstance(v, dict)), {})
    driver = ""
    for v in data.values():
        if isinstance(v, dict):
            for k, val in v.items():
                if k.lower().startswith("driver version"):
                    driver = str(val)
    clocks: dict = {}
    for key in ("sclk", "mclk", "fclk"):
        for k, val in card.items():
            if k.startswith(f"{key} clock speed"):
                _put(clocks, f"{key}_mhz", _num(val))
    return {
        "gpu": card.get("Card Series") or card.get("Card series", ""),
        "driver": driver,
        "clocks": clocks,
    }


def _out(run, cmd: list[str]) -> str:
    return run(cmd, capture_output=True, text=True, check=True).stdout


def _prefix(m: Machine) -> list[str]:
    return ["srun", *m.srun] if m.srun else []


def gpu_info(m: Machine, run: Callable = subprocess.run) -> dict:
    cuda = m.executor == "cuda"
    var = "CUDA_VISIBLE_DEVICES" if cuda else "ROCR_VISIBLE_DEVICES"
    info: dict = {}
    try:
        if cuda:
            cmd = ["nvidia-smi", "-i", "0",
                   "--query-gpu=name,driver_version,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,power.limit",
                   "--format=csv"]
            info = parse_nvidia_smi(_out(run, [*_prefix(m), *cmd]))
        else:
            cmd = ["rocm-smi", "-d", "0", "--showproductname", "--showdriverversion",
                   "--showclocks", "--json"]
            info = parse_rocm_smi(_out(run, [*_prefix(m), *cmd]))
        if not info["gpu"]:
            info["error"] = "no GPU name in smi output"
    except ERRORS as e:
        info["error"] = f"{type(e).__name__}: {e}"
    binding = {"srun": _prefix(m), "visible_devices": "", "cpus": ""}
    probe = f'echo "${{{var}:-}}"; taskset -cp $$ | sed "s/.*: //"'
    try:
        lines = _out(run, [*_prefix(m), "sh", "-c", probe]).split("\n")
        binding["visible_devices"] = lines[0].strip()
        binding["cpus"] = lines[1].strip()
        if not binding["cpus"]:
            binding["error"] = "taskset reported no CPU list"
    except ERRORS as e:
        binding["error"] = f"{type(e).__name__}: {e}"
    info["binding"] = binding
    return info


def parse_smi_query(text: str) -> dict:
    """`nvidia-smi -q`'s indented `key : value` lines as nested dicts.

    A line without " : " (e.g. "GPU 00000009:01:00.0" or "Clocks") opens a section.
    """
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for line in text.splitlines():
        item = line.strip()
        if not item or item.startswith("="):
            continue
        indent = len(line) - len(line.lstrip())
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        kv = re.match(r"(.*?)\s+:\s*(.*)$", item)
        if kv:
            parent[kv.group(1)] = kv.group(2)
        else:
            stack.append((indent, parent.setdefault(item, {})))
    return root


def gpu_state(m: Machine, run: Callable = subprocess.run) -> dict:
    """Clocks, power, temperature and settings of GPU 0, as the smi tools report them.

    Read after the stream run (bench/stream.py), to explain a node that is slower.
    Each query that fails gets an "error" instead.
    """
    if m.executor == "cuda":
        queries = {"nvidia-smi": (["nvidia-smi", "-q", "-i", "0", "-d",
                                   "CLOCK,POWER,TEMPERATURE,PERFORMANCE,ECC"], parse_smi_query)}
    else:
        queries = {
            "rocm-smi": (["rocm-smi", "-d", "0", "--showclocks", "--showpower",
                          "--showmaxpower", "--showtemp", "--json"], json.loads),
            "partition": (["rocm-smi", "-d", "0", "--showcomputepartition",
                           "--showmemorypartition", "--json"], json.loads),
        }
    state: dict = {}
    for key, (cmd, parse) in queries.items():
        try:
            state[key] = parse(_out(run, [*_prefix(m), *cmd]))
        except ERRORS as e:
            state[key] = {"error": f"{type(e).__name__}: {e}"}
    return state


def toolchain_info(m: Machine, run: Callable = subprocess.run) -> dict:
    def first_last(cmd, idx):
        return _out(run, [*_prefix(m), *cmd]).strip().splitlines()[idx]

    items = [("g++", ["g++", "--version"], 0)]
    items.append(("cuda", ["nvcc", "--version"], -1) if m.executor == "cuda"
                 else ("clang++", ["clang++", "--version"], 0))
    items.append(("cmake", ["cmake", "--version"], 0))
    t: dict = {}
    errors = []
    for key, cmd, idx in items:
        try:
            t[key] = first_last(cmd, idx)
        except ERRORS as e:
            errors.append(f"{key}: {type(e).__name__}: {e}")
    if errors:
        t["error"] = "; ".join(errors)
    return t


def script_version(run: Callable, repo: Path) -> str | dict:
    try:
        return run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                   capture_output=True, text=True, check=True).stdout.strip()
    except ERRORS as e:
        return {"error": f"{type(e).__name__}: {e}"}


def run_record(m: Machine, commit: Commit, ref: str, build: dict, manifest: dict,
               groups: dict, started: str, finished: str | None,
               run: Callable = subprocess.run, repo: Path = REPO,
               gpu: dict | None = None, toolchain: dict | None = None) -> dict:
    script = script_version(run, repo)
    return {
        "machine": m.name,
        "node": socket.gethostname(),
        "gpu": gpu if gpu is not None else gpu_info(m, run),
        "toolchain": toolchain if toolchain is not None else toolchain_info(m, run),
        "uenv": m.uenv,
        "cmake_cache": build["cmake_cache"],
        "ginkgo": {"commit": commit.sha, "commit_date": commit.date,
                   "develop_base": commit.develop_base, "ref": ref},
        "dataset_version": manifest["version"],
        "script_version": script,
        "started": started,
        "finished": finished,
        "groups": groups,
    }
