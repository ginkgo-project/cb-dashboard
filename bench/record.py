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


def _num(s: str) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        raise ValueError(f"no number in {s!r}")
    return float(m.group())


def parse_nvidia_smi(csv_text: str) -> dict:
    rows = list(csv.reader(io.StringIO(csv_text.strip()), skipinitialspace=True))
    if len(rows) < 2 or len(rows[1]) < 7:
        raise ValueError("unexpected nvidia-smi output")
    name, driver, sm, mem, max_sm, max_mem, power = [c.strip() for c in rows[1][:7]]
    return {
        "gpu": name,
        "driver": driver,
        "clocks": {
            "sm_mhz": int(_num(sm)),
            "mem_mhz": int(_num(mem)),
            "max_sm_mhz": int(_num(max_sm)),
            "max_mem_mhz": int(_num(max_mem)),
            "power_limit_w": _num(power),
        },
    }


def parse_rocm_smi(json_text: str) -> dict:
    data = json.loads(json_text)
    card = next(v for k, v in data.items() if k.startswith("card"))
    driver = ""
    for v in data.values():
        if isinstance(v, dict):
            for k, val in v.items():
                if k.lower().startswith("driver version"):
                    driver = str(val)
    clocks = {}
    for key in ("sclk", "mclk", "fclk"):
        for k, val in card.items():
            if k.startswith(f"{key} clock speed"):
                clocks[f"{key}_mhz"] = int(_num(str(val)))
    return {
        "gpu": card.get("Card Series") or card.get("Card series", ""),
        "driver": driver,
        "clocks": clocks,
    }


def _out(run, cmd: list[str]) -> str:
    return run(cmd, capture_output=True, text=True, check=True).stdout


def gpu_info(m: Machine, run: Callable = subprocess.run) -> dict:
    if m.executor == "cuda":
        cmd = ["nvidia-smi", "-i", "0",
               "--query-gpu=name,driver_version,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,power.limit",
               "--format=csv"]
        info = parse_nvidia_smi(_out(run, [*m.srun, *cmd]))
        var = "CUDA_VISIBLE_DEVICES"
    else:
        cmd = ["rocm-smi", "-d", "0", "--showproductname", "--showdriverversion",
               "--showclocks", "--json"]
        info = parse_rocm_smi(_out(run, [*m.srun, *cmd]))
        var = "ROCR_VISIBLE_DEVICES"
    probe = f'echo "${{{var}:-}}"; taskset -cp $$ | sed "s/.*: //"'
    lines = _out(run, [*m.srun, "sh", "-c", probe]).split("\n")
    info["binding"] = {
        "srun": m.srun,
        "visible_devices": lines[0].strip(),
        "cpus": lines[1].strip() if len(lines) > 1 else "",
    }
    return info


def toolchain_info(m: Machine, run: Callable = subprocess.run) -> dict:
    def lines(cmd):
        return _out(run, [*m.srun, *cmd]).strip().splitlines()

    t = {"g++": lines(["g++", "--version"])[0]}
    if m.executor == "cuda":
        t["cuda"] = lines(["nvcc", "--version"])[-1]
    else:
        t["clang++"] = lines(["clang++", "--version"])[0]
    t["cmake"] = lines(["cmake", "--version"])[0]
    return t


def run_record(m: Machine, commit: Commit, ref: str, build: dict, manifest: dict,
               groups: dict, started: str, finished: str,
               run: Callable = subprocess.run, repo: Path = REPO) -> dict:
    script = run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                 capture_output=True, text=True, check=True).stdout.strip()
    return {
        "machine": m.name,
        "node": socket.gethostname(),
        "gpu": gpu_info(m, run),
        "toolchain": toolchain_info(m, run),
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
