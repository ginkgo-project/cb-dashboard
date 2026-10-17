"""Machine configurations for the benchmark run scripts."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEFAULT = Path(__file__).parent / "machines"
DEFAULT_ROOT = "/capstor/store/cscs/userlab/d131/cb"


@dataclass
class Machine:
    name: str
    executor: str
    uenv: str
    sbatch: list[str]
    srun: list[str]
    cmake_flags: list[str]
    needs_gcc_root: bool
    arch: str
    root: Path
    build_dir: Path
    results_dir: Path


def _known(machines_dir: Path) -> list[str]:
    return sorted(p.stem for p in machines_dir.glob("*.toml") if p.stem != "common")


def load_machine(
    name: str,
    machines_dir: Path = DEFAULT,
    env: Mapping[str, str] = os.environ,
) -> Machine:
    machines_dir = Path(machines_dir)
    path = machines_dir / f"{name}.toml"
    if name == "common" or not path.is_file():
        raise KeyError(f"unknown machine {name!r}; known machines: {', '.join(_known(machines_dir))}")
    with path.open("rb") as f:
        data = tomllib.load(f)
    flags = list(data.get("cmake_flags", []))
    if "include" in data:
        with (machines_dir / data["include"]).open("rb") as f:
            flags = list(tomllib.load(f).get("cmake_flags", [])) + flags
    if "SCRATCH" not in env:
        raise KeyError("SCRATCH is not set in the environment; it is needed for build and results directories")
    scratch = Path(env["SCRATCH"])
    return Machine(
        name=name,
        executor=data["executor"],
        uenv=data["uenv"],
        sbatch=list(data["sbatch"]),
        srun=list(data["srun"]),
        cmake_flags=flags,
        needs_gcc_root=bool(data.get("needs_gcc_root", False)),
        arch=data["arch"],
        root=Path(data.get("root", DEFAULT_ROOT)),
        build_dir=scratch / "cb" / "build" / name,
        results_dir=scratch / "cb" / "results" / name,
    )
