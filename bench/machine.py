"""Machine configurations for the benchmark run scripts."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

DEFAULT = Path(__file__).parent / "machines"
DEFAULT_ROOT = "/capstor/store/cscs/userlab/d131/cb"


@dataclass
class Distributed:
    """The weekly distributed run's settings (a machine file's [distributed])."""
    nodes: int
    gpus_per_node: int
    srun: list[str]  # per rank; bench.distributed adds the nodes and tasks
    cmake_flags: list[str]  # after the machine's, so a repeated -D overrides
    env: dict[str, str]
    mpibench_cmake_flags: list[str]
    build_dir: Path
    mpibench_build_dir: Path
    results_dir: Path
    # the command each rank runs its benchmark through, after srun's options
    wrapper: list[str] = field(default_factory=list)


KERNEL_TOOLS = ("ncu", "rocprofv3")


@dataclass
class Kernels:
    """The kernel counters' settings (a machine file's [kernels])."""
    tool: str  # one of KERNEL_TOOLS: how the counters are read and parsed
    profiler: list[str]  # the profiler's command, e.g. ["ncu"]
    tracer: list[str]  # the tracer's command for the counts, or [] if the profiler traces too
    metrics: list[list[str]]  # the counters, one list per profiler pass
    budget_s: int = 45 * 60  # the whole stage, per run
    timeout_s: int = 15 * 60  # one pass of one case


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
    stream_cmake_flags: list[str] = field(default_factory=list)
    disabled_preconditioners: list[str] = field(default_factory=list)
    distributed: Distributed | None = None
    kernels: Kernels | None = None


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
        stream_cmake_flags=list(data.get("stream_cmake_flags", [])),
        disabled_preconditioners=list(data.get("disabled_preconditioners", [])),
        distributed=_distributed(name, data.get("distributed"), scratch),
        kernels=_kernels(name, data.get("kernels")),
    )


def _distributed(name: str, data: dict | None, scratch: Path) -> Distributed | None:
    if data is None:
        return None
    return Distributed(
        nodes=int(data["nodes"]),
        gpus_per_node=int(data["gpus_per_node"]),
        srun=list(data["srun"]),
        cmake_flags=list(data.get("cmake_flags", [])),
        env={k: str(v) for k, v in data.get("env", {}).items()},
        mpibench_cmake_flags=list(data.get("mpibench_cmake_flags", [])),
        build_dir=scratch / "cb" / "build" / f"{name}-distributed",
        mpibench_build_dir=scratch / "cb" / "build" / f"{name}-mpibench",
        results_dir=scratch / "cb" / "results" / "distributed" / name,
        wrapper=list(data.get("wrapper", [])),
    )


def _kernels(name: str, data: dict | None) -> Kernels | None:
    if data is None:
        return None
    if data.get("tool") not in KERNEL_TOOLS:
        raise ValueError(f"{name}: [kernels] tool must be one of {', '.join(KERNEL_TOOLS)}, "
                         f"not {data.get('tool')!r}")
    return Kernels(
        tool=data["tool"],
        profiler=list(data["profiler"]),
        tracer=list(data.get("tracer", [])),
        metrics=[list(g) for g in data["metrics"]],
        budget_s=int(data.get("budget_s", 45 * 60)),
        timeout_s=int(data.get("timeout_s", 15 * 60)),
    )
