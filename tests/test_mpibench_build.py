import dataclasses
import shutil
import subprocess

import pytest

from bench import mpibench
from bench.machine import load_machine


@pytest.mark.skipif(not all(shutil.which(t) for t in ("mpicxx", "cmake")),
                    reason="needs MPI and CMake")
def test_build_with_host_buffers(tmp_path):
    m = load_machine("daint-gh200", env={"SCRATCH": str(tmp_path)})
    m = dataclasses.replace(m, distributed=dataclasses.replace(
        m.distributed, mpibench_cmake_flags=["-DMPIBENCH_GPU=none"]))
    exe = mpibench.build(m)
    assert exe == m.distributed.mpibench_build_dir / "mpibench"
    assert subprocess.run(["mpirun", "-n", "1", str(exe), "--max-allreduce-k", "0",
                           "--max-alltoallv-k", "0", "--min-time", "0"],
                          capture_output=True).returncode == 0


def test_gcc_root_is_filled_in(tmp_path, monkeypatch):
    m = load_machine("beverin-mi300a", env={"SCRATCH": str(tmp_path)})
    calls = []
    monkeypatch.setattr(mpibench, "detect_gcc_root", lambda: tmp_path / "gcc")
    monkeypatch.setattr(mpibench.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    mpibench.build(m)
    assert f"-DCMAKE_CXX_COMPILER={tmp_path}/gcc/bin/g++" in calls[0]
    assert not any("{gcc_root}" in a for a in calls[0])
    assert calls[1][:2] == ["cmake", "--build"]
