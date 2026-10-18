from pathlib import Path

import pytest

from bench.machine import load_machine

COMMON = (
    "-G Ninja -DCMAKE_BUILD_TYPE=Release -DGINKGO_BUILD_REFERENCE=ON "
    "-DGINKGO_BUILD_OMP=OFF -DGINKGO_BUILD_MPI=OFF -DGINKGO_BUILD_SYCL=OFF "
    "-DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_EXAMPLES=OFF -DGINKGO_BUILD_DOC=OFF "
    "-DGINKGO_DEVEL_TOOLS=OFF -DGINKGO_BUILD_BENCHMARKS=ON -DGINKGO_ENABLE_HALF=OFF "
    "-DGINKGO_ENABLE_BFLOAT16=OFF -DGINKGO_MIXED_PRECISION=OFF "
    "-DGINKGO_BUILD_PAPI_SDE=OFF -DCMAKE_DISABLE_FIND_PACKAGE_Kokkos=ON"
).split()


def test_load_daint():
    m = load_machine("daint-gh200", env={"SCRATCH": "/s"})
    assert m.name == "daint-gh200"
    assert m.executor == "cuda"
    assert m.uenv == "prgenv-gnu/26.3:v1"
    assert m.arch == "aarch64"
    assert m.needs_gcc_root is False
    assert m.sbatch == "--partition=normal --nodes=1 --exclusive --time=04:00:00 --account=d131".split()
    assert m.srun == "--ntasks=1 --gpus-per-task=1 --cpus-per-task=72 --cpu-bind=cores".split()
    assert m.cmake_flags == COMMON + (
        "-DGINKGO_BUILD_CUDA=ON -DGINKGO_CUDA_ARCHITECTURES=90 -DGINKGO_BUILD_HIP=OFF "
        "-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_HOST_COMPILER=g++"
    ).split()
    assert m.root == Path("/capstor/store/cscs/userlab/d131/cb")
    assert m.build_dir == Path("/s/cb/build/daint-gh200")
    assert m.results_dir == Path("/s/cb/results/daint-gh200")


def test_beverin_has_gcc_placeholder():
    m = load_machine("beverin-mi300a", env={"SCRATCH": "/s"})
    assert m.needs_gcc_root is True
    assert m.executor == "hip"
    assert m.arch == "x86_64"
    assert m.uenv == "prgenv-gnu/25.07-6.3.3:v12"
    assert m.sbatch[0] == "--partition=mi300"
    assert "--cpus-per-task=24" in m.srun
    assert m.cmake_flags[: len(COMMON)] == COMMON
    assert "-DCMAKE_C_COMPILER={gcc_root}/bin/gcc" in m.cmake_flags
    assert "-DCMAKE_HIP_FLAGS=--gcc-toolchain={gcc_root}" in m.cmake_flags
    assert "-DCMAKE_HIP_COMPILER=/user-environment/env/default/bin/clang++" in m.cmake_flags
    assert m.build_dir == Path("/s/cb/build/beverin-mi300a")


def test_unknown_machine():
    with pytest.raises(KeyError) as e:
        load_machine("x", env={"SCRATCH": "/s"})
    assert "daint-gh200" in str(e.value) and "beverin-mi300a" in str(e.value)


def test_missing_scratch():
    with pytest.raises(KeyError, match="SCRATCH"):
        load_machine("daint-gh200", env={})
