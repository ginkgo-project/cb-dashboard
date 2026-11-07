from pathlib import Path

import pytest

from bench.machine import load_machine

COMMON = (
    "-G Ninja -DCMAKE_BUILD_TYPE=Release -DGINKGO_BUILD_REFERENCE=ON "
    "-DGINKGO_BUILD_OMP=ON -DGINKGO_BUILD_MPI=OFF -DGINKGO_BUILD_SYCL=OFF "
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
    assert m.srun == ("--ntasks=1 --gpus-per-task=1 --cpus-per-task=72 --cpu-bind=cores "
                      "--export=ALL,OMP_NUM_THREADS=64").split()
    assert m.cmake_flags == COMMON + (
        "-DGINKGO_BUILD_CUDA=ON -DGINKGO_CUDA_ARCHITECTURES=90 -DGINKGO_BUILD_HIP=OFF "
        "-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_HOST_COMPILER=g++"
    ).split()
    assert m.root == Path("/capstor/store/cscs/userlab/d131/cb")
    assert m.build_dir == Path("/s/cb/build/daint-gh200")
    assert m.results_dir == Path("/s/cb/results/daint-gh200")
    assert m.disabled_preconditioners == []


def test_beverin_has_gcc_placeholder():
    m = load_machine("beverin-mi300a", env={"SCRATCH": "/s"})
    assert m.needs_gcc_root is True
    assert m.disabled_preconditioners == ["ilu-syncfree", "ic-syncfree"]
    assert m.executor == "hip"
    assert m.arch == "x86_64"
    assert m.uenv == "prgenv-gnu/25.07-6.3.3:v12"
    assert m.sbatch[0] == "--partition=mi300"
    assert m.srun == ("--ntasks=1 --gpus-per-task=1 --cpus-per-task=24 --hint=nomultithread "
                      "--export=ALL,OMP_NUM_THREADS=20").split()
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


@pytest.mark.parametrize("name", ["daint-gh200", "beverin-mi300a"])
def test_omp_threads_fit_in_cpus_per_task(name):
    """Fewer threads than cores leaves a few for the runtime daemons (CSCS GROMACS docs)."""
    srun = load_machine(name, env={"SCRATCH": "/s"}).srun
    cpus = int(next(a.split("=", 1)[1] for a in srun if a.startswith("--cpus-per-task=")))
    export = next(a for a in srun if a.startswith("--export="))
    threads = int(export.split("OMP_NUM_THREADS=", 1)[1].split(",")[0])
    assert export.startswith("--export=ALL,") and 0 < threads < cpus


@pytest.mark.parametrize("name", ["daint-gh200", "beverin-mi300a"])
def test_distributed_section_loaded(name):
    d = load_machine(name, env={"SCRATCH": "/s"}).distributed
    assert d.nodes == 16 and d.gpus_per_node == 4
    assert d.env["MPICH_GPU_SUPPORT_ENABLED"] == "1"
    assert "-DGINKGO_BUILD_MPI=ON" in d.cmake_flags
    assert "-DGINKGO_FORCE_GPU_AWARE_MPI=ON" in d.cmake_flags
    assert any(f.startswith("-DMPIBENCH_GPU=") for f in d.mpibench_cmake_flags)
    assert "--ntasks=1" not in d.srun and "--gpus-per-task=1" in d.srun


@pytest.mark.parametrize("name", ["daint-gh200", "beverin-mi300a"])
def test_distributed_paths_differ_from_single_node(name):
    m = load_machine(name, env={"SCRATCH": "/s"})
    d = m.distributed
    assert d.build_dir == Path(f"/s/cb/build/{name}-distributed")
    assert d.mpibench_build_dir == Path(f"/s/cb/build/{name}-mpibench")
    assert d.results_dir == Path(f"/s/cb/results/distributed/{name}")
    assert len({d.build_dir, d.mpibench_build_dir, m.build_dir}) == 3
    assert d.results_dir != m.results_dir


def test_machine_without_distributed_section(tmp_path):
    (tmp_path / "m.toml").write_text(
        'executor = "reference"\nuenv = "u"\narch = "x86_64"\nsbatch = []\nsrun = []\n')
    assert load_machine("m", machines_dir=tmp_path, env={"SCRATCH": "/s"}).distributed is None
