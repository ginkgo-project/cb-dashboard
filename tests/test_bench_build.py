import json
import os
import subprocess
from pathlib import Path

import pytest

from bench import build as B
from bench.machine import load_machine


def machine(name, tmp_path):
    return load_machine(name, env={"SCRATCH": str(tmp_path / "scratch")})


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    for tool in ("cmake", "ninja"):
        p = bindir / tool
        p.write_text(f'#!/bin/sh\necho "{tool} $*" >> "{log}"\n')
        p.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setattr(B, "compiler_versions", lambda: dict(COMPILERS))
    return log


COMPILERS = {"g++": "g++ (GCC) 13.2.0", "nvcc": "Cuda compilation tools, release 12.6",
             "clang++": None}


def test_targets():
    assert B.TARGETS == ["blas", "spmv", "sparse_blas", "solver", "preconditioner",
                         "matrix_statistics", "mtx_to_binary"]


def test_exe(tmp_path):
    m = machine("daint-gh200", tmp_path)
    assert B.exe(m, "spmv") == m.build_dir / "benchmark" / "spmv" / "spmv"


def test_configure_args_daint(tmp_path):
    m = machine("daint-gh200", tmp_path)
    args = B.configure_args(m, Path("/src"), None)
    assert args == ["cmake", "-S", "/src", "-B", str(m.build_dir), *m.cmake_flags]
    assert args[5:7] == ["-G", "Ninja"]
    assert "-DGINKGO_CUDA_ARCHITECTURES=90" in args
    assert "-DCMAKE_C_COMPILER=gcc" in args


def test_configure_args_beverin_fills_gcc_root(tmp_path):
    m = machine("beverin-mi300a", tmp_path)
    args = B.configure_args(m, Path("/src"), Path("/g"))
    assert "-DCMAKE_HIP_FLAGS=--gcc-toolchain=/g" in args
    assert "-DCMAKE_CXX_COMPILER=/g/bin/g++" in args
    assert not any("{gcc_root}" in a for a in args)


def test_configure_args_beverin_requires_gcc_root(tmp_path):
    m = machine("beverin-mi300a", tmp_path)
    with pytest.raises(ValueError):
        B.configure_args(m, Path("/src"), None)


def test_fresh_configure_when_no_ninja_file(tmp_path, fakes):
    m = machine("daint-gh200", tmp_path)
    m.build_dir.mkdir(parents=True)  # empty, half configured
    r = B.build(m, Path("/src"), jobs=7)
    calls = fakes.read_text().splitlines()
    assert calls[0].startswith("cmake -S /src -B " + str(m.build_dir))
    assert calls[1] == (
        f"cmake --build {m.build_dir} --parallel 7 --target " + " ".join(B.TARGETS)
    )
    assert r["configured"] is True
    assert r["seconds"] >= 0


def test_compilers_and_flags_in_the_environment_are_dropped(tmp_path, fakes, monkeypatch):
    """A CC left in the submit shell (e.g. from another uenv) must not reach CMake."""
    (tmp_path / "bin" / "cmake").write_text(
        f'#!/bin/sh\necho "CC=${{CC-unset}} CXXFLAGS=${{CXXFLAGS-unset}} HOME=${{HOME:+set}}" >> "{fakes}"\n')
    monkeypatch.setenv("CC", "/old-uenv/bin/gcc")
    monkeypatch.setenv("CXXFLAGS", "-O0")
    monkeypatch.setenv("HOME", "/home/u")
    B.build(machine("daint-gh200", tmp_path), Path("/src"))
    assert fakes.read_text().splitlines() == ["CC=unset CXXFLAGS=unset HOME=set"] * 2


def test_fresh_configure_when_build_dir_missing(tmp_path, fakes):
    m = machine("daint-gh200", tmp_path)
    assert B.build(m, Path("/src"))["configured"] is True


def configured_once(m, fakes):
    """A first build, as cmake would leave it: build.ninja and CMakeCache.txt."""
    B.build(m, Path("/src"))
    (m.build_dir / "build.ninja").write_text("")
    (m.build_dir / "CMakeCache.txt").write_text("")
    fakes.write_text("")


def test_no_configure_when_ninja_file_exists(tmp_path, fakes):
    m = machine("daint-gh200", tmp_path)
    configured_once(m, fakes)
    r = B.build(m, Path("/src"))
    calls = fakes.read_text().splitlines()
    assert len(calls) == 1 and calls[0].startswith("cmake --build ")
    assert "-S" not in calls[0]
    assert r["configured"] is False and r["fresh"] is False


def test_fingerprint_written(tmp_path, fakes):
    m = machine("daint-gh200", tmp_path)
    B.build(m, Path("/src"))
    fp = json.loads((m.build_dir / "cb-config.json").read_text())
    assert fp == {"uenv": m.uenv, "configure_args": B.configure_args(m, Path("/src"), None),
                  "compilers": COMPILERS}


def test_fingerprint_has_gcc_root_substituted(tmp_path, fakes, monkeypatch):
    monkeypatch.setattr(B, "detect_gcc_root", lambda: Path("/g"))
    m = machine("beverin-mi300a", tmp_path)
    B.build(m, Path("/src"))
    args = json.loads((m.build_dir / "cb-config.json").read_text())["configure_args"]
    assert "-DCMAKE_CXX_COMPILER=/g/bin/g++" in args
    assert not any("{gcc_root}" in a for a in args)


def assert_fresh_configure(m, fakes, r):
    calls = fakes.read_text().splitlines()
    assert calls[0].startswith(f"cmake --fresh -S /src -B {m.build_dir} ")
    assert calls[1].startswith("cmake --build ")
    assert r["configured"] is True and r["fresh"] is True


def test_changed_flag_reconfigures_fresh(tmp_path, fakes):
    m = machine("daint-gh200", tmp_path)
    configured_once(m, fakes)
    m.cmake_flags = [*m.cmake_flags, "-DGINKGO_MIXED_PRECISION=ON"]
    r = B.build(m, Path("/src"))
    assert_fresh_configure(m, fakes, r)
    assert "-DGINKGO_MIXED_PRECISION=ON" in fakes.read_text().splitlines()[0]
    # The new fingerprint is stored, so the next build doesn't configure.
    fakes.write_text("")
    assert B.build(m, Path("/src"))["configured"] is False


def test_changed_uenv_reconfigures_fresh(tmp_path, fakes):
    m = machine("daint-gh200", tmp_path)
    configured_once(m, fakes)
    m.uenv = "prgenv-gnu/26.9:v1"
    assert_fresh_configure(m, fakes, B.build(m, Path("/src")))


def test_changed_compiler_reconfigures_fresh(tmp_path, fakes, monkeypatch):
    m = machine("daint-gh200", tmp_path)
    configured_once(m, fakes)
    monkeypatch.setattr(B, "compiler_versions", lambda: {**COMPILERS, "nvcc": "release 12.8"})
    assert_fresh_configure(m, fakes, B.build(m, Path("/src")))


def test_old_build_dir_without_fingerprint_reconfigures_fresh(tmp_path, fakes):
    m = machine("daint-gh200", tmp_path)
    m.build_dir.mkdir(parents=True)
    (m.build_dir / "build.ninja").write_text("")
    (m.build_dir / "CMakeCache.txt").write_text("")
    assert_fresh_configure(m, fakes, B.build(m, Path("/src")))


def test_failed_configure_leaves_no_fingerprint(tmp_path, fakes, monkeypatch):
    m = machine("daint-gh200", tmp_path)
    configured_once(m, fakes)
    m.uenv = "other"
    bindir = tmp_path / "bin"
    (bindir / "cmake").write_text("#!/bin/sh\nexit 1\n")
    with pytest.raises(subprocess.CalledProcessError):
        B.build(m, Path("/src"))
    assert not (m.build_dir / "cb-config.json").exists()


def test_compiler_versions_probe():
    def run(cmd, **kw):
        if cmd[0] == "clang++":
            raise FileNotFoundError(cmd[0])
        out = {"g++": "g++ (GCC) 13.2.0\nCopyright\n",
               "nvcc": "nvcc: NVIDIA\nCuda compilation tools, release 12.6\n"}[cmd[0]]
        return subprocess.CompletedProcess(cmd, 0, out, "")
    assert B.compiler_versions(run) == {"g++": "g++ (GCC) 13.2.0",
                                        "nvcc": "Cuda compilation tools, release 12.6",
                                        "clang++": None}


def test_build_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "compiler_versions", lambda: dict(COMPILERS))
    bindir = tmp_path / "bin"
    bindir.mkdir()
    p = bindir / "cmake"
    p.write_text("#!/bin/sh\nexit 3\n")
    p.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    m = machine("daint-gh200", tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        B.build(m, Path("/src"))


def test_cmake_cache_subset(tmp_path, fakes):
    m = machine("daint-gh200", tmp_path)
    configured_once(m, fakes)
    (m.build_dir / "CMakeCache.txt").write_text(
        "# comment\n// help\n"
        "CMAKE_BUILD_TYPE:STRING=Release\n"
        "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++\n"
        "CMAKE_CXX_FLAGS:STRING=\n"
        "CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG\n"
        "GINKGO_BUILD_CUDA:BOOL=ON\n"
        "CMAKE_INSTALL_PREFIX:PATH=/usr/local\n"
        "CMAKE_GENERATOR:INTERNAL=Ninja\n"
        "OTHER_GINKGO_X:BOOL=ON\n"
    )
    assert B.build(m, Path("/src"))["cmake_cache"] == {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CXX_COMPILER": "/usr/bin/g++",
        "CMAKE_CXX_FLAGS": "",
        "CMAKE_CXX_FLAGS_RELEASE": "-O3 -DNDEBUG",
        "GINKGO_BUILD_CUDA": "ON",
    }


def test_missing_cache_gives_empty(tmp_path):
    assert B.read_cmake_cache(tmp_path / "nope") == {}


def make_gcc(tmp_path, version="12.3.0", under="user-environment"):
    root = tmp_path / under / "linux-sapphirerapids" / "gcc-12"
    (root / "bin").mkdir(parents=True)
    g = root / "bin" / "g++"
    g.write_text("")
    g.chmod(0o755)
    return root, g


def test_detect_gcc_root_resolves_symlink(tmp_path):
    root, g = make_gcc(tmp_path)
    view = tmp_path / "view"
    view.mkdir()
    (view / "g++").symlink_to(g)
    got = B.detect_gcc_root(
        candidates=[tmp_path / "missing", view / "g++", g],
        run=lambda cmd: "12.3.0\n",
        prefix=tmp_path / "user-environment",
    )
    assert got == root.resolve()


def test_detect_gcc_root_rejects_old_gcc(tmp_path):
    _, g = make_gcc(tmp_path)
    with pytest.raises(RuntimeError, match="GCC 9"):
        B.detect_gcc_root([g], run=lambda cmd: "7.5.0\n", prefix=tmp_path / "user-environment")


def test_detect_gcc_root_rejects_outside_uenv(tmp_path):
    _, g = make_gcc(tmp_path, under="elsewhere")
    with pytest.raises(RuntimeError, match="GCC 9"):
        B.detect_gcc_root([g], run=lambda cmd: "12.3.0\n", prefix=tmp_path / "user-environment")


def test_detect_gcc_root_none_found(tmp_path):
    with pytest.raises(RuntimeError, match="No GCC"):
        B.detect_gcc_root([tmp_path / "x"], run=lambda cmd: "12\n", prefix=tmp_path)


def test_build_beverin_detects_gcc_root(tmp_path, fakes, monkeypatch):
    monkeypatch.setattr(B, "detect_gcc_root", lambda: Path("/g"))
    m = machine("beverin-mi300a", tmp_path)
    B.build(m, Path("/src"))
    assert "--gcc-toolchain=/g" in fakes.read_text()
