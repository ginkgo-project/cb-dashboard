"""mpibench/, built with host buffers and run with the local MPI."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from bench.runner import find_failures

REPO = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(
    not all(shutil.which(t) for t in ("mpicxx", "mpirun", "cmake")),
    reason="needs MPI and CMake")

SMALL = ["--max-allreduce-k", "3", "--max-alltoallv-k", "2", "--min-time", "0.001"]


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    build = tmp_path_factory.mktemp("mpibench")
    for cmd in (["cmake", "-S", str(REPO / "mpibench"), "-B", str(build), "-DMPIBENCH_GPU=none"],
                ["cmake", "--build", str(build), "-j4"]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
    return build / "mpibench"


def run(exe, n, *args):
    r = subprocess.run(["mpirun", "--oversubscribe", "-n", str(n), str(exe), *args],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def cases(data, op):
    return [c for c in data if c["operation"] == op]


@pytest.fixture(scope="module")
def four(exe):
    return run(exe, 4, *SMALL)


def test_allreduce_sweep(four):
    ar = cases(four, "allreduce")
    assert [c["count"] for c in ar] == [1, 4, 16, 64]
    for c in ar:
        res = c["mpi"]["allreduce"]
        assert c["bytes"] == 8 * c["count"] and c["ranks"] == 4
        assert res["completed"] is True
        assert 0 < res["min_time"] <= res["time"]
        assert res["repetitions"] >= 3


def test_alltoallv_and_neighbor(four):
    for op in ("alltoallv", "neighbor_alltoallv"):
        cs = cases(four, op)
        assert [c["count"] for c in cs] == [1, 4, 16]
        assert all(c["mpi"][op]["completed"] is True for c in cs)


def test_no_pingpong_without_flag(four):
    assert not [c for c in four if c["operation"].startswith("pingpong")]


def test_pingpong_pairs(exe):
    data = run(exe, 8, *SMALL, "--pingpong", "--max-pingpong-bytes", "64")
    for op in ("pingpong_intra", "pingpong_inter"):
        cs = cases(data, op)
        assert [c["bytes"] for c in cs] == [8, 32]
        assert all(c["mpi"][op]["completed"] is True for c in cs)


def test_single_rank(exe):
    data = run(exe, 1, *SMALL)
    assert [c["count"] for c in cases(data, "allreduce")] == [1, 4, 16, 64]
    assert all(c["mpi"][c["operation"]]["completed"] is True for c in data)


def test_output_passes_find_failures(four):
    assert find_failures(four) == (len(four), [])
