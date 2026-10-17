import json
import subprocess
from pathlib import Path

from bench.checkout import Commit
from bench.machine import Machine
from bench.record import (
    gpu_info, parse_nvidia_smi, parse_rocm_smi, results_dir_name,
    run_record, toolchain_info,
)

DATA = Path(__file__).parent / "data"


def machine(executor="cuda"):
    return Machine("daint-gh200", executor, "u/1", [], ["srun", "-n1"], [], False,
                   "90", Path("/r"), Path("/b"), Path("/res"))


def test_dir_name_and_suffix(tmp_path):
    c = Commit("abcdef1234567890", "2026-10-16", "base")
    first = results_dir_name(tmp_path, c)
    assert first == tmp_path / "2026-10-16_abcdef1234"
    first.mkdir()
    second = results_dir_name(tmp_path, c)
    assert second == tmp_path / "2026-10-16_abcdef1234_2"
    second.mkdir()
    assert results_dir_name(tmp_path, c) == tmp_path / "2026-10-16_abcdef1234_3"


def test_parse_nvidia_smi_sample():
    r = parse_nvidia_smi((DATA / "nvidia-smi.csv").read_text())
    assert r == {
        "gpu": "NVIDIA GH200 480GB",
        "driver": "550.54.15",
        "clocks": {"sm_mhz": 1980, "mem_mhz": 2619, "max_sm_mhz": 1980,
                   "max_mem_mhz": 2619, "power_limit_w": 900.0},
    }


def test_parse_rocm_smi_sample():
    r = parse_rocm_smi((DATA / "rocm-smi.json").read_text())
    assert r == {
        "gpu": "AMD Instinct MI300A",
        "driver": "6.10.5",
        "clocks": {"sclk_mhz": 2100, "mclk_mhz": 1300, "fclk_mhz": 2100},
    }


def fake_run(calls):
    def run(cmd, **kw):
        calls.append(cmd)
        if "git" in cmd:
            out = "f" * 40 + "\n"
        elif "nvidia-smi" in cmd and "--query-gpu" in " ".join(cmd):
            out = (DATA / "nvidia-smi.csv").read_text()
        elif "rocm-smi" in cmd:
            out = (DATA / "rocm-smi.json").read_text()
        elif "nvcc" in cmd:
            out = "nvcc: NVIDIA\nCuda compilation tools, release 12.6, V12.6.20\n"
        elif "clang++" in cmd:
            out = "AMD clang version 18\nTarget: x86_64\n"
        elif "g++" in cmd:
            out = "g++ (GCC) 13.2.0\nCopyright\n"
        elif "cmake" in cmd:
            out = "cmake version 3.30.1\n\nfoo\n"
        else:  # binding probe
            out = "0\n0-71\n"
        return subprocess.CompletedProcess(cmd, 0, out, "")
    return run


def test_gpu_info_cuda_and_binding():
    calls = []
    g = gpu_info(machine(), run=fake_run(calls))
    assert g["gpu"] == "NVIDIA GH200 480GB"
    assert g["binding"] == {"srun": ["srun", "-n1"], "visible_devices": "0", "cpus": "0-71"}
    assert all(c[:2] == ["srun", "-n1"] for c in calls)


def test_gpu_info_hip():
    g = gpu_info(machine("hip"), run=fake_run([]))
    assert g["gpu"] == "AMD Instinct MI300A"
    assert g["binding"]["cpus"] == "0-71"


def test_toolchain_info():
    t = toolchain_info(machine(), run=fake_run([]))
    assert t["g++"] == "g++ (GCC) 13.2.0"
    assert t["cuda"] == "Cuda compilation tools, release 12.6, V12.6.20"
    assert t["cmake"] == "cmake version 3.30.1"
    t = toolchain_info(machine("hip"), run=fake_run([]))
    assert t["clang++"] == "AMD clang version 18"


def test_run_record_keys():
    c = Commit("a" * 40, "2026-10-16", "b" * 40)
    rec = run_record(
        machine(), c, "origin/develop",
        {"configured": True, "seconds": 3.0, "cmake_cache": {"X": "1"}},
        {"version": "v7"}, {"blas": {"status": "ok"}}, "t0", "t1",
        run=fake_run([]), repo=Path(__file__).parent.parent,
    )
    for k in ["machine", "node", "gpu", "toolchain", "uenv", "cmake_cache", "ginkgo",
              "dataset_version", "script_version", "started", "finished", "groups"]:
        assert k in rec, k
    assert rec["ginkgo"] == {"commit": "a" * 40, "commit_date": "2026-10-16",
                             "develop_base": "b" * 40, "ref": "origin/develop"}
    assert rec["dataset_version"] == "v7"
    assert rec["machine"] == "daint-gh200"
    assert len(rec["script_version"]) == 40
    json.dumps(rec)


NV_HEADER = (DATA / "nvidia-smi.csv").read_text().splitlines()[0]


def test_nvidia_na_values():
    text = NV_HEADER + "\nNVIDIA GH200, 550.1, [N/A], 2619 MHz, 1980 MHz, 2619 MHz, [Not Supported]\n"
    r = parse_nvidia_smi(text)
    assert r["gpu"] == "NVIDIA GH200"
    assert r["clocks"] == {"mem_mhz": 2619, "max_sm_mhz": 1980, "max_mem_mhz": 2619}


def test_nvidia_short_row_and_empty():
    r = parse_nvidia_smi(NV_HEADER + "\nNVIDIA GH200, 550.1\n")
    assert r == {"gpu": "NVIDIA GH200", "driver": "550.1", "clocks": {}}
    assert parse_nvidia_smi("")["gpu"] == ""


def test_rocm_without_card_key():
    r = parse_rocm_smi('{"system": {"Driver version": "6.1"}}')
    assert r == {"gpu": "", "driver": "6.1", "clocks": {}}


def test_gpu_info_empty_output_records_error():
    def run(cmd, **kw):
        out = "" if "nvidia-smi" in cmd else "0\n0-3\n"
        return subprocess.CompletedProcess(cmd, 0, out, "")
    g = gpu_info(machine(), run=run)
    assert "error" in g
    assert g["binding"]["cpus"] == "0-3"


def test_gpu_info_smi_fails_binding_kept():
    good = fake_run([])

    def run(cmd, **kw):
        if "nvidia-smi" in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        return good(cmd, **kw)
    g = gpu_info(machine(), run=run)
    assert "CalledProcessError" in g["error"]
    assert g["binding"]["cpus"] == "0-71"
    assert "error" not in g["binding"]


def test_gpu_info_bad_rocm_json_and_probe_failure():
    def run(cmd, **kw):
        if "rocm-smi" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "not json", "")
        raise OSError("no srun")
    g = gpu_info(machine("hip"), run=run)
    assert "error" in g and "error" in g["binding"]


def test_toolchain_failure_recorded():
    good = fake_run([])

    def run(cmd, **kw):
        if "nvcc" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return good(cmd, **kw)
    t = toolchain_info(machine(), run=run)
    assert t["g++"] == "g++ (GCC) 13.2.0" and "cuda" not in t
    assert "cuda" in t["error"]


def test_script_version_outside_git(tmp_path):
    c = Commit("a" * 40, "2026-10-16", "b" * 40)
    rec = run_record(machine(), c, "r", {"cmake_cache": {}}, {"version": "v"}, {}, "t0", "t1",
                     run=lambda cmd, **kw: (_ for _ in ()).throw(subprocess.CalledProcessError(128, cmd))
                     if "git" in cmd else fake_run([])(cmd, **kw), repo=tmp_path)
    assert "error" in rec["script_version"]
