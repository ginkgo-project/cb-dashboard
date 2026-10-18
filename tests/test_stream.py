import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from bench import stream as S
from bench.machine import Machine, load_machine
from bench.runner import Killed

# As the HIP build prints it: device lines on stdout before the table.
OUT = """Using HIP device AMD Instinct MI300A
Driver: 60342134
Memory: DEFAULT
function,num_times,n_elements,sizeof,max_MB_per_sec,min_runtime,max_runtime,avg_runtime
Copy,100,134217728,8,3114000,0.000690,0.000720,0.000700
Mul,100,134217728,8,3100000,0.000693,0.000730,0.000710
Add,100,134217728,8,3140000,0.001026,0.001060,0.001040
Triad,100,134217728,8,3145000,0.001024,0.001050,0.001030
Dot,100,134217728,8,2983000,0.000720,0.000750,0.000730
"""


def machine(tmp_path, flags=("-DMODEL=cuda",), srun=()):
    return Machine("daint-gh200", "cuda", "u/1", [], list(srun), [], False, "aarch64",
                   tmp_path / "root", tmp_path / "s" / "build" / "daint-gh200",
                   tmp_path / "res", list(flags))


def fake(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "stream-exe"
    p.write_text(f"#!{sys.executable}\nimport sys\n{body}\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def test_parse_csv_skips_device_lines_and_gives_bytes_per_second():
    r = S.parse_csv(OUT)
    assert list(r) == ["copy", "mul", "add", "triad", "dot"]
    t = r["triad"]
    assert t["bandwidth"] == 3145000e6
    assert t["avg_bandwidth"] == pytest.approx(3145000e6 * 0.001024 / 0.001030)
    assert (t["min_runtime"], t["max_runtime"], t["avg_runtime"]) == (0.001024, 0.00105, 0.00103)


def test_parse_csv_other_unit():
    out = OUT.replace("max_MB_per_sec", "max_GB_per_sec").replace(",3145000,", ",3145,")
    assert S.parse_csv(out)["triad"]["bandwidth"] == 3145e9


@pytest.mark.parametrize("text, msg", [
    ("Using HIP device\n", "no CSV header"),
    (OUT.replace("max_MB_per_sec", "max_bananas_per_sec"), "no known bandwidth column"),
    ("\n".join(l for l in OUT.splitlines() if not l.startswith("Triad")), "no Triad"),
])
def test_parse_csv_rejects(text, msg):
    with pytest.raises(ValueError, match=msg):
        S.parse_csv(text)


def test_configure_args_daint():
    m = load_machine("daint-gh200", env={"SCRATCH": "/s"})
    assert S.configure_args(m, Path("/src"), Path("/b"), None) == [
        "cmake", "-S", "/src", "-B", "/b", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        "-DMODEL=cuda", "-DCUDA_ARCH=sm_90", "-DCMAKE_CUDA_COMPILER=nvcc",
        "-DCMAKE_CXX_COMPILER=g++", "-DCMAKE_CUDA_HOST_COMPILER=g++"]


def test_configure_args_beverin_fills_in_gcc_root():
    m = load_machine("beverin-mi300a", env={"SCRATCH": "/s"})
    args = S.configure_args(m, Path("/src"), Path("/b"), Path("/g"))
    assert args[8:] == ["-DMODEL=hip", "-DCMAKE_CXX_COMPILER=/user-environment/env/default/bin/hipcc",
                        "-DCXX_EXTRA_FLAGS=--offload-arch=gfx942;--gcc-toolchain=/g"]
    with pytest.raises(ValueError, match="gcc_root"):
        S.configure_args(m, Path("/src"), Path("/b"), None)


def test_machine_without_flags_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no stream_cmake_flags"):
        S.configure_args(machine(tmp_path, flags=()), Path("/src"), Path("/b"), None)


def test_paths(tmp_path):
    clone, worktree, build_dir = S.paths(machine(tmp_path))
    assert clone == tmp_path / "root/babelstream/clone"
    assert worktree == tmp_path / "root/babelstream/worktrees/daint-gh200"
    assert build_dir == tmp_path / "s/build/daint-gh200-babelstream"


def test_build_configures_only_when_something_changed(tmp_path, monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd[1])
        if cmd[1] == "-S":  # configure
            (Path(cmd[cmd.index("-B") + 1]) / "build.ninja").touch()
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(S.subprocess, "run", run)
    m, b = machine(tmp_path), tmp_path / "b"
    assert S.build(m, tmp_path / "src", b) == b / "cuda-stream"
    assert json.loads((b / S.FINGERPRINT).read_text())["commit"] == S.COMMIT
    S.build(m, tmp_path / "src", b)
    assert calls == ["-S", "--build", "--build"]
    (b / "stale.o").touch()
    m.stream_cmake_flags.append("-DUNROLL_FACTOR=2")
    S.build(m, tmp_path / "src", b)
    assert calls[3:] == ["-S", "--build"] and not (b / "stale.o").exists()


def test_run_parses_and_keeps_the_output(tmp_path):
    exe = fake(tmp_path, f"open({str(tmp_path / 'argv')!r}, 'w').write(' '.join(sys.argv[1:]))\n"
                         f"sys.stderr.write('note\\n'); print({OUT!r}, end='')")
    out = tmp_path / "out"
    r = S.run(machine(tmp_path), exe, out)
    assert r["triad"]["bandwidth"] == 3145000e6
    assert (out / "stream.txt").read_text() == OUT
    assert (out / "logs/stream.log").read_text() == "note\n"
    assert (tmp_path / "argv").read_text() == f"--csv --arraysize {2**27} --numtimes 100"


def test_run_failure_keeps_the_output(tmp_path):
    exe = fake(tmp_path, "print('FAILED validation'); sys.exit(1)")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="exit code 1"):
        S.run(machine(tmp_path), exe, out)
    assert (out / "stream.txt").read_text() == "FAILED validation\n"


def test_run_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "TIMEOUT_S", 1)
    monkeypatch.setattr(S, "KILL_GRACE_S", 1)
    exe = fake(tmp_path, "import time; time.sleep(60)")
    with pytest.raises(RuntimeError, match="timeout"):
        S.run(machine(tmp_path), exe, tmp_path / "out")


def test_srun_step_has_a_time_limit(tmp_path, monkeypatch):
    seen = []

    class Proc:
        returncode = 0
        pid = 0

        def __init__(self, cmd, **kw):
            seen.append(cmd)

        def communicate(self, timeout=None):
            return OUT.encode(), None
    monkeypatch.setattr(S.subprocess, "Popen", Proc)
    monkeypatch.setattr(S, "_killpg", lambda proc, sig: None)
    S.run(machine(tmp_path, srun=["-n1"]), Path("/x/cuda-stream"), tmp_path / "out")
    assert seen[0][:4] == ["srun", "-n1", f"--time={S.TIMEOUT_S // 60 + 1}", "/x/cuda-stream"]


def stub_measure(monkeypatch, tmp_path, fail=None):
    def clone(url, path):
        if fail:
            raise fail
    monkeypatch.setattr(S, "ensure_clone", clone)
    monkeypatch.setattr(S, "prepare_worktree", lambda clone, wt, sha: None)
    monkeypatch.setattr(S, "build", lambda m, src, b: b / "cuda-stream")
    monkeypatch.setattr(S, "run", lambda m, exe, out: {"triad": {"bandwidth": 1.0}, "exe": str(exe)})
    monkeypatch.setattr(S, "gpu_state", lambda m: {"nvidia-smi": {}})


def test_measure_records_results_and_gpu_state(tmp_path, monkeypatch):
    stub_measure(monkeypatch, tmp_path)
    m = machine(tmp_path)
    rec = S.measure(m, tmp_path / "out")
    assert rec["results"]["exe"] == str(S.paths(m)[2] / "cuda-stream")
    assert (rec["url"], rec["commit"], rec["array_size"], rec["num_times"]) == (
        S.URL, S.COMMIT, 2**27, 100)
    assert rec["gpu_after"] == {"nvidia-smi": {}} and "error" not in rec


def test_measure_records_an_error_and_still_reads_the_gpu(tmp_path, monkeypatch):
    stub_measure(monkeypatch, tmp_path, fail=subprocess.CalledProcessError(128, ["git"]))
    rec = S.measure(machine(tmp_path), tmp_path / "out")
    assert rec["error"].startswith("CalledProcessError") and "results" not in rec
    assert rec["gpu_after"] == {"nvidia-smi": {}}


def test_measure_lets_the_job_kill_through(tmp_path, monkeypatch):
    stub_measure(monkeypatch, tmp_path, fail=Killed())
    with pytest.raises(Killed):
        S.measure(machine(tmp_path), tmp_path / "out")
