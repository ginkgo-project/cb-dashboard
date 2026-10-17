import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bench import runner
from bench.groups import Invocation
from bench.runner import run_group


def fake(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(f"#!{sys.executable}\nimport sys\n{body}\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def inv(exe, label="x", timeout=30, missing=None, args=None):
    return Invocation("g", label, exe, args or ["-a", "1"], None, timeout, missing)


def test_concatenates_arrays(tmp_path):
    a = fake(tmp_path, "a", 'print(\'[{"a":1}]\')')
    b = fake(tmp_path, "b", 'print(\'[{"b":2}]\')')
    out = tmp_path / "out"
    r = run_group("g", [inv(a, "a"), inv(b, "b")], [], out)
    assert r["status"] == "ok"
    assert json.loads((out / "g.json").read_text()) == [{"a": 1}, {"b": 2}]
    assert r["invocations"][0]["args"] == ["-a", "1"]
    assert r["invocations"][0]["exit_code"] == 0


def test_crash_is_partial(tmp_path):
    ok = fake(tmp_path, "ok", 'print(\'[{"a":1}]\')')
    bad = fake(tmp_path, "bad", 'sys.stderr.write("boom\\n"); sys.exit(3)')
    out = tmp_path / "out"
    r = run_group("g", [inv(bad, "bad"), inv(ok, "ok")], [], out)
    assert r["status"] == "partial"
    assert r["invocations"][0]["status"] == "failed"
    assert r["invocations"][0]["exit_code"] == 3
    assert "boom" in (out / "logs/g/bad.log").read_text()
    assert json.loads((out / "g.json").read_text()) == [{"a": 1}]


def test_truncated_json_is_failed(tmp_path):
    t = fake(tmp_path, "t", 'sys.stdout.write(\'[{"a":1},{"b"\')')
    out = tmp_path / "out"
    r = run_group("g", [inv(t)], [], out)
    assert r["status"] == "failed"
    assert r["invocations"][0]["status"] == "failed"
    assert json.loads((out / "g.json").read_text()) == []


def test_timeout(tmp_path):
    s = fake(tmp_path, "s", "import time; time.sleep(5)")
    r = run_group("g", [inv(s, timeout=1)], [], tmp_path / "out")
    assert r["invocations"][0]["reason"] == "timeout"
    assert r["status"] == "failed"
    assert r["seconds"] < 4


def test_missing_input_not_started(tmp_path):
    marker = tmp_path / "started"
    s = fake(tmp_path, "s", f'open({str(marker)!r}, "w")')
    r = run_group("g", [inv(s, missing=tmp_path / "nope.mtx")], [], tmp_path / "out")
    assert r["invocations"][0]["reason"] == "missing input"
    assert r["invocations"][0]["status"] == "failed"
    assert not marker.exists()


def test_stdin_and_large_output(tmp_path):
    s = fake(tmp_path, "s",
             'd = sys.stdin.read(); print(json.dumps([{"d": d}] + [{"i": "x" * 100}] * 200000))'
             .replace("print(json", "import json; print(json"))
    i = inv(s)
    i.stdin = "hello"
    out = tmp_path / "out"
    r = run_group("g", [i], [], out)
    assert r["status"] == "ok"
    assert json.loads((out / "g.json").read_text())[0] == {"d": "hello"}


def test_empty_group_writes_empty_array(tmp_path):
    out = tmp_path / "out"
    r = run_group("g", [], [], out)
    assert r == {"status": "ok", "seconds": 0.0, "invocations": []}
    assert json.loads((out / "g.json").read_text()) == []


def test_mismatched_group_rejected(tmp_path):
    s = fake(tmp_path, "s", "pass")
    with pytest.raises(ValueError):
        run_group("other", [inv(s)], [], tmp_path / "out")


# Ginkgo's output shapes come from benchmark/utils/runner.hpp: an operation that
# throws gets {"completed": false, "error_type", "error"}, a case whose setup
# throws gets "error_type" and "error" on the case itself; the exit code is 0.
def ginkgo_fake(tmp_path, name, cases):
    return fake(tmp_path, name, f"import json; print(json.dumps({cases!r}))")


def test_failed_operation_is_partial(tmp_path):
    cases = [{"filename": "/m/web.bin", "spmv": {
        "csr": {"time": 1.0, "completed": True},
        "coo": {"completed": False, "error_type": "gko::OutOfBoundsError", "error": "oob"}}}]
    r = run_group("g", [inv(ginkgo_fake(tmp_path, "f", cases))], [], tmp_path / "out")
    i = r["invocations"][0]
    assert i["status"] == "partial" and i["exit_code"] == 0
    assert i["failures"] == [{"case": "/m/web.bin", "op": "coo",
                              "error_type": "gko::OutOfBoundsError", "error": "oob"}]
    assert r["status"] == "partial"
    # The raw output is kept unchanged.
    assert json.loads((tmp_path / "out/g.json").read_text()) == cases


def test_case_level_error_is_failed(tmp_path):
    cases = [{"size": 8000000, "stencil": "27pt", "optimal": {"spmv": "csr"}, "solver": {},
              "error_type": "gko::AllocationError", "error": "out of memory"}]
    r = run_group("g", [inv(ginkgo_fake(tmp_path, "f", cases))], [], tmp_path / "out")
    i = r["invocations"][0]
    assert i["status"] == "failed" and i["exit_code"] == 0
    assert i["failures"] == [{"case": "stencil(8000000, 27pt)", "op": None,
                              "error_type": "gko::AllocationError", "error": "out of memory"}]
    assert r["status"] == "failed"


def test_all_operations_failed_is_failed(tmp_path):
    cases = [{"n": 100, "blas": {"dot": {"completed": False, "error": "x"},
                                 "axpy": {"completed": False, "error": "y"}}}]
    r = run_group("g", [inv(ginkgo_fake(tmp_path, "f", cases))], [], tmp_path / "out")
    i = r["invocations"][0]
    assert i["status"] == "failed" and i["reason"] == "2 of 2 operations failed"
    assert [f["op"] for f in i["failures"]] == ["dot", "axpy"]
    assert i["failures"][0]["case"] == "n = 100"


def test_partial_invocation_and_ok_one_make_partial_group(tmp_path):
    ok = fake(tmp_path, "ok", 'print(\'[{"n": 1, "blas": {"dot": {"completed": true}}}]\')')
    bad = ginkgo_fake(tmp_path, "bad", [{"n": 2, "blas": {"dot": {"completed": False}}}])
    r = run_group("g", [inv(ok, "ok"), inv(bad, "bad")], [], tmp_path / "out")
    assert [i["status"] for i in r["invocations"]] == ["ok", "failed"]
    assert r["invocations"][0]["failures"] == []
    assert r["status"] == "partial"


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"


def _wait_gone(pid: int, seconds: float = 5.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


# Stands in for srun: the step's task runs outside srun's process group (under
# slurmstepd on the cluster), so it ends only if srun forwards SIGTERM to it.
FORWARDING_SRUN = '''
import os, signal, subprocess, time
p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                     start_new_session=True)
open({pidfile!r}, "w").write(str(p.pid))
def forward(signum, frame):
    p.send_signal(signum)
    p.wait()
    sys.exit(143)
signal.signal(signal.SIGTERM, forward)
time.sleep(120)
'''

# A child and grandchild in one process group that both ignore SIGTERM.
STUBBORN = '''
import signal, subprocess, time
child = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"
p = subprocess.Popen([sys.executable, "-c", child])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
open({pidfile!r}, "w").write(str(p.pid))
time.sleep(120)
'''


@pytest.mark.parametrize("body", [FORWARDING_SRUN, STUBBORN], ids=["forwarded", "sigkill"])
def test_timeout_stops_grandchild(tmp_path, monkeypatch, body):
    # SIGTERM first, so srun can forward it; SIGKILL after the grace period.
    monkeypatch.setattr(runner, "KILL_GRACE_S", 1)
    pidfile = tmp_path / "pid"
    s = fake(tmp_path, "s", body.format(pidfile=str(pidfile)))
    t = time.monotonic()
    try:
        r = run_group("g", [inv(s, timeout=2)], [], tmp_path / "out")
        assert r["invocations"][0]["reason"] == "timeout"
        assert time.monotonic() - t < 2 + 1 + 3
        pid = int(pidfile.read_text())
        assert _wait_gone(pid), f"grandchild {pid} survived the timeout"
    finally:
        if pidfile.exists() and _alive(int(pidfile.read_text())):
            os.kill(int(pidfile.read_text()), signal.SIGKILL)


def test_srun_gets_step_time_limit(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "srun.args"
    fake(bindir, "srun", f'''import os
args = sys.argv[1:]
i = 0
while args[i].startswith("-"):
    i += 1
open({str(log)!r}, "w").write(" ".join(args[:i]))
os.execv(args[i], args[i:])''')
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    ok = fake(tmp_path, "ok", 'print("[]")')
    r = run_group("g", [inv(ok, timeout=90)], ["--ntasks=1", "--gpus-per-task=1"], tmp_path / "out")
    assert r["status"] == "ok"
    assert log.read_text() == "--ntasks=1 --gpus-per-task=1 --time=3"
    assert [runner.srun_time_minutes(t) for t in (60, 61, 900, 3600)] == [2, 3, 16, 61]


def test_not_enough_job_time_skips(tmp_path):
    marker = tmp_path / "started"
    short = fake(tmp_path, "short", 'print("[]")')
    long = fake(tmp_path, "long", f'open({str(marker)!r}, "w")')
    r = run_group("g", [inv(short, "short", timeout=10), inv(long, "long", timeout=3600)], [],
                  tmp_path / "out", deadline=time.time() + 600)
    assert [i["status"] for i in r["invocations"]] == ["ok", "skipped"]
    assert r["invocations"][1]["reason"] == "not enough job time"
    assert not marker.exists()
    assert r["status"] == "partial"
    r = run_group("g", [inv(long, timeout=3600)], [], tmp_path / "out", deadline=time.time() + 5)
    assert r["status"] == "skipped"


def test_killed_keeps_finished_invocations(tmp_path):
    ok = fake(tmp_path, "ok", 'print(\'[{"a":1}]\')')
    term = fake(tmp_path, "term", "import os, signal, time\n"
                "os.kill(os.getppid(), signal.SIGTERM)\ntime.sleep(30)")

    def handler(signum, frame):
        raise runner.Killed(signum)
    previous = signal.signal(signal.SIGTERM, handler)
    t = time.monotonic()
    try:
        with pytest.raises(runner.Killed) as e:
            run_group("g", [inv(ok, "ok"), inv(term, "term"), inv(ok, "never")], [],
                      tmp_path / "out")
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert time.monotonic() - t < 10
    g = e.value.group_result
    assert g["status"] == "killed"
    assert [(i["label"], i["status"]) for i in g["invocations"]] == [("ok", "ok"), ("term", "killed")]
    assert json.loads((tmp_path / "out/g.json").read_text()) == [{"a": 1}]


@pytest.mark.parametrize("text,seconds", [
    ("1-02:03:04", 93784), ("2:03:04", 7384), ("03:04", 184), ("59", 59),
    ("UNLIMITED", None), ("NOT_SET", None), ("", None)])
def test_parse_time_left(text, seconds):
    assert runner.parse_time_left(text) == seconds


def test_job_deadline_sources():
    def no_run(cmd, **kw):
        raise AssertionError("ran squeue")
    assert runner.job_deadline({"SLURM_JOB_END_TIME": "1700000000"}, run=no_run) == 1700000000.0
    assert runner.job_deadline({}, run=no_run) is None
    assert runner.job_deadline({"SLURM_JOB_END_TIME": "0"}, run=no_run) is None
    assert runner.job_deadline({"SLURM_JOB_END_TIME": "junk"}, run=no_run) is None
    calls = []

    def squeue(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "1:00:00\n", "")
    assert runner.job_deadline({"SLURM_JOB_ID": "42"}, run=squeue, now=lambda: 1000.0) == 4600.0
    assert calls == [["squeue", "-h", "-j", "42", "-o", "%L"]]

    def broken(cmd, **kw):
        raise OSError("no squeue")
    assert runner.job_deadline({"SLURM_JOB_ID": "42"}, run=broken) is None
