import json
import stat
import sys
from pathlib import Path

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
    r = run_group([inv(a, "a"), inv(b, "b")], [], out)
    assert r["status"] == "ok"
    assert json.loads((out / "g.json").read_text()) == [{"a": 1}, {"b": 2}]
    assert r["invocations"][0]["args"] == ["-a", "1"]
    assert r["invocations"][0]["exit_code"] == 0


def test_crash_is_partial(tmp_path):
    ok = fake(tmp_path, "ok", 'print(\'[{"a":1}]\')')
    bad = fake(tmp_path, "bad", 'sys.stderr.write("boom\\n"); sys.exit(3)')
    out = tmp_path / "out"
    r = run_group([inv(bad, "bad"), inv(ok, "ok")], [], out)
    assert r["status"] == "partial"
    assert r["invocations"][0]["status"] == "failed"
    assert r["invocations"][0]["exit_code"] == 3
    assert "boom" in (out / "logs/g/bad.log").read_text()
    assert json.loads((out / "g.json").read_text()) == [{"a": 1}]


def test_truncated_json_is_failed(tmp_path):
    t = fake(tmp_path, "t", 'sys.stdout.write(\'[{"a":1},{"b"\')')
    out = tmp_path / "out"
    r = run_group([inv(t)], [], out)
    assert r["status"] == "failed"
    assert r["invocations"][0]["status"] == "failed"
    assert json.loads((out / "g.json").read_text()) == []


def test_timeout(tmp_path):
    s = fake(tmp_path, "s", "import time; time.sleep(5)")
    r = run_group([inv(s, timeout=1)], [], tmp_path / "out")
    assert r["invocations"][0]["reason"] == "timeout"
    assert r["status"] == "failed"
    assert r["seconds"] < 4


def test_missing_input_not_started(tmp_path):
    marker = tmp_path / "started"
    s = fake(tmp_path, "s", f'open({str(marker)!r}, "w")')
    r = run_group([inv(s, missing=tmp_path / "nope.mtx")], [], tmp_path / "out")
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
    r = run_group([i], [], out)
    assert r["status"] == "ok"
    assert json.loads((out / "g.json").read_text())[0] == {"d": "hello"}
