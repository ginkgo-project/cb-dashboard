"""Run a group's invocations; one failure never stops the others."""
from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Mapping

from bench.groups import Invocation

# On a timeout, srun gets SIGTERM (which it forwards to the step) and this long
# to end before the process group gets SIGKILL.
KILL_GRACE_S = 30
# The same, when this script is itself being stopped (Slurm's KillWait is short).
INTERRUPT_GRACE_S = 5


class Killed(BaseException):
    """The job got SIGTERM. A BaseException, so `except Exception` lets it through.

    run_group sets `group_result` to the group's record so far before re-raising.
    """

    def __init__(self, signum: int = signal.SIGTERM):
        super().__init__(signum)
        self.signum = signum
        self.group_result: dict | None = None


def srun_time_minutes(timeout_s: int) -> int:
    """srun's --time for one step: the timeout rounded up to minutes, plus one."""
    return math.ceil(timeout_s / 60) + 1


def _killpg(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _stop(proc: subprocess.Popen, grace: float) -> None:
    """SIGTERM the process group, wait up to `grace` s, then SIGKILL what is left."""
    try:
        _killpg(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
    finally:
        _killpg(proc, signal.SIGKILL)
        proc.wait()


def describe_case(case: dict) -> str:
    """The test case as Ginkgo's describe_config names it."""
    if "filename" in case:
        return str(case["filename"])
    if "stencil" in case:
        return f"stencil({case.get('size')}, {case['stencil']})"
    keys = [k for k in ("n", "k", "m", "r") if isinstance(case.get(k), int)]
    if keys:
        return " ".join(f"{k} = {case[k]}" for k in keys)
    return json.dumps({k: v for k, v in case.items() if not isinstance(v, (dict, list))},
                      sort_keys=True)


def find_failures(data: list) -> tuple[int, list[dict]]:
    """Return (checked units, failures) of Ginkgo's output array.

    Ginkgo catches errors and still exits with 0 (benchmark/utils/runner.hpp):
    an operation that throws gets `<benchmark>.<op> = {"completed": false,
    "error_type", "error"}`, and a test case whose setup throws gets
    `error_type` and `error` on the case itself. Each operation with a
    `completed` field is one unit, and so is each case with an error.
    """
    units = 0
    failures: list[dict] = []
    for case in data:
        if not isinstance(case, dict):
            continue
        desc = describe_case(case)
        if "error" in case:
            units += 1
            failures.append({"case": desc, "op": None, "error_type": case.get("error_type"),
                             "error": case["error"]})
        for value in case.values():
            if not isinstance(value, dict):
                continue
            for op, result in value.items():
                if not isinstance(result, dict) or "completed" not in result:
                    continue
                units += 1
                if result["completed"] is not True:
                    failures.append({"case": desc, "op": op,
                                     "error_type": result.get("error_type"),
                                     "error": result.get("error")})
    return units, failures


def _run_one(inv: Invocation, prefix: list[str], log: Path) -> tuple[list, dict]:
    """Return (records, status fields) for one invocation."""
    cmd = [*prefix, str(inv.exe), *inv.args]
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "wb") as errf, tempfile.TemporaryFile() as outf:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE if inv.stdin is not None else subprocess.DEVNULL,
            stdout=outf, stderr=errf, start_new_session=True)
        try:
            proc.communicate(inv.stdin.encode() if inv.stdin is not None else None,
                             timeout=inv.timeout_s)
        except subprocess.TimeoutExpired:
            _stop(proc, KILL_GRACE_S)
            return [], {"status": "failed", "exit_code": None, "reason": "timeout"}
        except BaseException:
            _stop(proc, INTERRUPT_GRACE_S)
            raise
        # Reap any stragglers left in the group (e.g. srun children).
        _killpg(proc, signal.SIGKILL)
        if proc.returncode != 0:
            return [], {"status": "failed", "exit_code": proc.returncode,
                        "reason": f"exit code {proc.returncode}"}
        outf.seek(0)
        try:
            data = json.load(outf)
        except (ValueError, UnicodeDecodeError) as e:
            return [], {"status": "failed", "exit_code": 0, "reason": f"invalid JSON: {e}"}
        if not isinstance(data, list):
            return [], {"status": "failed", "exit_code": 0, "reason": "output is not a JSON array"}
        units, failures = find_failures(data)
        if not failures:
            return data, {"status": "ok", "exit_code": 0, "reason": None, "failures": []}
        status = "failed" if len(failures) >= units else "partial"
        return data, {"status": status, "exit_code": 0,
                      "reason": f"{len(failures)} of {units} operations failed",
                      "failures": failures}


def _aggregate(records: list[dict]) -> str:
    statuses = [r["status"] for r in records]
    if all(s == "ok" for s in statuses):
        return "ok"
    if all(s == "skipped" for s in statuses):
        return "skipped"
    if any(s in ("ok", "partial") for s in statuses):
        return "partial"
    return "failed"


def run_group(group: str, invs: list[Invocation], srun: list[str], out_dir: Path,
              deadline: float | None = None) -> dict:
    """Run the invocations of one group.

    `deadline` is the job's end as a Unix time, or None if unknown. An
    invocation whose timeout doesn't fit before it is skipped.
    """
    for inv in invs:
        if inv.group != group:
            raise ValueError(f"invocation {inv.label!r} is in group {inv.group!r}, not {group!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    records: list[dict] = []
    results: list = []
    try:
        for inv in invs:
            t = time.monotonic()
            prefix = ["srun", *srun, f"--time={srun_time_minutes(inv.timeout_s)}"] if srun else []
            try:
                if inv.missing_input is not None:
                    data, st = [], {"status": "failed", "exit_code": None, "reason": "missing input"}
                elif deadline is not None and deadline - time.time() < inv.timeout_s:
                    data, st = [], {"status": "skipped", "exit_code": None,
                                    "reason": "not enough job time"}
                else:
                    try:
                        data, st = _run_one(inv, prefix,
                                            out_dir / "logs" / inv.group / f"{inv.label}.log")
                    except OSError as e:
                        data, st = [], {"status": "failed", "exit_code": None,
                                        "reason": f"cannot start: {e}"}
            except Killed as k:
                records.append({"label": inv.label, "args": list(inv.args), "status": "killed",
                                "exit_code": None, "reason": "job killed (SIGTERM)",
                                "failures": [], "seconds": time.monotonic() - t})
                k.group_result = {"status": "killed", "seconds": time.monotonic() - t0,
                                  "invocations": records}
                raise
            results.extend(data)
            rec = {"label": inv.label, "args": list(inv.args), **st}
            rec.setdefault("failures", [])
            rec["seconds"] = time.monotonic() - t
            records.append(rec)
    finally:
        (out_dir / f"{group}.json").write_text(json.dumps(results))
    if not records:
        return {"status": "ok", "seconds": 0.0, "invocations": []}
    return {"status": _aggregate(records), "seconds": time.monotonic() - t0, "invocations": records}


def parse_time_left(text: str) -> int | None:
    """Seconds in squeue's %L ([days-][hours:]minutes:seconds), or None."""
    m = re.fullmatch(r"(?:(\d+)-)?(\d+(?::\d+){0,2})", text.strip())
    if not m:
        return None  # UNLIMITED, NOT_SET, INVALID, empty
    parts = [int(p) for p in m.group(2).split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, mins, s = parts
    return int(m.group(1) or 0) * 86400 + h * 3600 + mins * 60 + s


def job_deadline(env: Mapping[str, str] = os.environ, run: Callable = subprocess.run,
                 now: Callable[[], float] = time.time) -> float | None:
    """The job's end as a Unix time: SLURM_JOB_END_TIME, else squeue; None if unknown."""
    end = env.get("SLURM_JOB_END_TIME", "").strip()
    if end:
        try:
            return float(end)
        except ValueError:
            pass
    job = env.get("SLURM_JOB_ID", "").strip()
    if not job:
        return None
    try:
        out = run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True, text=True,
                  check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    left = parse_time_left(out.strip().splitlines()[0] if out.strip() else "")
    return None if left is None else now() + left
