"""Run a group's invocations; one failure never stops the others."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from bench.groups import Invocation


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
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return [], {"status": "failed", "exit_code": None, "reason": "timeout"}
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
        # Reap any stragglers left in the group (e.g. srun children).
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
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
        return data, {"status": "ok", "exit_code": 0, "reason": None}


def run_group(invs: list[Invocation], srun: list[str], out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = ["srun", *srun] if srun else []
    t0 = time.monotonic()
    records: list[dict] = []
    results: list = []
    for inv in invs:
        t = time.monotonic()
        if inv.missing_input is not None:
            data, st = [], {"status": "failed", "exit_code": None, "reason": "missing input"}
        else:
            try:
                data, st = _run_one(inv, prefix, out_dir / "logs" / inv.group / f"{inv.label}.log")
            except OSError as e:
                data, st = [], {"status": "failed", "exit_code": None, "reason": f"cannot start: {e}"}
        results.extend(data)
        records.append({"label": inv.label, "args": list(inv.args), **st,
                        "seconds": time.monotonic() - t})
    group = invs[0].group if invs else None
    if group is not None:
        (out_dir / f"{group}.json").write_text(json.dumps(results))
    n_ok = sum(r["status"] == "ok" for r in records)
    status = "ok" if n_ok == len(records) else ("failed" if n_ok == 0 else "partial")
    return {"status": status, "seconds": time.monotonic() - t0, "invocations": records}
