"""The node's energy counters (HPE Cray pm_counters), read around each benchmark run.

Each `*energy` file counts joules for the node ("energy") or a part of it (e.g.
"accel0_energy"); which parts exist depends on the node type. They update about
10 times a second, and "freshness" changes with every update.
"""
from __future__ import annotations

from pathlib import Path

PM_COUNTERS = Path("/sys/cray/pm_counters")


def _text(path: Path) -> str:
    return path.read_text()


def read(base: Path = PM_COUNTERS) -> dict[str, int]:
    """Every energy counter in joules, all from one update; {} if there are none.

    A file reads e.g. "1234 J 1697041234567890 us". The counters are read again
    (at most twice) if "freshness" changed meanwhile.
    """
    fresh = base / "freshness"
    out: dict[str, int] = {}
    for _ in range(3):
        try:
            before = _text(fresh) if fresh.exists() else None
            out = {}
            for p in sorted(base.glob("*energy")):
                value, unit = _text(p).split()[:2]
                if unit == "J":
                    out[p.name] = int(value)
            if before is None or _text(fresh) == before:
                return out
        except (OSError, ValueError):
            return {}
    return out


def used(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Joules between two readings, for the counters in both."""
    return {k: after[k] - before[k] for k in before if k in after}


def snapshot(base: Path = PM_COUNTERS) -> dict:
    """Every pm_counters file as text, to see which counters a node has."""
    try:
        return {p.name: _text(p).strip() for p in sorted(base.iterdir()) if p.is_file()}
    except OSError as e:
        return {"error": f"{type(e).__name__}: {e}"}
