"""Structure Ginkgo decides, frozen from the first committed run of a case."""
from __future__ import annotations

import json
from pathlib import Path

from .cases import Measurement, measurements
from .load import Run, columns
from .models import model_name

FROZEN_JSON = Path(__file__).resolve().parent / "frozen.json"

_HYBRID = ("ell_num_stored_elements_per_row", "coo_num_stored_elements")
_SELLP = ("slice_size", "total_cols")
_LEVEL = ("fine_rows", "coarse_rows", "coarse_nonzeros")


def structure_of(m: Measurement) -> dict | None:
    """The structure a model needs from this measurement, if any."""
    if m.status != "ok":
        return None
    name = model_name(m)
    if name in ("spmv-hybrid", "spmv-sellp"):
        keys = _HYBRID if name == "spmv-hybrid" else _SELLP
        params = m.extra.get("format_parameters") or {}
        if all(k in params for k in keys):
            return {k: params[k] for k in keys}
        return None
    if m.group != "preconditioner" or m.labels.get("stage") != "apply":
        return None
    if name == "jacobi":
        storage = m.extra.get("storage")
        return {"storage": storage} if storage is not None else None
    if name == "amg":
        levels = m.extra.get("levels")
        if not levels:
            return None
        return {"levels": [{k: lv[k] for k in _LEVEL} for lv in levels]}
    return None


def load_frozen(path: Path = FROZEN_JSON) -> dict[str, dict[str, dict]]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def freeze(runs: list[Run], frozen: dict) -> dict:
    """`frozen` plus the structure of every (machine, key) not yet in it."""
    out = {machine: dict(keys) for machine, keys in frozen.items()}
    for machine in sorted({r.machine for r in runs}):
        have = out.setdefault(machine, {})
        seen = set(have)
        for col in columns(runs, machine):
            for run in col.runs:
                for key, m in measurements(run).items():
                    if key in seen:
                        continue
                    s = structure_of(m)
                    if s is not None:
                        have[key] = s
                        seen.add(key)
    return {m: k for m, k in out.items() if k or m in frozen}


def structures(runs: list[Run], frozen: dict) -> tuple[dict, list[str]]:
    new = freeze(runs, frozen)
    warnings = [f"not frozen yet: {machine} {key}; run python -m dashboard freeze"
                for machine in sorted(new) for key in sorted(new[machine])
                if key not in frozen.get(machine, {})]
    return new, warnings
