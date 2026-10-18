"""Fixed theoretical bytes and flops models per benchmark, and machine peaks.

Values are 8 bytes, indices 4 bytes. Notation: n rows, c cols, z nonzeros,
it iterations, S = 12z + 4(n+1) + 8n (CSR matrix plus scalar Jacobi).
"""
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dashboard.cases import Measurement

VALUE, INDEX, CACHE_FACTOR = 8, 4, 2
MACHINES_TOML = Path(__file__).resolve().parent / "machines.toml"
_REV = "9cb13ee6ab"


@dataclass(frozen=True)
class Model:
    name: str
    bytes_text: str
    flops_text: str
    source: str
    bandwidth_bound: bool


def _gmres_source(kind):
    if kind == "mgs":
        return f"core/solver/gmres.cpp:492 @ {_REV}"
    return f"derived from orthogonalize_{kind}"


_GM = {
    "mgs": ("(5k+12)n", "4k+7"),
    "cgs": ("(4k+12)n", "4k+7"),
    "cgs2": ("(8k+17)n", "8k+11"),
}

_AMG_BYTES = ("per level: 2s(72r + S_l + J_l) + (40r + S_l) + "
              "(12r + 4(r'+1) + 8r') + 28r; coarsest: c(144r_L + S_L + J_L)")
_AMG_FLOPS = ("per level: 2s(2z + 6r) + (2z + r) + 2r; "
              "coarsest: c(2z_L + 13r_L)")
_ILU_BYTES = "12(z+n) + 8(n+1) + 32n"


def _models():
    ms = [
        Model("blas", "bandwidth_rate x time", "flops_rate x time",
              f"benchmark/blas/blas_common.hpp @ {_REV}", True),
        Model("spmv-csr", "12z + 4(n+1) + 8c + 8n", "2z", "derived", True),
        Model("spmv-coo", "16z + 8c + 8n", "2z", "derived", True),
        Model("spmv-hybrid", "12*n*e + 16*k + 8c + 8n (e = ell stored "
              "elements per row, k = coo stored elements)", "2z",
              "derived", True),
        Model("spmv-sellp", "12*s*t + 8c + 8n (s = slice size, "
              "t = total cols)", "2z", "derived", True),
        Model("spgemm", "bandwidth_rate x time", "flops_rate x time",
              f"benchmark/sparse_blas/operations.cpp:152 @ {_REV}", False),
        Model("cg", "it * (144n + S)", "it * (2z + 13n)",
              f"core/solver/cg.cpp:145 @ {_REV}", True),
        Model("bicgstab", "it * (248n + 2S)", "it * (4z + 26n)",
              f"core/solver/bicgstab.cpp:165 @ {_REV}", True),
        Model("ir", "it * (72n + S)", "it * (2z + 6n)",
              "derived: copy b, residual SpMV, norm, Jacobi update "
              "(9n values)", True),
        Model("jacobi", "s + 16n (s = storage)", "s / 4", "derived", True),
        Model("ilu", _ILU_BYTES, "2(z+n)",
              "derived: L and U have A's sparsity plus the diagonal", False),
        Model("spd-isai", _ILU_BYTES, "2(z+n)",
              "derived: two SpMVs with the triangular inverse", True),
        Model("general-isai", "12z + 4(n+1) + 16n", "2z",
              "derived: one SpMV with A's pattern", True),
        Model("amg", _AMG_BYTES, _AMG_FLOPS, "derived: one V-cycle", True),
        Model("generate", "None", "None", "time only", False),
    ]
    for kind, (vals, f) in _GM.items():
        ms.append(Model(
            f"gmres-{kind}",
            f"sum_j (8*{vals} + S) + sum_r (8(m_r+14)n + S)",
            f"sum_j (2z + n + ({f})n) + sum_r (2z + (2m_r+10)n)",
            _gmres_source(kind), True))
    return {m.name: m for m in ms}


MODELS = _models()
_AMG_NAMES = ("mg-pgm", "mg-pmis", "mg-uniform")
_ILU_NAMES = ("ilu", "ilu-syncfree", "ic", "ic-syncfree")


def model_name(m: Measurement) -> str:
    g = m.group
    if g == "blas":
        return "blas"
    if g == "spgemm":
        return "spgemm"
    if g == "spmv":
        return "spmv-" + m.labels.get("format", "")
    if g == "solver":
        return m.labels.get("solver", "")
    if g == "preconditioner":
        if m.labels.get("stage") == "generate":
            return "generate"
        p = m.labels.get("preconditioner", "")
        if p in _ILU_NAMES:
            return "ilu"
        if p in _AMG_NAMES:
            return "amg"
        return p
    return ""


def stencil_nonzeros(stencil: str, rows: int) -> int:
    if stencil == "5pt":
        g = round(rows ** 0.5)
        return 5 * g * g - 4 * g
    if stencil == "7pt":
        g = round(rows ** (1 / 3))
        return 7 * g ** 3 - 6 * g ** 2
    if stencil == "27pt":
        g = round(rows ** (1 / 3))
        return (3 * g - 2) ** 3
    raise ValueError(f"unknown stencil {stencil!r}")


def _s(n, z):
    return 12 * z + 4 * (n + 1) + 8 * n


def _gmres(kind, n, z, it, d):
    s = _s(n, z)
    vals = {"mgs": lambda k: 5 * k + 12, "cgs": lambda k: 4 * k + 12,
            "cgs2": lambda k: 8 * k + 17}[kind]
    f = {"mgs": lambda k: 4 * k + 7, "cgs": lambda k: 4 * k + 7,
         "cgs2": lambda k: 8 * k + 11}[kind]
    b = fl = 0
    for j in range(it):
        k = j % d
        b += 8 * vals(k) * n + s
        fl += 2 * z + n + f(k) * n
    restarts = math.ceil(it / d)
    for r in range(restarts):
        m = d if r < restarts - 1 else it - (restarts - 1) * d
        b += 8 * (m + 14) * n + s
        fl += 2 * z + (2 * m + 10) * n
    return b, fl


def _amg(n, z, levels, params):
    s = params.get("smoother_iters", 1)
    try:
        c = params["coarsest_solver"][0]["criteria"][0]["max_iters"]
    except (KeyError, IndexError, TypeError):
        c = 20
    b = fl = 0
    r, zz = n, z
    for lv in levels:
        sl = 12 * zz + 4 * (r + 1)
        jl = 8 * r
        r1 = lv["coarse_rows"]
        b += (2 * s * (72 * r + sl + jl) + (40 * r + sl)
              + (12 * r + 4 * (r1 + 1) + 8 * r1) + 28 * r)
        fl += 2 * s * (2 * zz + 6 * r) + (2 * zz + r) + r + r
        r, zz = r1, lv["coarse_nonzeros"]
    b += c * (144 * r + 12 * zz + 4 * (r + 1) + 8 * r)
    return b, fl, c, r, zz


def cost(m: Measurement, structure: dict | None):
    name = model_name(m)
    e = m.extra
    if name == "generate" or name not in MODELS:
        return None, None
    if name in ("blas", "spgemm"):
        br, fr = e.get("bandwidth_rate"), e.get("flops_rate")
        if br is None or fr is None or m.time is None:
            return None, None
        return br * m.time, fr * m.time
    if m.group == "spmv":
        n, c, z = e["rows"], e["cols"], e["nonzeros"]
        if name == "spmv-csr":
            return 12 * z + 4 * (n + 1) + 8 * c + 8 * n, 2 * z
        if name == "spmv-coo":
            return 16 * z + 8 * c + 8 * n, 2 * z
        if structure is None:
            return None, None
        if name == "spmv-hybrid":
            ee = structure["ell_num_stored_elements_per_row"]
            k = structure["coo_num_stored_elements"]
            return 12 * n * ee + 16 * k + 8 * c + 8 * n, 2 * z
        if name == "spmv-sellp":
            sl, t = structure["slice_size"], structure["total_cols"]
            return 12 * sl * t + 8 * c + 8 * n, 2 * z
        return None, None
    if m.group == "solver":
        n = e["rows"]
        z = stencil_nonzeros(m.labels["stencil"], n)
        it = e["iterations"]
        if it is None:
            return None, None
        s = _s(n, z)
        if name == "cg":
            return it * (144 * n + s), it * (2 * z + 13 * n)
        if name == "bicgstab":
            return it * (248 * n + 2 * s), it * (4 * z + 26 * n)
        if name == "ir":
            return it * (72 * n + s), it * (2 * z + 6 * n)
        if name.startswith("gmres-"):
            d = (e.get("parameters") or {}).get("krylov_dim")
            if not d:
                return None, None
            return _gmres(name[6:], n, z, it, d)
        return None, None
    if m.group == "preconditioner":
        n, z = e["rows"], e["nonzeros"]
        if name == "jacobi":
            if structure is None:
                return None, None
            s = structure["storage"]
            return s + 16 * n, s / 4
        if name in ("ilu", "spd-isai"):
            return 12 * (z + n) + 8 * (n + 1) + 32 * n, 2 * (z + n)
        if name == "general-isai":
            return 12 * z + 4 * (n + 1) + 16 * n, 2 * z
        if name == "amg":
            if structure is None:
                return None, None
            b, fl, c, r, zz = _amg(n, z, structure["levels"],
                                   e.get("parameters") or {})
            fl += c * (2 * zz + 13 * r)
            return b, fl
    return None, None


def compare_kind(m: Measurement, nbytes, machine: dict) -> str:
    name = model_name(m)
    model = MODELS.get(name)
    if model is None or not model.bandwidth_bound:
        return "time"
    if m.group in ("blas", "spmv") and (
            nbytes is None or nbytes < CACHE_FACTOR * machine["largest_cache"]):
        return "time"
    return "bandwidth"


def load_machines(path: Path = MACHINES_TOML) -> dict[str, dict]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    keys = ("label", "fp64_peak", "largest_cache", "cache_name", "citations")
    return {name: {k: v[k] for k in keys} for name, v in raw.items()}
