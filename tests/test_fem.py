"""FEM problem generators, at small sizes (about 1e4 rows)."""
import functools

import numpy as np
import pytest
import scipy.sparse as sp
import skfem
from scipy.sparse.linalg import eigsh

from datasets import fem
from datasets.assemble import row_length_stats
from datasets.catalog import load_catalog

SMALL = {
    "porous_pressure": 22,  # 9261 rows
    "anisotropic_heat": 101,  # 10000 rows
    "elasticity_p2": 7,  # 9450 rows
    "convection_diffusion": 101,  # 10000 rows
    "curl_curl": 12,  # 10836 rows
    "poisson_p1": 22,  # 9261 rows
    # The smallest size at which all four refinement passes mark elements.
    "elasticity_p2_graded": 8,
}
SPD = [p for p in SMALL if p != "convection_diffusion"]


@functools.cache
def _matrix(problem):
    return fem.generate(problem, SMALL[problem])


def _cv(A):
    return row_length_stats(A.shape[0], A.tocoo().row)["cv"]


def test_catalog_problems():
    entries = [e for e in load_catalog().entries if e.set == "fem"]
    assert {e.fem["problem"] for e in entries} == set(SMALL)
    assert {e.fem["problem"] for e in entries if e.spd} == set(SPD)
    assert all(e.fem["size"] >= SMALL[e.fem["problem"]] for e in entries)


def test_unknown_problem():
    with pytest.raises(ValueError, match="no_such_problem"):
        fem.generate("no_such_problem", 4)


def test_size_too_small():
    # n = 1 leaves the P1 problems without interior unknowns.
    with pytest.raises(ValueError, match="at least 2"):
        fem.generate("poisson_p1", 1)


@pytest.mark.parametrize("problem", list(SMALL))
def test_format(problem):
    A = _matrix(problem)
    assert type(A) is sp.csr_matrix
    assert A.dtype == np.float64
    assert A.shape[0] == A.shape[1] > 0
    # Column indices strictly increase within each row: sorted, no duplicates.
    step = np.diff(A.indices)
    row_start = A.indptr[1:-1] - 1
    row_start = row_start[(row_start >= 0) & (row_start < len(step))]
    step[row_start] = 1
    assert np.all(step > 0)
    assert np.all(A.data != 0)
    assert np.all(np.isfinite(A.data))


@pytest.mark.parametrize("problem", SPD)
def test_spd(problem):
    A = _matrix(problem)
    assert abs(A - A.T).max() < 1e-12 * abs(A).max()
    lam = eigsh(A, k=1, sigma=0, return_eigenvectors=False)[0]
    assert lam > 0


@pytest.mark.parametrize("problem", SPD)
def test_symmetric_pattern(problem):
    P = _matrix(problem).copy()
    P.data[:] = 1.0
    assert (P - P.T).nnz == 0


def test_convdiff_unsymmetric():
    A = _matrix("convection_diffusion")
    assert abs(A - A.T).max() > 1e-3 * abs(A).max()


def test_row_balance():
    assert _cv(_matrix("poisson_p1")) <= 0.2
    assert _cv(_matrix("elasticity_p2_graded")) >= 0.4


def test_porous_contrast():
    d = _matrix("porous_pressure").diagonal()
    assert d.max() / d.min() > 1e5


def test_graded_mesh_is_refined():
    n = SMALL["elasticity_p2_graded"]
    # P2 vector unknowns of the unrefined mesh, with the face x = 0 clamped.
    unrefined = 3 * (2 * n + 1) ** 2 * (2 * n)
    # The region within 0.5 of the corner alone is 6.5% of the cube.
    assert _matrix("elasticity_p2_graded").shape[0] > 1.05 * unrefined


def test_graded_mesh_is_graded():
    # The four nested passes leave the elements near the corner several
    # bisections smaller than those away from it.
    m = fem._graded_cube(SMALL["elasticity_p2_graded"])
    vol = np.abs(_volumes(m.p, m.t))
    dist = np.linalg.norm(m.p[:, m.t].mean(axis=1), axis=0)
    assert vol[dist <= 0.0625].mean() < vol[dist > 0.5].mean() / 4


def test_p1_not_stencil():
    lens = np.diff(_matrix("poisson_p1").indptr)
    assert lens.max() > 7


def _volumes(p, t):
    """Signed volumes (3D) or areas (2D) of the simplices."""
    x = p[:, t]
    e = x[:, 1:] - x[:, :1]
    return np.linalg.det(np.moveaxis(e, -1, 0)) / (6 if len(p) == 3 else 2)


@pytest.mark.parametrize("dim", [2, 3])
def test_mesh_perturbation(dim):
    n = 9
    x = np.linspace(0.0, 1.0, n + 1)
    if dim == 2:
        m, grid = fem._square(n), skfem.MeshTri.init_tensor(x, x)
    else:
        m, grid = fem._cube(n), skfem.MeshTet.init_tensor(x, x, x)
    assert np.array_equal(m.t, grid.t)
    on_boundary = np.any((grid.p == 0) | (grid.p == 1), axis=0)
    # Boundary nodes stay exactly where they were.
    assert np.array_equal(m.p[:, on_boundary], grid.p[:, on_boundary])
    # Interior nodes move, by at most 0.2 h per coordinate.
    d = np.abs(m.p[:, ~on_boundary] - grid.p[:, ~on_boundary])
    assert d.max() <= 0.2 / n and np.all(d.max(axis=0) > 0)
    # No element is inverted.
    assert np.all(_volumes(m.p, m.t) / _volumes(grid.p, grid.t) > 0)


@pytest.mark.parametrize("problem", list(SMALL))
def test_deterministic(problem):
    A = _matrix(problem)
    B = fem.generate(problem, SMALL[problem])
    assert A.shape == B.shape
    assert A.indptr.tobytes() == B.indptr.tobytes()
    assert A.indices.tobytes() == B.indices.tobytes()
    assert A.data.tobytes() == B.data.tobytes()


def _with_and_without_drop(monkeypatch, problem):
    """The matrix, and the same one with only the explicit zeros dropped."""
    A = _matrix(problem)  # first: _matrix caches, and must not see the patch
    with monkeypatch.context() as mp:
        mp.setattr(fem, "_residue", lambda B: B.data == 0)
        raw = fem.generate(problem, SMALL[problem])
    return A, raw


def test_residue_rule_is_per_pair():
    A = sp.csr_matrix(np.array([[4.0, 1e-20, 1e-20],
                                [0.0, 9.0, 2.0],
                                [1.0, 2.0, 1.0]]))
    A.sort_indices()
    # (0,1): max(1e-20, 0) <= 1e-14 * sqrt(4 * 9), dropped. (0,2): its mirror
    # (2,0) is 1, so both stay.
    kept = A.copy()
    kept.data[fem._residue(A)] = 0.0
    kept.eliminate_zeros()
    assert kept[0, 1] == 0 and kept[0, 2] == 1e-20 and kept[2, 0] == 1.0
    assert kept.nnz == A.nnz - 1
    assert fem._residue(sp.csr_matrix((3, 3))).size == 0


@pytest.mark.parametrize("problem", ["porous_pressure", "anisotropic_heat", "poisson_p1",
                                     "convection_diffusion", "curl_curl"])
def test_p1_and_n0_drop_nothing(monkeypatch, problem):
    A, raw = _with_and_without_drop(monkeypatch, problem)
    assert A.nnz == raw.nnz
    assert A.indices.tobytes() == raw.indices.tobytes()
    assert A.data.tobytes() == raw.data.tobytes()


def test_elasticity_residue_dropped_symmetrically(monkeypatch):
    A, raw = _with_and_without_drop(monkeypatch, "elasticity_p2")
    P, R = A.copy(), raw.copy()
    P.data[:] = 1.0
    R.data[:] = 1.0
    assert (R - R.T).nnz > 0  # the residue alone leaves the pattern unsymmetric
    assert (P - P.T).nnz == 0
    assert 0 < raw.nnz - A.nnz < 0.05 * raw.nnz
    # Only rounding residue goes: the dropped entries are tiny next to max|A|.
    dropped = abs(raw - raw.multiply(P.astype(bool)))
    assert dropped.max() <= 1e-14 * abs(A).max()
