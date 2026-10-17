"""FEM problem generators, at small sizes (about 1e4 rows)."""
import functools

import numpy as np
import pytest
import scipy.sparse as sp
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
    # Below 10 cells per direction no centroid lies within 0.1 of the corner,
    # so nothing would be refined. 10 gives about 26500 rows.
    "elasticity_p2_graded": 10,
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


def test_unknown_problem():
    with pytest.raises(ValueError, match="no_such_problem"):
        fem.generate("no_such_problem", 4)


def test_size_too_small():
    # The catalog's placeholder size 1 must not build silently.
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
    assert _matrix("elasticity_p2_graded").shape[0] > unrefined


@pytest.mark.parametrize("problem", list(SMALL))
def test_deterministic(problem):
    A = _matrix(problem)
    B = fem.generate(problem, SMALL[problem])
    assert A.shape == B.shape
    assert A.indptr.tobytes() == B.indptr.tobytes()
    assert A.indices.tobytes() == B.indices.tobytes()
    assert A.data.tobytes() == B.data.tobytes()
