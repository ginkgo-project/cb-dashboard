"""FEM matrices for the preconditioner and SpGEMM benchmarks, from scikit-fem.

``generate(problem, size)`` assembles one problem, removes the Dirichlet
unknowns with ``skfem.condense`` and returns the float64 CSR matrix with sorted
indices and no explicit zeros. The values are the assembled ones.

``size`` is n, the number of cells per direction of the tensor mesh on the unit
square or cube, before each cell is split into simplices (2 triangles or 6
tetrahedra). The interior nodes of every tensor mesh are then moved by a seeded
random offset of at most ``_JITTER * h`` per coordinate (h = 1/n); boundary
nodes stay exactly where they are. Without the offsets, couplings cancel
exactly on these meshes and the P1 matrices collapse to the 7-point (3D) and
5-point (2D) stencil patterns of the solver benchmarks.

The nonzeros grow like n^2 (2D) or n^3 (3D). The first numbers are measured at
the test sizes; the growth follows from the interior rows:

====================  ===  =========  =======================  ===============
problem               dim  element    test size: rows, nnz     nnz, large n
====================  ===  =========  =======================  ===============
porous_pressure       3D   P1         n=22: 9261, 128581       15 n^3
poisson_p1            3D   P1         n=22: 9261, 128581       15 n^3
anisotropic_heat      2D   P1         n=101: 10000, 69202      7 n^2
convection_diffusion  2D   P1         n=101: 10000, 69202      7 n^2
curl_curl             3D   Nedelec    n=12: 10836, 163428      110 n^3
elasticity_p2         3D   vector P2  n=7: 9450, 678916        about 2000 n^3
elasticity_p2_graded  3D   vector P2  n=8: 15162, 1111796      about 2200 n^3
====================  ===  =========  =======================  ===============

``elasticity_p2_graded`` refines in four nested passes: pass k bisects
(``refined(marked)``) the elements whose centroid is within 0.5 / 2^k of the
corner (0, 0, 0), for k = 0..3. So the elements nearest the corner are bisected
four times. At n = 8 the passes mark 202, 53, 13 and 6 elements and the mesh
grows from 3072 to 3380 elements (about 10% for every n from 6 to 14). n = 8
is the smallest size at which every pass marks something.

In ``porous_pressure`` each tetrahedron takes the coefficient of the layer that
holds the centroid of its unperturbed position, so the 8 layers follow the cell
faces exactly when n is a multiple of 8.

Small entries: next to the boundary, where the nodes don't move, P2 elasticity
still has couplings that cancel exactly. In floating point they leave a rounding
residue instead of a zero: 12782 of 691698 entries at n = 7, at most
1.3e-17 * max|A|. Without the drop, the pattern of the symmetric matrix wouldn't
be symmetric (2200 entries at n = 7). Every entry at most ``_RESIDUE * max|A|``
is therefore dropped with the explicit zeros. At the test sizes the smallest
kept entry is 2.5e-8 * max|A| in elasticity and 1.8e-12 * max|A| in
porous_pressure (its 1e6 contrast). The other problems have no residue.
"""
from __future__ import annotations

import operator

import numpy as np
import scipy.sparse as sp
import skfem as fem
from skfem.helpers import curl, dot, grad
from skfem.models.elasticity import lame_parameters, linear_elasticity

_RESIDUE = 1e-14
# Interior nodes move by at most _JITTER * h per coordinate. The ruling allows
# 0.2 h, but at 0.2 h the worst case inverts a Kuhn tetrahedron (volume -0.2
# times the original); at 0.15 h every one keeps at least 0.1 of its volume.
_JITTER = 0.15
_SEED = 20261017

_ELASTICITY = linear_elasticity(*lame_parameters(1.0, 0.3))


@fem.BilinearForm
def _laplace(u, v, w):
    return dot(grad(u), grad(v))


@fem.BilinearForm
def _layered_laplace(u, v, w):
    return w.kappa * dot(grad(u), grad(v))


@fem.BilinearForm
def _anisotropic(u, v, w):
    return grad(u)[0] * grad(v)[0] + 1e-3 * grad(u)[1] * grad(v)[1]


@fem.BilinearForm
def _convection_diffusion(u, v, w):
    return (1e-2 * dot(grad(u), grad(v))
            + (1.0 * grad(u)[0] + 0.5 * grad(u)[1]) * v)


@fem.BilinearForm
def _curl_curl(u, v, w):
    return dot(curl(u), curl(v)) + dot(u, v)


def _perturbed(m, n: int):
    """Move the interior nodes by a seeded random offset, at most _JITTER / n."""
    p = m.p.copy()
    interior = np.all((p > 0.0) & (p < 1.0), axis=0)
    offset = np.random.default_rng(_SEED).uniform(-_JITTER, _JITTER, p.shape) / n
    p[:, interior] += offset[:, interior]
    return type(m)(p, m.t)


def _square(n: int) -> fem.MeshTri:
    x = np.linspace(0.0, 1.0, n + 1)
    return _perturbed(fem.MeshTri.init_tensor(x, x), n)


def _cube(n: int) -> fem.MeshTet:
    x = np.linspace(0.0, 1.0, n + 1)
    return _perturbed(fem.MeshTet.init_tensor(x, x, x), n)


def _graded_cube(n: int) -> fem.MeshTet:
    """Bisect the elements whose centroid is within 0.5 / 2^k of the origin."""
    m = _cube(n)
    for k in range(4):
        centroid = m.p[:, m.t].mean(axis=1)
        marked = np.nonzero(np.linalg.norm(centroid, axis=0) <= 0.5 * 0.5**k)[0]
        if marked.size:
            m = m.refined(marked)
    return m


def _finish(A, D) -> sp.csr_matrix:
    A = sp.csr_matrix(fem.condense(A, D=D, expand=False), dtype=np.float64)
    A.sum_duplicates()
    A.data[np.abs(A.data) <= _RESIDUE * np.abs(A.data).max()] = 0.0
    A.eliminate_zeros()
    return A


def _porous_pressure(n):
    m = _cube(n)
    basis = fem.Basis(m, fem.ElementTetP1())
    # Layer of each tetrahedron by the centroid of its unperturbed cell
    # position, in integer arithmetic so that no centroid on an interface is
    # rounded either way: 8 * z_centroid = 2 * sum(z_index) / n. The offsets are
    # below h / 2, so rounding recovers each node's grid index.
    iz = np.rint(m.p[2] * n).astype(np.int64)
    layer = 2 * iz[m.t].sum(axis=0) // n
    kappa = np.where(layer % 2 == 0, 1.0, 1e6)
    k = basis.with_element(fem.ElementTetP0()).interpolate(kappa)
    return _finish(_layered_laplace.assemble(basis, kappa=k), basis.get_dofs())


def _poisson_p1(n):
    basis = fem.Basis(_cube(n), fem.ElementTetP1())
    return _finish(_laplace.assemble(basis), basis.get_dofs())


def _anisotropic_heat(n):
    basis = fem.Basis(_square(n), fem.ElementTriP1())
    return _finish(_anisotropic.assemble(basis), basis.get_dofs())


def _convdiff(n):
    basis = fem.Basis(_square(n), fem.ElementTriP1())
    return _finish(_convection_diffusion.assemble(basis), basis.get_dofs())


def _curl_curl_n0(n):
    basis = fem.Basis(_cube(n), fem.ElementTetN0())
    return _finish(_curl_curl.assemble(basis), basis.get_dofs())


def _clamped_elasticity(m):
    basis = fem.Basis(m, fem.ElementVector(fem.ElementTetP2()))
    clamped = basis.get_dofs(m.facets_satisfying(lambda x: x[0] == 0.0))
    return _finish(_ELASTICITY.assemble(basis), clamped)


_GENERATORS = {
    "porous_pressure": _porous_pressure,
    "anisotropic_heat": _anisotropic_heat,
    "elasticity_p2": lambda n: _clamped_elasticity(_cube(n)),
    "convection_diffusion": _convdiff,
    "curl_curl": _curl_curl_n0,
    "poisson_p1": _poisson_p1,
    "elasticity_p2_graded": lambda n: _clamped_elasticity(_graded_cube(n)),
}


def generate(problem: str, size: int) -> sp.csr_matrix:
    """Assemble ``problem`` with ``size`` cells per direction (see the module)."""
    if problem not in _GENERATORS:
        raise ValueError(f"unknown FEM problem {problem!r}, "
                         f"expected one of {sorted(_GENERATORS)}")
    n = operator.index(size)
    if n < 2:
        raise ValueError(f"size must be at least 2, got {n}")
    return _GENERATORS[problem](n)
