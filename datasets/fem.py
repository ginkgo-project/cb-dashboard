"""FEM matrices for the preconditioner and SpGEMM benchmarks, from scikit-fem.

``generate(problem, size)`` assembles one problem, removes the Dirichlet
unknowns with ``skfem.condense`` and returns the float64 CSR matrix with sorted
indices and no explicit zeros. The values are the assembled ones.

``size`` is n, the number of cells per direction of the tensor mesh on the unit
square or cube, before each cell is split into simplices (2 triangles or 6
tetrahedra). The nonzeros grow like n^2 (2D) or n^3 (3D), measured:

====================  ====  ============  ================  =================
problem               dim   element       unknowns          nonzeros, large n
====================  ====  ============  ================  =================
porous_pressure       3D    P1            (n-1)^3           7 n^3
poisson_p1            3D    P1            (n-1)^3           7 n^3
anisotropic_heat      2D    P1            (n-1)^2           5 n^2
convection_diffusion  2D    P1            (n-1)^2           7 n^2
curl_curl             3D    Nedelec N0    about 6.8 n^3     110 n^3
elasticity_p2         3D    vector P2     3 (2n+1)^2 2n     1560 n^3
elasticity_p2_graded  3D    vector P2     a few more        1560 n^3
====================  ====  ============  ================  =================

``elasticity_p2_graded`` bisects (``refined(marked)``) the elements whose
centroid is within 0.1 / 2^k of the corner (0, 0, 0), for k = 0..3. Nothing is
refined below n = 10. The radii don't shrink with n, so the refinement stays a
small part of the mesh: it adds 6 elements to 6000 at n = 10 and 30 to 48000 at
n = 20.
In ``porous_pressure`` each tetrahedron takes the coefficient of the layer that
holds its centroid, so the 8 layers follow the cell faces exactly when n is a
multiple of 8.

On these structured meshes many couplings cancel exactly: the P1 matrices
have the 7-point (3D) and 5-point (2D) stencil pattern, and convection adds the
diagonal of each square. In P2 elasticity the cancellation leaves a rounding
residue instead of an exact zero in about 15% of the assembled entries, at most
3e-17 * max|A|, while the smallest real entries are above 1e-3 * max|A| (above
1e-7 in the other problems). Entries at most ``_RESIDUE * max|A|`` are
therefore dropped with the explicit zeros, so the pattern doesn't depend on
rounding and stays symmetric.
"""
from __future__ import annotations

import operator

import numpy as np
import scipy.sparse as sp
import skfem as fem
from skfem.helpers import curl, dot, grad
from skfem.models.elasticity import lame_parameters, linear_elasticity

_RESIDUE = 1e-14

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


def _square(n: int) -> fem.MeshTri:
    x = np.linspace(0.0, 1.0, n + 1)
    return fem.MeshTri.init_tensor(x, x)


def _cube(n: int) -> fem.MeshTet:
    x = np.linspace(0.0, 1.0, n + 1)
    return fem.MeshTet.init_tensor(x, x, x)


def _graded_cube(n: int) -> fem.MeshTet:
    """Bisect the elements whose centroid is within 0.1 / 2^k of the origin."""
    m = _cube(n)
    for k in range(4):
        centroid = m.p[:, m.t].mean(axis=1)
        marked = np.nonzero(np.linalg.norm(centroid, axis=0) <= 0.1 * 0.5**k)[0]
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
    # Layer of each tetrahedron by its centroid, in integer arithmetic so that
    # no centroid on an interface is rounded either way:
    # 8 * z_centroid = 8 * sum(z_vertex) / 4 = 2 * sum(z_index) / n.
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
