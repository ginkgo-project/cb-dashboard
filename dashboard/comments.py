"""The three Memory movement summary comments copied from Ginkgo.

The models stay fixed; a changed comment only raises a warning.
"""
from pathlib import Path

_REV = "9cb13ee6ab"

COPIED = {
    "cg": (
        "core/solver/cg.cpp",
        "\n".join(
            [
                '    /* Memory movement summary:',
                '     * 18n * values + matrix/preconditioner storage',
                '     * 1x SpMV:           2n * values + storage',
                '     * 1x Preconditioner: 2n * values + storage',
                '     * 2x dot             4n',
                '     * 1x step 1 (axpy)   3n',
                '     * 1x step 2 (axpys)  6n',
                '     * 1x norm2 residual   n',
                '     */',
            ]
        ),
    ),
    "bicgstab": (
        "core/solver/bicgstab.cpp",
        "\n".join(
            [
                '    /* Memory movement summary:',
                '     * 31n * values + 2 * matrix/preconditioner storage',
                '     * 2x SpMV:                4n * values + 2 * storage',
                '     * 2x Preconditioner:      4n * values + 2 * storage',
                '     * 3x dot                  6n',
                '     * 1x norm2                 n',
                '     * 1x step 1 (fused axpys) 4n',
                '     * 1x step 2 (axpy)        3n',
                '     * 1x step 3 (fused axpys) 7n',
                '     * 2x norm2 residual       2n',
                '     */',
            ]
        ),
    ),
    "gmres-mgs": (
        "core/solver/gmres.cpp",
        "\n".join(
            [
                '    /* Memory movement summary for average iteration with krylov_dim d:',
                '     * (5/2d+21/2+14/d)n * values + (1+1/d) * matrix/preconditioner storage',
                '     * 1x SpMV:                2n * values + storage',
                '     * 1x Preconditioner:      2n * values + storage',
                '     * MGS:          (5/2d+11/2)n = sum k=0 to d-1 of (5k+8)n/d',
                '     *       1x dots           2(k+1)n in iteration k (0-based)',
                '     *       1x axpys          3(k+1)n in iteration k (0-based)',
                '     *       1x norm2               n',
                '     *       1x scal               2n',
                '     * Restart:         (1+14/d)n  (every dth iteration)',
                '     *       1x gemv           (d+1)n',
                '     *       1x Preconditioner     2n * values + storage',
                '     *       1x axpy               3n',
                '     *       1x copy               2n',
                '     *       1x Advanced SpMV      3n * values + storage',
                '     *       1x norm2               n',
                '     *       1x scal               2n',
                '     */',
            ]
        ),
    ),
}


def _norm(text):
    lines = (line.lstrip(" \t*").rstrip() for line in text.splitlines())
    return [line for line in lines if line]


def _contains(haystack, needle):
    n = len(needle)
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def check_comments(ginkgo: Path) -> list[str]:
    warnings = []
    for model, (path, text) in COPIED.items():
        file = Path(ginkgo) / path
        if not file.is_file():
            warnings.append(f"{model}: {path} is missing in the Ginkgo checkout")
        elif not _contains(_norm(file.read_text(errors="replace")), _norm(text)):
            warnings.append(
                f"{model}: the memory movement comment in {path} changed "
                f"since {_REV}; the model stays as it is"
            )
    return warnings
