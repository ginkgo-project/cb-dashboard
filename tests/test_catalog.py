import shutil
from pathlib import Path

from datasets.catalog import DEFAULT, load_catalog


def test_spmv_set():
    cat = load_catalog()
    spmv = [e for e in cat.entries if e.set == "spmv"]
    assert len(spmv) == 10
    assert len({e.name for e in spmv}) == 10
    assert all(e.expected_nnz < 2**31 for e in spmv)
    assert cat.version == "1" and cat.seed == 20261017


def test_fem_roles():
    cat = load_catalog()
    fem = [e for e in cat.entries if e.set == "fem"]
    assert len(fem) == 7
    spgemm = sorted(e.name for e in fem if "spgemm" in e.roles)
    assert spgemm == ["elasticity_p2_graded", "poisson_p1"]
    assert len([e for e in fem if "precond" in e.roles]) == 5
    for e in fem:
        assert e.url is None and e.nnz_check == "none"
        assert e.fem == {"problem": e.name, "size": e.fem["size"]}
        assert e.spd == (e.name != "convection_diffusion")


def test_sha_changes_with_content(tmp_path):
    p = tmp_path / "catalog.toml"
    shutil.copy(DEFAULT, p)
    a = load_catalog(p).sha256
    assert a == load_catalog(DEFAULT).sha256
    p.write_text(p.read_text() + "\n# edit\n")
    assert load_catalog(p).sha256 != a
