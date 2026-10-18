import re
import shutil
from pathlib import Path

from datasets.catalog import DEFAULT, load_catalog


def test_spmv_set():
    cat = load_catalog()
    spmv = [e for e in cat.entries if e.set == "spmv"]
    assert len(spmv) == 10
    assert len({e.name for e in spmv}) == 10
    assert all(e.expected_nnz < 2**31 for e in spmv)
    assert cat.version == "2" and cat.seed == 20261017


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


def test_only_com_orkut_compacts_ids():
    # SNAP's com-Orkut file skips 185 ids; SuiteSparse's com-Orkut has 3072441 rows.
    cat = load_catalog()
    assert [e.name for e in cat.entries if e.compact_ids] == ["com-Orkut"]


def test_tsopf_gets_its_diagonal():
    (e,) = [e for e in load_catalog().entries if e.name == "TSOPF_RS_b2383_c1"]
    assert e.add_diagonal is True and e.nnz_check == "plus_diagonal"
    assert e.expected_nnz == 16171169


def test_entry_sha256_follows_only_its_table_and_the_seed(tmp_path):
    p = tmp_path / "catalog.toml"
    text = DEFAULT.read_text()

    def hashes(t):
        p.write_text(t)
        return {e.name: e.sha256 for e in load_catalog(p).entries}

    a = hashes(text)
    assert a == {e.name: e.sha256 for e in load_catalog().entries}
    assert len(set(a.values())) == len(a)
    assert hashes("# new\n" + re.sub(r'^version = ".*"', 'version = "x"', text, flags=re.M)) == a
    assert text.count("size = 19}") == 1
    b = hashes(text.replace("size = 19}", "size = 20}"))
    assert [n for n in a if a[n] != b[n]] == ["elasticity_p2_graded"]
    c = hashes(text.replace("seed = 20261017", "seed = 1"))
    assert all(a[n] != c[n] for n in a)
