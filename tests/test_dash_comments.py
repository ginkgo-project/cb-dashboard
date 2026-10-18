from pathlib import Path

import pytest

from dashboard.comments import COPIED, check_comments


def _fake(root, edit=None, skip=None):
    for model, (path, text) in COPIED.items():
        if model == skip:
            continue
        if model == edit:
            text = text.replace("2n", "3n", 1)
        f = root / path
        f.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join("        " + line for line in text.splitlines())
        f.write_text("void f() {\n    int iter = -1;\n" + body + "\n    while (true) {}\n}\n")


def test_unchanged_comments_give_no_warning(tmp_path):
    _fake(tmp_path)
    assert check_comments(tmp_path) == []


def test_changed_and_missing(tmp_path):
    _fake(tmp_path, edit="cg", skip="gmres-mgs")
    assert check_comments(tmp_path) == [
        "cg: the memory movement comment in core/solver/cg.cpp changed "
        "since 9cb13ee6ab; the model stays as it is",
        "gmres-mgs: core/solver/gmres.cpp is missing in the Ginkgo checkout",
    ]


def test_real_worktree():
    wt = Path(__file__).resolve().parents[2] / "2ginkgo-cb"
    if not wt.is_dir():
        pytest.skip("no Ginkgo worktree")
    assert check_comments(wt) == []
