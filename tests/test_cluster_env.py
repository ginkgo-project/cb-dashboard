import os
import subprocess
from pathlib import Path

import pytest

from datasets.__main__ import DEFAULT_WEBGRAPH_CMD

DS = Path(__file__).resolve().parent.parent / "datasets"
X86 = os.uname().machine == "x86_64"


def run(script, *args, env=None):
    return subprocess.run(["bash", str(DS / script), *args], capture_output=True,
                          text=True, env={**os.environ, **(env or {})})


@pytest.mark.parametrize("s", ["setup_env.sh", "setup_webgraph.sh", "build.sbatch"])
def test_bash_syntax(s):
    assert subprocess.run(["bash", "-n", str(DS / s)]).returncode == 0


@pytest.mark.skipif(not X86, reason="needs an x86_64 host")
def test_beverin_dry_run(tmp_path):
    r = run("setup_env.sh", "beverin-mi300a",
            env={"CB_ROOT": str(tmp_path), "CB_DRY_RUN": "1"})
    assert r.returncode == 0, r.stderr
    assert f"venv --python 3.12 '{tmp_path}/venv/beverin-mi300a'" in r.stdout
    assert "astral.sh/uv/install.sh" in r.stdout
    assert "import skfem, numpy, scipy, datasets" in r.stdout
    assert f"UV_PYTHON_INSTALL_DIR={tmp_path}/tools/python" in r.stdout
    assert f"UV_CACHE_DIR={tmp_path}/tools/uv-cache" in r.stdout
    assert f"-e '{DS.parent}'" in r.stdout
    assert not (tmp_path / "venv" / "beverin-mi300a").exists()


@pytest.mark.skipif(not X86, reason="needs an x86_64 host")
def test_daint_refused_on_x86(tmp_path):
    r = run("setup_env.sh", "daint-gh200",
            env={"CB_ROOT": str(tmp_path), "CB_DRY_RUN": "1"})
    assert r.returncode != 0
    assert "aarch64" in r.stderr


def test_unknown_machine(tmp_path):
    assert run("setup_env.sh", "laptop", env={"CB_ROOT": str(tmp_path)}).returncode != 0


def test_webgraph_cmd_default():
    assert "{basename}" in DEFAULT_WEBGRAPH_CMD and "{out}" in DEFAULT_WEBGRAPH_CMD
