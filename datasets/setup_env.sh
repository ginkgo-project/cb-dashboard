#!/usr/bin/env bash
# Usage: datasets/setup_env.sh <daint-gh200|beverin-mi300a>
# Creates $CB_ROOT/venv/<machine> (one venv per machine: aarch64 vs x86_64).
# CB_DRY_RUN=1 only prints what would be done (the arch check always runs).
set -euo pipefail

ROOT="${CB_ROOT:-/capstor/store/cscs/userlab/d131/cb}"
MACHINE="${1:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$MACHINE" in
  daint-gh200) WANT=aarch64 ;;
  beverin-mi300a) WANT=x86_64 ;;
  *) echo "usage: $0 <daint-gh200|beverin-mi300a>" >&2; exit 2 ;;
esac

HAVE="$(uname -m)"
if [ "$HAVE" != "$WANT" ]; then
  echo "error: $MACHINE needs $WANT but this host is $HAVE; run this on a $MACHINE node" >&2
  exit 1
fi

run() {
  if [ "${CB_DRY_RUN:-0}" = 1 ]; then echo "[dry-run] $*"; else eval "$@"; fi
}

TOOLS="$ROOT/tools/bin"
VENV="$ROOT/venv/$MACHINE"
UV="$TOOLS/uv"

# Keep uv's managed CPython and cache on the shared root, not in one user's home.
export UV_PYTHON_INSTALL_DIR="$ROOT/tools/python"
export UV_CACHE_DIR="$ROOT/tools/uv-cache"
echo "UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"

run "mkdir -p '$TOOLS' '$ROOT/venv' '$UV_PYTHON_INSTALL_DIR' '$UV_CACHE_DIR'"
if [ ! -x "$UV" ]; then
  run "curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR='$TOOLS' UV_NO_MODIFY_PATH=1 sh"
else
  echo "uv already installed: $UV"
fi

if [ ! -x "$VENV/bin/python" ]; then
  run "'$UV' venv --python 3.12 '$VENV'"
else
  echo "venv already exists: $VENV"
fi
run "'$UV' pip install --python '$VENV/bin/python' -e '$REPO'"
run "'$VENV/bin/python' -c 'import skfem, numpy, scipy, datasets'"
echo "ok: $VENV"
