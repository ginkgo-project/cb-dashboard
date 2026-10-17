#!/usr/bin/env bash
# Installs the Rust WebGraph CLI (crate webgraph-cli, binary `webgraph`) into
# $CB_ROOT/tools/<arch> (aarch64 on Daint, x86_64 on Beverin). The binary ends
# up in $CB_ROOT/tools/<arch>/cargo/bin/webgraph. The dataset build runs on
# Daint, so run this there. CB_DRY_RUN=1 only prints what would be done.
set -euo pipefail

ROOT="${CB_ROOT:-/capstor/store/cscs/userlab/d131/cb}"
TOOLS="$ROOT/tools/$(uname -m)"
export RUSTUP_HOME="$TOOLS/rustup"
export CARGO_HOME="$TOOLS/cargo"

run() {
  if [ "${CB_DRY_RUN:-0}" = 1 ]; then echo "[dry-run] $*"; else eval "$@"; fi
}

run "mkdir -p '$RUSTUP_HOME' '$CARGO_HOME'"
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
  run "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path"
else
  echo "rustup/cargo already installed in $CARGO_HOME"
fi
run "'$CARGO_HOME/bin/cargo' install webgraph-cli"
run "'$CARGO_HOME/bin/webgraph' --help"
echo "now run: $CARGO_HOME/bin/webgraph to arcs --help"
