#!/usr/bin/env bash
# Installs the Rust WebGraph CLI (crate webgraph-cli, binary `webgraph`) into
# $CB_ROOT/tools. Binary ends up in $CB_ROOT/tools/cargo/bin/webgraph.
set -euo pipefail

ROOT="${CB_ROOT:-/capstor/store/cscs/userlab/d131/cb}"
export RUSTUP_HOME="$ROOT/tools/rustup"
export CARGO_HOME="$ROOT/tools/cargo"

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
