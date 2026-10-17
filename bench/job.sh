#!/usr/bin/env bash
set -euo pipefail
CB_ROOT="${CB_ROOT:-/capstor/store/cscs/userlab/d131/cb}"
exec "$CB_ROOT/venv/$1/bin/python" -m bench run "$@"
