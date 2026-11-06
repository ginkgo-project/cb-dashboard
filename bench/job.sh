#!/usr/bin/env bash
set -euo pipefail
CB_ROOT="${CB_ROOT:-/capstor/store/cscs/userlab/d131/cb}"
# state/ and the clone are shared with the CSCS CI's service account
umask 0002
exec "$CB_ROOT/venv/$1/bin/python" -m bench run "$@"
