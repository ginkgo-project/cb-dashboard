#!/usr/bin/env bash
# Usage: bench/submit.sh <machine> [--ref REF] [--force] [--groups ...]
# SUBMIT_DRY_RUN=1 prints the sbatch command instead of running it.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MACHINE="${1:?usage: $0 <machine> [args...]}"
shift
CB_ROOT="${CB_ROOT:-/capstor/store/cscs/userlab/d131/cb}"
PYTHON="${CB_PYTHON:-$CB_ROOT/venv/$MACHINE/bin/python}"
SCRATCH="${SCRATCH:?SCRATCH is not set}"
cd "$REPO"
read -r -a SB < <("$PYTHON" -m bench sbatch-args "$MACHINE")
# singleton: the jobs of one machine share its worktree and build directory, so
# they run one after the other.
CMD=(sbatch "${SB[@]}" --view=default "--job-name=cb-$MACHINE" --dependency=singleton
     "--output=$SCRATCH/cb/logs/%j.out" bench/job.sh "$MACHINE" "$@")
if [ "${SUBMIT_DRY_RUN:-0}" = 1 ]; then
  echo "${CMD[*]}"
else
  mkdir -p "$SCRATCH/cb/logs"
  exec "${CMD[@]}"
fi
