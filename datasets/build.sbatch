#!/usr/bin/env bash
# Submit from the repository checkout:  sbatch datasets/build.sbatch [extra args]
# Submit with an explicit log path (SBATCH lines cannot expand $SCRATCH):
#   mkdir -p $SCRATCH/cb/logs && sbatch -o $SCRATCH/cb/logs/datasets-%j.out datasets/build.sbatch
#SBATCH --job-name=datasets-build
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --account=d131
set -euo pipefail

ROOT="${CB_ROOT:-/capstor/store/cscs/userlab/d131/cb}"
export PATH="$ROOT/tools/$(uname -m)/cargo/bin:$PATH"  # webgraph (setup_webgraph.sh)
cd "${SLURM_SUBMIT_DIR:-.}"
"$ROOT/venv/daint-gh200/bin/python" -m datasets build --root "$ROOT" "$@"
