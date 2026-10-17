#!/usr/bin/env bash
# Submit from the repository checkout:  sbatch datasets/build.sbatch [extra args]
# Adjust the --output path below to your $SCRATCH (SBATCH lines cannot expand
# variables) and create the directory first: mkdir -p $SCRATCH/cb/logs
#SBATCH --job-name=datasets-build
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --account=d131
#SBATCH --output=/capstor/scratch/cscs/CHANGE_ME/cb/logs/datasets-%j.out
set -euo pipefail

ROOT="${CB_ROOT:-/capstor/store/cscs/userlab/d131/cb}"
export PATH="$ROOT/tools/cargo/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-.}"
"$ROOT/venv/daint-gh200/bin/python" -m datasets build --root "$ROOT" "$@"
