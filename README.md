# cb-dashboard

## Datasets

The SpMV matrix set is built by `python -m datasets build` into
`/capstor/store/cscs/userlab/d131/cb` (override with `CB_ROOT`). Environments are
per machine (`<root>/venv/<machine>`) because Daint is aarch64 and Beverin x86_64.

### 0. Setup of the shell

    export CB_ROOT=/capstor/store/cscs/userlab/d131/cb
    cd <path to the cb-dashboard checkout>   # all commands below run from here

The package is installed editable, so `catalog.toml` and `manifest.json` always
come from this checkout.

### 1. Environment (once per machine, on a node of that machine)

    datasets/setup_env.sh daint-gh200      # on Daint (refuses unless aarch64)
    datasets/setup_env.sh beverin-mi300a   # on Beverin (refuses unless x86_64)

Reruns reuse `uv` and the venv. `CB_DRY_RUN=1` only prints the steps.

### 2. WebGraph (twitter-2010 and the other LAW graphs)

    datasets/setup_webgraph.sh
    $CB_ROOT/tools/cargo/bin/webgraph to arcs --help

The default `--webgraph-cmd` is `sh -c "webgraph to arcs {basename} > {out}"`
(crate `webgraph-cli`; `to arcs` writes TAB-separated arcs to stdout, per
docs.rs). It was read from the docs, not run: confirm with `--help` above, and
override `--webgraph-cmd` if it differs. Check: twitter-2010 has 1,468,365,182 arcs.

### 3. FEM sizes

Edit the `fem = {problem, size}` entries in `datasets/catalog.toml` so each FEM
matrix has 20-100M nonzeros. Test on a compute node:

    srun -A d131 -p normal -N1 --exclusive -t 01:00:00 \
      $CB_ROOT/venv/daint-gh200/bin/python -m datasets build --root $CB_ROOT \
      --manifest $SCRATCH/cb/tuning-manifest.json --only <fem names>

`--only` on a missing manifest writes a manifest holding just those entries, so
always pass the separate `--manifest` while tuning; the committed
`datasets/manifest.json` must come only from the full build.

Changing a size changes `catalog_sha256`. No manifest exists yet, so `version`
stays `"1"`; after the first manifest is committed, any catalog change needs a
version bump. Also check that `test_row_balance`'s thresholds still hold.

### 4. Full build

    mkdir -p $SCRATCH/cb/logs && sbatch -o $SCRATCH/cb/logs/datasets-%j.out datasets/build.sbatch

Expect: finishes within 6 h, `datasets/manifest.json` has 17 entries, every
nonzero check passes, `du -sh $CB_ROOT/matrices` is about 60 GB. Then:

    git add datasets/manifest.json datasets/catalog.toml README.md
    git commit -m "datasets: manifest of dataset version 1"

## Running the benchmarks

`python -m bench run <machine>` builds a Ginkgo commit and runs the five
benchmark groups (blas, spmv, spgemm, solver, preconditioner) on one node.

### Setup

Use the venv (`datasets/setup_env.sh`) and the checkout of this repository from
the "Datasets" section above. The dataset manifest `datasets/manifest.json` must
exist.

### Dry run (login node)

    cd $CB_ROOT/cb-dashboard
    $CB_ROOT/venv/daint-gh200/bin/python -m bench run daint-gh200 --dry-run

Prints the machine, the ref and one line per invocation. It runs no git, build
or srun.

### Submit

    bench/submit.sh <machine> [--ref REF] [--force]

The default ref is `origin/develop`. Without `--force`, a run of develop's head
that was already benchmarked exits with "nothing to do". `last_commit` is only
updated after a completed run of `origin/develop` with all groups, in any order
(a `--groups` subset is a debugging run). The groups always run in the order
blas, spmv, spgemm, solver, preconditioner. `SUBMIT_DRY_RUN=1` prints the
sbatch command instead of submitting it. Exit codes of the job: 0 on success or
nothing to do, 1 on a build failure, 2 on an architecture mismatch, 143 when
Slurm stopped the job with SIGTERM (time limit or `scancel`).

### Logs and results

- Slurm logs: `$SCRATCH/cb/logs/<jobid>.out`
- Results: `$SCRATCH/cb/results/<machine>/<date>_<sha10>/` (a rerun of the same
  commit gets `_2`, `_3`, ...) with `run.json`, one `<group>.json` per group and
  `logs/<group>/<label>.log` per invocation.
- `run.json` is rewritten after the build and after every group, so a killed
  job still has one. `finished` stays `null` until the run ends. A group is
  `pending`, `running`, `ok`, `partial`, `failed`, `killed` (the job got
  SIGTERM while it ran) or `skipped` (not requested, build failed, job killed).
  An invocation is `ok`, `partial` or `failed` (Ginkgo's own errors are listed
  under `failures`), or `skipped` when its timeout no longer fits in the job.

### Copy back

    rsync -a daint:/ritom/scratch/cscs/<user>/cb/results/daint-gh200/ results/daint-gh200/
    rsync -a beverin:/capstor/scratch/cscs/<user>/cb/results/beverin-mi300a/ results/beverin-mi300a/

Replace `<user>` with your cluster user name. `$SCRATCH` would expand on the
laptop, so the remote paths are written out.
