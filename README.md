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

`uv`, its CPython and its cache go to `$CB_ROOT/tools/<arch>/` (`aarch64` or
`x86_64`), so the two machines never share a binary. Reruns reuse `uv` and the
venv. `CB_DRY_RUN=1` only prints the steps.

### 2. WebGraph (twitter-2010 and the other LAW graphs)

On Daint, where the dataset build runs:

    datasets/setup_webgraph.sh
    $CB_ROOT/tools/aarch64/cargo/bin/webgraph to arcs --help

Rust and the binary go to `$CB_ROOT/tools/$(uname -m)/{rustup,cargo}`, and
`datasets/build.sbatch` puts `$CB_ROOT/tools/$(uname -m)/cargo/bin` on `PATH`.

The default `--webgraph-cmd` is `sh -c "webgraph to arcs {basename} > {out}"`
(crate `webgraph-cli`; `to arcs` writes TAB-separated arcs to stdout, per
docs.rs). It was read from the docs, not run: confirm with `--help` above, and
override `--webgraph-cmd` if it differs. Check: twitter-2010 has 1,468,365,182 arcs.

### 3. FEM sizes

Edit the `fem = {problem, size}` entries in `datasets/catalog.toml` so each FEM
matrix has 20-100M nonzeros. Test on a compute node:

    srun -A <account> -p normal -N1 --exclusive -t 01:00:00 \
      $CB_ROOT/venv/daint-gh200/bin/python -m datasets build --root $CB_ROOT \
      --manifest $SCRATCH/cb/tuning-manifest.json --only <fem names>

`--only` on a missing manifest writes a manifest holding just those entries, so
always pass the separate `--manifest` while tuning; the committed
`datasets/manifest.json` must come only from the full build.

Changing a size changes `catalog_sha256`. No manifest exists yet, so `version`
stays `"1"`; after the first manifest is committed, any catalog change needs a
version bump. So does a converter change that alters the data: a rebuilt matrix
whose `binary_sha256` differs from the manifest of the same version stops the
build with exit code 2. Each entry records its `converter_version` (this
repository's commit, `-dirty` if a tracked file changed, the numpy, scipy and
scikit-fem versions, and `webgraph --version` for twitter-2010). Also check
that `test_row_balance`'s thresholds still hold.

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

### Build

The build directory is `$SCRATCH/cb/build/<machine>`, and rebuilds are
incremental. A missing or half configured directory is configured again. So is
one whose `cb-config.json` (the uenv, the CMake arguments, the `g++`, `nvcc`
and `clang++` versions and the Ginkgo commit) doesn't match the current run,
then with `cmake --fresh`. Besides the five benchmarks, the build makes
`matrix_statistics` and `mtx_to_binary`.

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
Slurm stopped the job with SIGTERM (time limit or `scancel`). The jobs of one
machine and account share its worktree (`$CB_ROOT/worktrees/<user>/<machine>`)
and build directory (`$SCRATCH` is per account), so they are submitted with
`--dependency=singleton` and run one after the other. The job runs with
`umask 0002`, so the shared clone and `state/` stay writable for the other
accounts that run benchmarks.

### Older Ginkgo versions

To fill the history, older releases and develop commits run with today's
benchmark code. Each has a branch `cb/<point>` in the Ginkgo repository: the
point plus one commit that only changes `benchmark/`. That commit puts in the
benchmark directory of `cb/harness`, which is `update/benchmarks` without
`write_config` and `get_actual_strategy` (core changes; such runs record no
`parameters`), and undoes the API changes of develop after the point. Its
committer date is the point's, so the results directory and the dashboard
column carry the point's date. Run one with

    bench/submit.sh <machine> --ref origin/cb/<point>

A preconditioner that the commit lacks (see `ADDED` in `bench/groups.py`) isn't
run: its invocations are `not_available`, hatched on the dashboard.

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
  under `failures`), `skipped` when its timeout no longer fits in the job,
  `disabled` on this machine, or `not_available` in this Ginkgo commit.

### Copy back

    rsync -a daint:/ritom/scratch/cscs/<user>/cb/results/daint-gh200/ results/daint-gh200/
    rsync -a beverin:/ritom/scratch/cscs/<user>/cb/results/beverin-mi300a/ results/beverin-mi300a/

Replace `<user>` with your cluster user name. `$SCRATCH` would expand on the
laptop, so the remote paths are written out. `results/` is the worktree of the
results branch (see "Results"), so commit and push the new run there:

    git -C results add <machine>/<dir>
    git -C results commit -m "results: <machine> <sha10>"
    git -C results push origin results

## Results

The runs are on the orphan branch `results`, whose root holds
`<machine>/<date>_<sha10>[_n]/`. Main has only the code. In a checkout of main,
the branch is the worktree `results/`, which main ignores:

    git fetch origin results
    git worktree add results results

Only full runs of develop go on the branch: the weekly ones arrive
automatically (see "Weekly automation"), and manual runs are committed by
hand. Each run's `logs/` stays out of git (`results/.gitignore`). The logs of
the CSCS CI runs are in the job artifacts for 30 days.

To take out an imported run, delete its directory and add its CI job id
(`ci.job_id` in its `run.json`) to `removed-jobs.txt` at the branch's root, one
per line, `#` starting a comment. Otherwise the import takes it again while
its pipeline is recent.

## Dashboard

`python -m dashboard build` reads the runs under `results/` whose `run.json` is
tracked by git, together with `dashboard/machines.toml` (per-machine peaks),
`dashboard/models.py` and `dashboard/frozen.json`, and writes the page data to
`site/data/`. `--ginkgo PATH` points it at a Ginkgo checkout. To look at the
page locally:

    python -m dashboard build
    python3 -m http.server -d site 8000

After committing a run with new cases, run `python -m dashboard freeze` and
commit the updated `dashboard/frozen.json`.

The models in `dashboard/models.py` (bytes and flops per case) are fixed per
benchmark: the history is recomputed from them at every build, so correcting a
model changes every past point. Change one only to fix an error.

`site/data/` is generated and git-ignored. `.github/workflows/deploy.yml`
runs the tests, builds the data from main and the `results` branch, and deploys
`site/` to GitHub Pages at `cb.gko-project.org` (`site/CNAME`). It deploys:

- on every push to `main`, and by hand;
- hourly, when the `results` branch differs from the commit the live page was
  built from (`results_commit` in `data/meta.json`). A push to `results` by
  hand is therefore live within the hour.

## Weekly automation

1. **Trigger:** `.github/workflows/benchmark.yml` in Ginkgo runs twice a week.
   It runs `python -m automation trigger` from this repository, with develop's
   head. For the machines without a run of that
   commit on the `results` branch, it starts the CSCS CI pipeline `benchmark`
   of Ginkgo with `ref: sha:<head>`.
2. **Benchmark:** `.gitlab/cscs-benchmark.yml` in Ginkgo has one job per
   machine. It runs only when the trigger sets `CB_BENCHMARK=1`. It clones
   this repository and runs `bench/ci.sh <machine>`, with the same clone, venv
   and build setup as a manual run.
3. **Import:** the hourly run of `deploy.yml` runs
   `python -m automation ingest`. It finds the finished jobs on gitlab.com,
   where their artifacts are public. It checks each run: a single run
   directory, no paths outside it, the ref `origin/develop`, all groups run,
   the commit in develop's history. Then it commits the run to `results`, and
   the page redeploys.

Still done by hand: bisects, pull request and history runs (`submit.sh`), and
`python -m dashboard freeze` after a run with new cases.

`python -m automation trigger --sha <sha> --dry-run` prints what would be
triggered without sending anything.

### Configuration

- **Ginkgo's CSCS CI:** a pipeline `benchmark` with the entry point
  `.gitlab/cscs-benchmark.yml`, and the Actions secrets
  `CSCS_CI_REPOSITORY_ID` and `CSCS_CI_WEBHOOK_SECRET` for the trigger. Give
  the pipeline its own list of CI-enabled branches, naming a branch that
  doesn't exist. An empty list means the global one, so a push to develop
  would start the pipeline without the trigger's variables: it has no jobs
  then, and its status fails with "The resulting pipeline would have been
  empty".
- **This repository:** the Actions variable `CSCS_CI_BENCHMARK_PROJECT`. It is
  the pipeline's gitlab.com mirror,
  `cscs-ci/ci-testing/webhook-ci/mirrors/<project_id>/<pipeline_id>`. The
  import is skipped while it is not set.
- **The store:** every account that runs benchmarks must be able to write the
  shared clone and `state/` under `$CB_ROOT`. Use a common group, group-writable
  setgid directories, and `git config core.sharedRepository group` on the clone.

## Distributed benchmarks

Once a week, a 16-node job per machine measures how Ginkgo's distributed
operations scale from 1 to 64 GPUs. It is weak scaling: every GPU gets 256³
rows of a 3D 7-point stencil. The operations are:

- SpMV and SpGEMM (A·A);
- CG and GMRES with a Schwarz-Jacobi preconditioner;
- the multigrid preconditioners PGM, uniform coarsening and PMIS, with generate
  and apply timed separately.

The benchmarks are Ginkgo's distributed ones under `benchmark/*/distributed`.

First, in the same allocation, `mpibench/` times `MPI_Allreduce`,
`MPI_Alltoallv`, `MPI_Neighbor_alltoallv` and a ping-pong. It sweeps the
message sizes, up to 2 GiB for the allreduce, on GPU buffers with GPU-aware
MPI. It doesn't use Ginkgo, so if a benchmark gets slower together with the
MPI operation it depends on, the cause is MPI or the network.

### Run

    python -m bench distributed <machine> [--gpus 1,8] [--dry-run]

Inside an allocation of the machine's nodes, this builds Ginkgo with MPI and
builds `mpibench`. Then it runs every GPU count in turn, each as `srun` steps
on the nodes it needs. The settings are in the `[distributed]` table of
`bench/machines/<machine>.toml`:

- nodes and GPUs per node;
- the srun flags per rank;
- the extra CMake flags (MPI, GPU-aware MPI);
- the environment;
- `mpibench`'s CMake flags.

It has its own build directory (`$SCRATCH/cb/build/<machine>-distributed`),
worktree (`$CB_ROOT/worktrees/<user>/<machine>-distributed`) and state
(`state/distributed/<machine>/`), so it can run next to a single-node job.

**Results:** the runs go to `$SCRATCH/cb/results/distributed/<machine>/<dir>/`,
with `run.json` and `gpus-<n>/<group>.json`. The groups are `mpi`, `spmv`,
`spgemm`, `solver` and `preconditioner`.

**Failures:** a failed GPU count or benchmark doesn't stop the run. It is
recorded as failed.

`mpibench` also builds and runs on its own, with host buffers, for a test:

    cmake -S mpibench -B build-mpibench -DMPIBENCH_GPU=none && cmake --build build-mpibench
    mpirun -n 4 build-mpibench/mpibench --max-allreduce-k 3 --max-alltoallv-k 2

### Automation and the tab

1. **Trigger:** on Saturday nights, Ginkgo's `.github/workflows/benchmark.yml`
   runs `python -m automation trigger --distributed`. It starts the CSCS CI
   jobs `benchmark-distributed/<machine>` (`CB_DISTRIBUTED=1`) for the
   machines without a distributed run of develop's head.
2. **Benchmark:** each job runs `bench/ci.sh <machine> distributed`.
3. **Import:** the hourly import puts the runs on the `results` branch under
   `distributed/<machine>/`.

**The Distributed tab:**

- **Change table:** compares two runs of a machine, by default the last two. It
  has one row per benchmark and phase, and three rows per MPI operation (8 B,
  128 KiB, its largest size). Each cell is the change in time at one GPU count.
- **Colours:** slower and faster are judged against the noise of the earlier
  weekly runs, with the same rule as the other tabs.
- **Opening a benchmark row** shows:
  - its time and its weak-scaling efficiency against the GPU count;
  - the numbers;
  - the MPI operations at the same GPU counts;
  - for multigrid, the hierarchy;
  - for the solvers, the iterations.
- **Opening an MPI row** shows a change map of message size against GPU count,
  and the time and bandwidth for one GPU count.
