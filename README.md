# cb-dashboard

## Datasets

The SpMV matrix set is built by `python -m datasets build` into
`/capstor/store/cscs/userlab/d131/cb` (override with `CB_ROOT`). Environments are
per machine (`<root>/venv/<machine>`) because Daint is aarch64 and Beverin x86_64.

### 1. Environment (once per machine, on a node of that machine)

    git clone <repo> && cd cb-dashboard
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
      $CB_ROOT/venv/daint-gh200/bin/python -m datasets build --root $CB_ROOT --only <fem names>

Changing a size changes `catalog_sha256`. No manifest exists yet, so `version`
stays `"1"`; after the first manifest is committed, any catalog change needs a
version bump. Also check that `test_row_balance`'s thresholds still hold.

### 4. Full build

    mkdir -p $SCRATCH/cb/logs
    # edit the --output path in datasets/build.sbatch to your $SCRATCH
    sbatch datasets/build.sbatch

Expect: finishes within 6 h, `datasets/manifest.json` has 17 entries, every
nonzero check passes, `du -sh $CB_ROOT/matrices` is about 60 GB. Then:

    git add datasets/manifest.json datasets/catalog.toml README.md
    git commit -m "datasets: manifest of dataset version 1"
