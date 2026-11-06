#!/usr/bin/env bash
# Usage: bench/ci.sh <machine>
#
# The body of a job of Ginkgo's CSCS CI pipeline "benchmark"
# (.gitlab/cscs-benchmark.yml in ginkgo-project/ginkgo). It runs on the
# allocated node, as the CI's service account. It benchmarks $CI_COMMIT_SHA
# with the shared clone, the venv and the per-account build of bench/job.sh,
# and copies the run into $CI_PROJECT_DIR/results/<machine>/ for the job's
# artifacts. The dashboard imports those (python -m automation ingest).
#
# CB_REF is the ref to record, origin/develop when the trigger started the job;
# without it the run records the sha, so a test by hand is never imported.
# CI_DRY_RUN=1 prints the uenv and benchmark commands instead of running them.
set -euo pipefail
# state/ and the clone are shared with the maintainer's account
umask 0002
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MACHINE="${1:?usage: $0 <machine>}"
: "${CI_COMMIT_SHA:?CI_COMMIT_SHA is not set: run this in a CSCS CI job}"
: "${CI_PROJECT_DIR:?CI_PROJECT_DIR is not set: run this in a CSCS CI job}"
: "${SCRATCH:?SCRATCH is not set}"
CB_ROOT="${CB_ROOT:-/capstor/store/cscs/userlab/d131/cb}"
PYTHON="${CB_PYTHON:-$CB_ROOT/venv/$MACHINE/bin/python}"
cd "$REPO"
UENV="$("$PYTHON" -m bench uenv "$MACHINE")"
OUT="$CI_PROJECT_DIR/results"
# the project directory can be left over from an earlier job
rm -rf "$OUT"

run() {
  if [ "${CI_DRY_RUN:-0}" = 1 ]; then echo "$*"; else "$@"; fi
}

# Pull the uenv once per version, as Ginkgo's test jobs do (.gitlab/cscs.yml):
# the images are in the service account's $SCRATCH/.uenv-images.
run flock "$SCRATCH/.ginkgo-ci-uenv-pull.lock" bash -c \
  "uenv image inspect $UENV > /dev/null 2>&1 || uenv image pull $UENV"

PATH_FILE="$(mktemp)"
trap 'rm -f "$PATH_FILE"' EXIT
status=0
run uenv run --view=default "$UENV" -- "$PYTHON" -m bench run "$MACHINE" \
  --ref "$CI_COMMIT_SHA" --record-ref "${CB_REF:-$CI_COMMIT_SHA}" --force \
  --path-file "$PATH_FILE" || status=$?

if [ -s "$PATH_FILE" ]; then
  mkdir -p "$OUT/$MACHINE"
  cp -r "$(cat "$PATH_FILE")" "$OUT/$MACHINE/"
fi
exit "$status"
