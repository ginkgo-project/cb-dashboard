"""Import finished jobs of Ginkgo's CSCS CI pipeline "benchmark" into the results.

The jobs' artifacts are on gitlab.com, in the pipeline's mirror project, and
anyone can read them. Each finished job that is not imported yet is
downloaded and checked. If it is a full run of a develop commit, it is
committed to the work tree of the results branch as <machine>/<dir>/, one
commit per run. Nothing is pushed: .github/workflows/deploy.yml does that.

The artifacts are data. They are checked member by member, never executed,
and never extracted blindly.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping
from urllib.parse import quote

from automation.http import HTTPError, URLError, download, get_json
from automation.trigger import MACHINES
from bench.checkout import DEVELOP

GITLAB = "https://gitlab.com/api/v4"
COMPARE = "https://api.github.com/repos/ginkgo-project/ginkgo/compare/{sha}...develop"
RUN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2}_[0-9a-f]{10})(_\d+)?")
JOB_PREFIX = "benchmark/"
FINISHED = ("success", "failed")  # a failed job still has its run.json
NETWORK = (HTTPError, URLError, OSError, ValueError)


class Rejected(Exception):
    """An artifact that is not a run to import."""


def imported_job_ids(results: Path) -> set[int]:
    """The CI job ids recorded in the runs on the results branch."""
    ids = set()
    for p in Path(results).glob("*/*/run.json"):
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        job = (rec.get("ci") or {}).get("job_id") if isinstance(rec, dict) else None
        if isinstance(job, int):
            ids.add(job)
    return ids


def select_jobs(jobs: list[dict], machines: list[str], imported: set[int]) -> list[dict]:
    """The finished benchmark jobs of known machines not imported yet, in order."""
    names = {f"{JOB_PREFIX}{m}" for m in machines}
    return [j for j in jobs if j.get("name") in names and j.get("status") in FINISHED
            and j.get("id") not in imported]


def unpack(data: bytes, machine: str, limit: int) -> tuple[str, dict[str, bytes]]:
    """The run directory name and its files, by path inside it, from a job's artifacts."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise Rejected(f"not a zip file: {e}") from None
    members = [i for i in zf.infolist() if not i.is_dir()]
    if sum(i.file_size for i in members) > limit:
        raise Rejected(f"more than {limit} bytes unpacked")
    prefix = ("results", machine)
    dirs, files = set(), {}
    for info in members:
        name = info.filename
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise Rejected(f"{name}: a symlink")
        parts = PurePosixPath(name).parts
        if name.startswith("/") or "\\" in name or ".." in parts:
            raise Rejected(f"{name}: a path outside the run directory")
        if parts[:2] != prefix or len(parts) < 4:
            raise Rejected(f"{name}: not in results/{machine}/<run>/")
        dirs.add(parts[2])
        files["/".join(parts[3:])] = zf.read(info)
    if len(dirs) != 1:
        raise Rejected(f"{len(dirs)} run directories, not one")
    (run,) = dirs
    if not RUN_NAME.fullmatch(run):
        raise Rejected(f"{run}: not a run directory name")
    if "run.json" not in files:
        raise Rejected(f"{run}: no run.json")
    try:
        rec = json.loads(files["run.json"])
    except ValueError:
        raise Rejected(f"{run}: run.json is not JSON") from None
    if not isinstance(rec, dict):
        raise Rejected(f"{run}: run.json is not an object")
    return run, files


def accept(record: dict, machine: str, in_develop: Callable[[str], bool]) -> str | None:
    """None for a full run of a develop commit on `machine`, else why not."""
    if record.get("machine") != machine:
        return f"run.json is of machine {record.get('machine')!r}, the job of {machine!r}"
    ginkgo = record.get("ginkgo") or {}
    if ginkgo.get("ref") != DEVELOP:
        return f"not a develop run: ref {ginkgo.get('ref')!r}"
    groups = record.get("groups") or {}
    skipped = sorted(g for g, v in groups.items() if isinstance(v, dict)
                     and v.get("status") == "skipped" and v.get("reason") == "not requested")
    if skipped:
        return f"groups not requested: {', '.join(skipped)}"
    sha = ginkgo.get("commit")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        return f"no commit sha: {sha!r}"
    if not in_develop(sha):
        return f"{sha[:10]} is not in develop's history"
    return None


def in_develop(sha: str, get_json, token: str | None = None) -> bool:
    """Whether `sha` is develop's head or one of its ancestors."""
    try:
        r = get_json(COMPARE.format(sha=sha), token)
    except HTTPError as e:
        if e.code == 404:  # unknown commit
            return False
        raise
    return isinstance(r, dict) and r.get("status") in ("ahead", "identical")


def place(results: Path, machine: str, name: str, files: dict[str, bytes]) -> Path:
    """Write the run as results/<machine>/<name>, or the first free `_n` of its base name."""
    base = Path(results) / machine
    path = base / name
    if path.exists():
        stem = RUN_NAME.fullmatch(name).group(1)
        n = 2
        while (base / f"{stem}_{n}").exists():
            n += 1
        path = base / f"{stem}_{n}"
    for rel, data in files.items():
        f = path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
    return path


def finished_jobs(project: str, since: str) -> list[dict]:
    """The jobs of the mirror project's pipelines updated since `since`."""
    p = quote(project, safe="")
    jobs = []
    for pipeline in get_json(f"{GITLAB}/projects/{p}/pipelines?updated_after={since}&per_page=100"):
        jobs += get_json(f"{GITLAB}/projects/{p}/pipelines/{pipeline['id']}/jobs?per_page=100")
    return jobs


def _commit(results: Path, path: Path, message: str) -> None:
    rel = path.relative_to(results).as_posix()
    subprocess.run(["git", "-C", str(results), "add", "--", rel], check=True)
    subprocess.run(["git", "-C", str(results), "commit", "-q", "-m", message], check=True)


def _warn(message: str) -> None:
    print(f"::warning::{message}")  # a GitHub Actions annotation


def main(argv: list[str], env: Mapping[str, str]) -> int:
    p = argparse.ArgumentParser(prog="python -m automation ingest",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--results", required=True, help="a work tree of the results branch")
    p.add_argument("--project", required=True,
                   help="the gitlab.com mirror, cscs-ci/ci-testing/webhook-ci/mirrors/<id>/<id>")
    p.add_argument("--days", type=int, default=14, help="pipelines updated this recently")
    p.add_argument("--limit-mb", type=int, default=50, help="largest artifact")
    p.add_argument("--dry-run", action="store_true", help="check, but write nothing")
    args = p.parse_args(argv)
    results = Path(args.results)
    limit = args.limit_mb * 10**6
    token = env.get("GITHUB_TOKEN") or None
    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat(timespec="seconds")
    try:
        jobs = finished_jobs(args.project, quote(since))
    except NETWORK as e:
        print(f"error: listing the jobs of {args.project}: {e}", file=sys.stderr)
        return 1
    todo = select_jobs(jobs, MACHINES, imported_job_ids(results))
    done = 0
    for job in todo:
        machine = job["name"][len(JOB_PREFIX):]
        what = f"job {job['id']} ({machine})"
        url = f"{GITLAB}/projects/{quote(args.project, safe='')}/jobs/{job['id']}/artifacts"
        try:
            name, files = unpack(download(url, limit), machine, limit)
            rec = json.loads(files["run.json"])
            if (rec.get("ci") or {}).get("job_id") != job["id"]:
                # imported_job_ids would not find it again: it would be imported every hour
                raise Rejected(f"run.json's ci.job_id is {(rec.get('ci') or {}).get('job_id')!r}")
            reason = accept(rec, machine, lambda sha: in_develop(sha, get_json, token))
        except Rejected as e:
            _warn(f"{what}: {e}")
            continue
        except NETWORK as e:  # no artifacts (expired, or the job died before the upload)
            _warn(f"{what}: {e}")
            continue
        if reason:
            _warn(f"{what}: {reason}")
            continue
        if args.dry_run:
            print(f"would import {what}: {machine}/{name}")
            continue
        path = place(results, machine, name, files)
        _commit(results, path,
                f"results: {machine} {rec['ginkgo']['commit'][:10]}, ci job {job['id']}")
        print(f"imported {what}: {machine}/{path.name}")
        done += 1
    print(f"{done} of {len(todo)} new jobs imported")
    return 0
