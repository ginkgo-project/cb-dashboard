import io
import json
import subprocess
import urllib.error
import zipfile

import pytest

from automation import ingest
from automation.ingest import (Rejected, accept, imported_job_ids, in_develop, main, place,
                               select_jobs, unpack)

NAME = "2026-11-05_655362e804"
SHA = "655362e804" + "0" * 30
PROJECT = "cscs-ci/ci-testing/webhook-ci/mirrors/1/2"


def zip_of(members: dict[str, bytes], symlink: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (0o120777 << 16)
            z.writestr(info, "/etc/passwd")
    return buf.getvalue()


def record(machine="daint-gh200", job=9, ref="origin/develop", groups=None):
    return {"machine": machine, "ginkgo": {"commit": SHA, "ref": ref},
            "groups": groups or {"blas": {"status": "ok"}}, "ci": {"job_id": job}}


def good(machine="daint-gh200", name=NAME, rec=None) -> bytes:
    rec = rec or record(machine)
    return zip_of({f"results/{machine}/{name}/run.json": json.dumps(rec).encode(),
                   f"results/{machine}/{name}/blas.json": b"[]",
                   f"results/{machine}/{name}/logs/blas.log": b"log"})


def test_unpack_accepts_one_run_directory():
    name, files = unpack(good(), "daint-gh200", 10**6)
    assert name == NAME and set(files) == {"run.json", "blas.json", "logs/blas.log"}
    assert files["logs/blas.log"] == b"log"


@pytest.mark.parametrize("members", [
    {f"results/daint-gh200/{NAME}/run.json": b"{}",
     "results/daint-gh200/2026-11-06_1111111111/run.json": b"{}"},
    {"results/daint-gh200/not-a-run/run.json": b"{}"},
    {f"results/daint-gh200/{NAME}/../../x": b"", f"results/daint-gh200/{NAME}/run.json": b"{}"},
    {f"/results/daint-gh200/{NAME}/run.json": b"{}"},
    {f"results/beverin-mi300a/{NAME}/run.json": b"{}"},
    {f"results/daint-gh200/{NAME}/blas.json": b"[]"},
    {f"results/daint-gh200/{NAME}/run.json": b"not json"},
    {f"results/daint-gh200/{NAME}/run.json": b"[1]"},
    {},
])
def test_unpack_rejects(members):
    with pytest.raises(Rejected):
        unpack(zip_of(members), "daint-gh200", 10**6)


def test_unpack_rejects_symlinks_size_and_garbage():
    with pytest.raises(Rejected, match="symlink"):
        unpack(zip_of({f"results/daint-gh200/{NAME}/run.json": b"{}"},
                      symlink=f"results/daint-gh200/{NAME}/link"), "daint-gh200", 10**6)
    with pytest.raises(Rejected, match="bytes"):
        unpack(good(), "daint-gh200", 10)
    with pytest.raises(Rejected):
        unpack(b"not a zip", "daint-gh200", 10**6)


def test_accept():
    yes, no = (lambda s: True), (lambda s: False)
    rec = record()
    assert accept(rec, "daint-gh200", yes) is None
    assert "machine" in accept(rec, "beverin-mi300a", yes)
    assert "develop" in accept(record(ref=SHA), "daint-gh200", yes)
    assert "not requested" in accept(record(groups={
        "blas": {"status": "skipped", "reason": "not requested"}}), "daint-gh200", yes)
    assert "history" in accept(rec, "daint-gh200", no)
    # a build failure is a run like any other: the page says develop didn't build
    assert accept(record(groups={"blas": {"status": "skipped", "reason": "build failed"}}),
                  "daint-gh200", yes) is None


def test_select_jobs():
    jobs = [{"id": 1, "name": "benchmark/daint-gh200", "status": "success"},
            {"id": 2, "name": "benchmark/daint-gh200", "status": "failed"},
            {"id": 3, "name": "benchmark/daint-gh200", "status": "running"},
            {"id": 4, "name": "benchmark/other", "status": "success"},
            {"id": 5, "name": "benchmark/beverin-mi300a", "status": "success"}]
    assert [j["id"] for j in select_jobs(jobs, ["beverin-mi300a", "daint-gh200"], {5})] == [1, 2]


def test_place_keeps_a_free_name_and_renumbers_a_taken_one(tmp_path):
    assert place(tmp_path, "m", NAME, {"run.json": b"{}"}).name == NAME
    assert place(tmp_path, "m", NAME, {"run.json": b"{}"}).name == f"{NAME}_2"
    assert place(tmp_path, "m", f"{NAME}_2", {"run.json": b"{}", "logs/a": b"a"}).name == f"{NAME}_3"
    assert (tmp_path / "m" / f"{NAME}_3" / "logs" / "a").read_bytes() == b"a"
    digits = "2026-11-05_1234567890"  # a sha10 of digits only is not a suffix
    place(tmp_path, "m", digits, {"run.json": b"{}"})
    assert place(tmp_path, "m", digits, {"run.json": b"{}"}).name == f"{digits}_2"


def test_imported_job_ids(tmp_path):
    for d, rec in (("a", {"ci": {"job_id": 4}}), ("b", {"ci": {"job_id": 9}}), ("c", {}),
                   ("d", None)):
        p = tmp_path / "m" / d
        p.mkdir(parents=True)
        (p / "run.json").write_text("not json" if rec is None else json.dumps(rec))
    assert imported_job_ids(tmp_path) == {4, 9}


@pytest.mark.parametrize("status, expected", [("ahead", True), ("identical", True),
                                              ("diverged", False), ("behind", False)])
def test_in_develop(status, expected):
    seen = []
    def get_json(url, token=None):
        seen.append(url)
        return {"status": status}
    assert in_develop(SHA, get_json) is expected
    assert seen == [f"https://api.github.com/repos/ginkgo-project/ginkgo/compare/{SHA}...develop"]


# end to end: a real git work tree, faked gitlab.com and GitHub

@pytest.fixture
def repo(tmp_path, monkeypatch):
    for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                 "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_COUNT": "1",
                 "GIT_CONFIG_KEY_0": "commit.gpgsign", "GIT_CONFIG_VALUE_0": "false"}.items():
        monkeypatch.setenv(k, v)
    results = tmp_path / "results"
    results.mkdir()
    subprocess.run(["git", "init", "-q", str(results)], check=True)
    (results / ".gitignore").write_text("\n")
    subprocess.run(["git", "-C", str(results), "add", ".gitignore"], check=True)
    subprocess.run(["git", "-C", str(results), "commit", "-q", "-m", "start"], check=True)
    return results


@pytest.fixture
def gitlab(monkeypatch):
    """One pipeline with the jobs in `jobs`; artifacts by job id in `zips`."""
    state = {"jobs": [], "zips": {}, "develop": True, "fail_listing": False}

    def get_json(url, token=None):
        if url.startswith("https://api.github.com/"):
            return {"status": "ahead" if state["develop"] else "diverged"}
        if state["fail_listing"]:
            raise urllib.error.URLError("down")
        assert "/projects/cscs-ci%2Fci-testing%2Fwebhook-ci%2Fmirrors%2F1%2F2/" in url
        if "/pipelines?" in url:
            return [{"id": 77}]
        assert "/pipelines/77/jobs" in url
        return state["jobs"]

    def download(url, limit):
        job = int(url.rsplit("/jobs/", 1)[1].split("/")[0])
        if job not in state["zips"]:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return state["zips"][job]

    monkeypatch.setattr(ingest, "get_json", get_json)
    monkeypatch.setattr(ingest, "download", download)
    return state


def log(results):
    return subprocess.run(["git", "-C", str(results), "log", "--format=%s"], check=True,
                          capture_output=True, text=True).stdout.splitlines()


def run_ingest(results, *extra):
    return main(["--results", str(results), "--project", PROJECT, *extra], {})


def test_main_imports_commits_and_is_idempotent(repo, gitlab):
    gitlab["jobs"] = [{"id": 9, "name": "benchmark/daint-gh200", "status": "success"}]
    gitlab["zips"][9] = good()
    assert run_ingest(repo) == 0
    assert log(repo)[0] == "results: daint-gh200 655362e804, ci job 9"
    assert (repo / "daint-gh200" / NAME / "logs" / "blas.log").read_bytes() == b"log"
    assert run_ingest(repo) == 0
    assert len(log(repo)) == 2  # nothing new the second time


def test_main_skips_rejected_and_expired(repo, gitlab, capsys):
    gitlab["jobs"] = [{"id": 1, "name": "benchmark/daint-gh200", "status": "success"},
                      {"id": 2, "name": "benchmark/daint-gh200", "status": "success"},
                      {"id": 3, "name": "benchmark/beverin-mi300a", "status": "failed"},
                      {"id": 4, "name": "benchmark/beverin-mi300a", "status": "success"}]
    gitlab["zips"][2] = good(rec=record(job=2, ref=SHA))      # a test by hand
    gitlab["zips"][3] = good("beverin-mi300a", rec=record("beverin-mi300a", job=99))
    gitlab["zips"][4] = good("beverin-mi300a", rec=record("beverin-mi300a", job=4))
    assert run_ingest(repo) == 0
    out = capsys.readouterr().out
    assert out.count("::warning::") == 3
    assert "job 1" in out and "job 2" in out and "job 3" in out
    assert log(repo)[0] == "results: beverin-mi300a 655362e804, ci job 4"
    assert len(log(repo)) == 2


def test_main_skips_commits_not_in_develop(repo, gitlab, capsys):
    gitlab["jobs"] = [{"id": 9, "name": "benchmark/daint-gh200", "status": "success"}]
    gitlab["zips"][9] = good()
    gitlab["develop"] = False
    assert run_ingest(repo) == 0
    assert "::warning::" in capsys.readouterr().out and len(log(repo)) == 1


def test_main_api_failure_exits_1(repo, gitlab):
    gitlab["fail_listing"] = True
    assert run_ingest(repo) == 1


def test_dry_run_commits_nothing(repo, gitlab, capsys):
    gitlab["jobs"] = [{"id": 9, "name": "benchmark/daint-gh200", "status": "success"}]
    gitlab["zips"][9] = good()
    assert run_ingest(repo, "--dry-run") == 0
    assert "would import job 9" in capsys.readouterr().out
    assert len(log(repo)) == 1 and not (repo / "daint-gh200").exists()
