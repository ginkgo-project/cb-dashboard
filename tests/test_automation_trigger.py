import json
import urllib.error

import pytest

from automation import trigger
from automation.trigger import MACHINES, TRIGGER_URL, body, has_run, main, pending

SHA = "655362e804" + "0" * 30
ENV = {"CSCS_CI_REPOSITORY_ID": "123", "CSCS_CI_WEBHOOK_SECRET": "s3cr3t"}


def test_machines_are_the_harness_machines():
    assert MACHINES == ["beverin-mi300a", "daint-gh200"]


def test_has_run():
    assert has_run(["2026-11-05_655362e804"], SHA)
    assert has_run(["2026-11-05_655362e804_2"], SHA)
    assert not has_run(["2026-11-05_655362e805", "655362e804", "x_655362e804",
                        "2026-11-05_655362e804_x"], SHA)


def test_pending_lists_the_machines_without_the_commit():
    listing = {"daint-gh200": ["2026-11-05_655362e804"], "beverin-mi300a": []}
    assert pending(SHA, listing.__getitem__, ["beverin-mi300a", "daint-gh200"]) == [
        "beverin-mi300a"]


def test_body():
    assert body(SHA, ["daint-gh200"]) == {
        "ref": f"sha:{SHA}", "pipeline": "benchmark",
        "variables": {"CB_BENCHMARK": "1", "CB_MACHINES": "daint-gh200",
                      "CB_REF": "origin/develop"}}


@pytest.fixture
def fake(monkeypatch):
    """Results with SHA on daint only; beverin has no directory (404)."""
    state = {"dirs": {"daint-gh200": [f"2026-11-05_{SHA[:10]}"]}, "posts": [],
             "reply": (200, "pipeline created"), "tokens": []}

    def get_json(url, token=None):
        state["tokens"].append(token)
        machine = url.split("/contents/")[1].split("?")[0]
        assert url.endswith("?ref=results")
        if machine not in state["dirs"]:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return [{"name": n, "type": "dir"} for n in state["dirs"][machine]]

    def post(url, data, auth):
        state["posts"].append((url, data, auth))
        return state["reply"]

    monkeypatch.setattr(trigger, "get_json", get_json)
    monkeypatch.setattr(trigger, "post", post)
    return state


def test_nothing_to_do_posts_nothing(fake, capsys):
    fake["dirs"]["beverin-mi300a"] = [f"2026-11-06_{SHA[:10]}_2"]
    assert main(["--sha", SHA], ENV) == 0
    assert "nothing to do" in capsys.readouterr().out
    assert fake["posts"] == []


def test_missing_machine_directory_is_pending(fake):
    assert main(["--sha", SHA], ENV) == 0
    [(_, data, _)] = fake["posts"]
    assert json.loads(data)["variables"]["CB_MACHINES"] == "beverin-mi300a"


def test_posts_with_basic_auth(fake):
    fake["dirs"]["daint-gh200"] = []
    assert main(["--sha", SHA], {**ENV, "GITHUB_TOKEN": "gh"}) == 0
    [(url, data, auth)] = fake["posts"]
    assert url == TRIGGER_URL == "https://cicd-ext-mw.cscs.ch/ci/pipeline/trigger"
    assert auth == ("123", "s3cr3t")
    assert json.loads(data) == body(SHA, ["beverin-mi300a", "daint-gh200"])
    assert set(fake["tokens"]) == {"gh"}


def test_force_ignores_the_results(fake):
    assert main(["--sha", SHA, "--force", "--machines", "daint-gh200"], ENV) == 0
    [(_, data, _)] = fake["posts"]
    assert json.loads(data)["variables"]["CB_MACHINES"] == "daint-gh200"


def test_machines_limits_the_check(fake, capsys):
    assert main(["--sha", SHA, "--machines", "daint-gh200"], ENV) == 0
    assert fake["posts"] == [] and "nothing to do" in capsys.readouterr().out


def test_dry_run_prints_the_body_and_posts_nothing(fake, capsys):
    assert main(["--sha", SHA, "--dry-run"], {}) == 0
    assert json.loads(capsys.readouterr().out.split("\n", 1)[1]) == body(
        SHA, ["beverin-mi300a"])
    assert fake["posts"] == []


def test_missing_secret_exits_2(fake, capsys):
    assert main(["--sha", SHA], {"CSCS_CI_REPOSITORY_ID": "123"}) == 2
    assert "CSCS_CI_WEBHOOK_SECRET" in capsys.readouterr().err
    assert fake["posts"] == []


def test_http_error_exits_1(fake):
    fake["reply"] = (500, "boom")
    assert main(["--sha", SHA], ENV) == 1


def test_bad_sha_and_unknown_machine_exit_2(fake):
    assert main(["--sha", "abc"], ENV) == 2
    assert main(["--sha", SHA, "--machines", "nope"], ENV) == 2
    assert fake["posts"] == []


def test_secret_is_never_printed(fake, capsys):
    for reply, rc in (((200, "ok"), 0), ((403, "denied"), 1)):
        fake["reply"] = reply
        assert main(["--sha", SHA], ENV) == rc
    assert main(["--sha", SHA], {"CSCS_CI_WEBHOOK_SECRET": "s3cr3t"}) == 2
    out = capsys.readouterr()
    assert "s3cr3t" not in out.out + out.err
