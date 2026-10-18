import json

from dash_fakes import write_run

from dashboard.__main__ import main
from dashboard.frozen import freeze, load_frozen, structure_of, structures
from dashboard.load import load_runs

KEY = "spmv/hybrid/cage15"


def _entry(ell, coo):
    return {"filename": "/x/cage15.bin", "rows": 5, "cols": 5, "nonzeros": 9,
            "spmv": {"hybrid": {
                "time": 1e-3, "repetition_times": [1e-3], "completed": True,
                "format_parameters": {
                    "ell_num_stored_elements_per_row": ell,
                    "coo_num_stored_elements": coo}}}}


def _write(tmp, name, commit, date, ell, coo, machine="m"):
    write_run(tmp, machine, name, commit=commit, date=date,
              started=date + "T10:00", data={"spmv": [_entry(ell, coo)]})


def test_freeze_takes_the_earliest_and_keeps_existing(tmp_path):
    _write(tmp_path, "B", "c2", "2026-10-19", 9, 90)
    _write(tmp_path, "A", "c1", "2026-10-18", 4, 40)
    frozen = freeze(load_runs(tmp_path), {})
    assert frozen == {"m": {KEY: {"ell_num_stored_elements_per_row": 4,
                                  "coo_num_stored_elements": 40}}}
    _write(tmp_path, "A", "c1", "2026-10-18", 7, 70)
    assert freeze(load_runs(tmp_path), frozen) == frozen
    kept = {"m": {KEY: {"ell_num_stored_elements_per_row": 1,
                        "coo_num_stored_elements": 2}}}
    assert freeze(load_runs(tmp_path), kept) == kept


def test_structure_of_other_kinds(tmp_path):
    pre = {"filename": "/x/p.bin", "rows": 4, "cols": 4, "nonzeros": 8,
           "preconditioner": {
               "jacobi-32-0,0": {"completed": True, "storage": 100,
                                 "apply": {"time": 1.0}, "generate": {"time": 1.0}},
               "mg-pgm": {"completed": True, "storage": 5,
                          "levels": [{"fine_rows": 4, "coarse_rows": 2,
                                      "coarse_nonzeros": 3, "x": 1}],
                          "apply": {"time": 1.0}, "generate": {"time": 1.0}},
               "ilu": {"completed": True, "apply": {"time": 1.0}}}}
    write_run(tmp_path, "m", "A", commit="c", date="2026-10-18",
              started="2026-10-18T10:00", data={"preconditioner": [pre]})
    frozen = freeze(load_runs(tmp_path), {})["m"]
    assert frozen["preconditioner/apply/jacobi/p"] == {"storage": 100}
    assert frozen["preconditioner/apply/mg-pgm/p"] == {"levels": [
        {"fine_rows": 4, "coarse_rows": 2, "coarse_nonzeros": 3}]}
    assert not any("generate" in k or "ilu" in k for k in frozen)


def test_structures_warns_for_unfrozen(tmp_path):
    _write(tmp_path, "A", "c1", "2026-10-18", 4, 40)
    out, warnings = structures(load_runs(tmp_path), {})
    assert len(warnings) == 1 and KEY in warnings[0]
    assert warnings[0] == f"not frozen yet: m {KEY}; run python -m dashboard freeze"
    assert KEY in out["m"]


def test_load_frozen_missing(tmp_path):
    assert load_frozen(tmp_path / "none.json") == {}


def test_cli_freeze_writes_the_file(tmp_path):
    res = tmp_path / "res"
    _write(res, "A", "c1", "2026-10-18", 4, 40)
    out = tmp_path / "f.json"
    assert main(["freeze", "--results", str(res), "--frozen", str(out)]) == 0
    assert KEY in json.loads(out.read_text())["m"]
