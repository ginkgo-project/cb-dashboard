from bench import energy as E


def counters(tmp_path, **files):
    for name, text in files.items():
        (tmp_path / name).write_text(text + "\n")
    return tmp_path


def test_read_takes_the_energy_counters_in_joules(tmp_path):
    base = counters(tmp_path, energy="1000 J 1697041234567890 us",
                    accel0_energy="300 J 1697041234567890 us", cpu_energy="200 J",
                    power="350 W 1697041234567890 us", freshness="5")
    assert E.read(base) == {"accel0_energy": 300, "cpu_energy": 200, "energy": 1000}


def test_read_without_counters_is_empty(tmp_path):
    assert E.read(tmp_path / "missing") == {}
    assert E.read(counters(tmp_path, energy="garbage")) == {}


def test_read_again_when_freshness_changed(tmp_path, monkeypatch):
    base = counters(tmp_path, energy="1000 J", freshness="5")
    seen = []

    def text(path):
        if path.name == "freshness":
            seen.append(path)
            return str(len(seen) // 2 if len(seen) < 3 else 9)  # 0, 1 (changed), 9, 9
        return path.read_text()
    monkeypatch.setattr(E, "_text", text)
    assert E.read(base) == {"energy": 1000}
    assert len(seen) == 4


def test_used():
    assert E.used({"energy": 10, "accel0_energy": 4}, {"energy": 25, "cpu_energy": 1}) == {
        "energy": 15}


def test_snapshot(tmp_path):
    base = counters(tmp_path, energy="1000 J", version="3")
    assert E.snapshot(base) == {"energy": "1000 J", "version": "3"}
    assert E.snapshot(tmp_path / "missing")["error"].startswith("FileNotFoundError")
