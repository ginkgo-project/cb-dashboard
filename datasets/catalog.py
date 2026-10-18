"""Matrix catalog: every benchmark matrix and where it comes from."""
from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT = Path(__file__).with_name("catalog.toml")


@dataclass
class Entry:
    name: str
    set: str  # "spmv" or "fem"
    source: str
    url: str | None
    format: str  # metis | dimacs9 | edgelist | mtx-zip | webgraph | fem
    symmetric: bool
    add_diagonal: bool
    expected_rows: int | None
    expected_nnz: int | None
    nnz_check: str  # exact | plus_diagonal | at_least | none
    roles: list[str]
    spd: bool | None
    fem: dict | None  # {"problem": str, "size": int}
    index_base: int = 0  # smallest vertex id in an edge-list file
    compact_ids: bool = False  # renumber an edge list's ids, dropping unused ones


@dataclass
class Catalog:
    version: str
    seed: int
    entries: list[Entry]
    sha256: str


def load_catalog(path: Path = DEFAULT) -> Catalog:
    raw = Path(path).read_bytes()
    doc = tomllib.loads(raw.decode("utf-8"))
    entries = [
        Entry(
            name=e["name"],
            set=e["set"],
            source=e["source"],
            url=e.get("url"),
            format=e["format"],
            symmetric=e["symmetric"],
            add_diagonal=e["add_diagonal"],
            expected_rows=e.get("expected_rows"),
            expected_nnz=e.get("expected_nnz"),
            nnz_check=e["nnz_check"],
            roles=list(e.get("roles", [])),
            spd=e.get("spd"),
            fem=e.get("fem"),
            index_base=e.get("index_base", 0),
            compact_ids=e.get("compact_ids", False),
        )
        for e in doc["entry"]
    ]
    return Catalog(
        version=doc["version"],
        seed=doc["seed"],
        entries=entries,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
