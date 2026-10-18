"""Build every catalog matrix into Ginkgo binaries and write the manifest."""
from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from . import fem, readers
from .assemble import assemble, row_length_stats, sellp_padding_ratio
from .binary import write_binary
from .catalog import Catalog, Entry
from .fetch import _sha256_file, download, fetch_webgraph

MANIFEST_VERSION_KEY = "version"
SELLP_LIMIT = 2.0
REPO = Path(__file__).resolve().parent.parent
PACKAGES = ("numpy", "scipy", "scikit-fem")


class BuildError(RuntimeError):
    """One or more entries failed; the others were built and recorded."""


class VersionError(RuntimeError):
    """The catalog or the converted data changed but the version did not."""


def _git(args: list[str]) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                          check=True, timeout=60).stdout


def _repository_version(manifest_path: Path) -> str | None:
    """HEAD of this repository, plus "-dirty" if a tracked file changed.

    Untracked files don't count (as in `git describe --dirty`), and neither does
    the manifest, which the build itself rewrites.
    """
    try:
        sha = _git(["rev-parse", "HEAD"]).strip()
        spec = ["--", "."]
        try:
            spec.append(f":(exclude){Path(manifest_path).resolve().relative_to(REPO.resolve())}")
        except ValueError:
            pass  # the manifest is outside the repository
        dirty = _git(["status", "--porcelain", "--untracked-files=no", *spec]).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return sha + ("-dirty" if dirty else "")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def webgraph_version() -> str | None:
    """`webgraph --version` of the webgraph on PATH, best effort."""
    exe = shutil.which("webgraph")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                             check=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out.splitlines()[0] if out else None


def converter_version(manifest_path: Path) -> dict:
    """What converted the data: this repository and the packages it uses."""
    return {"repository": _repository_version(manifest_path),
            **{name: _package_version(name) for name in PACKAGES}}


def check_nnz(entry: Entry, nnz: int, rows: int) -> None:
    exp = entry.expected_nnz
    kind = entry.nnz_check
    if kind == "none" or exp is None:
        return
    ok = {
        "exact": nnz == exp,
        "plus_diagonal": exp <= nnz <= exp + rows,
        "at_least": nnz >= exp,
    }.get(kind)
    if ok is None:
        raise ValueError(f"{entry.name}: unknown nnz_check {kind!r}")
    if not ok:
        raise ValueError(f"{entry.name}: nnz check '{kind}' failed: got {nnz}, "
                         f"expected {exp} (rows {rows})")


def _mtx_from_zip(zip_path: Path, dest: Path) -> tuple[Path, bool]:
    """Extract the .mtx member next to the zip; return (path, is_symmetric)."""
    if not dest.exists():
        with zipfile.ZipFile(zip_path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".mtx")]
            if len(names) != 1:
                raise ValueError(f"{zip_path}: expected one .mtx member, found {names}")
            tmp = dest.with_name(dest.name + ".tmp")
            with z.open(names[0]) as src, open(tmp, "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            os.replace(tmp, dest)
    with open(dest, "rb") as f:
        banner = f.readline().lower().split()
    return dest, len(banner) > 4 and banner[4] == b"symmetric"


def _pattern_for(entry: Entry, raw: Path, webgraph_cmd: str | None):
    """Download and read a spmv entry; return (Pattern, symmetric, downloads)."""
    fmt = entry.format
    if fmt == "webgraph":
        if not webgraph_cmd:
            raise ValueError(f"{entry.name}: webgraph entry needs --webgraph-cmd "
                             "(a format string with {basename} and {out})")
        base = raw / entry.url.rsplit("/", 1)[-1]
        shas = {f"{base.name}.{ext}": h
                for ext, h in fetch_webgraph(entry.url, raw).items()}
        arcs = raw / f"{entry.name}.arcs"
        if not arcs.exists():
            tmp = arcs.with_name(arcs.name + ".tmp")
            cmd = webgraph_cmd.format(basename=shlex.quote(str(base)),
                                      out=shlex.quote(str(tmp)))
            subprocess.run(shlex.split(cmd), check=True)
            os.replace(tmp, arcs)
        n = readers.webgraph_nodes(base.with_name(base.name + ".properties"))
        return readers.read_webgraph_arcs(arcs, n, min_id=entry.index_base), \
            entry.symmetric, shas
    fname = entry.url.rsplit("/", 1)[-1]
    shas = {fname: download(entry.url, raw / fname)}
    path = raw / fname
    if fmt == "metis":
        return readers.read_metis(path), entry.symmetric, shas
    if fmt == "dimacs9":
        return readers.read_dimacs9(path), entry.symmetric, shas
    if fmt == "edgelist":
        return readers.read_edge_list(path, min_id=entry.index_base,
                                      compact=entry.compact_ids), entry.symmetric, shas
    if fmt == "mtx-zip":
        mtx, is_sym = _mtx_from_zip(path, raw / (entry.name + ".mtx"))
        return readers.read_mtx(mtx), is_sym, shas
    raise ValueError(f"{entry.name}: unknown format {fmt!r}")


def _build_entry(entry: Entry, catalog: Catalog, root: Path,
                 webgraph_cmd: str | None) -> dict:
    out = root / "matrices" / entry.set / f"{entry.name}.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    shas: dict[str, str] = {}
    if entry.set == "fem":
        a = fem.generate(entry.fem["problem"], entry.fem["size"]).tocsr()
        a.sum_duplicates()
        a.sort_indices()
        shape = a.shape
        rows = np.repeat(np.arange(shape[0], dtype=np.int32), np.diff(a.indptr))
        cols = a.indices.astype(np.int32)
        vals = a.data.astype(np.float64)
        del a
    else:
        pattern, symmetric, shas = _pattern_for(entry, root / "raw", webgraph_cmd)
        shape = pattern.shape
        rows, cols, vals = assemble(pattern, symmetric=symmetric,
                                    add_diagonal=entry.add_diagonal, seed=catalog.seed)
        del pattern
    if entry.expected_rows is not None and shape[0] != entry.expected_rows:
        raise ValueError(f"{entry.name}: row check failed: got {shape[0]}, "
                         f"expected {entry.expected_rows}")
    nnz = len(rows)
    check_nnz(entry, nnz, shape[0])
    stats = row_length_stats(shape[0], rows)
    ratio = sellp_padding_ratio(shape[0], rows, 64)
    digest = write_binary(out, shape, rows, cols, vals)
    del rows, cols, vals
    gc.collect()
    return {
        "name": entry.name, "set": entry.set, "roles": list(entry.roles),
        "spd": entry.spd, "source": entry.source, "url": entry.url,
        "download_sha256": shas,
        "path": f"matrices/{entry.set}/{entry.name}.bin", "binary_sha256": digest,
        "rows": int(shape[0]), "cols": int(shape[1]), "nonzeros": int(nnz),
        "row_length": stats, "sellp_padding_ratio": ratio,
        "run_sellp": bool(ratio <= SELLP_LIMIT),
        "nnz_check": entry.nnz_check, "expected_nnz": entry.expected_nnz,
    }


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build(catalog: Catalog, root: Path, manifest_path: Path,
          only: list[str] | None, webgraph_cmd: str | None) -> dict:
    root, manifest_path = Path(root), Path(manifest_path)
    done: dict[str, dict] = {}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("version") == catalog.version:
            if old.get("catalog_sha256") != catalog.sha256:
                raise VersionError(
                    f"catalog changed but version is still {catalog.version!r}; "
                    "bump `version` in catalog.toml")
            done = {m["name"]: m for m in old["matrices"]}
    names = {e.name for e in catalog.entries}
    if only:
        unknown = sorted(set(only) - names)
        if unknown:
            raise ValueError(f"unknown entries: {unknown}")
    manifest = {"version": catalog.version, "catalog_sha256": catalog.sha256,
                "seed": catalog.seed, "matrices": []}

    def snapshot():
        manifest["matrices"] = [done[e.name] for e in catalog.entries if e.name in done]
        _write_manifest(manifest_path, manifest)

    snapshot()
    converter = converter_version(manifest_path)
    webgraph: list[str | None] = []  # looked up once, for the first webgraph entry
    failures = []
    for entry in catalog.entries:
        if only and entry.name not in only:
            continue
        prev = done.get(entry.name)
        binary = root / "matrices" / entry.set / f"{entry.name}.bin"
        if prev and binary.exists() and _sha256_file(binary) == prev["binary_sha256"]:
            continue
        done.pop(entry.name, None)
        try:
            new = _build_entry(entry, catalog, root, webgraph_cmd)
        except Exception as e:  # keep going; report all failures at the end
            failures.append(f"{entry.name}: {e}")
        else:
            if prev and new["binary_sha256"] != prev["binary_sha256"]:
                # The manifest on disk keeps the old entry until the version is bumped.
                raise VersionError(
                    f"{entry.name}: the rebuilt binary differs from the one in the "
                    f"manifest of version {catalog.version!r} (sha256 "
                    f"{new['binary_sha256']} != {prev['binary_sha256']}); changed data "
                    "needs a version bump in catalog.toml")
            new["converter_version"] = dict(converter)
            if entry.format == "webgraph":
                if not webgraph:
                    webgraph.append(webgraph_version())
                new["converter_version"]["webgraph-cli"] = webgraph[0]
            done[entry.name] = new
        gc.collect()
        snapshot()
    if failures:
        raise BuildError("; ".join(failures))
    return manifest
