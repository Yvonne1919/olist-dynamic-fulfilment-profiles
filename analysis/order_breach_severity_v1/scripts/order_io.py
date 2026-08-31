from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/order_breach_severity_v1"
WORK = OUT / "working"
FIGURES = OUT / "figures"
FIGURE_SOURCES = OUT / "figure_sources"
CONFIG_PATH = OUT / "ORDER_FROZEN_CONFIG.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recursive_hashes(path: str | Path) -> dict[str, str]:
    root = Path(path)
    if not root.exists():
        return {}
    if root.is_file():
        return {root.name: sha256_file(root)}
    return {
        item.relative_to(root).as_posix(): sha256_file(item)
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }


def protected_roots() -> dict[str, Path]:
    result: dict[str, Path] = {}
    analysis = ROOT / "analysis"
    for item in sorted(analysis.iterdir()):
        if item.name == OUT.name:
            continue
        result[f"analysis/{item.name}"] = item
    for relative in ("docs", "report", "results", "src"):
        path = ROOT / relative
        if path.exists():
            result[relative] = path
    for relative in ("AGENTS.md", "PROJECT_CONTEXT.md", "RESULTS_REGISTRY.md", "DECISION_LOG.md"):
        path = ROOT / relative
        if path.exists():
            result[relative] = path
    return result


def hash_protected_roots() -> dict[str, dict[str, str]]:
    return {name: recursive_hashes(path) for name, path in protected_roots().items()}


def compare_hash_maps(before: Mapping[str, Mapping[str, str]], after: Mapping[str, Mapping[str, str]]) -> tuple[bool, dict[str, object]]:
    detail: dict[str, object] = {}
    ok = True
    for root in sorted(set(before) | set(after)):
        old = dict(before.get(root, {}))
        new = dict(after.get(root, {}))
        added = sorted(set(new) - set(old))
        removed = sorted(set(old) - set(new))
        changed = sorted(key for key in set(old) & set(new) if old[key] != new[key])
        root_ok = not (added or removed or changed)
        detail[root] = {"added": added, "removed": removed, "changed": changed, "unchanged": root_ok}
        ok &= root_ok
    return ok, detail


def repository_state() -> dict[str, object]:
    status = subprocess.check_output(["git", "status", "--porcelain=v1", "-uall"], cwd=ROOT, text=True)
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "dirty": bool(status.strip()),
        "status_porcelain": status.splitlines(),
    }


def exact_command() -> str:
    argv = list(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])
    return shlex.join(argv)


def package_versions(names: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    result["python"] = sys.version.replace("\n", " ")
    return result


def ensure_directories() -> None:
    for path in (OUT, WORK, FIGURES, FIGURE_SOURCES):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
        handle.write(payload)
        temp_name = handle.name
    os.replace(temp_name, destination)


def write_csv(path: str | Path, frame: pd.DataFrame, *, columns: Iterable[str] | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = frame.loc[:, list(columns)].copy() if columns is not None else frame.copy()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=destination.parent, delete=False) as handle:
        output.to_csv(handle, index=False, float_format="%.12g", date_format="%Y-%m-%d", na_rep="")
        temp_name = handle.name
    os.replace(temp_name, destination)


def write_parquet(
    path: str | Path,
    frame: pd.DataFrame,
    *,
    sort_by: Iterable[str],
    engine: str,
    compression: str,
    index: bool = False,
    sort_kind: str = "mergesort",
) -> None:
    """Atomically persist a deterministically sorted Parquet artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = list(sort_by)
    output = frame.sort_values(keys, kind=sort_kind).reset_index(drop=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=destination.parent, suffix=".parquet", delete=False
    ) as handle:
        temp_name = handle.name
    try:
        output.to_parquet(
            temp_name,
            engine=engine,
            index=index,
            compression=compression,
        )
        os.replace(temp_name, destination)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def write_gzip_csv(path: str | Path, frame: pd.DataFrame, *, columns: Iterable[str] | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = frame.loc[:, list(columns)].copy() if columns is not None else frame.copy()
    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as raw:
        temp_name = raw.name
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="", write_through=True) as text:
                output.to_csv(text, index=False, float_format="%.12g", date_format="%Y-%m-%d", na_rep="")
    os.replace(temp_name, destination)


def append_run_event(event: str, **payload: object) -> None:
    ensure_directories()
    path = WORK / "RUN_STATE.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"events": [], "commands": []}
    state.setdefault("commands", []).append(exact_command())
    state.setdefault("events", []).append({"sequence": len(state.get("events", [])) + 1, "event": event, **payload})
    write_json(path, state)


def output_inventory(*, exclude: Iterable[str] = ()) -> dict[str, dict[str, object]]:
    excluded = set(exclude)
    result: dict[str, dict[str, object]] = {}
    for path in sorted(OUT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(OUT).as_posix()
        if relative in excluded or relative.startswith("working/"):
            continue
        result[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


__all__ = [
    "CONFIG_PATH", "FIGURES", "FIGURE_SOURCES", "OUT", "ROOT", "WORK",
    "append_run_event", "compare_hash_maps", "ensure_directories", "exact_command",
    "hash_protected_roots", "load_config", "output_inventory", "package_versions",
    "protected_roots", "recursive_hashes", "repository_state", "sha256_file",
    "write_csv", "write_gzip_csv", "write_json", "write_parquet",
]
