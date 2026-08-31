"""Deterministic I/O helpers confined to the supplementary RQ1 workspace."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
FIGURE_DIR = WORKSPACE / "figures"
FIGURE_SOURCE_DIR = WORKSPACE / "figure_sources"
WORKING_DIR = WORKSPACE / "working"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalise_json(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | Path, value: Any) -> None:
    payload = json.dumps(
        _normalise_json(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)


def write_text(path: str | Path, value: str) -> None:
    atomic_write_bytes(path, (value.rstrip() + "\n").encode("utf-8"))


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        date_format="%Y-%m-%dT%H:%M:%S",
    ).encode("utf-8")
    atomic_write_bytes(path, payload)


def table_receipt(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    result: dict[str, Any] = {
        "path": str(target.relative_to(WORKSPACE)),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }
    lower_name = target.name.lower()
    if lower_name.endswith(".csv"):
        with target.open("r", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n\r").split(",")
            rows = sum(1 for _ in handle)
        result.update({"rows": rows, "columns": header})
    return result


def ensure_workspace_dirs() -> None:
    for directory in (FIGURE_DIR, FIGURE_SOURCE_DIR, WORKING_DIR):
        directory.mkdir(parents=True, exist_ok=True)
