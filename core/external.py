"""Repository-local access to locked third-party source trees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "external" / "lock.json"


def locked_sources() -> dict[str, dict[str, Any]]:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))["sources"]


def external_path(name: str, *, required: bool = True) -> Path:
    try:
        relative_path = locked_sources()[name]["path"]
    except KeyError as exc:
        raise ValueError(f"unknown external source: {name}") from exc
    path = ROOT / relative_path
    if required and not path.is_dir():
        raise FileNotFoundError(
            f"missing external source {name!r} at {path}; run "
            "git submodule update --init --recursive"
        )
    return path
