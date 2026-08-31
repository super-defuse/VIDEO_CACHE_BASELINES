#!/usr/bin/env python3
"""Materialize a read-only view of the selected shared-cache entries.

The original shared cache is intentionally left untouched.  The view contains
relative symlinks to selected entry directories plus filtered metadata, so it
does not duplicate the large source videos or tensor artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = ROOT / "artifacts" / "shared_cross_request_cache" / "vidprom_cluster_02_2170_first1000"
DEFAULT_SELECTION = Path("/home/tangxuebin/datasets/1000cache_300input_noun_match/cluster_02_2170_first1000_cached.csv")
DEFAULT_OUTPUT = ROOT / "artifacts" / "vidprom_1000cache_300input_noun_match" / "source_view"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_selection(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"request_index", "uuid", "prompt"}
    if required - set(rows[0] if rows else {}):
        raise ValueError(f"selection CSV must contain {sorted(required)}: {path}")
    seen: set[str] = set()
    for row in rows:
        uuid = str(row["uuid"]).strip()
        if not uuid or Path(uuid).name != uuid or uuid in seen:
            raise ValueError(f"invalid or duplicate UUID in selection: {uuid!r}")
        seen.add(uuid)
    if not rows:
        raise ValueError(f"selection CSV is empty: {path}")
    return rows


def load_index(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        payload = json.loads(line)
        uuid = str(payload.get("uuid", ""))
        if not uuid or uuid in entries:
            raise ValueError(f"invalid or duplicate UUID in index: {uuid!r}")
        entries[uuid] = payload
    return entries


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cache_root = args.cache_root.expanduser().resolve()
    selection_path = args.selection.expanduser().resolve()
    output = args.output.expanduser().resolve()
    entries_root = cache_root / "entries"
    index_path = cache_root / "index.jsonl"
    if not entries_root.is_dir() or not index_path.is_file():
        raise SystemExit(f"shared cache is incomplete: {cache_root}")
    rows = load_selection(selection_path)
    index = load_index(index_path)
    missing = [str(row["uuid"]) for row in rows if str(row["uuid"]) not in index]
    if missing:
        raise SystemExit(f"selection UUIDs missing from shared index ({len(missing)}): {missing[:5]}")

    selected_ids = {str(row["uuid"]) for row in rows}
    selected = [index[str(row["uuid"])] for row in rows]
    excluded = [payload for payload in index.values() if str(payload["uuid"]) not in selected_ids]
    excluded.sort(key=lambda payload: int(payload.get("request_index", 0)))
    for payload in selected:
        uuid = str(payload["uuid"])
        entry = entries_root / uuid
        if not entry.is_dir():
            raise SystemExit(f"selected cache entry is missing: {entry}")
        for required in ("manifest.json", "status.json", "source.mp4"):
            if not (entry / required).is_file():
                raise SystemExit(f"selected entry lacks {required}: {entry}")

    output.mkdir(parents=True, exist_ok=True)
    view_entries = output / "entries"
    view_entries.mkdir(exist_ok=True)
    for payload in selected:
        uuid = str(payload["uuid"])
        link = view_entries / uuid
        target = os.path.relpath(entries_root / uuid, view_entries)
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.readlink() != Path(target):
                raise SystemExit(f"refusing to replace existing view entry: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)

    # Preserve the input bytes (including CRLF vs LF) for exact provenance.
    (output / "source_requests.csv").write_bytes(selection_path.read_bytes())
    (output / "index.jsonl").write_text(
        "".join(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n" for payload in selected),
        encoding="utf-8",
    )
    with (output / "excluded_requests.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("request_index", "uuid", "prompt"))
        writer.writeheader()
        writer.writerows({key: payload.get(key, "") for key in writer.fieldnames} for payload in excluded)
    write_json(output / "view_manifest.json", {
        "format": "shared-cross-request-cache-view/v1",
        "source_cache_root": str(cache_root),
        "source_index": str(index_path),
        "selection_csv": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "selected_requests": len(selected),
        "source_index_entries": len(index),
        "excluded_entries": len(index) - len(selected),
        "excluded_requests_csv": str(output / "excluded_requests.csv"),
        "entry_storage": "relative symlinks; source cache files are not copied",
        "uuid_order": "selection CSV order",
    })
    print(json.dumps({
        "view": str(output),
        "selected_requests": len(selected),
        "excluded_entries": len(index) - len(selected),
        "symlink_entries": sum((view_entries / str(item["uuid"])).is_symlink() for item in selected),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
