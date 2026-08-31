#!/usr/bin/env python3
"""Prune a shared cache to one source CSV before a resumable refresh."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.prepare_vidprom_cluster_cross_request_cache import (
    DEFAULT_OUTPUT,
    PromptRow,
    read_prompt_rows,
    sha256_file,
    strip_timing_from_entry,
    without_timing_metadata,
    write_json,
)

DEFAULT_SELECTION = ROOT.parent.parent / "datasets" / "1000cache_300input_noun_match" / "cluster_02_2170_first1000_cached.csv"


def _refresh_entry_metadata(entry: Path, row: PromptRow) -> None:
    strip_timing_from_entry(entry)
    manifest_path = entry / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["request_index"] = row.request_index
    manifest["prompt"] = row.prompt
    manifest.setdefault("source", {})["prompt"] = row.prompt
    carnator = manifest["artifacts"]["carnator"]
    for record in (carnator["source_manifest"], carnator["raw_block_mask_metadata"]):
        path = entry / str(record["path"])
        record["sha256"] = sha256_file(path)
        record["bytes"] = path.stat().st_size
    write_json(manifest_path, without_timing_metadata(manifest))


def _write_partial_index(root: Path, rows: list[PromptRow]) -> int:
    index_path = root / "index.jsonl"
    temporary = index_path.with_suffix(".refresh.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            entry = root / "entries" / row.uuid
            if not entry.is_dir():
                continue
            manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
            stream.write(json.dumps({
                "request_index": row.request_index,
                "uuid": row.uuid,
                "prompt": row.prompt,
                "entry": str(entry.relative_to(root)),
                "config_fingerprint": manifest["config_fingerprint"],
                "consumers": manifest["consumers"],
            }, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(index_path)
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--confirm", action="store_true", help="Required because non-selected cache entries are removed.")
    args = parser.parse_args(argv)
    if not args.confirm:
        raise RuntimeError("refusing to remove cache entries without --confirm")

    root = args.cache_root.expanduser().resolve()
    entries = root / "entries"
    if not entries.is_dir() or not (root / "run_manifest.json").is_file():
        raise RuntimeError(f"shared cache root is incomplete: {root}")
    rows = read_prompt_rows(args.selection.expanduser().resolve(), 1000)
    expected = {row.uuid for row in rows}
    if len(rows) != 1000 or len(expected) != len(rows):
        raise RuntimeError("selection must contain exactly 1000 unique cache UUIDs")

    removed = 0
    for entry in entries.iterdir():
        if entry.name == ".staging":
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            continue
        if not entry.is_dir():
            continue
        if entry.is_symlink():
            raise RuntimeError(f"refusing to prune symlinked cache entry: {entry}")
        if entry.name not in expected:
            shutil.rmtree(entry)
            removed += 1

    retained = 0
    for row in rows:
        entry = entries / row.uuid
        if entry.is_dir():
            _refresh_entry_metadata(entry, row)
            retained += 1

    for path in (root / "workers", root / "cached_requests_923"):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
    for path in (root / "full_run.log", root / "space_estimate.json", root / "pilot.json"):
        if path.is_file():
            path.unlink()

    run_manifest_path = root / "run_manifest.json"
    run_manifest = without_timing_metadata(json.loads(run_manifest_path.read_text(encoding="utf-8")))
    run_manifest.update({
        "input": str(args.selection.expanduser().resolve()),
        "input_sha256": sha256_file(args.selection.expanduser().resolve()),
        "requested_limit": len(rows),
        "selected_requests": len(rows),
        "status": "refresh_pending",
        "indexed_entries": retained,
    })
    run_manifest.pop("results", None)
    write_json(run_manifest_path, run_manifest)
    indexed = _write_partial_index(root, rows)
    print(json.dumps({
        "cache_root": str(root),
        "removed_entries": removed,
        "retained_entries": retained,
        "indexed_entries": indexed,
        "missing_entries": len(rows) - retained,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
