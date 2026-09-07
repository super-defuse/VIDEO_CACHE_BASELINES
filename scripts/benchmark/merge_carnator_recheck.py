#!/usr/bin/env python3
"""Validate and merge a focused Carnator recheck into a canonical 300-row run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

TASK_INFO = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
)
DISABLED = {
    "max_context_color_distance": 1_000_000.0,
    "max_log_area_ratio": 1_000_000.0,
    "max_shape_distance": 1_000_000.0,
    "max_changed_color_distance": 1_000_000.0,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--recheck-root", type=Path, required=True)
    parser.add_argument("--scored-results", type=Path, required=True)
    parser.add_argument("--backup-name", default="pre_recheck_20260906")
    args = parser.parse_args()
    canonical = args.canonical.expanduser().resolve()
    recheck = args.recheck_root.expanduser().resolve()
    scored_path = args.scored_results.expanduser().resolve()
    if not canonical.is_dir() or not recheck.is_dir():
        raise FileNotFoundError("canonical or recheck directory is missing")

    base_requests = read_jsonl(canonical / "requests.jsonl")
    if len(base_requests) != 300 or len({int(row["request_index"]) for row in base_requests}) != 300:
        raise ValueError("canonical requests.jsonl must contain 300 unique requests")
    base_by_index = {int(row["request_index"]): row for row in base_requests}
    target = {idx for idx, row in base_by_index.items() if not bool(row.get("hit", row.get("cache_match", {}).get("hit")))}
    if len(target) != 65:
        raise ValueError(f"expected 65 canonical misses before merge, found {len(target)}")

    shards = sorted(recheck.glob("gpu[3-7]"))
    if len(shards) != 5:
        raise ValueError(f"expected five gpu3-gpu7 shards, found {len(shards)}")
    shard_rows: dict[str, list[dict[str, Any]]] = {}
    merged: dict[str, dict[int, dict[str, Any]]] = {"requests": {}, "request_records": {}}
    for shard in shards:
        cfg = json.loads((shard / "run_config.json").read_text(encoding="utf-8"))
        if cfg.get("early_signature_disabled") is not True or cfg.get("early_signature_thresholds") != DISABLED:
            raise ValueError(f"{shard}: not an early-signature-disabled run")
        rows = read_jsonl(shard / "requests.jsonl")
        records = read_jsonl(shard / "request_records.jsonl")
        if len(rows) != 13 or len(records) != 13:
            raise ValueError(f"{shard}: expected 13 requests and records")
        shard_rows[shard.name] = rows
        for kind, values in (("requests", rows), ("request_records", records)):
            for row in values:
                idx = int(row["request_index"])
                if idx in merged[kind]:
                    raise ValueError(f"{kind}: duplicate request_index {idx}")
                merged[kind][idx] = row
    if set(merged["requests"]) != target or set(merged["request_records"]) != target:
        raise ValueError("focused shards do not exactly cover the canonical 65 misses")

    scored = read_jsonl(scored_path)
    if len(scored) != 65 or {int(row["request_index"]) for row in scored} != target:
        raise ValueError("scored results do not exactly cover the canonical 65 misses")
    scored_by_index: dict[int, dict[str, Any]] = {}
    for row in scored:
        idx = int(row["request_index"])
        dimensions = row.get("vbench_dimensions")
        if row.get("status") != "ok" or not row.get("video_path") or not isinstance(dimensions, dict):
            raise ValueError(f"scored result {idx} is not a successful generated result")
        if not isinstance(row.get("clip_t"), (int, float)) or not isinstance(row.get("vbench_q"), (int, float)):
            raise ValueError(f"scored result {idx} is missing CLIP-T/VBench-Q")
        if any(not isinstance(dimensions.get(name), (int, float)) for name in TASK_INFO):
            raise ValueError(f"scored result {idx} is missing a VBench dimension")
        if idx in scored_by_index:
            raise ValueError(f"scored results: duplicate request_index {idx}")
        scored_by_index[idx] = row

    backup = canonical / args.backup_name
    backup.mkdir(parents=False, exist_ok=False)
    for name in ("requests.jsonl", "request_records.jsonl", "results.jsonl", "quality_input.jsonl", "run_config.json"):
        source = canonical / name
        if source.is_file():
            shutil.copy2(source, backup / name)

    def replace_rows(name: str, replacements: dict[int, dict[str, Any]]) -> None:
        original = read_jsonl(canonical / name)
        output = []
        for row in original:
            idx = int(row["request_index"])
            if idx in replacements:
                replacement = dict(replacements[idx])
                replacement["arrival_index"] = row["arrival_index"]
                replacement["request_index"] = row["request_index"]
                replacement["uuid"] = row["uuid"]
                replacement["prompt"] = row["prompt"]
                output.append(replacement)
            else:
                output.append(row)
        if len(output) != 300 or {int(row["request_index"]) for row in output} != set(base_by_index):
            raise ValueError(f"merged {name} failed 300-row invariant")
        write_atomic(canonical / name, output)

    replace_rows("requests.jsonl", merged["requests"])
    replace_rows("request_records.jsonl", merged["request_records"])
    replace_rows("results.jsonl", scored_by_index)
    replace_rows("quality_input.jsonl", scored_by_index)
    manifest = {
        "status": "merged",
        "target_requests": sorted(target),
        "shards": [str(path) for path in shards],
        "scored_results": str(scored_path),
        "backup": str(backup),
        "early_signature_disabled": True,
        "early_signature_thresholds": DISABLED,
        "requests": 300,
        "hits_after_merge": 300,
    }
    (canonical / "recheck_merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
