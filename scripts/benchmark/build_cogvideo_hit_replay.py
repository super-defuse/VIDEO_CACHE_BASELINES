#!/usr/bin/env python3
"""Build immutable CogVideo hit/miss decisions from the old Wan run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHODS = ("nirvana_vid", "teacache", "chorus", "chorus2", "carnator_boundary_b5")
TERMINAL = {"ok", "cache_miss", "error"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSONL: {path}") from exc


def _source_uuid(row: dict[str, Any]) -> str | None:
    match = row.get("cache_match")
    if isinstance(match, dict):
        return match.get("source_uuid")
    return row.get("hit_object")


def build_method_replay(path: Path, *, method: str, limit: int = 300) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    if len(rows) != limit:
        raise ValueError(f"{method} must contain exactly {limit} records, found {len(rows)}")
    replay: list[dict[str, Any]] = []
    seen: set[int] = set()
    for expected, row in enumerate(rows):
        arrival = int(row.get("arrival_index", expected))
        if arrival != expected or arrival in seen:
            raise ValueError(f"{method} arrival_index must be the unique sequence 0..{limit - 1}")
        seen.add(arrival)
        status = str(row.get("status", ""))
        if status not in TERMINAL:
            raise ValueError(f"{method} arrival {arrival} has non-terminal status {status!r}")
        if method == "teacache":
            hit: bool | None = None
            decision = "target_only"
            source_uuid = None
        else:
            hit = bool(row.get("hit", status == "ok" and _source_uuid(row) is not None))
            decision = "hit" if hit else "miss"
            source_uuid = _source_uuid(row) if hit else None
            if hit and not source_uuid:
                raise ValueError(f"{method} arrival {arrival} is a hit without source_uuid")
        replay.append({
            "schema_version": "cogvideox15-hit-replay/v1",
            "method": method,
            "arrival_index": arrival,
            "request_index": int(row["request_index"]),
            "uuid": str(row["uuid"]),
            "prompt": str(row["prompt"]),
            "decision": decision,
            "hit": hit,
            "source_uuid": source_uuid,
            "source_request_index": row.get("cache_match", {}).get("source_request_index") if isinstance(row.get("cache_match"), dict) else None,
            "legacy_status": status,
            "legacy_error": row.get("error"),
        })
    return replay


def build_replay(legacy_root: Path, output: Path, *, limit: int = 300) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for method in METHODS:
        rows = build_method_replay(legacy_root / method / "request_records.jsonl", method=method, limit=limit)
        target = output / f"{method}.jsonl"
        target.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        counts[method] = len(rows)
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": "cogvideox15-hit-replay/v1",
        "legacy_root": str(legacy_root.resolve()),
        "methods": counts,
        "requests": limit,
        "teacache_hit_rate": "N/A",
    }, indent=2) + "\n", encoding="utf-8")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(build_replay(args.legacy_root, args.output, limit=args.limit), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
