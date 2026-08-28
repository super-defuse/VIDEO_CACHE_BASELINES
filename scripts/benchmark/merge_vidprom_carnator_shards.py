#!/usr/bin/env python3
"""Merge disjoint Carnator VidProM worker outputs into the canonical run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def merge(kind: str, shard_dirs: list[Path], expected_requests: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        rows.extend(read_jsonl(shard_dir / f"{kind}.jsonl"))
    by_arrival: dict[int, dict[str, Any]] = {}
    for row in rows:
        arrival = row.get("arrival_index")
        if not isinstance(arrival, int) or arrival in by_arrival:
            raise ValueError(f"{kind}: duplicate or invalid arrival index {arrival!r}")
        by_arrival[arrival] = row
    expected = set(range(expected_requests))
    if set(by_arrival) != expected:
        missing = sorted(expected - set(by_arrival))[:10]
        unexpected = sorted(set(by_arrival) - expected)[:10]
        raise ValueError(f"{kind}: arrivals do not cover 0..{expected_requests - 1}; missing={missing}, unexpected={unexpected}")
    return [by_arrival[index] for index in range(expected_requests)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-requests", type=int, default=300)
    args = parser.parse_args()
    if args.expected_requests < 1:
        raise ValueError("--expected-requests must be positive")
    output = args.output.expanduser().resolve()
    shard_dirs = [path.expanduser().resolve() for path in args.shard_dir]
    if len(set(shard_dirs)) != len(shard_dirs):
        raise ValueError("--shard-dir values must be unique")
    for name in ("requests.jsonl", "request_records.jsonl", "results.jsonl", "run_config.json"):
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite merged Carnator output: {output / name}")

    requests = merge("requests", shard_dirs, args.expected_requests)
    records = merge("request_records", shard_dirs, args.expected_requests)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "requests.jsonl", requests)
    write_jsonl(output / "request_records.jsonl", records)
    write_jsonl(output / "results.jsonl", requests)
    (output / "run_config.json").write_text(
        json.dumps(
            {
                "method": "carnator_boundary_b5",
                "requests": args.expected_requests,
                "shards": [str(path) for path in shard_dirs],
                "merged": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "merged", "output": str(output), "requests": len(requests)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
