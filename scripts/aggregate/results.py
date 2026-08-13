from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate request JSONL without hand-entered values.")
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line]
    ok = [row for row in rows if row["status"] == "ok"]
    hits = [row for row in ok if row["cache_match"]["hit"]]
    summary = {"requests": len(rows), "ok": len(ok), "errors": len(rows) - len(ok),
               "hit_rate": len(hits) / len(ok) if ok else None,
               "latency_s": sum(row["latency_s"] for row in ok) / len(ok) if ok else None,
               "cache_bytes": max((row["cache_bytes"] for row in rows), default=0)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
