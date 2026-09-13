#!/usr/bin/env python
"""Verify a replay run actually followed its fixed decisions.

The experiment worker does not exit non-zero for a request that errored -- it
records ``status="error"`` on that row and carries on -- so a run in which every
request failed still looks like a success to the shell.  This reads each
method's ``requests.jsonl`` back and checks the terminal status distribution
against the decision file it was replaying:

* every one of the ``limit`` arrivals reached a terminal status,
* no request errored,
* the hit/miss split equals the decisions in ``<replay-root>/<method>.jsonl``,
* the dense baseline, which has no decision file, ran every request.

Exits 1 with a per-method report when anything disagrees.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TERMINAL = {"ok", "cache_miss", "error"}
# Methods the replay protocol drives; the rest run every request densely.
REPLAYED_METHODS = {"nirvana_vid", "chorus", "chorus2", "carnator_boundary_b5", "teacache"}


def _statuses(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing results file: {path}")
    return [
        json.loads(line).get("status")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _decisions(path: Path, limit: int) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing replay decision file: {path}")
    decisions = [
        json.loads(line).get("decision")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:limit]
    if len(decisions) < limit:
        raise ValueError(f"{path} has {len(decisions)} rows, fewer than the {limit} requested")
    return decisions


def check_method(output_root: Path, replay_root: Path, method: str, limit: int) -> dict[str, int]:
    statuses = _statuses(output_root / method / "requests.jsonl")
    if len(statuses) != limit:
        raise ValueError(f"{method}: expected {limit} rows, found {len(statuses)}")
    unknown = sorted({status for status in statuses if status not in TERMINAL})
    if unknown:
        raise ValueError(f"{method}: non-terminal statuses {unknown}")
    errors = sum(status == "error" for status in statuses)
    if errors:
        # The first failing row's message is the useful part of the report.
        rows = [
            json.loads(line)
            for line in (output_root / method / "requests.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        first = next(row for row in rows if row.get("status") == "error")
        raise ValueError(f"{method}: {errors} request(s) errored, e.g. {first.get('error')!r}")
    ok = sum(status == "ok" for status in statuses)
    misses = sum(status == "cache_miss" for status in statuses)
    if method in REPLAYED_METHODS:
        decisions = _decisions(replay_root / f"{method}.jsonl", limit)
        unknown_decisions = sorted({decision for decision in decisions if decision not in {"hit", "miss"}})
        if unknown_decisions:
            raise ValueError(f"{method}: unsupported replay decisions {unknown_decisions}")
        expected_hits = sum(decision == "hit" for decision in decisions)
        expected_misses = sum(decision == "miss" for decision in decisions)
        if (ok, misses) != (expected_hits, expected_misses):
            raise ValueError(
                f"{method}: ran {ok} hit(s) and {misses} miss(es), "
                f"but the replay file fixes {expected_hits} hit(s) and {expected_misses} miss(es)"
            )
    elif misses:
        raise ValueError(f"{method}: the dense baseline cannot report a cache miss")
    return {"ok": ok, "cache_miss": misses, "error": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    args = parser.parse_args()

    failures: list[str] = []
    for method in args.methods:
        try:
            counts = check_method(args.output_root, args.replay_root, method, args.limit)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            failures.append(str(exc))
            print(f"FAIL {method}: {exc}", file=sys.stderr)
        else:
            print(
                f"OK   {method}: {counts['ok']} ok, {counts['cache_miss']} cache_miss, {counts['error']} error"
            )
    if failures:
        print(f"status distribution check failed for {len(failures)} method(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
