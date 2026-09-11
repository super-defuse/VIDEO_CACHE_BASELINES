#!/usr/bin/env python3
"""Compare a Wan2.1 rerun against the archived 10-request experiment.

The source-stage optimization (bit-packed block masks) is only acceptable if it
is bit-neutral for the target artefacts.  This script joins the archived
``results_wan21_test`` rows with a rerun produced under identical settings and
reports, per method:

* ``output_hash`` equality for every request that ran in both;
* sha256 equality of the emitted ``videos/*.mp4``;
* the ``source_prepare_s`` distribution before and after.

Usage:
    python scripts/benchmark/compare_wan21_rerun.py \
        --baseline results_wan21_test --rerun results_wan21_rerun
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "artifacts" / "vidprom_1000cache_300input_noun_match"
METHODS = ("chorus2", "carnator_boundary_b5")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scored_rows(method_root: Path) -> list[dict]:
    """Prefer the quality-annotated results.jsonl, then the canonical logs.

    The canonical ``requests.jsonl`` / ``results.jsonl`` are only written once
    ``wan21_run`` returns, so a rerun that was stopped early leaves just the
    runner's streaming ``requests.raw.jsonl``.  Reading that lets a partial
    rerun still be compared.
    """
    for name in ("results.jsonl", "requests.jsonl", "requests.raw.jsonl"):
        rows = read_jsonl(method_root / name)
        if rows:
            return rows
    return []


def source_prepare_times(method_root: Path) -> dict[int, float]:
    return {
        int(row["request_index"]): float(row["source_prepare_s"])
        for row in read_jsonl(method_root / "source_prepare" / "events.jsonl")
    }


def describe(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (f"n={len(values)} mean={statistics.fmean(values):.1f}s "
            f"min={min(values):.1f}s max={max(values):.1f}s")


def compare_method(baseline_root: Path, rerun_root: Path, method: str) -> bool:
    print(f"\n=== {method} ===")
    baseline_rows = {int(r.get("request_index", -1)): r for r in scored_rows(baseline_root / method)}
    rerun_rows = {int(r.get("request_index", -1)): r for r in scored_rows(rerun_root / method)}
    if not baseline_rows:
        print("  baseline rows missing; nothing to compare", file=sys.stderr)
        return False
    if not rerun_rows:
        print("  rerun rows missing; nothing to compare", file=sys.stderr)
        return False

    shared = sorted(set(baseline_rows) & set(rerun_rows))
    only_baseline = sorted(set(baseline_rows) - set(rerun_rows))
    if only_baseline:
        print(f"  rerun is partial: {len(only_baseline)} archive requests absent "
              f"(indices {only_baseline})")

    ok = True
    hash_matches = hash_mismatches = skipped = 0
    video_matches = video_mismatches = video_missing = 0
    for index in shared:
        old, new = baseline_rows[index], rerun_rows[index]
        old_status, new_status = old.get("status"), new.get("status")
        if old_status != new_status:
            print(f"  [{index}] STATUS CHANGED: {old_status} -> {new_status}")
            ok = False
            continue
        if new_status != "ok":
            skipped += 1
            continue
        if old.get("output_hash") == new.get("output_hash"):
            hash_matches += 1
        else:
            hash_mismatches += 1
            ok = False
            print(f"  [{index}] OUTPUT HASH MISMATCH")
            print(f"        archive {old.get('output_hash')}")
            print(f"        rerun   {new.get('output_hash')}")

        # The canonical logs carry video_path; the runner's streaming raw rows
        # only carry output_path.
        old_video = Path(str(old.get("video_path") or old.get("output_path") or ""))
        new_video = Path(str(new.get("video_path") or new.get("output_path") or ""))
        if not old_video.is_file() or not new_video.is_file():
            # Fall back to the videos/ directory.  The adapter's own filename
            # wins over the runner's "{index:04d}_{uuid}.mp4" default, so match
            # on the UUID suffix rather than assuming a prefix.
            uuid = str(new.get("uuid"))
            old_match = sorted((baseline_root / method / "videos").glob(f"*{uuid}.mp4"))
            new_match = sorted((rerun_root / method / "videos").glob(f"*{uuid}.mp4"))
            old_video = old_match[0] if old_match else old_video
            new_video = new_match[0] if new_match else new_video
        if not old_video.is_file() or not new_video.is_file():
            video_missing += 1
            continue
        if sha256_file(old_video) == sha256_file(new_video):
            video_matches += 1
        else:
            video_mismatches += 1
            ok = False
            print(f"  [{index}] VIDEO SHA256 MISMATCH: {new_video.name}")

    print(f"  output_hash : {hash_matches} identical, {hash_mismatches} different "
          f"({skipped} non-ok requests)")
    print(f"  video sha256: {video_matches} identical, {video_mismatches} different, "
          f"{video_missing} not found on disk")

    old_times = source_prepare_times(baseline_root / method)
    new_times = source_prepare_times(rerun_root / method)
    shared_times = {k: (old_times[k], new_times[k]) for k in sorted(set(old_times) & set(new_times))}
    print(f"  source_prepare_s archive: {describe([v[0] for v in shared_times.values()])}")
    print(f"  source_prepare_s rerun  : {describe([v[1] for v in shared_times.values()])}")
    if shared_times:
        deltas = [b - a for a, b in shared_times.values()]
        print(f"  delta (rerun - archive) : mean={statistics.fmean(deltas):+.1f}s "
              f"(negative means the rerun is faster)")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_ROOT / "results_wan21_test")
    parser.add_argument("--rerun", type=Path, default=DEFAULT_ROOT / "results_wan21_rerun")
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()

    methods = [value for value in args.methods.split(",") if value.strip()]
    all_ok = True
    for method in methods:
        all_ok &= compare_method(args.baseline, args.rerun, method)

    print("\n=== verdict ===")
    print("bit-identical" if all_ok else "DIFFERENCES FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
