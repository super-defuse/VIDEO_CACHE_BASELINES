#!/usr/bin/env python3
"""Build a NIRVANA CacheSelector profile from offline quality observations.

Input CSV columns: ``similarity,reuse_step,quality``.  Observations should be
collected with the intended diffusion backbone and quality metric before this
script is run.  The script derives a quality-safe threshold for each measured
restart step; it does not search for a workload-optimal configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.retrieval import CacheSelectorObservation, CacheSelectorProfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-threshold", type=float, default=0.90)
    parser.add_argument("--quality-metric", default="CLIPScore")
    args = parser.parse_args()

    with args.observations.open(newline="", encoding="utf-8") as handle:
        observations = [
            CacheSelectorObservation(
                similarity=float(row["similarity"]),
                reuse_step=int(row["reuse_step"]),
                quality=float(row["quality"]),
            )
            for row in csv.DictReader(handle)
        ]
    profile = CacheSelectorProfile.from_observations(
        observations,
        quality_threshold=args.quality_threshold,
        quality_metric=args.quality_metric,
    )
    payload = profile.to_payload()
    payload["source"] = f"offline_observations:{args.observations.name}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "steps": len(profile.thresholds)}, indent=2))


if __name__ == "__main__":
    main()
