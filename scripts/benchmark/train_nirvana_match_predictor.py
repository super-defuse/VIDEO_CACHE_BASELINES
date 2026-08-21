#!/usr/bin/env python3
"""Train NIRVANA's optional SGDOneClassSVM Match Predictor offline.

The workload prompt embeddings are one-class inliers.  This only creates the
paper-aligned predictor interface; use a separately curated production prompt
population when reproducing reported predictor precision/recall.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.retrieval import SgdOneClassMatchPredictor, _make_embedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("clip", "test_lexical"), default="clip")
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nu", type=float, default=0.05)
    args = parser.parse_args()
    if not 0.0 < args.nu <= 1.0:
        raise SystemExit("--nu must be in (0, 1]")

    config = {"backend": args.backend, "model_id": args.model_id, "device": args.device}
    embedder = _make_embedder(config, dry_run=args.backend == "test_lexical")
    with args.workload.open(newline="", encoding="utf-8") as handle:
        prompts = [row["prompt"] for row in csv.DictReader(handle) if row.get("prompt")]
    predictor = SgdOneClassMatchPredictor.fit(
        embedder.encode(prompts), embedder.signature, nu=args.nu
    )
    predictor.save(args.output)
    print({"output": str(args.output), "inliers": len(prompts), "embedding": embedder.signature})


if __name__ == "__main__":
    main()
