#!/usr/bin/env python3
"""Run Carnator's retrieval-only acceptance path.

No Wan model, denoising loop, CTAP, latent tensor, or sparse-attention tensor
is loaded here.  Source records and signatures come from the local fixture
JSON, while cache-ready manifests are treated as decision metadata only.
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

from carnator.retrieval import (  # noqa: E402
    CARNATOR_IMPLEMENTATION_ID,
    CARNATOR_SIGNATURE_PROFILE_ID,
    CarnatorNounAnnotations,
    CarnatorRetriever,
    CarnatorSignatureBuilder,
    CarnatorThresholdConfig,
)
from core.retrieval import LexicalTestEmbedder, VectorRecord  # noqa: E402
from core.types import Request  # noqa: E402


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _records(path: Path | None, embedder: LexicalTestEmbedder, builder: CarnatorSignatureBuilder) -> list[VectorRecord]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_records = payload if isinstance(payload, list) else payload.get("records", [])
    records: list[VectorRecord] = []
    for item in raw_records:
        if not isinstance(item, dict) or not item.get("uuid"):
            raise ValueError("fixture records need a uuid")
        prompt = str(item.get("prompt", item.get("term", "")))
        nouns = item.get("trf_nouns", [])
        metadata = dict(item.get("metadata") or {})
        metadata.setdefault("source_request_index", int(item.get("request_index", 0)))
        metadata.setdefault("trf_nouns", nouns)
        signature = item.get("signature")
        if signature is None and item.get("metrics") is not None:
            signature = {
                "implementation_id": "video-cache-baselines.carnator-two-stage.v1",
                "signature_profile_id": "ets-v1",
                "completed_step": 5,
                "trf_nouns": nouns,
                "metrics": item["metrics"],
            }
        metadata.setdefault("signature", signature or builder.build(prompt, nouns))
        if "cache_ready" in item:
            metadata["cache_ready"] = bool(item["cache_ready"])
        if "manifest" in item:
            metadata["manifest"] = item["manifest"]
        vector = item.get("vector") or embedder.encode([prompt])[0]
        records.append(VectorRecord(str(item["uuid"]), prompt, [float(value) for value in vector], metadata,
                                    int(item.get("request_index", 0))))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=ROOT / "datasets" / "vbench_cross_request_prompts.csv")
    parser.add_argument("--noun-annotations", type=Path,
                        default=ROOT / "datasets" / "carnator_vbench_trf_nouns.csv")
    parser.add_argument("--threshold-config", type=Path,
                        default=ROOT / "configs" / "retrieval" / "carnator_ets_thresholds.json")
    parser.add_argument("--fixture", type=Path,
                        help="JSON source-vector/signature/cache-ready fixture")
    parser.add_argument("--output", type=Path, default=Path("retrieval_results.jsonl"))
    parser.add_argument("--candidate-top-k", type=int, default=20)
    args = parser.parse_args()

    config = CarnatorThresholdConfig.load(args.threshold_config)
    embedder = LexicalTestEmbedder()
    builder = CarnatorSignatureBuilder()
    retriever = CarnatorRetriever(
        embedder,
        CarnatorNounAnnotations(args.noun_annotations),
        config,
        candidate_top_k=args.candidate_top_k,
        retrieval_only=True,
        signature_builder=builder,
    )
    records = _records(args.fixture, embedder, builder)
    workload = _rows(args.workload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(workload):
            request = Request(int(row["request_index"]), row["uuid"], row["prompt"], index)
            match = retriever.search(request, records)
            metadata = match.metadata
            result = {
                "request_index": request.request_index,
                "uuid": request.uuid,
                "top_k": metadata.get("top_k", []),
                "noun_prefilter": metadata.get("noun_prefilter", []),
                "hit": bool(match.hit),
                "source_uuid": match.source_uuid,
                "cache_ready": bool(metadata.get("cache_ready", False)),
                "reason": metadata.get("reason"),
                "retriever": metadata.get("retriever"),
                "threshold_config_id": metadata.get("threshold_config_id"),
                "implementation_id": CARNATOR_IMPLEMENTATION_ID,
                "signature_profile_id": CARNATOR_SIGNATURE_PROFILE_ID,
                "stage_2": metadata.get("stage_2", []),
            }
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"requests": len(workload), "output": str(args.output)}))


if __name__ == "__main__":
    main()
