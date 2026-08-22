#!/usr/bin/env python3
"""Replay cross-request retrieval with progressive Carnator source collection.

The benchmark never preloads a corpus of Carnator signatures. Every source
sidecar is created only after that request misses in its own arrival-order
cache. Native Carnator work retains the 50-step scheduler but stops at step
10 without decoding a video.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carnator.online_collector import NativeOnlineCollector, SyntheticOnlineCollector  # noqa: E402
from carnator.retrieval import CarnatorNounExtractor, CarnatorRetriever, make_carnator_retriever  # noqa: E402
from core.retrieval import (  # noqa: E402
    RetrievalStrategy,
    VectorRecord,
    make_chai_retriever,
    make_chorus_retriever,
    make_nirvana_retriever,
)
from core.types import Request  # noqa: E402


METHODS = ("nirvana_vid", "chai", "chorus", "carnator")
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "cross_request_retrieval"
MANAGED_OUTPUTS = ("results", "indices", "reports", "benchmark.json", "carnator_sources")


@dataclass
class SourceEntry:
    uuid: str
    request_index: int
    prompt: str
    metadata: dict[str, Any] | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


def _resolve_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    values = config.get("retrieval", {}).get("chai", {})
    checkpoint = values.get("longclip_checkpoint")
    if checkpoint and not Path(checkpoint).is_absolute():
        values["longclip_checkpoint"] = str((ROOT / checkpoint).resolve())
    return config


def _load_workload(path: Path, shuffle_seed: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"request_index", "uuid", "prompt"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"workload must contain {sorted(required)}")
    seen: set[str] = set()
    for row in rows:
        uuid = str(row["uuid"])
        if not uuid or uuid in seen or Path(uuid).name != uuid:
            raise ValueError(f"workload UUIDs must be unique, non-empty file names: {uuid!r}")
        seen.add(uuid)
    random.Random(shuffle_seed).shuffle(rows)
    return rows


def _make_retriever(method: str, config: dict[str, Any], *, dry_run: bool) -> RetrievalStrategy:
    retrieval = config["retrieval"]
    if method == "nirvana_vid":
        return make_nirvana_retriever(retrieval[method], dry_run=dry_run)
    if method == "chai":
        return make_chai_retriever(retrieval[method], dry_run=dry_run)
    if method == "chorus":
        return make_chorus_retriever(retrieval[method], dry_run=dry_run)
    if method == "carnator":
        return make_carnator_retriever(retrieval[method], dry_run=dry_run, retrieval_only=True)
    raise ValueError(f"unsupported method: {method}")


def _warm(retriever: RetrievalStrategy, prompt: str) -> None:
    retriever.embedder.encode([prompt])
    extractor = getattr(retriever, "extractor", None)
    if extractor is not None:
        extractor.extract(prompt)


def _records_for_miss(
    retriever: RetrievalStrategy,
    request: Request,
    existing_records: list[VectorRecord],
    *,
    method: str,
    signature: dict[str, Any] | None = None,
    noun_extractor: CarnatorNounExtractor | None = None,
) -> list[VectorRecord]:
    if method != "carnator":
        return retriever.records_for_add(SourceEntry(request.uuid, request.request_index, request.prompt), existing_records)
    if signature is None or noun_extractor is None:
        raise RuntimeError("a Carnator miss must create its source signature before entering the cache")
    metadata = {
        "source_request_index": request.request_index,
        "trf_nouns": list(noun_extractor.for_request(request)),
        "cache_ready": True,
        "signature": signature,
    }
    return retriever.records_for_add(
        SourceEntry(request.uuid, request.request_index, request.prompt, metadata), existing_records
    )


def _summary(method: str, requests: int, hits: int, lookup_s: float, peak_records: int, final_records: int) -> dict[str, Any]:
    return {
        "method": method,
        "requests": requests,
        "hits": hits,
        "hit_rate": hits / requests if requests else 0.0,
        "retrieval_total_s": lookup_s,
        "retrieval_mean_ms": lookup_s * 1000.0 / requests if requests else 0.0,
        "peak_cache_records": peak_records,
        "final_cache_records": final_records,
    }


def _run_standard_method(
    method: str,
    rows: list[dict[str, str]],
    config: dict[str, Any],
    *,
    seed: int,
    dry_run: bool,
    index_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retriever = _make_retriever(method, config, dry_run=dry_run)
    retriever.bind_vector_store(index_dir)
    if rows:
        _warm(retriever, str(rows[0]["prompt"]))
    records: list[VectorRecord] = []
    source_prompts: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    hits = 0
    total_lookup_s = 0.0
    peak_records = 0
    for arrival_index, row in enumerate(rows):
        request = Request(int(row["request_index"]), str(row["uuid"]), str(row["prompt"]), seed + int(row["request_index"]))
        started = time.perf_counter()
        match = retriever.search(request, records)
        lookup_s = time.perf_counter() - started
        total_lookup_s += lookup_s
        if match.hit:
            hits += 1
        else:
            records.extend(_records_for_miss(retriever, request, records, method=method))
            source_prompts[request.uuid] = request.prompt
            retriever.sync_vector_store(records)
        cache_records = len({record.uuid for record in records})
        peak_records = max(peak_records, cache_records)
        results.append({
            "arrival_index": arrival_index,
            "request_index": request.request_index,
            "uuid": request.uuid,
            "prompt": request.prompt,
            "hit": bool(match.hit),
            "source_uuid": match.source_uuid,
            "source_prompt": source_prompts.get(match.source_uuid or ""),
            "score": match.score,
            "retrieval_s": lookup_s,
            "stage1_retrieval_s": lookup_s,
            "signature_gate_s": 0.0,
            "early_probe_s": 0.0,
            "source_cache_s": 0.0,
            "probe_collected": False,
            "source_cached": not match.hit,
            "completed_steps": 0,
            "cache_records": cache_records,
            "cache_match": match.metadata,
        })
    return _summary(method, len(rows), hits, total_lookup_s, peak_records, len({record.uuid for record in records})), results


def _run_carnator(
    rows: list[dict[str, str]],
    config: dict[str, Any],
    *,
    seed: int,
    dry_run: bool,
    index_dir: Path,
    collector: NativeOnlineCollector | SyntheticOnlineCollector,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retriever = _make_retriever("carnator", config, dry_run=dry_run)
    assert isinstance(retriever, CarnatorRetriever)
    retriever.bind_vector_store(index_dir)
    if rows:
        _warm(retriever, str(rows[0]["prompt"]))
    records: list[VectorRecord] = []
    source_prompts: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    hits = 0
    total_lookup_s = 0.0
    total_stage1_s = 0.0
    total_gate_s = 0.0
    total_probe_s = 0.0
    total_source_cache_s = 0.0
    probe_requests = 0
    peak_records = 0
    try:
        for arrival_index, row in enumerate(rows):
            request = Request(
                int(row["request_index"]), str(row["uuid"]), str(row["prompt"]), seed + int(row["request_index"])
            )
            stage1_started = time.perf_counter()
            preflight = retriever.search(request, records)
            stage1_s = time.perf_counter() - stage1_started
            evaluator = None
            if preflight.kind == "probe_required":
                evaluator = lambda signature: retriever.search(request, records, target_signature=signature)
            collection = collector.process(
                request,
                retriever.noun_extractor.for_request(request),
                evaluate_probe=evaluator,
            )
            match = collection.match or preflight
            lookup_s = stage1_s + collection.signature_gate_s
            total_lookup_s += lookup_s
            total_stage1_s += stage1_s
            total_gate_s += collection.signature_gate_s
            total_probe_s += collection.early_probe_s
            total_source_cache_s += collection.source_cache_s
            probe_requests += int(collection.probe_collected)
            if match.hit:
                hits += 1
            else:
                records.extend(
                    _records_for_miss(
                        retriever,
                        request,
                        records,
                        method="carnator",
                        signature=collection.source_signature,
                        noun_extractor=retriever.noun_extractor,
                    )
                )
                source_prompts[request.uuid] = request.prompt
                retriever.sync_vector_store(records)
            cache_records = len({record.uuid for record in records})
            peak_records = max(peak_records, cache_records)
            results.append({
                "arrival_index": arrival_index,
                "request_index": request.request_index,
                "uuid": request.uuid,
                "prompt": request.prompt,
                "hit": bool(match.hit),
                "source_uuid": match.source_uuid,
                "source_prompt": source_prompts.get(match.source_uuid or ""),
                "score": match.score,
                "retrieval_s": lookup_s,
                "stage1_retrieval_s": stage1_s,
                "signature_gate_s": collection.signature_gate_s,
                "early_probe_s": collection.early_probe_s,
                "source_cache_s": collection.source_cache_s,
                "probe_collected": collection.probe_collected,
                "source_cached": collection.source_cached,
                "completed_steps": collection.completed_steps,
                "cache_records": cache_records,
                "cache_match": match.metadata,
            })
    finally:
        collector.close()
    summary = _summary("carnator", len(rows), hits, total_lookup_s, peak_records, len({record.uuid for record in records}))
    summary.update({
        "stage_2": "progressive_online_signature",
        "stage1_retrieval_total_s": total_stage1_s,
        "signature_gate_total_s": total_gate_s,
        "early_probe_total_s": total_probe_s,
        "early_probe_requests": probe_requests,
        "source_cache_total_s": total_source_cache_s,
        "source_cache_entries": len({record.uuid for record in records}),
    })
    return summary, results


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_reports(reports_dir: Path, summaries: dict[str, dict[str, Any]], results: dict[str, list[dict[str, Any]]]) -> None:
    requests = results[METHODS[0]]
    match_rows: list[dict[str, Any]] = []
    for position, request in enumerate(requests):
        row: dict[str, Any] = {
            "arrival_index": request["arrival_index"], "request_index": request["request_index"],
            "uuid": request["uuid"], "prompt": request["prompt"],
        }
        for method in METHODS:
            item = results[method][position]
            row.update({
                f"{method}_hit": item["hit"], f"{method}_source_uuid": item["source_uuid"],
                f"{method}_source_prompt": item["source_prompt"], f"{method}_score": item["score"],
            })
        match_rows.append(row)
    fields = ["arrival_index", "request_index", "uuid", "prompt"]
    for method in METHODS:
        fields.extend((f"{method}_hit", f"{method}_source_uuid", f"{method}_source_prompt", f"{method}_score"))
    _write_csv(reports_dir / "request_matches.csv", fields, match_rows)

    hit_rate_rows: list[dict[str, Any]] = []
    cache_growth_rows: list[dict[str, Any]] = []
    for method in METHODS:
        hits = 0
        for item in results[method]:
            hits += int(bool(item["hit"]))
            arrived = int(item["arrival_index"]) + 1
            hit_rate_rows.append({
                "method": method, "arrival_index": item["arrival_index"], "requests_seen": arrived,
                "cumulative_hits": hits, "cumulative_hit_rate": hits / arrived,
            })
            cache_growth_rows.append({
                "method": method, "arrival_index": item["arrival_index"], "request_index": item["request_index"],
                "uuid": item["uuid"], "cache_records": item["cache_records"],
            })
    _write_csv(
        reports_dir / "hit_rate_over_time.csv",
        ["method", "arrival_index", "requests_seen", "cumulative_hits", "cumulative_hit_rate"], hit_rate_rows,
    )
    _write_csv(
        reports_dir / "cache_growth.csv",
        ["method", "arrival_index", "request_index", "uuid", "cache_records"], cache_growth_rows,
    )
    summary_fields = [
        "method", "requests", "hits", "hit_rate", "retrieval_total_s", "retrieval_mean_ms",
        "peak_cache_records", "final_cache_records", "stage_2", "stage1_retrieval_total_s",
        "signature_gate_total_s", "early_probe_total_s", "early_probe_requests", "source_cache_total_s",
        "source_cache_entries",
    ]
    _write_csv(reports_dir / "retrieval_summary.csv", summary_fields, [summaries[method] for method in METHODS])


def _prepare_artifact_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in MANAGED_OUTPUTS:
        target = path / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    for name in ("results", "indices", "reports", "carnator_sources"):
        (path / name).mkdir()


def _native_args(args: argparse.Namespace):
    from carnator.adapters import wan_ti2v_5b as native

    native_args = native.build_parser().parse_args([])
    native_args.no_env_setup = True
    native_args.model_dir = args.carnator_model_dir
    native_args.device = args.carnator_device
    native_args.model_dtype = args.carnator_model_dtype
    native_args.steps = 50
    native_args.mask_probe_step = 5
    native_args.reuse_start_step = 10
    native_args.offload_model = False
    native_args.disable_progress = True
    return native_args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=ROOT / "datasets" / "vbench_cross_request_prompts.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "methods" / "default.yaml")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--shuffle-seed", type=int, default=20260821)
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic retrieval and collector fixtures.")
    parser.add_argument("--carnator-model-dir", type=Path, default=Path("/home/tangxuebin/models/Wan2.2-TI2V-5B"))
    parser.add_argument("--carnator-device", default="cuda:4")
    parser.add_argument("--carnator-model-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--carnator-probe-seed", type=int, default=20260218)
    args = parser.parse_args(argv)

    rows = _load_workload(args.workload.expanduser().resolve(), args.shuffle_seed)
    config = _resolve_config_paths(_load_yaml(args.config.expanduser().resolve()))
    artifact_dir = args.artifact_dir.expanduser().resolve()
    _prepare_artifact_dir(artifact_dir)
    summaries: dict[str, dict[str, Any]] = {}
    results: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS[:-1]:
        summaries[method], results[method] = _run_standard_method(
            method, rows, config, seed=args.seed, dry_run=args.dry_run, index_dir=artifact_dir / "indices" / method
        )

    carnator_retriever = _make_retriever("carnator", config, dry_run=args.dry_run)
    assert isinstance(carnator_retriever, CarnatorRetriever)
    noun_signature = carnator_retriever.noun_extractor.extractor.signature
    del carnator_retriever
    collector = (
        SyntheticOnlineCollector(
            artifact_dir / "carnator_sources", probe_seed=args.carnator_probe_seed, noun_extractor=noun_signature
        )
        if args.dry_run
        else NativeOnlineCollector(
            _native_args(args), artifact_dir / "carnator_sources", probe_seed=args.carnator_probe_seed,
            noun_extractor=noun_signature,
        )
    )
    summaries["carnator"], results["carnator"] = _run_carnator(
        rows, config, seed=args.seed, dry_run=args.dry_run, index_dir=artifact_dir / "indices" / "carnator", collector=collector
    )
    for method in METHODS:
        with (artifact_dir / "results" / f"{method}.jsonl").open("w", encoding="utf-8") as handle:
            for item in results[method]:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    _write_reports(artifact_dir / "reports", summaries, results)
    benchmark = {
        "schema_version": "cross-request-retrieval/v3-progressive-online",
        "workload": str(args.workload.expanduser().resolve()),
        "requests": len(rows),
        "seed": args.seed,
        "shuffle_seed": args.shuffle_seed,
        "dry_run": args.dry_run,
        "carnator": {
            "sidecar_mode": "arrival_order_miss_only",
            "scheduler_total_steps": 50,
            "target_probe_completed_step": 5,
            "source_cache_completed_step": 10,
            "video_decoded": False,
            "probe_seed": args.carnator_probe_seed,
        },
        "methods": summaries,
        "reports": {
            "request_matches": "reports/request_matches.csv",
            "hit_rate_over_time": "reports/hit_rate_over_time.csv",
            "retrieval_summary": "reports/retrieval_summary.csv",
            "cache_growth": "reports/cache_growth.csv",
        },
    }
    (artifact_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(benchmark, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
