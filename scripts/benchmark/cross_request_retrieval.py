#!/usr/bin/env python3
"""Benchmark all cross-request retrieval baselines without video generation.

The script replays one request arrival order through isolated NIRVANA-VID,
CHAI, Chorus, and Carnator caches. It always writes the current benchmark to
``artifacts/cross_request_retrieval``; reruns replace only this script's
``results``, ``indices``, ``reports``, and ``benchmark.json`` outputs.
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

from carnator.retrieval import CarnatorNounAnnotations, CarnatorRetriever, make_carnator_retriever  # noqa: E402
from carnator.online_signature import SignatureError, validate_online_signature  # noqa: E402
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
MANAGED_OUTPUTS = ("results", "indices", "reports", "benchmark.json")


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
    for section, keys in (("chai", ("longclip_checkpoint",)), ("carnator", ("noun_annotations",))):
        values = config.get("retrieval", {}).get(section, {})
        for key in keys:
            value = values.get(key)
            if value and not Path(value).is_absolute():
                values[key] = str((ROOT / value).resolve())
    return config


def _load_workload(path: Path, shuffle_seed: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"request_index", "uuid", "prompt"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"workload must contain {sorted(required)}")
    seen: set[str] = set()
    for row in rows:
        uuid = str(row["uuid"])
        if not uuid or uuid in seen:
            raise ValueError(f"workload UUIDs must be unique and non-empty: {uuid!r}")
        seen.add(uuid)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
    return rows


def _load_signatures(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            uuid = str(row.get("uuid", ""))
            signature = row.get("signature")
            if not uuid or not isinstance(signature, dict) or uuid in records:
                raise ValueError(f"{path}:{line_number} needs a unique uuid and signature object")
            records[uuid] = signature
    return records


def _usable_carnator_signatures(
    rows: list[dict[str, str]],
    signatures: dict[str, dict[str, Any]],
    *,
    allow_missing: bool,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    usable: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for row in rows:
        uuid = str(row["uuid"])
        signature = signatures.get(uuid)
        path_value = signature.get("signature_path") if isinstance(signature, dict) else None
        if not isinstance(path_value, str) or not Path(path_value).is_file():
            missing.append(uuid)
            continue
        try:
            payload = validate_online_signature(signature, require_cache=True, check_cache_artifacts=False)
        except (OSError, RuntimeError, SignatureError, ValueError):
            missing.append(uuid)
            continue
        if payload.get("prompt") != str(row["prompt"]):
            missing.append(uuid)
            continue
        usable[uuid] = signature
    if missing and not allow_missing:
        raise ValueError(
            "Carnator needs a readable object-level signature sidecar for every workload request; "
            f"first missing UUID: {missing[0]}"
        )
    return usable, missing


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
    signatures: dict[str, dict[str, Any]],
    annotations: CarnatorNounAnnotations | None,
) -> list[VectorRecord]:
    if method != "carnator":
        return retriever.records_for_add(SourceEntry(request.uuid, request.request_index, request.prompt), existing_records)

    assert isinstance(retriever, CarnatorRetriever)
    assert annotations is not None
    signature = signatures.get(request.uuid)
    metadata: dict[str, Any] = {
        "source_request_index": request.request_index,
        "trf_nouns": list(annotations.for_request(request)),
        # This benchmark measures retrieval authorization. Artifact presence is
        # represented by the sidecar and is not materialized by a model run.
        "cache_ready": signature is not None,
    }
    if signature is not None:
        metadata["signature"] = signature
    else:
        metadata["signature_missing_reason"] = "online_signature_sidecar_unavailable"
    return retriever.records_for_add(
        SourceEntry(request.uuid, request.request_index, request.prompt, metadata), existing_records
    )


def _run_method(
    method: str,
    rows: list[dict[str, str]],
    config: dict[str, Any],
    *,
    seed: int,
    dry_run: bool,
    index_dir: Path,
    signatures: dict[str, dict[str, Any]],
    missing_signatures: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retriever = _make_retriever(method, config, dry_run=dry_run)
    annotations = retriever.noun_annotations if method == "carnator" else None
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
        request = Request(
            int(row["request_index"]), str(row["uuid"]), str(row["prompt"]), seed + int(row["request_index"])
        )
        started = time.perf_counter()
        match = (
            retriever.search(request, records, target_signature=signatures.get(request.uuid))
            if method == "carnator"
            else retriever.search(request, records)
        )
        lookup_s = time.perf_counter() - started
        total_lookup_s += lookup_s
        if match.hit:
            hits += 1
        else:
            records.extend(
                _records_for_miss(
                    retriever, request, records, method=method, signatures=signatures, annotations=annotations
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
            "cache_records": cache_records,
            "cache_match": match.metadata,
        })

    summary: dict[str, Any] = {
        "method": method,
        "requests": len(rows),
        "hits": hits,
        "hit_rate": hits / len(rows) if rows else 0.0,
        "retrieval_total_s": total_lookup_s,
        "retrieval_mean_ms": total_lookup_s * 1000.0 / len(rows) if rows else 0.0,
        "peak_cache_records": peak_records,
        "final_cache_records": len({record.uuid for record in records}),
    }
    if method == "carnator":
        summary["missing_online_signatures"] = missing_signatures
        summary["stage_2"] = "online_signature"
    return summary, results


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_reports(
    reports_dir: Path,
    summaries: dict[str, dict[str, Any]],
    results: dict[str, list[dict[str, Any]]],
) -> None:
    requests = results[METHODS[0]]
    match_rows: list[dict[str, Any]] = []
    for position, request in enumerate(requests):
        row: dict[str, Any] = {
            "arrival_index": request["arrival_index"],
            "request_index": request["request_index"],
            "uuid": request["uuid"],
            "prompt": request["prompt"],
        }
        for method in METHODS:
            item = results[method][position]
            row.update({
                f"{method}_hit": item["hit"],
                f"{method}_source_uuid": item["source_uuid"],
                f"{method}_source_prompt": item["source_prompt"],
                f"{method}_score": item["score"],
            })
        match_rows.append(row)
    match_fields = ["arrival_index", "request_index", "uuid", "prompt"]
    for method in METHODS:
        match_fields.extend((f"{method}_hit", f"{method}_source_uuid", f"{method}_source_prompt", f"{method}_score"))
    _write_csv(reports_dir / "request_matches.csv", match_fields, match_rows)

    hit_rate_rows: list[dict[str, Any]] = []
    cache_growth_rows: list[dict[str, Any]] = []
    for method in METHODS:
        hits = 0
        for item in results[method]:
            hits += int(bool(item["hit"]))
            arrived = int(item["arrival_index"]) + 1
            hit_rate_rows.append({
                "method": method,
                "arrival_index": item["arrival_index"],
                "requests_seen": arrived,
                "cumulative_hits": hits,
                "cumulative_hit_rate": hits / arrived,
            })
            cache_growth_rows.append({
                "method": method,
                "arrival_index": item["arrival_index"],
                "request_index": item["request_index"],
                "uuid": item["uuid"],
                "cache_records": item["cache_records"],
            })
    _write_csv(
        reports_dir / "hit_rate_over_time.csv",
        ["method", "arrival_index", "requests_seen", "cumulative_hits", "cumulative_hit_rate"],
        hit_rate_rows,
    )
    _write_csv(
        reports_dir / "cache_growth.csv",
        ["method", "arrival_index", "request_index", "uuid", "cache_records"],
        cache_growth_rows,
    )
    summary_rows = [summaries[method] for method in METHODS]
    _write_csv(
        reports_dir / "retrieval_summary.csv",
        [
            "method", "requests", "hits", "hit_rate", "retrieval_total_s", "retrieval_mean_ms",
            "peak_cache_records", "final_cache_records", "missing_online_signatures", "stage_2",
        ],
        summary_rows,
    )


def _prepare_artifact_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in MANAGED_OUTPUTS:
        target = path / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    (path / "results").mkdir()
    (path / "indices").mkdir()
    (path / "reports").mkdir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=ROOT / "datasets" / "vbench_cross_request_prompts.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "methods" / "default.yaml")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--online-signatures", type=Path, help="JSONL: {uuid, signature: {signature_path}}.")
    parser.add_argument("--allow-missing-carnator-signatures", action="store_true")
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--shuffle-seed", type=int, default=20260821)
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic lexical encoders for protocol tests.")
    args = parser.parse_args(argv)

    rows = _load_workload(args.workload.expanduser().resolve(), args.shuffle_seed)
    config = _resolve_config_paths(_load_yaml(args.config.expanduser().resolve()))
    raw_signatures = _load_signatures(args.online_signatures.expanduser().resolve() if args.online_signatures else None)
    signatures, missing = _usable_carnator_signatures(
        rows, raw_signatures, allow_missing=args.allow_missing_carnator_signatures
    )
    artifact_dir = args.artifact_dir.expanduser().resolve()
    _prepare_artifact_dir(artifact_dir)

    summaries: dict[str, dict[str, Any]] = {}
    results: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        summary, method_results = _run_method(
            method,
            rows,
            config,
            seed=args.seed,
            dry_run=args.dry_run,
            index_dir=artifact_dir / "indices" / method,
            signatures=signatures,
            missing_signatures=len(missing),
        )
        summaries[method] = summary
        results[method] = method_results
        with (artifact_dir / "results" / f"{method}.jsonl").open("w", encoding="utf-8") as handle:
            for item in method_results:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    _write_reports(artifact_dir / "reports", summaries, results)
    benchmark = {
        "schema_version": "cross-request-retrieval/v2",
        "workload": str(args.workload.expanduser().resolve()),
        "requests": len(rows),
        "seed": args.seed,
        "shuffle_seed": args.shuffle_seed,
        "dry_run": args.dry_run,
        "online_signatures": str(args.online_signatures.expanduser().resolve()) if args.online_signatures else None,
        "carnator_missing_online_signatures": len(missing),
        "methods": summaries,
        "reports": {
            "request_matches": "reports/request_matches.csv",
            "hit_rate_over_time": "reports/hit_rate_over_time.csv",
            "retrieval_summary": "reports/retrieval_summary.csv",
            "cache_growth": "reports/cache_growth.csv",
        },
    }
    (artifact_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(benchmark, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
