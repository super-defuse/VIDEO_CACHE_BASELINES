#!/usr/bin/env python3
"""Replay a prompt dataset through cross-request retrieval without video generation.

This is the dataset counterpart to ``source_target_matrix.py``.  It calls only
the retrievers: every cache miss adds a lightweight source record, every hit
selects an earlier source, and no latent/video/model denoising is loaded.  A
Carnator run requires precomputed native step-5 ETS signatures for every UUID.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "cross_request_retrieval"

from carnator.retrieval import (  # noqa: E402
    CARNATOR_IMPLEMENTATION_ID,
    CARNATOR_SIGNATURE_PROFILE_ID,
    CarnatorNounAnnotations,
    CarnatorRetriever,
    make_carnator_retriever,
    prompt_sha256,
)
from core.retrieval import (  # noqa: E402
    RetrievalStrategy,
    VectorRecord,
    make_chai_retriever,
    make_chorus_retriever,
    make_nirvana_retriever,
)
from core.types import Request  # noqa: E402


METHODS = ("nirvana_vid", "chai", "chorus", "carnator")


@dataclass
class SourceEntry:
    uuid: str
    request_index: int
    prompt: str
    metadata: dict[str, Any] | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _safe_component(value: str, label: str) -> str:
    component = Path(value)
    if not value or component.name != value or value in {".", ".."}:
        raise ValueError(f"{label} must be a single directory name: {value!r}")
    return value


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        if args.run_id is not None or args.run_name is not None:
            raise ValueError("--output-dir cannot be combined with --run-id or --run-name")
        return args.output_dir.expanduser().resolve()
    if args.run_id is None or args.run_name is None:
        raise ValueError("provide --output-dir, or provide both --run-id and --run-name")
    run_id = _safe_component(args.run_id, "run-id")
    run_name = _safe_component(args.run_name, "run-name")
    return (args.artifact_root.expanduser().resolve() / run_id / "runs" / run_name).resolve()


def _resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    for section, keys in (
        ("chai", ("longclip_checkpoint",)),
        ("carnator", ("noun_annotations", "threshold_config")),
    ):
        values = config.get("retrieval", {}).get(section, {})
        for key in keys:
            value = values.get(key)
            if value and not Path(value).is_absolute():
                values[key] = str((ROOT / value).resolve())
    return config


def _workload(path: Path, shuffle_seed: int | None) -> list[dict[str, str]]:
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


def _signature_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            uuid = str(row.get("uuid", ""))
            signature = row.get("signature", row.get("ets"))
            if not uuid or not isinstance(signature, dict):
                raise ValueError(f"{path}:{line_number} needs uuid and signature")
            if uuid in records:
                raise ValueError(f"duplicate ETS UUID in {path}: {uuid}")
            records[uuid] = signature
    return records


def _validate_carnator_signatures(
    rows: list[dict[str, str]],
    signatures: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> None:
    for row in rows:
        uuid = str(row["uuid"])
        signature = signatures.get(uuid)
        if signature is None:
            raise ValueError(f"Carnator dataset mode needs native ETS for every request; first missing UUID: {uuid}")
        if signature.get("signature_scope") == "prompt":
            if signature.get("prompt_sha256") != prompt_sha256(str(row["prompt"])):
                raise ValueError(f"Carnator ETS prompt hash does not match workload UUID: {uuid}")
            capture = signature.get("native_capture")
            expected_seed = seed + int(row["request_index"])
            if not isinstance(capture, dict) or capture.get("request_seed") != expected_seed:
                raise ValueError(f"Carnator ETS request seed does not match workload UUID: {uuid}")


def _inline_carnator_manifest(uuid: str, threshold_config_id: str) -> dict[str, Any]:
    return {
        "implementation_id": CARNATOR_IMPLEMENTATION_ID,
        "signature_profile_id": CARNATOR_SIGNATURE_PROFILE_ID,
        "threshold_config_id": threshold_config_id,
        "uuid": uuid,
        "namespace": "retrieval-only",
        "completed_step": 5,
        "reuse_anchor_step": 10,
    }


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


def _warm_retriever(retriever: RetrievalStrategy, prompt: str) -> None:
    """Initialize lazy text/entity models outside the per-request timing window."""

    retriever.embedder.encode([prompt])
    extractor = getattr(retriever, "extractor", None)
    if extractor is not None:
        extractor.extract(prompt)


def _source_records(
    retriever: RetrievalStrategy,
    request: Request,
    *,
    existing_records: list[VectorRecord],
    method: str,
    signatures: dict[str, dict[str, Any]],
    annotations: CarnatorNounAnnotations | None,
) -> list:
    if method != "carnator":
        return retriever.records_for_add(
            SourceEntry(request.uuid, request.request_index, request.prompt),
            existing_records,
        )
    assert isinstance(retriever, CarnatorRetriever)
    assert annotations is not None
    signature = signatures.get(request.uuid)
    if signature is None:
        raise ValueError(f"missing native step-5 ETS for source UUID {request.uuid}")
    nouns = list(annotations.for_request(request))
    metadata = {
        "source_request_index": request.request_index,
        "trf_nouns": nouns,
        "signature": signature,
        # This mode evaluates retrieval authorization only. It deliberately
        # does not assert that a step-10 latent exists on disk.
        "cache_ready": True,
        "manifest": _inline_carnator_manifest(request.uuid, retriever.threshold_config.threshold_config_id),
        "artifact_ready_assumed": True,
    }
    return retriever.records_for_add(
        SourceEntry(request.uuid, request.request_index, request.prompt, metadata),
        existing_records,
    )


def _run_method(
    method: str,
    rows: list[dict[str, str]],
    config: dict[str, Any],
    output_dir: Path,
    *,
    seed: int,
    dry_run: bool,
    signatures: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    retriever = _make_retriever(method, config, dry_run=dry_run)
    annotations = (
        CarnatorNounAnnotations(config["retrieval"]["carnator"]["noun_annotations"])
        if method == "carnator" else None
    )
    retriever.bind_vector_store(output_dir / "vector_store")
    if rows:
        _warm_retriever(retriever, str(rows[0]["prompt"]))
    if method == "carnator":
        _validate_carnator_signatures(rows, signatures, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    results: dict[str, dict[str, Any]] = {}
    arrival_results: list[dict[str, Any]] = []
    source_prompts: dict[str, str] = {}
    total_lookup_s = 0.0
    hits = 0
    peak_records = 0
    result_path = output_dir / "results.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for arrival_index, row in enumerate(rows):
            request = Request(
                int(row["request_index"]),
                str(row["uuid"]),
                str(row["prompt"]),
                seed + int(row["request_index"]),
            )
            started = time.perf_counter()
            if method == "carnator":
                match = retriever.search(request, records, target_signature=signatures[request.uuid])
            else:
                match = retriever.search(request, records)
            lookup_s = time.perf_counter() - started
            total_lookup_s += lookup_s
            if match.hit:
                hits += 1
            else:
                records.extend(_source_records(
                    retriever,
                    request,
                    existing_records=records,
                    method=method,
                    signatures=signatures,
                    annotations=annotations,
                ))
                source_prompts[request.uuid] = request.prompt
                retriever.sync_vector_store(records)
            peak_records = max(peak_records, len({record.uuid for record in records}))
            source_prompt = source_prompts.get(match.source_uuid or "")
            payload = {
                "arrival_index": arrival_index,
                "request_index": request.request_index,
                "uuid": request.uuid,
                "prompt": request.prompt,
                "hit": bool(match.hit),
                "source_uuid": match.source_uuid,
                "source_prompt": source_prompt,
                "score": match.score,
                "retrieval_s": lookup_s,
                "cache_records": len({record.uuid for record in records}),
                "cache_match": match.metadata,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            results[request.uuid] = payload
            arrival_results.append(payload)

    summary = {
        "method": method,
        "requests": len(rows),
        "hits": hits,
        "hit_rate": hits / len(rows) if rows else 0.0,
        "retrieval_total_s": total_lookup_s,
        "retrieval_mean_ms": total_lookup_s * 1000.0 / len(rows) if rows else 0.0,
        "peak_cache_records": peak_records,
        "final_cache_records": len({record.uuid for record in records}),
        "results": str(result_path),
        "carnator_artifact_ready_assumed": method == "carnator",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary, results, arrival_results


def _write_hit_rate_by_group(
    results: dict[str, list[dict[str, Any]]],
    output: Path,
    *,
    group_size: int,
) -> dict[str, list[dict[str, Any]]]:
    if group_size <= 0:
        raise ValueError("group size must be positive")
    rows: dict[str, list[dict[str, Any]]] = {}
    for method, method_results in results.items():
        groups: list[dict[str, Any]] = []
        for start in range(0, len(method_results), group_size):
            group = method_results[start:start + group_size]
            hits = sum(bool(item["hit"]) for item in group)
            groups.append({
                "method": method,
                "group_start": start + 1,
                "end_request": start + len(group),
                "requests_in_group": len(group),
                "hits": hits,
                "hit_rate": hits / len(group),
            })
        rows[method] = groups
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "group_start", "end_request", "requests_in_group", "hits", "hit_rate"),
            lineterminator="\n",
        )
        writer.writeheader()
        for method in results:
            writer.writerows(rows[method])
    return rows


def _write_hit_rate_svg(groups: dict[str, list[dict[str, Any]]], output: Path) -> None:
    """Write a dependency-free comparison chart beside the CSV data."""

    width, height = 960, 480
    left, right, top, bottom = 74, 28, 28, 58
    chart_width, chart_height = width - left - right, height - top - bottom
    largest_x = max((item["end_request"] for rows in groups.values() for item in rows), default=1)
    colors = {
        "nirvana_vid": "#1f77b4",
        "chai": "#d62728",
        "chorus": "#2ca02c",
        "carnator": "#9467bd",
    }

    def x(value: int) -> float:
        return left + chart_width * value / largest_x

    def y(value: float) -> float:
        return top + chart_height * (1.0 - value)

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#1f2937"/>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#1f2937"/>',
        '<text x="18" y="22" font-family="sans-serif" font-size="16" fill="#111827">Hit rate by arrival group</text>',
        f'<text x="{left + chart_width / 2}" y="{height - 16}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#374151">Processed requests</text>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = y(fraction)
        pieces.extend((
            f'<line x1="{left}" y1="{yy:.1f}" x2="{left + chart_width}" y2="{yy:.1f}" stroke="#e5e7eb"/>',
            f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#4b5563">{fraction:.2f}</text>',
        ))
    for method, rows in groups.items():
        if not rows:
            continue
        color = colors.get(method, "#4b5563")
        points = " ".join(f"{x(item['end_request']):.1f},{y(item['hit_rate']):.1f}" for item in rows)
        pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for item in rows:
            pieces.append(
                f'<circle cx="{x(item["end_request"]):.1f}" cy="{y(item["hit_rate"]):.1f}" r="3" fill="{color}"/>'
            )
        legend_y = top + 18 + 20 * list(groups).index(method)
        pieces.extend((
            f'<line x1="{left + chart_width - 180}" y1="{legend_y}" x2="{left + chart_width - 160}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
            f'<text x="{left + chart_width - 154}" y="{legend_y + 4}" font-family="sans-serif" font-size="12" fill="#111827">{escape(method)}</text>',
        ))
    pieces.append("</svg>")
    output.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def _write_selection_comparison(
    rows: list[dict[str, str]],
    results: dict[str, dict[str, dict[str, Any]]],
    output: Path,
) -> None:
    fields = ["request_index", "uuid", "prompt"]
    for method in METHODS:
        if method in results:
            fields.extend((f"{method}_hit", f"{method}_source_uuid", f"{method}_source_prompt", f"{method}_score"))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            payload = {key: row[key] for key in ("request_index", "uuid", "prompt")}
            for method, method_results in results.items():
                result = method_results[row["uuid"]]
                payload.update({
                    f"{method}_hit": result["hit"],
                    f"{method}_source_uuid": result["source_uuid"],
                    f"{method}_source_prompt": result["source_prompt"],
                    f"{method}_score": result["score"],
                })
            writer.writerow(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=ROOT / "datasets" / "vbench_cross_request_prompts.csv")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit output directory (legacy-compatible; cannot be combined with --run-id/--run-name).",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Root used by the organized layout when --run-id and --run-name are provided.",
    )
    parser.add_argument("--run-id", help="Date or experiment identifier for the organized artifact layout.")
    parser.add_argument("--run-name", help="Named retrieval run under <artifact-root>/<run-id>/runs/.")
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--shuffle-seed", type=int, default=20260821)
    parser.add_argument("--group-size", type=int, default=50)
    parser.add_argument("--native-ets-jsonl", type=Path,
                        help="One JSON object per UUID: {uuid, signature}; required when Carnator is selected.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use deterministic lexical retrieval encoders for protocol testing only.")
    args = parser.parse_args(argv)

    try:
        args.output_dir = _resolve_output_dir(args)
    except ValueError as exc:
        parser.error(str(exc))

    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    if not methods or any(method not in METHODS for method in methods):
        raise SystemExit(f"--methods must be a non-empty subset of {METHODS}")
    if args.group_size <= 0:
        raise SystemExit("--group-size must be positive")
    if "carnator" in methods and args.native_ets_jsonl is None:
        raise SystemExit("Carnator dataset mode requires --native-ets-jsonl from native step-5 probes")

    config = _resolve_paths(_load_yaml(ROOT / "configs" / "methods" / "default.yaml"))
    rows = _workload(args.workload, args.shuffle_seed)
    signatures = _signature_map(args.native_ets_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    order_path = args.output_dir / "arrival_order.csv"
    with order_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("arrival_index", "request_index", "uuid", "prompt"))
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"arrival_index": index, **{key: row[key] for key in ("request_index", "uuid", "prompt")}})

    summaries: dict[str, Any] = {}
    results: dict[str, dict[str, dict[str, Any]]] = {}
    arrival_results: dict[str, list[dict[str, Any]]] = {}
    for method in methods:
        summary, method_results, method_arrival_results = _run_method(
            method,
            rows,
            config,
            args.output_dir / method,
            seed=args.seed,
            dry_run=args.dry_run,
            signatures=signatures,
        )
        summaries[method] = summary
        results[method] = method_results
        arrival_results[method] = method_arrival_results
    selection_path = args.output_dir / "selection_comparison.csv"
    _write_selection_comparison(rows, results, selection_path)
    group_path = args.output_dir / "hit_rate_by_group.csv"
    groups = _write_hit_rate_by_group(arrival_results, group_path, group_size=args.group_size)
    chart_path = args.output_dir / "hit_rate_by_group.svg"
    _write_hit_rate_svg(groups, chart_path)
    manifest = {
        "mode": "dataset-retrieval-only",
        "artifact_layout": "cross-request-retrieval/v1",
        "run_id": args.run_id,
        "run_name": args.run_name,
        "output_dir": str(args.output_dir),
        "workload": str(args.workload.resolve()),
        "requests": len(rows),
        "seed": args.seed,
        "shuffle_seed": args.shuffle_seed,
        "dry_run": args.dry_run,
        "native_ets_jsonl": str(args.native_ets_jsonl.resolve()) if args.native_ets_jsonl else None,
        "arrival_order": str(order_path),
        "selection_comparison": str(selection_path),
        "hit_rate_by_group": str(group_path),
        "hit_rate_chart": str(chart_path),
        "group_size": args.group_size,
        "methods": summaries,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
