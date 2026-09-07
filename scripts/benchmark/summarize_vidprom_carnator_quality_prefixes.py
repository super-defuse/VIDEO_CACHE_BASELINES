#!/usr/bin/env python3
"""Summarize cumulative top-quality Carnator requests at fixed percentages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = (
    ROOT
    / "artifacts"
    / "vidprom_1000cache_300input_noun_match"
    / "early_signature_disabled_20260903_100305"
    / "carnator_boundary_b5"
    / "quality_input.jsonl"
)
DEFAULT_PERCENTAGES = tuple(range(5, 91, 5))
RANKING_METHODS = (
    ("clip_t", "CLIP-T descending"),
    ("vbench_q", "VBench-Q descending"),
    ("balanced_z", "equal-weight standardized CLIP-T and VBench-Q descending"),
)


@dataclass(frozen=True)
class QualityRecord:
    request_index: int
    uuid: str
    clip_t: float
    vbench_q: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"no JSONL records: {path}")
    return rows


def read_quality_records(path: Path) -> list[QualityRecord]:
    records: list[QualityRecord] = []
    seen_indices: set[int] = set()
    for row in read_jsonl(path):
        request_index = row.get("request_index")
        if not isinstance(request_index, int):
            raise ValueError("every input row must have an integer request_index")
        if request_index in seen_indices:
            raise ValueError(f"duplicate request_index: {request_index}")
        if not is_finite_number(row.get("clip_t")) or not is_finite_number(row.get("vbench_q")):
            raise ValueError(f"request {request_index} lacks a finite CLIP-T or VBench-Q score")
        seen_indices.add(request_index)
        records.append(
            QualityRecord(
                request_index=request_index,
                uuid=str(row.get("uuid", "")),
                clip_t=float(row["clip_t"]),
                vbench_q=float(row["vbench_q"]),
            )
        )
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_percentages(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise ValueError("--percentages must be a comma-separated list of integers") from exc
    if not values or any(not 0 < value <= 100 for value in values):
        raise ValueError("--percentages values must be integers from 1 through 100")
    if tuple(sorted(set(values))) != values:
        raise ValueError("--percentages must be strictly increasing without duplicates")
    return values


def prefix_count(total: int, percentage: int) -> int:
    return math.ceil(total * percentage / 100)


def balanced_scores(records: Sequence[QualityRecord]) -> dict[int, float]:
    clip_mean = fmean(record.clip_t for record in records)
    clip_stddev = pstdev(record.clip_t for record in records)
    vbench_mean = fmean(record.vbench_q for record in records)
    vbench_stddev = pstdev(record.vbench_q for record in records)
    if clip_stddev == 0 or vbench_stddev == 0:
        raise ValueError("cannot rank by balanced standardized quality when a metric has zero variation")
    return {
        record.request_index: ((record.clip_t - clip_mean) / clip_stddev + (record.vbench_q - vbench_mean) / vbench_stddev) / 2
        for record in records
    }


def ranked_records(records: Sequence[QualityRecord], method: str, composite: dict[int, float]) -> list[QualityRecord]:
    if method == "clip_t":
        return sorted(records, key=lambda record: (-record.clip_t, record.request_index))
    if method == "vbench_q":
        return sorted(records, key=lambda record: (-record.vbench_q, record.request_index))
    if method == "balanced_z":
        return sorted(records, key=lambda record: (-composite[record.request_index], record.request_index))
    raise ValueError(f"unknown ranking method: {method}")


def prefix_summary(records: Sequence[QualityRecord], percentages: Sequence[int]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty quality-record collection")
    composite = balanced_scores(records)
    full_clip_mean = fmean(record.clip_t for record in records)
    full_vbench_mean = fmean(record.vbench_q for record in records)
    rows: list[dict[str, Any]] = []
    selected_indices: dict[str, dict[str, list[int]]] = {}

    for method, description in RANKING_METHODS:
        ranked = ranked_records(records, method, composite)
        selected_indices[method] = {}
        for percentage in percentages:
            count = prefix_count(len(records), percentage)
            selected = ranked[:count]
            clip_mean = fmean(record.clip_t for record in selected)
            vbench_mean = fmean(record.vbench_q for record in selected)
            balanced_mean = fmean(composite[record.request_index] for record in selected)
            rows.append(
                {
                    "ranking_method": method,
                    "ranking_description": description,
                    "top_percent": percentage,
                    "selected_requests": count,
                    "total_requests": len(records),
                    "clip_t_mean": clip_mean,
                    "clip_t_delta_from_full": clip_mean - full_clip_mean,
                    "vbench_q_mean": vbench_mean,
                    "vbench_q_delta_from_full": vbench_mean - full_vbench_mean,
                    "balanced_z_mean": balanced_mean,
                }
            )
            selected_indices[method][str(percentage)] = [record.request_index for record in selected]

    return {
        "request_count": len(records),
        "full_population": {
            "clip_t_mean": full_clip_mean,
            "clip_t_population_stddev": pstdev(record.clip_t for record in records),
            "vbench_q_mean": full_vbench_mean,
            "vbench_q_population_stddev": pstdev(record.vbench_q for record in records),
        },
        "ranking_methods": {
            method: {
                "description": description,
                "tie_break": "request_index ascending",
            }
            for method, description in RANKING_METHODS
        },
        "percentages": list(percentages),
        "rows": rows,
        "selected_request_indices": selected_indices,
    }


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError("cannot write an empty summary CSV")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    population = summary["full_population"]
    lines = [
        "# Carnator 300-request quality ranked-prefix summary",
        "",
        "## Scope",
        "",
        "The completed 300-request Carnator result is ranked in descending quality order. Each row is a cumulative prefix: for example, top 10% contains the best 30 of 300 requests under that ranking. The two reported means are always calculated over the same selected requests.",
        "",
        f"- Requests: {summary['request_count']}",
        f"- Full-population CLIP-T / VBench-Q: {population['clip_t_mean']:.6f} / {population['vbench_q_mean']:.6f}",
        f"- Full-population standard deviations: {population['clip_t_population_stddev']:.6f} / {population['vbench_q_population_stddev']:.6f}",
        "- Ties: lower `request_index` first.",
        "",
        "## Ranking Rules",
        "",
        "- `clip_t`: CLIP-T descending.",
        "- `vbench_q`: VBench-Q descending.",
        "- `balanced_z`: equal-weight mean of CLIP-T and VBench-Q z-scores, each standardized using all 300 requests, descending.",
    ]
    for method, description in RANKING_METHODS:
        lines.extend([
            "",
            f"## {description}",
            "",
            "| Top requests | Selected | Mean CLIP-T | Delta CLIP-T | Mean VBench-Q | Delta VBench-Q | Mean balanced z |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in summary["rows"]:
            if row["ranking_method"] == method:
                lines.append(
                    f"| {row['top_percent']}% | {row['selected_requests']} | {row['clip_t_mean']:.6f} | "
                    f"{row['clip_t_delta_from_full']:+.6f} | {row['vbench_q_mean']:.6f} | "
                    f"{row['vbench_q_delta_from_full']:+.6f} | {row['balanced_z_mean']:+.6f} |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="completed Carnator quality_input.jsonl")
    parser.add_argument("--output-dir", type=Path, help="directory for ranked-prefix outputs")
    parser.add_argument(
        "--percentages",
        default=",".join(str(value) for value in DEFAULT_PERCENTAGES),
        help="strictly increasing cumulative top percentages (default: 5,10,...,90)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    percentages = parse_percentages(args.percentages)
    results_path = args.results.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_path.parent / "quality_threshold_optimization"
    )
    summary = prefix_summary(read_quality_records(results_path), percentages)
    summary["created_at"] = utc_now()
    summary["input"] = {"path": str(results_path), "sha256": sha256_file(results_path)}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "quality_ranked_prefix_summary.json", summary)
    write_csv(output_dir / "quality_ranked_prefix_summary.csv", summary["rows"])
    write_report(output_dir / "quality_ranked_prefix_summary.md", summary)
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "request_count": summary["request_count"],
                "percentages": list(percentages),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
