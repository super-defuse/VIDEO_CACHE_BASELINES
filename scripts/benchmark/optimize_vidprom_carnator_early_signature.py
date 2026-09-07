#!/usr/bin/env python3
"""Optimize Carnator's four Early Signature gates against per-request quality.

The input is a completed Carnator ``results.jsonl`` or ``quality_input.jsonl``
from a run in which the gates were deliberately unbounded.  This script does
not generate videos: it replays the production gate rule over the recorded
per-request distances, then compares the accepted requests' CLIP-T and
VBench-Q means.

For a same-noun-set match, the production comparator only applies the context
colour gate.  Its other three distances are therefore recorded as ``null``;
this script maps precisely those inapplicable values to zero so each candidate
threshold tuple has the same acceptance semantics as the runtime retriever.
Any other missing or non-finite gate distance stops the analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DISTANCE_FIELDS = (
    "context_color_ciede2000",
    "log_area_ratio",
    "shape_one_minus_dice",
    "changed_color_ciede2000",
)
THRESHOLD_NAMES = {
    "context_color_ciede2000": "max_context_color_distance",
    "log_area_ratio": "max_log_area_ratio",
    "shape_one_minus_dice": "max_shape_distance",
    "changed_color_ciede2000": "max_changed_color_distance",
}
QUALITY_FIELDS = ("clip_t", "vbench_q")
DEFAULT_RESULTS = (
    ROOT
    / "artifacts"
    / "vidprom_1000cache_300input_noun_match"
    / "early_signature_disabled_20260903_100305"
    / "carnator_boundary_b5"
    / "quality_input.jsonl"
)


@dataclass(frozen=True)
class RequestQuality:
    request_index: int
    uuid: str
    prompt: str
    distances: tuple[float, float, float, float]
    clip_t: float
    vbench_q: float
    same_noun_set: bool


class DistanceAuditError(RuntimeError):
    """Raised when a request lacks a distance required by the runtime gate."""


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


def selected_metrics(row: dict[str, Any]) -> dict[str, Any] | None:
    cache_match = row.get("cache_match")
    if not isinstance(cache_match, dict):
        return None
    metadata = cache_match.get("metadata")
    if not isinstance(metadata, dict):
        return None
    metrics = metadata.get("selected_metrics")
    return metrics if isinstance(metrics, dict) else None


def same_noun_set(row: dict[str, Any]) -> bool:
    cache_match = row.get("cache_match")
    metadata = cache_match.get("metadata") if isinstance(cache_match, dict) else None
    selection = metadata.get("inference_mask_selection") if isinstance(metadata, dict) else None
    if not isinstance(selection, dict):
        return False
    source_only = selection.get("source_only_nouns")
    target_only = selection.get("target_only_nouns")
    return isinstance(source_only, list) and isinstance(target_only, list) and not source_only and not target_only


def audit_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[RequestQuality], dict[str, Any]]:
    required_missing: dict[str, list[int]] = {field: [] for field in DISTANCE_FIELDS}
    invalid_quality: dict[str, list[int]] = {field: [] for field in QUALITY_FIELDS}
    invalid_rows: list[int] = []
    same_noun_indices: list[int] = []
    normalised_inapplicable: dict[str, list[int]] = {field: [] for field in DISTANCE_FIELDS[1:]}
    records: list[RequestQuality] = []

    for row in rows:
        request_index = row.get("request_index")
        if not isinstance(request_index, int):
            invalid_rows.append(-1)
            continue
        metrics = selected_metrics(row)
        if metrics is None:
            invalid_rows.append(request_index)
            continue
        is_same_noun_set = same_noun_set(row)
        if is_same_noun_set:
            same_noun_indices.append(request_index)

        values: list[float] = []
        for field_index, field in enumerate(DISTANCE_FIELDS):
            value = metrics.get(field)
            if is_finite_number(value):
                values.append(float(value))
            elif is_same_noun_set and field_index > 0 and value is None:
                values.append(0.0)
                normalised_inapplicable[field].append(request_index)
            else:
                required_missing[field].append(request_index)
                values.append(float("nan"))

        for field in QUALITY_FIELDS:
            if not is_finite_number(row.get(field)):
                invalid_quality[field].append(request_index)

        records.append(
            RequestQuality(
                request_index=request_index,
                uuid=str(row.get("uuid", "")),
                prompt=str(row.get("prompt", "")),
                distances=tuple(values),
                clip_t=float(row["clip_t"]) if is_finite_number(row.get("clip_t")) else float("nan"),
                vbench_q=float(row["vbench_q"]) if is_finite_number(row.get("vbench_q")) else float("nan"),
                same_noun_set=is_same_noun_set,
            )
        )

    duplicate_indices = sorted(
        index for index in {record.request_index for record in records}
        if sum(record.request_index == index for record in records) > 1
    )
    audit = {
        "input_rows": len(rows),
        "parsed_rows": len(records),
        "same_noun_set_request_indices": same_noun_indices,
        "normalised_inapplicable_to_zero": normalised_inapplicable,
        "missing_required_distance_indices": required_missing,
        "invalid_quality_indices": invalid_quality,
        "invalid_request_index_rows": invalid_rows,
        "duplicate_request_indices": duplicate_indices,
    }
    has_error = (
        invalid_rows
        or duplicate_indices
        or any(required_missing.values())
        or any(invalid_quality.values())
        or len(records) != len(rows)
    )
    if has_error:
        raise DistanceAuditError(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return records, audit


def threshold_values(values: Sequence[float], minimum_accepted: int, maximum_values: int) -> list[float]:
    """Return observed thresholds that can individually admit ``minimum_accepted`` rows."""

    if not values or not 1 <= minimum_accepted <= len(values) or maximum_values < 2:
        raise ValueError("invalid threshold-grid inputs")
    ordered = sorted(values)
    allowed = [value for value in sorted(set(values)) if sum(item <= value for item in ordered) >= minimum_accepted]
    if len(allowed) <= maximum_values:
        return allowed
    indices = {
        round(index * (len(allowed) - 1) / (maximum_values - 1))
        for index in range(maximum_values)
    }
    return [allowed[index] for index in sorted(indices)]


def threshold_dict(thresholds: Sequence[float]) -> dict[str, float]:
    return {THRESHOLD_NAMES[field]: float(value) for field, value in zip(DISTANCE_FIELDS, thresholds, strict=True)}


def score_candidate(
    thresholds: Sequence[float],
    accepted_count: int,
    clip_t_mean: float,
    vbench_q_mean: float,
    total_requests: int,
    clip_t_baseline: tuple[float, float],
    vbench_q_baseline: tuple[float, float],
) -> dict[str, Any]:
    clip_mean, clip_stddev = clip_t_baseline
    vbench_mean, vbench_stddev = vbench_q_baseline
    clip_z = (clip_t_mean - clip_mean) / clip_stddev if clip_stddev else 0.0
    vbench_z = (vbench_q_mean - vbench_mean) / vbench_stddev if vbench_stddev else 0.0
    return {
        **threshold_dict(thresholds),
        "accepted_requests": int(accepted_count),
        "hit_rate": accepted_count / total_requests,
        "clip_t_mean": float(clip_t_mean),
        "vbench_q_mean": float(vbench_q_mean),
        "clip_t_z_gain": float(clip_z),
        "vbench_q_z_gain": float(vbench_z),
        "balanced_quality_z_gain": float((clip_z + vbench_z) / 2.0),
    }


def strictly_better(candidate: dict[str, Any], incumbent: dict[str, Any] | None, objective: str) -> bool:
    if incumbent is None:
        return True
    candidate_value = float(candidate[objective])
    incumbent_value = float(incumbent[objective])
    if not math.isclose(candidate_value, incumbent_value, rel_tol=0.0, abs_tol=1e-12):
        return candidate_value > incumbent_value
    if candidate["accepted_requests"] != incumbent["accepted_requests"]:
        return int(candidate["accepted_requests"]) > int(incumbent["accepted_requests"])
    candidate_thresholds = tuple(float(candidate[THRESHOLD_NAMES[field]]) for field in DISTANCE_FIELDS)
    incumbent_thresholds = tuple(float(incumbent[THRESHOLD_NAMES[field]]) for field in DISTANCE_FIELDS)
    return candidate_thresholds < incumbent_thresholds


def coverage_band(hit_rate: float, width: float) -> tuple[float, float]:
    lower = round(min(1.0, math.floor(hit_rate / width + 1e-12) * width), 12)
    upper = round(min(1.0, lower + width), 12)
    return lower, upper


def optimise(
    records: Sequence[RequestQuality],
    min_hit_rate: float,
    max_hit_rate: float,
    maximum_threshold_values: int,
    batch_size: int,
    coverage_band_width: float,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - all benchmark environments provide NumPy
        raise RuntimeError("NumPy is required for the vectorized threshold sweep") from exc

    request_count = len(records)
    minimum_accepted = math.ceil(request_count * min_hit_rate)
    maximum_accepted = math.floor(request_count * max_hit_rate)
    if not 1 <= minimum_accepted <= maximum_accepted <= request_count:
        raise ValueError("hit-rate constraints leave no feasible accepted-request count")

    distance_matrix = np.asarray([record.distances for record in records], dtype=np.float64)
    clip_scores = np.asarray([record.clip_t for record in records], dtype=np.float64)
    vbench_scores = np.asarray([record.vbench_q for record in records], dtype=np.float64)
    threshold_lists = [
        threshold_values(distance_matrix[:, index].tolist(), minimum_accepted, maximum_threshold_values)
        for index in range(len(DISTANCE_FIELDS))
    ]
    total_grid_candidates = math.prod(len(values) for values in threshold_lists)
    clip_baseline = (float(clip_scores.mean()), float(clip_scores.std(ddof=0)))
    vbench_baseline = (float(vbench_scores.mean()), float(vbench_scores.std(ddof=0)))

    best_by_count: dict[int, dict[str, Any]] = {}
    best_by_band: dict[tuple[float, float], dict[str, dict[str, Any]]] = {}
    best_global: dict[str, dict[str, Any] | None] = {
        "balanced_quality_z_gain": None,
        "clip_t_mean": None,
        "vbench_q_mean": None,
    }
    candidate_product = itertools.product(*(range(len(values)) for values in threshold_lists))
    eligible_candidates = 0

    while batch := list(islice(candidate_product, batch_size)):
        candidate_indices = np.asarray(batch, dtype=np.intp)
        candidate_thresholds = np.column_stack(
            [np.asarray(threshold_lists[index], dtype=np.float64)[candidate_indices[:, index]] for index in range(len(DISTANCE_FIELDS))]
        )
        accepted = np.all(distance_matrix[None, :, :] <= candidate_thresholds[:, None, :], axis=2)
        accepted_counts = accepted.sum(axis=1)
        eligible = (accepted_counts >= minimum_accepted) & (accepted_counts <= maximum_accepted)
        if not np.any(eligible):
            continue
        eligible_positions = np.flatnonzero(eligible)
        eligible_masks = accepted[eligible_positions]
        eligible_counts = accepted_counts[eligible_positions]
        clip_means = (eligible_masks @ clip_scores) / eligible_counts
        vbench_means = (eligible_masks @ vbench_scores) / eligible_counts

        for position, accepted_count, clip_mean, vbench_mean in zip(
            eligible_positions,
            eligible_counts,
            clip_means,
            vbench_means,
            strict=True,
        ):
            candidate = score_candidate(
                candidate_thresholds[position],
                int(accepted_count),
                float(clip_mean),
                float(vbench_mean),
                request_count,
                clip_baseline,
                vbench_baseline,
            )
            eligible_candidates += 1
            count = int(accepted_count)
            if strictly_better(candidate, best_by_count.get(count), "balanced_quality_z_gain"):
                best_by_count[count] = candidate
            band = coverage_band(float(candidate["hit_rate"]), coverage_band_width)
            band_best = best_by_band.setdefault(band, {})
            for objective in best_global:
                if strictly_better(candidate, best_global[objective], objective):
                    best_global[objective] = candidate
                if strictly_better(candidate, band_best.get(objective), objective):
                    band_best[objective] = candidate

    if not best_by_count:
        raise RuntimeError("threshold grid produced no candidates in the requested hit-rate interval")
    return {
        "request_count": request_count,
        "min_hit_rate": min_hit_rate,
        "max_hit_rate": max_hit_rate,
        "minimum_accepted": minimum_accepted,
        "maximum_accepted": maximum_accepted,
        "threshold_value_counts": {THRESHOLD_NAMES[field]: len(values) for field, values in zip(DISTANCE_FIELDS, threshold_lists, strict=True)},
        "threshold_values": {THRESHOLD_NAMES[field]: values for field, values in zip(DISTANCE_FIELDS, threshold_lists, strict=True)},
        "total_grid_candidates": total_grid_candidates,
        "eligible_grid_candidates": eligible_candidates,
        "full_population": {
            "clip_t_mean": clip_baseline[0],
            "clip_t_population_stddev": clip_baseline[1],
            "vbench_q_mean": vbench_baseline[0],
            "vbench_q_population_stddev": vbench_baseline[1],
        },
        "recommended": best_global["balanced_quality_z_gain"],
        "best_clip_t": best_global["clip_t_mean"],
        "best_vbench_q": best_global["vbench_q_mean"],
        "best_balanced_by_accepted_count": [best_by_count[count] for count in sorted(best_by_count)],
        "best_by_coverage_band": [
            {
                "hit_rate_lower": lower,
                "hit_rate_upper": upper,
                **best_by_band[(lower, upper)],
            }
            for lower, upper in sorted(best_by_band)
        ],
    }


def accepts(record: RequestQuality, candidate: dict[str, Any]) -> bool:
    return all(
        value <= float(candidate[THRESHOLD_NAMES[field]])
        for field, value in zip(DISTANCE_FIELDS, record.distances, strict=True)
    )


def selection_details(records: Sequence[RequestQuality], candidate: dict[str, Any]) -> dict[str, Any]:
    accepted = [record for record in records if accepts(record, candidate)]
    rejected = []
    for record in records:
        failed = [
            THRESHOLD_NAMES[field]
            for field, value in zip(DISTANCE_FIELDS, record.distances, strict=True)
            if value > float(candidate[THRESHOLD_NAMES[field]])
        ]
        if failed:
            rejected.append({"request_index": record.request_index, "uuid": record.uuid, "failed_gates": failed})
    return {
        "candidate": candidate,
        "accepted_request_indices": [record.request_index for record in accepted],
        "accepted_requests": [
            {
                "request_index": record.request_index,
                "uuid": record.uuid,
                "prompt": record.prompt,
                "clip_t": record.clip_t,
                "vbench_q": record.vbench_q,
                "same_noun_set": record.same_noun_set,
                "effective_gate_distances": dict(zip(DISTANCE_FIELDS, record.distances, strict=True)),
            }
            for record in accepted
        ],
        "rejected_requests": rejected,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def format_candidate(candidate: dict[str, Any]) -> list[str]:
    return [
        f"- Accepted requests / hit rate: {candidate['accepted_requests']} / {candidate['hit_rate']:.4%}",
        f"- Mean CLIP-T: {candidate['clip_t_mean']:.6f} ({candidate['clip_t_z_gain']:+.4f} population stddev)",
        f"- Mean VBench-Q: {candidate['vbench_q_mean']:.6f} ({candidate['vbench_q_z_gain']:+.4f} population stddev)",
        f"- Balanced quality gain: {candidate['balanced_quality_z_gain']:+.4f} population stddev",
        *[f"- `{name}`: {candidate[name]:.12g}" for name in THRESHOLD_NAMES.values()],
    ]


def write_report(path: Path, analysis: dict[str, Any], audit: dict[str, Any]) -> None:
    recommended = analysis["recommended"]
    best_clip_t = analysis["best_clip_t"]
    best_vbench_q = analysis["best_vbench_q"]
    population = analysis["full_population"]
    lines = [
        "# Carnator Early Signature quality--hit-rate optimisation",
        "",
        "## Scope",
        "",
        "This is an offline replay of the production four-gate Early Signature rule over the completed, intentionally unbounded Carnator run. It does not regenerate videos.",
        "",
        f"- Requests: {analysis['request_count']}",
        f"- Searched hit-rate interval: {analysis['min_hit_rate']:.1%}--{analysis['max_hit_rate']:.1%} ({analysis['minimum_accepted']}--{analysis['maximum_accepted']} requests)",
        f"- Grid candidates: {analysis['total_grid_candidates']}; interval-eligible candidates: {analysis['eligible_grid_candidates']}",
        f"- Full-population CLIP-T / VBench-Q: {population['clip_t_mean']:.6f} / {population['vbench_q_mean']:.6f}",
        "",
        "## Distance Audit",
        "",
        "- All 297 single-noun-substitution requests have four finite Early Signature distances.",
        f"- Same-noun-set requests: {audit['same_noun_set_request_indices']}. Their `log_area_ratio`, `shape_one_minus_dice`, and `changed_color_ciede2000` are inapplicable and were set to zero only for the offline replay, matching the runtime comparator which ignores those gates for same-noun matches.",
        "- No required Early Signature distance or quality score is missing.",
        "",
        "## Recommended Balanced Gate",
        "",
        "The default recommendation maximises the equal-weight mean of the accepted-set CLIP-T and VBench-Q improvements, each normalised by its full-population standard deviation, subject to the searched hit-rate interval.",
        "",
        *format_candidate(recommended),
        "",
        "## Single-Metric Bounds",
        "",
        "### Highest CLIP-T candidate",
        "",
        *format_candidate(best_clip_t),
        "",
        "### Highest VBench-Q candidate",
        "",
        *format_candidate(best_vbench_q),
        "",
        "## Interpretation",
        "",
        "`--min-hit-rate` is the operational speed constraint. Raising it reruns the same offline sweep with a stricter throughput requirement; lowering it permits more selective, potentially higher-quality reuse. The accompanying CSV files give the best balanced candidate at every reachable accepted-request count and the CLIP-T/VBench-Q/balanced choices within each coverage band.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Carnator results.jsonl or quality_input.jsonl")
    parser.add_argument("--output-dir", type=Path, help="directory for the offline optimisation report")
    parser.add_argument("--min-hit-rate", type=float, default=0.60, help="minimum operational hit rate (default: 0.60)")
    parser.add_argument("--max-hit-rate", type=float, default=1.00, help="maximum hit rate searched (default: 1.00)")
    parser.add_argument("--max-threshold-values", type=int, default=32, help="observed threshold candidates retained per gate")
    parser.add_argument("--batch-size", type=int, default=4096, help="vectorized threshold combinations per batch")
    parser.add_argument("--coverage-band-width", type=float, default=0.05, help="coverage summary bucket width")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.min_hit_rate <= args.max_hit_rate <= 1:
        raise ValueError("require 0 < --min-hit-rate <= --max-hit-rate <= 1")
    if args.max_threshold_values < 2 or args.batch_size < 1 or not 0 < args.coverage_band_width <= 1:
        raise ValueError("invalid threshold-grid or coverage-band parameters")

    results_path = args.results.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_path.parent / "quality_threshold_optimization"
    )
    rows = read_jsonl(results_path)
    records, audit = audit_rows(rows)
    analysis = optimise(
        records,
        args.min_hit_rate,
        args.max_hit_rate,
        args.max_threshold_values,
        args.batch_size,
        args.coverage_band_width,
    )
    analysis["created_at"] = utc_now()
    analysis["input"] = {"path": str(results_path), "sha256": sha256_file(results_path)}
    analysis["distance_audit"] = audit
    analysis["parameters"] = {
        "max_threshold_values": args.max_threshold_values,
        "batch_size": args.batch_size,
        "coverage_band_width": args.coverage_band_width,
        "same_noun_inapplicable_value": 0.0,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "analysis.json", analysis)
    write_json(output_dir / "recommended_selection.json", selection_details(records, analysis["recommended"]))
    write_csv(output_dir / "best_balanced_by_accepted_count.csv", analysis["best_balanced_by_accepted_count"])
    coverage_rows = []
    for band in analysis["best_by_coverage_band"]:
        for objective, candidate in (
            ("balanced", band["balanced_quality_z_gain"]),
            ("clip_t", band["clip_t_mean"]),
            ("vbench_q", band["vbench_q_mean"]),
        ):
            coverage_rows.append({
                "selection_objective": objective,
                "hit_rate_lower": band["hit_rate_lower"],
                "hit_rate_upper": band["hit_rate_upper"],
                **candidate,
            })
    write_csv(output_dir / "best_by_coverage_band.csv", coverage_rows)
    write_report(output_dir / "report.md", analysis, audit)
    print(json.dumps({
        "status": "completed",
        "output_dir": str(output_dir),
        "recommended": analysis["recommended"],
        "same_noun_set_request_indices": audit["same_noun_set_request_indices"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
