from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy.stats import beta

from carnator.early_signature import compare_early_signatures


DISTANCE_FIELDS = (
    "context_color_ciede2000",
    "log_area_ratio",
    "shape_one_minus_dice",
    "changed_color_ciede2000",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "pair_id" not in rows[0]:
        raise ValueError(f"pairs CSV is invalid: {path}")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError(f"pairs CSV has duplicate pair IDs: {path}")
    return rows


def quality_root(run_root: Path) -> Path:
    return run_root / "quality"


def source_signature_path(run_root: Path, pair_id: str) -> Path:
    return run_root / "sources" / pair_id / "cache" / "source_dense" / "online_color_signature.json"


def target_signature_path(run_root: Path, pair_id: str) -> Path:
    return run_root / "targets" / pair_id / "cache" / "target_latent_reuse" / "online_color_signature.json"


def noun_substitution(source_path: Path, target_path: Path) -> tuple[str, str]:
    source_objects = {str(item["noun"]) for item in read_json(source_path).get("objects", [])}
    target_objects = {str(item["noun"]) for item in read_json(target_path).get("objects", [])}
    target_only = target_objects - source_objects
    source_only = source_objects - target_objects
    if len(target_only) != 1 or len(source_only) != 1:
        raise RuntimeError(
            f"expected exactly one noun substitution for source={source_path} target={target_path}; "
            f"source_only={sorted(source_only)} target_only={sorted(target_only)}"
        )
    return next(iter(source_only)), next(iter(target_only))


def compare_pair(run_root: Path, pair_id: str) -> dict[str, Any]:
    source_path = source_signature_path(run_root, pair_id)
    target_path = target_signature_path(run_root, pair_id)
    if not source_path.is_file() or not target_path.is_file():
        raise FileNotFoundError(f"missing signature sidecar for {pair_id}")
    source_only, target_only = noun_substitution(source_path, target_path)
    comparison = compare_early_signatures(
        {"signature_path": str(target_path)},
        {"signature_path": str(source_path)},
        query_changed_noun=target_only,
        candidate_changed_noun=source_only,
        max_context_color_distance=float("inf"),
        max_log_area_ratio=float("inf"),
        max_shape_distance=float("inf"),
        max_changed_color_distance=float("inf"),
        check_cache_artifacts=True,
    )
    if not comparison["metadata_compatible"] or not comparison["cache_ready"] or not comparison["signature_compatible"]:
        raise RuntimeError(f"unbounded signature comparison failed for {pair_id}: {comparison['reason']}")
    distances = comparison["distances"]
    if any(not isinstance(distances.get(field), (int, float)) or not math.isfinite(float(distances[field])) for field in DISTANCE_FIELDS):
        raise RuntimeError(f"invalid signature distances for {pair_id}: {distances}")
    return {
        "source_signature_path": str(source_path),
        "target_signature_path": str(target_path),
        "source_only_noun": source_only,
        "target_only_noun": target_only,
        "signature_comparison": comparison,
        "gate_distances": {field: float(distances[field]) for field in DISTANCE_FIELDS},
    }


def clopper_pearson(k: int, n: int, confidence: float) -> dict[str, float]:
    if n <= 0 or not 0 <= k <= n or not 0 < confidence < 1:
        raise ValueError(f"invalid binomial interval inputs: k={k}, n={n}, confidence={confidence}")
    alpha = 1.0 - confidence
    one_sided_lower = 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))
    two_sided_lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    two_sided_upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return {
        "success_rate": k / n,
        "one_sided_lower": one_sided_lower,
        "two_sided_lower": two_sided_lower,
        "two_sided_upper": two_sided_upper,
    }


def threshold_values(values: list[float], minimum_accepted: int) -> list[float]:
    ordered = sorted(values)
    if not 1 <= minimum_accepted <= len(ordered):
        raise ValueError("invalid minimum accepted count")
    return sorted(set(ordered[minimum_accepted - 1 :]))


def threshold_name(field: str) -> str:
    return {
        "context_color_ciede2000": "max_context_color_distance",
        "log_area_ratio": "max_log_area_ratio",
        "shape_one_minus_dice": "max_shape_distance",
        "changed_color_ciede2000": "max_changed_color_distance",
    }[field]


def evaluate_thresholds(rows: list[dict[str, Any]], thresholds: dict[str, float], confidence: float) -> dict[str, Any]:
    accepted: list[str] = []
    rejected: dict[str, list[str]] = {}
    for row in rows:
        failed = [field for field in DISTANCE_FIELDS if row["gate_distances"][field] > thresholds[field]]
        if failed:
            rejected[row["pair_id"]] = failed
        else:
            accepted.append(row["pair_id"])
    accepted_set = set(accepted)
    k = sum(1 for row in rows if row["pair_id"] in accepted_set and row["quality_pass"]["success"])
    interval = clopper_pearson(k, len(accepted), confidence)
    rejection_counts = {field: sum(field in failures for failures in rejected.values()) for field in DISTANCE_FIELDS}
    return {
        "thresholds": thresholds,
        "accepted_pair_ids": accepted,
        "rejected_pair_failures": rejected,
        "rejection_counts": rejection_counts,
        "n": len(accepted),
        "k": k,
        "coverage": len(accepted) / len(rows),
        **interval,
    }


def serialize_sweep_row(record: dict[str, Any], minimum_coverage: float, selection_target: float | None) -> dict[str, Any]:
    thresholds = record["thresholds"]
    return {
        **{threshold_name(field): f"{thresholds[field]:.12g}" for field in DISTANCE_FIELDS},
        "n": record["n"],
        "k": record["k"],
        "coverage": f"{record['coverage']:.12g}",
        "success_rate": f"{record['success_rate']:.12g}",
        "one_sided_cp_lower": f"{record['one_sided_lower']:.12g}",
        "two_sided_cp_lower": f"{record['two_sided_lower']:.12g}",
        "two_sided_cp_upper": f"{record['two_sided_upper']:.12g}",
        "coverage_eligible": record["coverage"] >= minimum_coverage,
        "selection_eligible": selection_target is not None and record["coverage"] >= minimum_coverage and record["one_sided_lower"] >= selection_target,
        "accepted_pair_ids_json": json.dumps(record["accepted_pair_ids"], ensure_ascii=False),
        "rejected_pair_failures_json": json.dumps(record["rejected_pair_failures"], ensure_ascii=False, sort_keys=True),
        "rejection_counts_json": json.dumps(record["rejection_counts"], ensure_ascii=False, sort_keys=True),
    }


def write_report(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]], sweep: list[dict[str, Any]], minimum_accepted: int, selection_target: float | None) -> None:
    quality = manifest["quality_thresholds"]
    coverage_eligible = [record for record in sweep if record["coverage"] >= quality["coverage_min"]]
    best_lower_value = max(record["one_sided_lower"] for record in coverage_eligible)
    best_lower_ties = [record for record in coverage_eligible if math.isclose(record["one_sided_lower"], best_lower_value, rel_tol=0.0, abs_tol=1e-12)]
    best_lower = min(best_lower_ties, key=lambda item: tuple(item["thresholds"][field] for field in DISTANCE_FIELDS))
    max_successes = max(record["k"] for record in coverage_eligible)
    target_text = "not pre-registered" if selection_target is None else f"{selection_target:.4f}"
    lines = [
        "# Carnator Early Signature calibration report",
        "",
        "## Frozen Inputs",
        "",
        f"- Pair count: {len(rows)}",
        f"- Q_min: {quality['q_min']:.4f}",
        f"- CLIP-T_min: {quality['clip_t_min']:.4f}",
        f"- C_min: {quality['coverage_min']:.4f} ({minimum_accepted}/{len(rows)})",
        f"- Confidence: {quality['confidence']:.4f}",
        f"- Required one-sided CP lower bound: {target_text}",
        "",
        "## Quality Success",
        "",
        f"- VBench-Q mean: {statistics.fmean(row['vbench_q_i'] for row in rows):.6f}",
        f"- CLIP-T mean: {statistics.fmean(row['clip_t'] for row in rows):.6f}",
        f"- Quality-success pairs: {sum(row['quality_pass']['success'] for row in rows)}/{len(rows)}",
        "",
        "## Gate Sweep",
        "",
        f"- Evaluated threshold tuples: {len(sweep)}",
        f"- Coverage-eligible tuples: {len(coverage_eligible)}",
        f"- Largest quality-success count among coverage-eligible tuples: {max_successes}",
        f"- Best one-sided {quality['confidence']:.0%} CP lower bound at coverage >= C_min: {best_lower['one_sided_lower']:.6f} ({best_lower['k']}/{best_lower['n']})",
        f"- Tied tuples at that lower bound: {len(best_lower_ties)}",
        "",
        "One deterministic diagnostic member of the tied maximum (not a selected gate):",
        "",
        "| threshold | value |",
        "| --- | ---: |",
        *[f"| {threshold_name(field)} | {best_lower['thresholds'][field]:.6f} |" for field in DISTANCE_FIELDS],
        "",
    ]
    if selection_target is None:
        lines.extend(
            [
                "## Selection Status",
                "",
                "No formal gate is selected. The pre-generation manifest did not freeze a required one-sided CP lower-bound target; choosing one after observing this sweep would invalidate the planned selection rule.",
            ]
        )
    else:
        qualified = [record for record in coverage_eligible if record["one_sided_lower"] >= selection_target]
        lines.extend(
            [
                "## Selection Status",
                "",
                f"Qualified tuples: {len(qualified)}",
                "",
            ]
        )
        if not qualified:
            lines.append("No threshold tuple meets the frozen coverage and CP lower-bound criteria.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run the offline Early Signature gate sweep over completed calibration artifacts.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--selection-min-lower-bound", type=float)
    args = parser.parse_args()
    if args.selection_min_lower_bound is not None and not 0 <= args.selection_min_lower_bound <= 1:
        parser.error("--selection-min-lower-bound must be in [0, 1]")

    manifest_path = args.run_root / "experiment_manifest.json"
    manifest = read_json(manifest_path)
    quality_thresholds = manifest.get("quality_thresholds")
    if not isinstance(quality_thresholds, dict):
        raise RuntimeError(f"missing frozen quality thresholds: {manifest_path}")
    q_min = float(quality_thresholds["q_min"])
    clip_t_min = float(quality_thresholds["clip_t_min"])
    coverage_min = float(quality_thresholds["coverage_min"])
    confidence = float(quality_thresholds["confidence"])
    if not 0 < coverage_min <= 1 or not 0 < confidence < 1:
        raise RuntimeError("invalid frozen coverage/confidence")

    pairs = load_pairs(args.pairs)
    quality_path = quality_root(args.run_root) / "pair_results.jsonl"
    quality_rows = [json.loads(line) for line in quality_path.read_text(encoding="utf-8").splitlines() if line]
    quality_by_pair = {row["pair_id"]: row for row in quality_rows}
    pair_ids = {row["pair_id"] for row in pairs}
    if pair_ids != set(quality_by_pair) or len(quality_rows) != len(quality_by_pair):
        raise RuntimeError("pair_results.jsonl does not provide exactly one quality record per frozen pair")

    enriched_rows: list[dict[str, Any]] = []
    for request_index, pair in enumerate(pairs):
        row = dict(quality_by_pair[pair["pair_id"]])
        if row.get("request_index") != request_index:
            raise RuntimeError(f"request_index mismatch for {pair['pair_id']}")
        row.update(compare_pair(args.run_root, pair["pair_id"]))
        q_pass = float(row["vbench_q_i"]) >= q_min
        clip_pass = float(row["clip_t"]) >= clip_t_min
        row["quality_pass"] = {
            "q_min": q_min,
            "clip_t_min": clip_t_min,
            "vbench_q_pass": q_pass,
            "clip_t_pass": clip_pass,
            "success": q_pass and clip_pass,
        }
        enriched_rows.append(row)
    write_jsonl(quality_path, enriched_rows)

    minimum_accepted = math.ceil(len(enriched_rows) * coverage_min)
    threshold_lists = {
        field: threshold_values([row["gate_distances"][field] for row in enriched_rows], minimum_accepted)
        for field in DISTANCE_FIELDS
    }
    sweep = [
        evaluate_thresholds(enriched_rows, dict(zip(DISTANCE_FIELDS, values)), confidence)
        for values in itertools.product(*(threshold_lists[field] for field in DISTANCE_FIELDS))
    ]
    quality_dir = quality_root(args.run_root)
    detail_rows = [
        {
            "thresholds": record["thresholds"],
            "n": record["n"],
            "k": record["k"],
            "coverage": record["coverage"],
            "success_rate": record["success_rate"],
            "one_sided_cp_lower": record["one_sided_lower"],
            "two_sided_cp_lower": record["two_sided_lower"],
            "two_sided_cp_upper": record["two_sided_upper"],
            "accepted_pair_ids": record["accepted_pair_ids"],
            "rejected_pair_failures": record["rejected_pair_failures"],
            "rejection_counts": record["rejection_counts"],
        }
        for record in sweep
    ]
    write_jsonl(quality_dir / "threshold_sweep_details.jsonl", detail_rows)
    csv_fields = [
        "max_context_color_distance",
        "max_log_area_ratio",
        "max_shape_distance",
        "max_changed_color_distance",
        "n",
        "k",
        "coverage",
        "success_rate",
        "one_sided_cp_lower",
        "two_sided_cp_lower",
        "two_sided_cp_upper",
        "coverage_eligible",
        "selection_eligible",
        "accepted_pair_ids_json",
        "rejected_pair_failures_json",
        "rejection_counts_json",
    ]
    write_csv(
        quality_dir / "threshold_sweep.csv",
        [serialize_sweep_row(record, coverage_min, args.selection_min_lower_bound) for record in sweep],
        csv_fields,
    )
    analysis_manifest = {
        "completed_at": utc_now(),
        "pair_count": len(enriched_rows),
        "quality_thresholds": quality_thresholds,
        "selection_min_lower_bound": args.selection_min_lower_bound,
        "selection_status": "withheld_missing_pre_registered_lower_bound" if args.selection_min_lower_bound is None else "evaluated",
        "minimum_accepted": minimum_accepted,
        "threshold_value_counts": {threshold_name(field): len(values) for field, values in threshold_lists.items()},
        "sweep_count": len(sweep),
        "wall_clock_seconds": time.perf_counter() - started,
    }
    write_json(quality_dir / "threshold_analysis_manifest.json", analysis_manifest)
    write_report(quality_dir / "report.md", manifest, enriched_rows, sweep, minimum_accepted, args.selection_min_lower_bound)
    print(
        json.dumps(
            {
                "status": "completed",
                "pair_count": len(enriched_rows),
                "minimum_accepted": minimum_accepted,
                "sweep_count": len(sweep),
                "selection_status": analysis_manifest["selection_status"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
