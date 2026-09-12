#!/usr/bin/env python3
"""Aggregate VidProM runner JSONL files and produce audit tables/curves."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODEL_METHODS = {
    "wan22": ("nirvana_vid", "teacache", "chorus", "chorus2", "carnator_boundary_b5"),
    "wan21": ("baseline", "nirvana_vid", "chorus", "chorus2", "carnator_boundary_b5"),
    "ltxvideo": ("baseline", "nirvana_vid", "chorus", "chorus2", "carnator_boundary_b5"),
}
# Methods without a cross-request cache: hit metrics are meaningless.
NO_HIT_METHODS = {"teacache", "baseline"}
DISPLAY = {
    "baseline": "Dense Baseline",
    "nirvana_vid": "NIRVANA-VID",
    "teacache": "TeaCache",
    "chorus": "Chorus",
    "chorus2": "Chorus2",
    "carnator_boundary_b5": "Carnator Boundary B5",
}


class AggregationError(RuntimeError):
    """The artifacts on disk cannot be summarized without misreporting them."""


def _describe_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ", timespec="seconds")


def _reject_stub_rows(rows: list[dict[str, Any]], scored: Path) -> None:
    """Reject dry-run / synthetic products that carry no real generation cost.

    A ``--dry-run`` writes the same statuses as a real run but every timing is a
    hash-derived placeholder, so the means it produces are meaningless and the
    speedups derived from them point the wrong way.  Such rows have no denoising
    time at all and complete far faster than any real diffusion step.
    """
    ok = [row for row in rows if row.get("status") == "ok"]
    if not ok:
        return
    denoising = [float(row.get("denoising_s") or 0.0) for row in ok]
    latency = [float(row.get("latency_s") or 0.0) for row in ok]
    if all(value == 0.0 for value in denoising) and max(latency) < 1.0:
        raise AggregationError(
            f"dry-run/stub results detected: {scored}\n"
            f"  all {len(ok)} 'ok' rows have denoising_s == 0 and max(latency_s) = {max(latency):.3g}s (< 1.0);\n"
            f"  these come from a --dry-run (adapters/synthetic.py) and were never re-generated.\n"
            f"  re-run the method without --dry-run."
        )


def rows_for(path: Path) -> list[dict[str, Any]]:
    # quality_runner rewrites results.jsonl with attached metrics while the
    # experiment's immutable audit log remains requests.jsonl.  Only the former
    # carries CLIP-T / VBench-Q, and it is written once the method has run to
    # completion, so a missing or stale copy means the run is not summarizable.
    scored = path.with_name("results.jsonl")
    if not scored.is_file():
        raise AggregationError(
            f"missing scored results: {scored}\n"
            f"  run_vidprom_300_experiment.py copies requests.jsonl to results.jsonl once the method\n"
            f"  finishes and quality_runner then attaches CLIP-T / VBench-Q.  The raw audit log\n"
            f"  ({path}) is appended during the run and has neither, so it is not a substitute.\n"
            f"  re-run the method instead of aggregating a partial log."
        )
    if path.is_file() and scored.stat().st_mtime < path.stat().st_mtime:
        raise AggregationError(
            f"stale scored results: {scored}\n"
            f"  mtime {_describe_mtime(scored)} is older than {path} mtime {_describe_mtime(path)}.\n"
            f"  results.jsonl is derived from requests.jsonl, so this predates the last run and\n"
            f"  still holds an earlier attempt's rows.  --overwrite does not clear it.\n"
            f"  delete it (or re-run the method) before aggregating."
        )
    rows = [json.loads(line) for line in scored.read_text(encoding="utf-8").splitlines() if line]
    _reject_stub_rows(rows, scored)
    return rows


def validate_quality_completeness(
    method: str, rows: list[dict[str, Any]], scored: Path, *, allow_incomplete: bool = False
) -> None:
    """Fail when a method's successful rows are missing CLIP-T or VBench-Q.

    ``aggregate_method()`` reads both columns row by row and simply skips absent
    values, so a method whose quality stage failed end to end (every row missing
    both) or partially (some rows missing) turns into a blank or deflated cell
    that reads like "not measured" rather than "broken".  Checked for every
    method: NO_HIT_METHODS have no cross-request cache, which is unrelated to
    whether quality was evaluated.
    """
    ok = [row for row in rows if row.get("status") == "ok"]
    if not ok:
        message = (
            f"{DISPLAY[method]}: no successful requests in {scored} — nothing was generated, "
            f"so there is no run to summarize"
        )
    else:
        with_clip = sum(1 for row in ok if row.get("clip_t") is not None)
        with_vbench = sum(1 for row in ok if row.get("vbench_q") is not None)
        if with_clip == len(ok) and with_vbench == len(ok):
            return
        message = (
            f"{DISPLAY[method]}: incomplete quality metrics in {scored} — "
            f"clip_t {with_clip}/{len(ok)}, vbench_q {with_vbench}/{len(ok)} successful rows\n"
            f"  the quality stage either failed (check for an OOM in the method's log) or was skipped"
        )
    if allow_incomplete:
        print(f"warning: {message}", file=sys.stderr)
        return
    raise AggregationError(message + "\n  pass --allow-incomplete-quality to summarize anyway.")


def aggregate_method(method: str, rows: list[dict[str, Any]], method_root: Path) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    completed = [row for row in rows if row.get("status") in {"ok", "cache_miss"}]
    errors = [row for row in rows if row.get("status") == "error"]
    hit_rows = [
        row for row in completed
        if bool(row.get("hit", row.get("cache_match", {}).get("hit")))
    ]
    hits = len(hit_rows)
    denoise = [float(row.get("denoising_s", 0.0)) for row in ok if row.get("denoising_s") is not None]
    latency = [float(row.get("latency_s", 0.0)) for row in ok]
    hit_latency = [
        float(row.get("hit_e2e_s") if row.get("hit_e2e_s") is not None else row.get("latency_s", 0.0))
        for row in hit_rows
    ]
    hit_denoise = [
        float(row.get("hit_denoising_s") if row.get("hit_denoising_s") is not None else row.get("denoising_s", 0.0))
        for row in hit_rows
    ]
    external = [float(row.get("external_s", 0.0)) for row in ok]
    clip = [float(row["clip_t"]) for row in ok if row.get("clip_t") is not None]
    vbench = [float(row["vbench_q"]) for row in ok if row.get("vbench_q") is not None]
    gpu = {"baseline": 3, "nirvana_vid": 3, "teacache": 4, "chorus": 5, "chorus2": 6,
           "carnator_boundary_b5": 7}[method]
    config_path = method_root / "run_config.json"
    if config_path.is_file():
        try:
            gpu = int(json.loads(config_path.read_text(encoding="utf-8")).get("gpu", gpu))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return {
        "baseline": DISPLAY[method], "method": method, "gpu": gpu,
        "requests": len(rows), "completed": len(completed), "errors": len(errors), "successful": len(ok),
        "hits": "N/A" if method in NO_HIT_METHODS else hits,
        "hit_rate": "N/A" if method in NO_HIT_METHODS else (hits / len(completed) if completed else None),
        "e2e_s_per_request": mean(latency) if latency else None,
        "hit_timed_requests": len(hit_latency),
        "hit_e2e_s_per_hit": mean(hit_latency) if hit_latency else None,
        "denoising_s_per_request": mean(denoise) if denoise else None,
        "hit_denoising_s_per_hit": mean(hit_denoise) if hit_denoise else None,
        "external_s_per_request": mean(external) if external else 0.0,
        "clip_t": mean(clip) if clip else None, "vbench_q": mean(vbench) if vbench else None,
    }


def attach_speedups(summaries: list[dict[str, Any]]) -> None:
    """Annotate each method with speedups relative to the dense baseline."""

    dense = next((item for item in summaries if item["method"] == "baseline"), None)

    def speedup(base: Any, value: Any) -> Any:
        if base in (None, "N/A") or value in (None, "N/A"):
            return "N/A"
        return float(base) / float(value) if value else None

    for item in summaries:
        if dense is None or item is dense:
            item.update({"e2e_speedup": "N/A", "denoising_speedup": "N/A",
                         "hit_e2e_speedup": "N/A", "hit_denoising_speedup": "N/A"})
            continue
        item.update({
            "e2e_speedup": speedup(dense["e2e_s_per_request"], item["e2e_s_per_request"]),
            "denoising_speedup": speedup(dense["denoising_s_per_request"], item["denoising_s_per_request"]),
            "hit_e2e_speedup": speedup(dense["e2e_s_per_request"], item["hit_e2e_s_per_hit"]),
            "hit_denoising_speedup": speedup(dense["denoising_s_per_request"], item["hit_denoising_s_per_hit"]),
        })


def write_hit_rates(root: Path, all_rows: dict[str, list[dict[str, Any]]]) -> None:
    path = root / "hit_rate_over_arrivals.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("baseline", "arrival_count", "window_start", "window_end", "window_hits", "window_hit_rate", "cumulative_hits", "cumulative_hit_rate"))
        writer.writeheader()
        for method, rows in all_rows.items():
            if method in NO_HIT_METHODS:
                for arrival in range(50, max(50, len(rows)) + 1, 50):
                    writer.writerow({"baseline": DISPLAY[method], "arrival_count": min(arrival, len(rows)), "window_start": "N/A", "window_end": "N/A", "window_hits": "N/A", "window_hit_rate": "N/A", "cumulative_hits": "N/A", "cumulative_hit_rate": "N/A"})
                continue
            endpoints = list(range(50, len(rows) + 1, 50))
            if rows and (not endpoints or endpoints[-1] != len(rows)):
                endpoints.append(len(rows))
            for end in endpoints:
                start = max(0, end - 50)
                window = rows[start:end]
                cumulative = rows[:end]
                # Errors are terminal audit records but are excluded from hit
                # rate denominators. Keep arrival_count on the original
                # request axis while calculating rates over completed rows.
                completed_window = [row for row in window if row.get("status") in {"ok", "cache_miss"}]
                completed_cumulative = [row for row in cumulative if row.get("status") in {"ok", "cache_miss"}]
                window_hits = sum(bool(row.get("hit", row.get("cache_match", {}).get("hit"))) for row in completed_window)
                cumulative_hits = sum(bool(row.get("hit", row.get("cache_match", {}).get("hit"))) for row in completed_cumulative)
                writer.writerow({"baseline": DISPLAY[method], "arrival_count": end, "window_start": start + 1, "window_end": end, "window_hits": window_hits, "window_hit_rate": window_hits / len(completed_window) if completed_window else "N/A", "cumulative_hits": cumulative_hits, "cumulative_hit_rate": cumulative_hits / len(completed_cumulative) if completed_cumulative else "N/A"})


def write_plot(root: Path, all_rows: dict[str, list[dict[str, Any]]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for method, rows in all_rows.items():
        if method in NO_HIT_METHODS:
            continue
        xs, ys = [], []
        endpoints = list(range(50, len(rows) + 1, 50))
        if rows and (not endpoints or endpoints[-1] != len(rows)):
            endpoints.append(len(rows))
        for end in endpoints:
            completed = [row for row in rows[:end] if row.get("status") in {"ok", "cache_miss"}]
            hits = sum(bool(row.get("hit", row.get("cache_match", {}).get("hit"))) for row in completed)
            xs.append(end); ys.append(hits / len(completed) if completed else 0.0)
        if xs:
            axis.plot(xs, ys, marker="o", label=DISPLAY[method])
    axis.set(xlabel="Arrived requests", ylabel="Cumulative hit rate", ylim=(0, 1))
    axis.grid(True, alpha=0.25); axis.legend(); figure.tight_layout()
    figure.savefig(root / "hit_rate_over_arrivals.png", dpi=160); plt.close(figure)


def write_results_table(root: Path, summaries: list[dict[str, Any]]) -> None:
    request_count = max((int(item.get("requests", 0)) for item in summaries), default=0)
    lines = [f"# VidProM results ({request_count} request{'s' if request_count != 1 else ''})", "", "Denoising is CUDA-synchronized DiT forward time; Carnator also includes its B5 probe. Source cache preparation is outside E2E and denoising. External is Chorus-only LLM + Grounded-SAM2 time and remains in E2E. Warm-up is unscored. Misses are recorded without generation; timing and quality use generated requests only. Errors are terminal audit records, excluded from hit-rate denominators and quality scoring. Speedups divide the dense-baseline mean (per-request over generated requests; hit columns compare the method's hit subset against the baseline mean).", "", "| Baseline | GPU | Requests | Completed | Errors | Hits | Hit rate | E2E s/request | E2E speedup | Denoising s/request | Denoising speedup | Hit E2E s/hit | Hit E2E speedup | Hit denoising s/hit | Hit denoising speedup | External s/request | CLIP-T | VBench-Q |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in summaries:
        def fmt(value: Any) -> str:
            return "N/A" if value in (None, "N/A") else f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append("| {baseline} | {gpu} | {requests} | {completed} | {errors} | {hits} | {hit_rate} | {e2e_s_per_request} | {e2e_speedup} | {denoising_s_per_request} | {denoising_speedup} | {hit_e2e_s_per_hit} | {hit_e2e_speedup} | {hit_denoising_s_per_hit} | {hit_denoising_speedup} | {external_s_per_request} | {clip_t} | {vbench_q} |".format(**{key: fmt(value) for key, value in item.items()}))
    (root / "VIDPROM_1000CACHE_300INPUT_RESULTS_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_METHODS), default="wan22",
                        help="Select the method set that matches the experiment runs.")
    parser.add_argument("--results-root", type=Path, default=ROOT / "artifacts" / "vidprom_1000cache_300input_noun_match" / "results")
    parser.add_argument(
        "--allow-incomplete-quality",
        action="store_true",
        help="Warn instead of failing when a method is missing CLIP-T / VBench-Q (e.g. --skip-quality runs).",
    )
    args = parser.parse_args()
    root = args.results_root.expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    try:
        all_rows = {method: rows_for(root / method / "requests.jsonl") for method in MODEL_METHODS[args.model]}
        # Validate every method before writing anything, so a rejected run leaves
        # no summary behind for someone to read as a real result.
        for method, rows in all_rows.items():
            validate_quality_completeness(
                method, rows, root / method / "results.jsonl", allow_incomplete=args.allow_incomplete_quality
            )
    except AggregationError as error:
        print(f"aggregation refused: {error}", file=sys.stderr)
        return 1
    summaries = [aggregate_method(method, rows, root / method) for method, rows in all_rows.items()]
    attach_speedups(summaries)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    write_hit_rates(root, all_rows); write_plot(root, all_rows); write_results_table(root, summaries)
    print(json.dumps({"summary": str(root / "summary.csv"), "hit_rates": str(root / "hit_rate_over_arrivals.csv"), "table": str(root / "VIDPROM_1000CACHE_300INPUT_RESULTS_TABLE.md")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
