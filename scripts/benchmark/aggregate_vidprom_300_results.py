#!/usr/bin/env python3
"""Aggregate VidProM runner JSONL files and produce audit tables/curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("nirvana_vid", "teacache", "chorus", "chorus2", "carnator_boundary_b5")
DISPLAY = {
    "nirvana_vid": "NIRVANA-VID",
    "teacache": "TeaCache",
    "chorus": "Chorus",
    "chorus2": "Chorus2",
    "carnator_boundary_b5": "Carnator Boundary B5",
}


def rows_for(path: Path) -> list[dict[str, Any]]:
    # quality_runner rewrites results.jsonl with attached metrics while the
    # experiment's immutable audit log remains requests.jsonl.
    scored = path.with_name("results.jsonl")
    source = scored if scored.is_file() else path
    if not source.is_file():
        raise FileNotFoundError(source)
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]


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
    gpu = {"nirvana_vid": 3, "teacache": 4, "chorus": 5, "chorus2": 6, "carnator_boundary_b5": 7}[method]
    config_path = method_root / "run_config.json"
    if config_path.is_file():
        try:
            gpu = int(json.loads(config_path.read_text(encoding="utf-8")).get("gpu", gpu))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return {
        "baseline": DISPLAY[method], "method": method, "gpu": gpu,
        "requests": len(rows), "completed": len(completed), "errors": len(errors), "successful": len(ok),
        "hits": "N/A" if method == "teacache" else hits,
        "hit_rate": "N/A" if method == "teacache" else (hits / len(completed) if completed else None),
        "e2e_s_per_request": mean(latency) if latency else None,
        "hit_timed_requests": len(hit_latency),
        "hit_e2e_s_per_hit": mean(hit_latency) if hit_latency else None,
        "denoising_s_per_request": mean(denoise) if denoise else None,
        "hit_denoising_s_per_hit": mean(hit_denoise) if hit_denoise else None,
        "external_s_per_request": mean(external) if external else 0.0,
        "clip_t": mean(clip) if clip else None, "vbench_q": mean(vbench) if vbench else None,
    }


def write_hit_rates(root: Path, all_rows: dict[str, list[dict[str, Any]]]) -> None:
    path = root / "hit_rate_over_arrivals.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("baseline", "arrival_count", "window_start", "window_end", "window_hits", "window_hit_rate", "cumulative_hits", "cumulative_hit_rate"))
        writer.writeheader()
        for method, rows in all_rows.items():
            if method == "teacache":
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
                window_hits = sum(bool(row.get("hit", row.get("cache_match", {}).get("hit"))) for row in window)
                cumulative_hits = sum(bool(row.get("hit", row.get("cache_match", {}).get("hit"))) for row in cumulative)
                writer.writerow({"baseline": DISPLAY[method], "arrival_count": end, "window_start": start + 1, "window_end": end, "window_hits": window_hits, "window_hit_rate": window_hits / len(window), "cumulative_hits": cumulative_hits, "cumulative_hit_rate": cumulative_hits / len(cumulative)})


def write_plot(root: Path, all_rows: dict[str, list[dict[str, Any]]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for method, rows in all_rows.items():
        if method == "teacache":
            continue
        xs, ys = [], []
        endpoints = list(range(50, len(rows) + 1, 50))
        if rows and (not endpoints or endpoints[-1] != len(rows)):
            endpoints.append(len(rows))
        for end in endpoints:
            hits = sum(bool(row.get("hit", row.get("cache_match", {}).get("hit"))) for row in rows[:end])
            xs.append(end); ys.append(hits / end)
        if xs:
            axis.plot(xs, ys, marker="o", label=DISPLAY[method])
    axis.set(xlabel="Arrived requests", ylabel="Cumulative hit rate", ylim=(0, 1))
    axis.grid(True, alpha=0.25); axis.legend(); figure.tight_layout()
    figure.savefig(root / "hit_rate_over_arrivals.png", dpi=160); plt.close(figure)


def write_results_table(root: Path, summaries: list[dict[str, Any]]) -> None:
    request_count = max((int(item.get("requests", 0)) for item in summaries), default=0)
    lines = [f"# VidProM results ({request_count} request{'s' if request_count != 1 else ''})", "", "Denoising is CUDA-synchronized DiT forward time; Carnator also includes its B5 probe. Source cache preparation is outside E2E and denoising. External is Chorus-only LLM + Grounded-SAM2 time and remains in E2E. Warm-up is unscored. Misses are recorded without generation; timing and quality use generated requests only. Errors are terminal audit records, excluded from hit-rate denominators and quality scoring.", "", "| Baseline | GPU | Requests | Completed | Errors | Hits | Hit rate | E2E s/request | Hit E2E s/hit | Denoising s/request | Hit denoising s/hit | External s/request | CLIP-T | VBench-Q |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in summaries:
        def fmt(value: Any) -> str:
            return "N/A" if value in (None, "N/A") else f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append("| {baseline} | {gpu} | {requests} | {completed} | {errors} | {hits} | {hit_rate} | {e2e_s_per_request} | {hit_e2e_s_per_hit} | {denoising_s_per_request} | {hit_denoising_s_per_hit} | {external_s_per_request} | {clip_t} | {vbench_q} |".format(**{key: fmt(value) for key, value in item.items()}))
    (root / "VIDPROM_1000CACHE_300INPUT_RESULTS_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "artifacts" / "vidprom_1000cache_300input_noun_match" / "results")
    args = parser.parse_args()
    root = args.results_root.expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    all_rows = {method: rows_for(root / method / "requests.jsonl") for method in METHODS}
    summaries = [aggregate_method(method, rows, root / method) for method, rows in all_rows.items()]
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    write_hit_rates(root, all_rows); write_plot(root, all_rows); write_results_table(root, summaries)
    print(json.dumps({"summary": str(root / "summary.csv"), "hit_rates": str(root / "hit_rate_over_arrivals.csv"), "table": str(root / "VIDPROM_1000CACHE_300INPUT_RESULTS_TABLE.md")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
