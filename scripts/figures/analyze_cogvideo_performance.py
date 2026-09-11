#!/usr/bin/env python3
"""Analyze CogVideoX 5-request test performance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def find_test_results():
    """Find all CogVideoX test results."""
    base_dir = ROOT / "artifacts" / "vidprom_1000cache_300input_noun_match"

    # Use the specific test directory mentioned in docs
    test_dir = base_dir / "early_signature_disabled_20260903_100305"
    if test_dir.exists():
        return test_dir

    if not base_dir.exists():
        return None

    # Look for the most recent test directory
    test_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()],
                      key=lambda x: x.stat().st_mtime, reverse=True)

    for test_dir in test_dirs:
        # Check if it has method subdirectories with results
        method_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
        if method_dirs:
            return test_dir

    return None


def analyze_method_results(method_dir: Path):
    """Analyze results for a single method."""
    records_file = method_dir / "request_records.jsonl"

    if not records_file.exists():
        return None

    records = []
    with open(records_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return None

    # Analyze first 5 records
    analysis = {
        "method": method_dir.name,
        "total_count": len(records),
        "first_5": []
    }

    for i, record in enumerate(records[:5]):
        req_analysis = {
            "index": i,
            "arrival_index": record.get("arrival_index", i),
            "status": record.get("status", "unknown"),
            "hit": record.get("hit", False),
        }

        # Timing metrics (handle different field names)
        if "hit_e2e_s" in record and record["hit_e2e_s"] is not None:
            req_analysis["e2e_seconds"] = round(record["hit_e2e_s"], 3)
        elif "e2e_seconds" in record and record["e2e_seconds"] is not None:
            req_analysis["e2e_seconds"] = round(record["e2e_seconds"], 3)
        elif "latency_s" in record and record["latency_s"] is not None:
            req_analysis["e2e_seconds"] = round(record["latency_s"], 3)

        if "dit_forward_s" in record and record["dit_forward_s"] is not None:
            req_analysis["dit_forward_s"] = round(record["dit_forward_s"], 3)

        if "hit_denoising_s" in record and record["hit_denoising_s"] is not None:
            req_analysis["denoising_seconds"] = round(record["hit_denoising_s"], 3)
        elif "denoising_s" in record and record["denoising_s"] is not None:
            req_analysis["denoising_seconds"] = round(record["denoising_s"], 3)
        elif "denoising_seconds" in record and record["denoising_seconds"] is not None:
            req_analysis["denoising_seconds"] = round(record["denoising_seconds"], 3)

        # Step information
        if "executed_steps" in record:
            req_analysis["executed_steps"] = record["executed_steps"]

        # Method-specific details
        extra = record.get("extra", {})

        if "chorus2" in extra:
            chorus2_info = extra["chorus2"]
            req_analysis["chorus2"] = {
                "mode": chorus2_info.get("mode"),
                "sparse_calls": chorus2_info.get("execution", {}).get("runs"),
                "avg_density": chorus2_info.get("execution", {}).get("average_runtime_density"),
                "refresh_events": chorus2_info.get("execution", {}).get("refresh_events"),
            }

        if "carnator" in extra:
            carnator_info = extra["carnator"]
            req_analysis["carnator"] = {
                "mode": carnator_info.get("mode"),
                "regional_calls": carnator_info.get("regional_attention", {}).get("calls"),
                "sparse_calls": carnator_info.get("outer_sparse", {}).get("runs"),
                "quantized_ffn": carnator_info.get("quantized_ffn", {}).get("enabled"),
            }

        if "chorus" in extra:
            chorus_info = extra["chorus"]
            req_analysis["chorus"] = {
                "mode": chorus_info.get("mode"),
                "regional_calls": chorus_info.get("regional_attention", {}).get("calls"),
                "tgaa_enabled": chorus_info.get("tgaa_enabled"),
            }

        if "teacache" in extra:
            teacache_info = extra["teacache"]
            req_analysis["teacache"] = {
                "actual_steps": teacache_info.get("actual_forward_count"),
                "skipped_steps": teacache_info.get("skipped_forward_count"),
                "skip_ratio": teacache_info.get("skip_ratio"),
            }

        if "nirvana_vid" in extra:
            nirvana_info = extra["nirvana_vid"]
            req_analysis["nirvana_vid"] = {
                "mode": nirvana_info.get("mode"),
                "reuse_step": nirvana_info.get("reuse_step"),
            }

        analysis["first_5"].append(req_analysis)

    return analysis


def print_analysis(analysis):
    """Pretty print the analysis."""
    print(f"\n{'='*80}")
    print(f"Method: {analysis['method'].upper()}")
    print(f"{'='*80}")
    print(f"Total requests in file: {analysis['total_count']}")
    print(f"\nFirst 5 requests analysis:")

    for req in analysis["first_5"]:
        print(f"\n  Request #{req['index']} (arrival: {req['arrival_index']})")
        print(f"    Status: {req['status']}")
        print(f"    Cache Hit: {req['hit']}")

        if "e2e_seconds" in req:
            print(f"    E2E Time: {req['e2e_seconds']}s")
        if "dit_forward_s" in req:
            print(f"    DiT Time: {req['dit_forward_s']}s")
        if "denoising_seconds" in req:
            print(f"    Denoising: {req['denoising_seconds']}s")
        if "executed_steps" in req:
            print(f"    Steps: {req['executed_steps']}")

        # Method-specific info
        for key in ["chorus2", "carnator", "chorus", "teacache", "nirvana_vid"]:
            if key in req and req[key]:
                print(f"    {key.upper()} Details:")
                for k, v in req[key].items():
                    if v is not None:
                        print(f"      {k}: {v}")


def main():
    print("\n" + "="*80)
    print("CogVideoX Baseline Performance Analysis - First 5 Requests")
    print("="*80)

    test_dir = find_test_results()

    if not test_dir:
        print("\n✗ No test results found")
        print("  Searched in: artifacts/vidprom_1000cache_300input_noun_match/")
        return

    print(f"\n✓ Found test results: {test_dir.relative_to(ROOT)}")

    # Analyze each method
    methods = ["nirvana_vid", "teacache", "chorus", "chorus2", "carnator_boundary_b5"]
    analyses = []

    for method in methods:
        method_dir = test_dir / method
        if method_dir.exists():
            analysis = analyze_method_results(method_dir)
            if analysis:
                analyses.append(analysis)

    if not analyses:
        print("\n✗ No valid results found in test directory")
        return

    # Print all analyses
    for analysis in analyses:
        print_analysis(analysis)

    # Summary comparison
    print(f"\n\n{'='*80}")
    print("SUMMARY COMPARISON (Average of successful requests)")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'Avg E2E (s)':<15} {'Avg DiT (s)':<15} {'Avg Steps':<12}")
    print("-"*80)

    for analysis in analyses:
        successful = [r for r in analysis["first_5"] if r["status"] == "ok"]
        if successful:
            avg_e2e = sum(r.get("e2e_seconds", 0) for r in successful) / len(successful)
            avg_dit = sum(r.get("dit_forward_s", 0) for r in successful) / len(successful)
            avg_steps = sum(r.get("executed_steps", 0) for r in successful) / len(successful)

            print(f"{analysis['method']:<25} {avg_e2e:<15.3f} {avg_dit:<15.3f} {avg_steps:<12.1f}")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
