#!/usr/bin/env python3
"""
WAN21 10-request test experiment runner.

This script runs all 5 baselines (baseline, nirvana_vid, chorus, chorus2, carnator_boundary_b5)
for WAN2.1 model with 10 requests for testing purposes.

Usage:
    # Test single method
    python scripts/benchmark/run_wan21_test.py --method baseline --gpu 3

    # Run all methods in parallel
    bash scripts/benchmark/run_wan21_all_test.sh
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import from the existing experiment script
from scripts.benchmark.run_vidprom_300_experiment import (
    resolved_config,
    load_view,
    read_rows,
    sha256_file,
    forced_smoke_config,
    EARLY_SIGNATURE_THRESHOLDS,
    DISABLED_EARLY_SIGNATURE_THRESHOLDS,
    DEFAULT_VIEW,
    DEFAULT_INPUT,
    GPU_COUNT,
)

METHODS = ("baseline", "nirvana_vid", "chorus", "chorus2", "carnator_boundary_b5")
GPU_BY_METHOD = {
    "baseline": 3,
    "nirvana_vid": 4,
    "chorus": 5,
    "chorus2": 6,
    "carnator_boundary_b5": 7
}
DEFAULT_OUTPUT = ROOT / "artifacts" / "vidprom_1000cache_300input_noun_match" / "results_wan21_test"


def run_experiment(args: argparse.Namespace) -> int:
    """Run the experiment by delegating to run_vidprom_300_experiment.py."""

    output = args.run_dir or args.output_root / args.method
    if output.exists() and args.overwrite and not args.resume:
        shutil.rmtree(output)
    elif output.exists() and not args.resume:
        raise FileExistsError(
            f"refusing to overwrite existing run: {output}; use --overwrite or --resume"
        )

    output.mkdir(parents=True, exist_ok=True)

    # Get git revision
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"

    # Read input rows
    rows = read_rows(
        args.input, args.limit,
        arrival_start=args.arrival_start,
        arrival_stop=args.arrival_stop,
        request_indices=args.request_indices
    )

    # Save run metadata
    run_config = {
        "model": "wan21",
        "method": args.method,
        "gpu": args.gpu,
        "seed": args.seed,
        "limit": args.limit,
        "arrival_start": args.arrival_start,
        "arrival_stop": args.arrival_stop,
        "request_indices": args.request_indices,
        "input": str(args.input),
        "cache_view": str(args.cache_view),
        "input_sha256": sha256_file(args.input),
        "test_mode": True,
        "git_revision": revision,
    }

    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    # Build the command to delegate to run_vidprom_300_experiment.py
    cmd = [
        sys.executable,
        str(ROOT / "scripts/benchmark/run_vidprom_300_experiment.py"),
        "--method", args.method,
        "--model", "wan21",
        "--gpu", str(args.gpu),
        "--limit", str(args.limit),
        "--input", str(args.input),
        "--cache-view", str(args.cache_view),
        "--output-root", str(args.output_root),
        "--seed", str(args.seed),
    ]

    if args.arrival_start:
        cmd.extend(["--arrival-start", str(args.arrival_start)])
    if args.arrival_stop:
        cmd.extend(["--arrival-stop", str(args.arrival_stop)])
    if args.request_indices:
        cmd.extend(["--request-indices", args.request_indices])
    if args.resume:
        cmd.append("--resume")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.allow_gpu_remap:
        cmd.append("--allow-gpu-remap")
    if args.skip_quality:
        cmd.append("--skip-quality")

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True,
                        help="Baseline method to run")
    parser.add_argument("--gpu", type=int, required=True,
                        help="GPU device ID")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of requests to process (default: 10)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Input CSV file")
    parser.add_argument("--cache-view", type=Path, default=DEFAULT_VIEW,
                        help="Cache view directory")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT,
                        help="Output root directory")
    parser.add_argument("--run-dir", type=Path,
                        help="Write results directly here instead of <output-root>/<method>")
    parser.add_argument("--arrival-start", type=int, default=0,
                        help="Inclusive global CSV arrival index")
    parser.add_argument("--arrival-stop", type=int,
                        help="Exclusive global CSV arrival index")
    parser.add_argument("--request-indices",
                        help="Comma-separated source row indices")
    parser.add_argument("--seed", type=int, default=20260218,
                        help="Random seed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run with synthetic model")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing output directory")
    parser.add_argument("--allow-gpu-remap", action="store_true",
                        help="Allow GPU outside 3-7 range")
    parser.add_argument("--skip-quality", action="store_true",
                        help="Skip quality evaluation")

    args = parser.parse_args()

    # Validation
    if args.gpu < 0 or args.gpu >= GPU_COUNT:
        raise SystemExit(f"--gpu must be in [0, {GPU_COUNT - 1}]")

    if args.gpu not in range(3, 8) and not args.allow_gpu_remap:
        raise SystemExit("--gpu outside 3-7 range requires --allow-gpu-remap")

    if args.gpu != GPU_BY_METHOD[args.method] and not args.allow_gpu_remap:
        raise SystemExit(
            f"{args.method} is assigned to GPU {GPU_BY_METHOD[args.method]}, "
            f"got {args.gpu}"
        )

    if args.limit < 1 or args.limit > 300:
        raise ValueError("--limit must be in [1, 300]")

    # Resolve paths
    args.input = args.input.expanduser().resolve()
    args.cache_view = args.cache_view.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None

    # Parse request indices
    if args.request_indices:
        args.request_indices = [
            int(item) for item in args.request_indices.split(",") if item.strip()
        ]
        if any(index < 0 or index >= args.limit for index in args.request_indices):
            raise ValueError("--request-indices must refer to rows below --limit")
    else:
        args.request_indices = None

    selected_count = len(args.request_indices) if args.request_indices else args.limit
    args.arrival_stop = selected_count if args.arrival_stop is None else args.arrival_stop

    if not 0 <= args.arrival_start < args.arrival_stop <= args.limit:
        raise ValueError(
            f"arrival range must satisfy 0 <= --arrival-start < --arrival-stop <= --limit ({args.limit})"
        )

    print("=" * 70)
    print(f"WAN21 Test Experiment: {args.method}")
    print("=" * 70)
    print(f"Model: wan21")
    print(f"Method: {args.method}")
    print(f"GPU: {args.gpu}")
    print(f"Requests: {args.limit}")
    print(f"Output: {args.output_root / args.method}")
    print("=" * 70)
    print()

    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
