#!/usr/bin/env python3
"""Complete CogVideoX experiment: Generation + Quality Evaluation + Aggregation."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASELINES = [
    "baseline",
    "carnator_boundary_b5",
    "chorus",
    "chorus2",
    "nirvana_vid",
]


def check_baseline_complete(baseline_dir: Path) -> tuple[bool, str]:
    """Check if a baseline has completed successfully."""
    results_file = baseline_dir / "results.jsonl"
    if not results_file.exists():
        return False, "results.jsonl not found"

    try:
        lines = results_file.read_text().strip().split('\n')
        if not lines:
            return False, "results.jsonl is empty"

        rows = [json.loads(line) for line in lines if line.strip()]
        if not rows:
            return False, "no valid JSON rows"

        # Check for any successful generations
        ok_count = sum(1 for r in rows if r.get("status") == "ok")
        if ok_count == 0:
            return False, f"no successful generations (0/{len(rows)})"

        return True, f"ok: {ok_count}/{len(rows)}"
    except Exception as e:
        return False, f"error reading results: {e}"


def has_quality_scores(results_file: Path) -> bool:
    """Check if quality scores are already computed."""
    if not results_file.exists():
        return False

    try:
        lines = results_file.read_text().strip().split('\n')
        if not lines:
            return False

        rows = [json.loads(line) for line in lines if line.strip()]
        ok_rows = [r for r in rows if r.get("status") == "ok"]

        if not ok_rows:
            return False

        # Check if first ok row has quality scores
        return all(
            ok_rows[0].get(key) is not None
            for key in ["clip_t", "vbench_q"]
        )
    except:
        return False


def run_generation(args: argparse.Namespace) -> int:
    """Phase 1: Run baseline generation."""
    print("\n" + "="*80)
    print("PHASE 1: Baseline Generation")
    print("="*80 + "\n")

    cmd = [
        "python3", str(ROOT / "scripts/dev/run_5baselines_10samples_fixed.py"),
        "--gpus", args.gpus,
        "--limit", str(args.limit),
        "--output-root", str(args.output_root),
    ]

    if args.dry_run:
        cmd.append("--dry-run")

    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def run_quality_evaluation(args: argparse.Namespace) -> int:
    """Phase 2: Quality evaluation for all baselines."""
    print("\n" + "="*80)
    print("PHASE 2: Quality Evaluation")
    print("="*80 + "\n")

    output_root = Path(args.output_root)

    # Find latest run directories for each baseline
    baseline_dirs = {}
    for baseline in BASELINES:
        pattern = f"{baseline}_*"
        matching = sorted(output_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matching:
            baseline_dirs[baseline] = matching[0]
            print(f"Found {baseline}: {matching[0].name}")

    if not baseline_dirs:
        print("✗ No baseline directories found")
        return 1

    print()

    # Check completion status
    incomplete = []
    for baseline, dir_path in baseline_dirs.items():
        complete, msg = check_baseline_complete(dir_path)
        status = "✓" if complete else "✗"
        print(f"{status} {baseline}: {msg}")
        if not complete:
            incomplete.append(baseline)

    if incomplete:
        print(f"\n⚠ Warning: {len(incomplete)} baselines incomplete: {', '.join(incomplete)}")
        if not args.force:
            print("Use --force to proceed anyway")
            return 1
        print("Continuing anyway (--force)")

    print()

    # Run quality evaluation for each baseline
    for baseline, dir_path in baseline_dirs.items():
        results_file = dir_path / "results.jsonl"

        # Skip if quality already computed
        if has_quality_scores(results_file) and not args.force_quality:
            print(f"✓ {baseline}: quality already computed (use --force-quality to recompute)")
            continue

        print(f"⏳ Evaluating {baseline}...")

        if args.dry_run:
            print(f"   [DRY RUN] Would run quality_runner on {results_file}")
            continue

        cmd = [
            "python3", str(ROOT / "scripts/benchmark/quality_runner.py"),
            "--results", str(results_file),
            "--output-dir", str(dir_path),
        ]

        log_file = output_root / f"quality_{baseline}.log"
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            print(f"✓ {baseline}: quality evaluation complete")
        else:
            print(f"✗ {baseline}: quality evaluation failed (see {log_file})")

    return 0


def run_aggregation(args: argparse.Namespace) -> int:
    """Phase 3: Aggregate results and generate summary."""
    print("\n" + "="*80)
    print("PHASE 3: Result Aggregation")
    print("="*80 + "\n")

    if args.dry_run:
        print("[DRY RUN] Would run aggregate_vidprom_300_results.py")
        return 0

    cmd = [
        "python3", str(ROOT / "scripts/benchmark/aggregate_vidprom_300_results.py"),
        "--results-root", str(args.output_root),
    ]

    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✓ Aggregation complete")
        print(f"  Summary: {args.output_root}/summary.csv")
        print(f"  Table: {args.output_root}/VIDPROM_1000CACHE_300INPUT_RESULTS_TABLE.md")
    else:
        print("\n✗ Aggregation failed")

    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="3,4,5,6,7", help="GPU IDs for parallel execution")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples per baseline")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/cogvideox15_5baselines_10samples",
        help="Output directory"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    parser.add_argument("--force", action="store_true", help="Proceed even if some baselines incomplete")
    parser.add_argument("--force-quality", action="store_true", help="Recompute quality even if already present")
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip generation phase (use existing results)"
    )
    parser.add_argument(
        "--skip-quality", action="store_true",
        help="Skip quality evaluation phase"
    )
    parser.add_argument(
        "--skip-aggregation", action="store_true",
        help="Skip aggregation phase"
    )

    args = parser.parse_args()

    print("="*80)
    print("CogVideoX Complete Experiment Pipeline")
    print("="*80)
    print(f"Output: {args.output_root}")
    print(f"GPUs: {args.gpus}")
    print(f"Samples: {args.limit}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Phase 1: Generation
    if not args.skip_generation:
        ret = run_generation(args)
        if ret != 0:
            print("\n✗ Generation phase failed")
            return ret
        print("\n✓ Generation phase complete")
    else:
        print("\n⏭ Skipping generation phase")

    # Phase 2: Quality Evaluation
    if not args.skip_quality:
        ret = run_quality_evaluation(args)
        if ret != 0:
            print("\n✗ Quality evaluation phase failed")
            return ret
        print("\n✓ Quality evaluation phase complete")
    else:
        print("\n⏭ Skipping quality evaluation phase")

    # Phase 3: Aggregation
    if not args.skip_aggregation:
        ret = run_aggregation(args)
        if ret != 0:
            print("\n✗ Aggregation phase failed")
            return ret
        print("\n✓ Aggregation phase complete")
    else:
        print("\n⏭ Skipping aggregation phase")

    print("\n" + "="*80)
    print("✓ All phases complete!")
    print("="*80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
