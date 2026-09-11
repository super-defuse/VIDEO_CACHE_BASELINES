#!/usr/bin/env python3
"""Run CogVideoX baselines on first 10 samples with replay decisions."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Get VIDEO_CACHE_BASELINES root directory
if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[2]
else:
    ROOT = Path.cwd()

# Ensure ROOT is in sys.path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Only baselines with replay decisions (exclude dense baseline)
BASELINES = [
    "carnator_boundary_b5",
    "chorus",
    "chorus2",
    "nirvana_vid",
]

def load_replay_decisions(replay_dir: Path, method: str, limit: int = 10):
    """Load first N replay decisions for a method."""
    replay_file = replay_dir / f"{method}.jsonl"
    if not replay_file.exists():
        raise FileNotFoundError(f"Replay file not found: {replay_file}")

    decisions = []
    with open(replay_file) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            decisions.append(json.loads(line))

    return decisions

def load_source_prompts(manifest_path: Path):
    """Load source prompts from manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    source_prompts = {}
    for item in manifest.get("sources", []):
        source_prompts[item["uuid"]] = item

    return source_prompts

def run_baseline_experiment(
    method: str,
    gpu: int,
    replay_dir: Path,
    manifest_path: Path,
    output_root: Path,
    limit: int = 10,
    dry_run: bool = False
):
    """Run one baseline method on GPU."""
    print(f"\n{'='*80}")
    print(f"Running {method} on GPU {gpu} (first {limit} samples)")
    print(f"{'='*80}\n")

    # Load replay decisions
    decisions = load_replay_decisions(replay_dir, method, limit)
    print(f"Loaded {len(decisions)} replay decisions for {method}")

    # Load source prompts
    source_prompts = load_source_prompts(manifest_path)
    print(f"Loaded {len(source_prompts)} source prompts")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{method}_{timestamp}_gpu{gpu}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save replay decisions for this run
    replay_subset = output_dir / "replay_decisions.jsonl"
    with open(replay_subset, 'w') as f:
        for dec in decisions:
            f.write(json.dumps(dec) + '\n')

    print(f"Output directory: {output_dir}")

    if dry_run:
        print("[DRY RUN] Would execute experiment here")
        return

    # Import required modules
    from core.cogvideo_runner import CogVideoReplayRunner
    from core.types import Request
    from adapters.cogvideox15 import CogVideoX15Adapter
    from methods import create_method

    # Build config
    config = {
        "model": "cogvideox15",
        "device": "cuda:0",  # Will be mapped by CUDA_VISIBLE_DEVICES
        "height": 480,
        "width": 720,
        "frames": 49,
        "baseline_steps": 50,
        "guidance_scale": 6.0,
        "seed": 20260218,
    }

    # Method-specific configs
    if method == "carnator_boundary_b5":
        config.update({
            "carnator": {
                "probe_step": 5,
                "reuse_step": 10,
                "region_layers": "10-30",
                "outer_layers": "0-9,31-39",
            }
        })
    elif method == "nirvana_vid":
        config.update({
            "nirvana_vid": {
                "reuse_start_step": 5,
                "cache_steps": [5],
            }
        })
    elif method in ["chorus", "chorus2"]:
        config.update({
            "chorus": {
                "k1": 5,
                "k2": 30,
            }
        })

    # Create adapter and method
    adapter = CogVideoX15Adapter(config)
    baseline_method = create_method(method, config)
    baseline_method.prepare(adapter, config)

    # Build replay index
    replay_index = {dec["request_index"]: dec for dec in decisions}

    # Create runner
    runner = CogVideoReplayRunner(
        adapter=adapter,
        method=baseline_method,
        output_dir=output_dir,
        replay=replay_index,
        source_prompts=source_prompts,
        config=config,
        model="CogVideoX1.5",
    )

    # Generate request list
    requests = [
        Request(
            request_index=dec["request_index"],
            uuid=dec["uuid"],
            prompt=dec["prompt"],
            seed=config["seed"],
        )
        for dec in decisions
    ]

    # Run experiment
    results_path = output_dir / "results.jsonl"
    try:
        results = runner.run(requests, results_path)
        print(f"\n✓ {method} completed successfully")
        print(f"  Processed {len(results)} requests")
        print(f"  Results: {results_path}")
    except Exception as e:
        print(f"\n✗ {method} failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=BASELINES,
        default=BASELINES,
        help="Baselines to run (default: all except teacache)"
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=ROOT / "artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay",
        help="Directory containing replay decision files"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay/manifest.json",
        help="Source prompts manifest file"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/cogvideox15_10samples_test",
        help="Output root directory"
    )
    parser.add_argument(
        "--gpus",
        default="3,4,5,6,7",
        help="Comma-separated GPU IDs (default: 3,4,5,6,7)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of samples to process (default: 10)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run baselines sequentially on one GPU instead of parallel"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.replay_dir.exists():
        print(f"Error: Replay directory not found: {args.replay_dir}")
        return 1

    if not args.manifest.exists():
        print(f"Error: Manifest file not found: {args.manifest}")
        return 1

    # Parse GPUs
    gpus = [int(g.strip()) for g in args.gpus.split(",")]

    if args.sequential:
        # Run all baselines sequentially on first GPU
        gpu = gpus[0]
        for baseline in args.baselines:
            run_baseline_experiment(
                method=baseline,
                gpu=gpu,
                replay_dir=args.replay_dir,
                manifest_path=args.manifest,
                output_root=args.output_root,
                limit=args.limit,
                dry_run=args.dry_run,
            )
    else:
        # Distribute baselines across GPUs using subprocess
        # Assign baselines to GPUs
        assignments = [(baseline, gpus[i % len(gpus)]) for i, baseline in enumerate(args.baselines)]

        print(f"\nRunning {len(args.baselines)} baselines across {len(gpus)} GPUs")
        for baseline, gpu in assignments:
            print(f"  {baseline} → GPU {gpu}")

        if args.dry_run:
            print("\n[DRY RUN] Would launch parallel processes here")
            return 0

        # Launch parallel subprocesses
        processes = []
        for baseline, gpu in assignments:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            # Ensure PYTHONPATH includes VIDEO_CACHE_BASELINES root
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--baselines", baseline,
                "--replay-dir", str(args.replay_dir.resolve()),
                "--manifest", str(args.manifest.resolve()),
                "--output-root", str(args.output_root.resolve()),
                "--limit", str(args.limit),
                "--sequential",  # Force sequential within this process
            ]

            log_path = args.output_root / f"{baseline}_gpu{gpu}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path, 'w')

            print(f"Starting {baseline} on GPU {gpu} (log: {log_path})...")
            proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(ROOT))
            processes.append((baseline, gpu, proc, log_file))

        # Wait for all processes to complete
        print(f"\nWaiting for {len(processes)} processes to complete...")
        for baseline, gpu, proc, log_file in processes:
            proc.wait()
            log_file.close()
            if proc.returncode == 0:
                print(f"✓ {baseline} on GPU {gpu} completed successfully")
            else:
                print(f"✗ {baseline} on GPU {gpu} failed with exit code {proc.returncode}")

    return 0

if __name__ == "__main__":
    main()
