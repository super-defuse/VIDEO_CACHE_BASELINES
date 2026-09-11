#!/usr/bin/env python3
"""Simple script to run 10-sample experiments sequentially."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.cogvideo_runner import CogVideoReplayRunner
from core.types import Request
from adapters.cogvideox15 import CogVideoX15Adapter
from core.registry import load_method

def run_experiment(method_name: str, gpu_id: int, limit: int = 10):
    """Run one baseline experiment."""

    print(f"\n{'='*80}")
    print(f"Running {method_name} (GPU {gpu_id}, first {limit} samples)")
    print(f"{'='*80}\n")

    # Paths
    replay_dir = ROOT / "artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay"
    manifest_path = replay_dir / "manifest.json"
    output_root = ROOT / "artifacts/cogvideox15_10samples_test"

    # Load replay decisions
    replay_file = replay_dir / f"{method_name}.jsonl"
    decisions = []
    with open(replay_file) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            decisions.append(json.loads(line))

    print(f"Loaded {len(decisions)} replay decisions")

    # Load source prompts
    with open(manifest_path) as f:
        manifest = json.load(f)
    source_prompts = {item["uuid"]: item for item in manifest.get("sources", [])}
    print(f"Loaded {len(source_prompts)} source prompts")

    # Create output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{method_name}_{timestamp}_gpu{gpu_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build config
    config = {
        "model": "cogvideox15",
        "device": f"cuda:{gpu_id}",
        "height": 480,
        "width": 720,
        "frames": 49,
        "baseline_steps": 50,
        "guidance_scale": 6.0,
        "seed": 20260218,
    }

    # Method-specific configs
    if method_name == "carnator_boundary_b5":
        config.update({
            "carnator": {
                "probe_step": 5,
                "reuse_step": 10,
                "region_layers": "10-30",
                "outer_layers": "0-9,31-39",
            }
        })
    elif method_name == "nirvana_vid":
        config.update({
            "nirvana_vid": {
                "reuse_start_step": 5,
                "cache_steps": [5],
            }
        })
    elif method_name in ["chorus", "chorus2"]:
        config.update({
            "chorus": {
                "k1": 5,
                "k2": 30,
            }
        })

    # Create adapter and method
    adapter = CogVideoX15Adapter(config)
    baseline_method = load_method(method_name, config)
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
        start = time.time()
        results = runner.run(requests, results_path)
        elapsed = time.time() - start

        print(f"\n✓ {method_name} completed successfully")
        print(f"  Processed {len(results)} requests in {elapsed:.1f}s")
        print(f"  Results: {results_path}")
        print(f"  Output: {output_dir}")
        return True
    except Exception as e:
        print(f"\n✗ {method_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    success = run_experiment(args.method, args.gpu, args.limit)
    sys.exit(0 if success else 1)
