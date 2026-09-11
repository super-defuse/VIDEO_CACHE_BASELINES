#!/usr/bin/env python3
"""
Prepare quality evaluation input for ablation study variants.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ABLATION_DIR = ROOT / "artifacts/ablation_req16_20260909_100759"
VARIANTS = ["full_computation", "latent_only", "attention_only", "joint"]

def main():
    print("=" * 80)
    print("Carnator Ablation Study - Quality Evaluation Preparation")
    print("=" * 80)
    print()

    # Collect all videos for quality evaluation
    videos = []

    for variant in VARIANTS:
        variant_dir = ABLATION_DIR / variant / "ablation"

        # Check for target_dense video
        dense_video = variant_dir / "videos/target_dense.mp4"
        if dense_video.exists():
            videos.append({
                "variant": variant,
                "type": "dense",
                "video_path": str(dense_video),
                "prompt": "A cute little lion and a wobbly penguin appear"
            })
            print(f"✓ Found: {variant}/target_dense.mp4")

        # Check for target_latent_reuse video
        reuse_video = variant_dir / "videos/target_latent_reuse.mp4"
        if reuse_video.exists():
            videos.append({
                "variant": variant,
                "type": "latent_reuse",
                "video_path": str(reuse_video),
                "prompt": "A cute little lion and a wobbly penguin appear"
            })
            print(f"✓ Found: {variant}/target_latent_reuse.mp4")

    print()
    print(f"Total videos to evaluate: {len(videos)}")
    print()

    if not videos:
        print("❌ No videos found for evaluation!")
        return 1

    # Create staging JSONL for quality evaluation
    staging_jsonl = ABLATION_DIR / "quality_evaluation_input.jsonl"

    with open(staging_jsonl, 'w') as f:
        for i, video in enumerate(videos):
            row = {
                "request_index": i,
                "uuid": f"{video['variant']}_{video['type']}",
                "status": "ok",
                "video_path": video["video_path"],
                "prompt": video["prompt"],
                "variant": video["variant"],
                "type": video["type"]
            }
            f.write(json.dumps(row) + '\n')

    print(f"✓ Created: {staging_jsonl}")
    print()
    print("=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print()
    print("Run quality evaluation with:")
    print()
    print(f"  python3 scripts/benchmark/quality_runner.py \\")
    print(f"    {staging_jsonl} \\")
    print(f"    --output-dir artifacts/ablation_req16_20260909_100759/quality")
    print()
    print("This will compute:")
    print("  • CLIP-T (text-video alignment)")
    print("  • VBench-Q (6-dimensional quality score)")
    print()

    return 0

if __name__ == "__main__":
    exit(main())
