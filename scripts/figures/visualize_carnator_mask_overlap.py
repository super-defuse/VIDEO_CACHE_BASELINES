#!/usr/bin/env python3
"""Test Carnator mask generation on CogVideoX and visualize the results."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_cogvideo_test_results():
    """Load the 5-request test results to check performance metrics."""
    base_dir = ROOT / "artifacts" / "vidprom_1000cache_300input_noun_match"

    # Try to find any cogvideo test results
    test_dirs = list(base_dir.glob("*"))

    results = {}
    for test_dir in test_dirs:
        if not test_dir.is_dir():
            continue

        # Look for request_records.jsonl in subdirectories
        for method_dir in test_dir.iterdir():
            if not method_dir.is_dir():
                continue

            records_file = method_dir / "request_records.jsonl"
            if records_file.exists():
                method_name = method_dir.name
                records = []
                with open(records_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))

                if records:
                    results[method_name] = {
                        "test_dir": str(test_dir.name),
                        "count": len(records),
                        "records": records[:5]  # First 5 requests
                    }

    return results


def extract_performance_metrics(results):
    """Extract performance metrics from test results."""
    metrics = {}

    for method, data in results.items():
        method_metrics = {
            "total_requests": data["count"],
            "requests_analyzed": len(data["records"]),
            "timings": []
        }

        for record in data["records"]:
            timing = {
                "arrival_index": record.get("arrival_index"),
                "status": record.get("status"),
                "hit": record.get("hit", False),
            }

            # Extract timing information
            if "dit_forward_s" in record:
                timing["dit_forward_s"] = record["dit_forward_s"]
            if "e2e_seconds" in record:
                timing["e2e_seconds"] = record["e2e_seconds"]
            if "executed_steps" in record:
                timing["executed_steps"] = record["executed_steps"]

            # Method-specific metrics
            extra = record.get("extra", {})
            if "chorus2" in extra:
                timing["chorus2"] = extra["chorus2"]
            if "carnator" in extra:
                timing["carnator"] = extra["carnator"]
            if "chorus" in extra:
                timing["chorus"] = extra["chorus"]

            method_metrics["timings"].append(timing)

        metrics[method] = method_metrics

    return metrics


def test_cat_dog_example():
    """Test the cat-dog example similar to WAN version."""
    print("\n" + "="*80)
    print("Testing Carnator Mask Generation on CogVideoX with Cat-Dog Example")
    print("="*80)

    # Check if we can import the necessary modules
    try:
        from methods.chorus.grounded_sam import GroundedSam2Segmenter
        from methods.chorus.cogvideo_mask import build_cogvideo_edit_mask
        print("✓ Successfully imported CogVideoX mask generation modules")
    except ImportError as e:
        print(f"✗ Failed to import modules: {e}")
        return None

    # Look for existing test videos
    test_videos = list(ROOT.glob("artifacts/**/source*.mp4"))
    if not test_videos:
        print("✗ No test videos found in artifacts directory")
        return None

    print(f"✓ Found {len(test_videos)} test videos")

    # Use the first available video
    test_video = test_videos[0]
    print(f"  Using: {test_video.relative_to(ROOT)}")

    # Try to generate mask for "dog" (or "cat" depending on video content)
    try:
        # CogVideoX latent shape: (frames, height, width)
        # For 81 frames at 1360x768, with 2x2x2 patch size
        latent_shape = (81, 170, 240)  # After VAE encoding
        patch_size = (1, 2, 2)  # CogVideoX patch size

        config = {
            "mask_dilation_edit": 8,
            "mask_dilation_see": 14,
            "grounded_sam2_model_cfg": "sam2.1_hiera_large.yaml",
            "grounded_sam2_checkpoint": str(ROOT / "models" / "Grounded-SAM-2-v1.0" / "checkpoints" / "sam2.1_hiera_large.pt"),
            "groundingdino_model_cfg": str(ROOT / "external" / "grounded-sam2" / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"),
            "groundingdino_checkpoint": str(ROOT / "models" / "Grounded-SAM-2-v1.0" / "gdino_checkpoints" / "groundingdino_swint_ogc.pth"),
        }

        print("\nGenerating mask for 'dog'...")
        masks, metadata = build_cogvideo_edit_mask(
            test_video,
            "dog",
            latent_shape=latent_shape,
            patch_size=patch_size,
            config=config,
        )

        print("✓ Successfully generated masks")
        print(f"  Metadata: {json.dumps(metadata, indent=2)}")
        print(f"  Token base shape: {masks['token_base'].shape}")
        print(f"  Token edit shape: {masks['token_edit'].shape}")
        print(f"  Token see shape: {masks['token_see'].shape}")

        return masks, metadata, test_video

    except Exception as e:
        print(f"✗ Failed to generate mask: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_mask_overlap(video_path: Path, masks: dict, output_path: Path):
    """Create visualization of first frame with mask overlay."""
    try:
        import cv2

        # Read first frame from video
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("✗ Failed to read video frame")
            return

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Get the latent masks (already at frame resolution)
        edit_mask = masks["latent_edit"][0].cpu().numpy()  # First frame
        see_mask = masks["latent_see"][0].cpu().numpy()

        # Resize masks to frame size
        h, w = frame_rgb.shape[:2]
        from scipy.ndimage import zoom

        edit_mask_resized = zoom(edit_mask.astype(float),
                                  (h / edit_mask.shape[0], w / edit_mask.shape[1]),
                                  order=0) > 0.5
        see_mask_resized = zoom(see_mask.astype(float),
                                 (h / see_mask.shape[0], w / see_mask.shape[1]),
                                 order=0) > 0.5

        # Create colored overlays
        overlay = frame_pil.copy()
        overlay_array = np.array(overlay)

        # Red for edit region (alpha blend)
        edit_color = np.array([255, 0, 0], dtype=np.uint8)
        overlay_array[edit_mask_resized] = (0.6 * overlay_array[edit_mask_resized] +
                                             0.4 * edit_color).astype(np.uint8)

        # Blue for see-only region (not in edit)
        see_only = see_mask_resized & ~edit_mask_resized
        see_color = np.array([0, 0, 255], dtype=np.uint8)
        overlay_array[see_only] = (0.8 * overlay_array[see_only] +
                                    0.2 * see_color).astype(np.uint8)

        result = Image.fromarray(overlay_array)

        # Add legend
        draw = ImageDraw.Draw(result)
        legend_y = 10
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        draw.rectangle([10, legend_y, 200, legend_y + 80], fill=(255, 255, 255, 200))
        draw.text((20, legend_y + 10), "Red: Edit region", fill=(255, 0, 0), font=font)
        draw.text((20, legend_y + 40), "Blue: See region", fill=(0, 0, 255), font=font)

        # Save result
        result.save(output_path)
        print(f"✓ Saved visualization to: {output_path.relative_to(ROOT)}")

        # Also save individual masks
        mask_dir = output_path.parent
        edit_img = Image.fromarray((edit_mask_resized * 255).astype(np.uint8))
        edit_img.save(mask_dir / "edit_mask.png")
        see_img = Image.fromarray((see_mask_resized * 255).astype(np.uint8))
        see_img.save(mask_dir / "see_mask.png")
        print(f"✓ Saved individual masks to: {mask_dir.relative_to(ROOT)}")

        return True

    except Exception as e:
        print(f"✗ Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("CogVideoX Baseline Performance Analysis")
    print("="*80)

    # Load test results
    print("\n1. Loading 5-request test results...")
    results = load_cogvideo_test_results()

    if not results:
        print("✗ No test results found")
    else:
        print(f"✓ Found results for {len(results)} methods")

        # Extract and display metrics
        metrics = extract_performance_metrics(results)

        for method, data in metrics.items():
            print(f"\n{method.upper()}")
            print("-" * 40)
            print(f"  Total requests: {data['total_requests']}")
            print(f"  Analyzed: {data['requests_analyzed']}")

            for i, timing in enumerate(data['timings']):
                print(f"\n  Request {i}:")
                print(f"    Status: {timing.get('status', 'N/A')}")
                print(f"    Hit: {timing.get('hit', False)}")
                if 'dit_forward_s' in timing:
                    print(f"    DiT forward time: {timing['dit_forward_s']:.3f}s")
                if 'e2e_seconds' in timing:
                    print(f"    End-to-end time: {timing['e2e_seconds']:.3f}s")
                if 'executed_steps' in timing:
                    print(f"    Executed steps: {timing['executed_steps']}")

                # Method-specific details
                for key in ['chorus2', 'carnator', 'chorus']:
                    if key in timing:
                        print(f"    {key} details: {json.dumps(timing[key], indent=6)}")

    # Test mask generation
    print("\n\n2. Testing Carnator Mask Generation...")
    mask_result = test_cat_dog_example()

    if mask_result:
        masks, metadata, video_path = mask_result

        # Create visualization
        output_dir = ROOT / "artifacts" / "mask_visualization"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "first_frame_with_mask.png"

        print("\n3. Creating mask visualization...")
        visualize_mask_overlap(video_path, masks, output_path)

    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
