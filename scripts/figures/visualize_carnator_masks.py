#!/usr/bin/env python3
"""Visualize Carnator fusion masks on the cat-dog example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def visualize_dog_cat_masks():
    """Visualize the dog-to-cat fusion masks."""
    import torch
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import cv2

    # Path to the diagnostic source (dog)
    source_dir = (ROOT / "artifacts" / "vidprom_1000cache_300input_noun_match" /
                  "carnator_mask_probe_gpu012_20260901" / "source_cache_grassland" /
                  "entries" / "diagnostic-source-dog")

    if not source_dir.exists():
        print(f"✗ Source directory not found: {source_dir}")
        return False

    print(f"✓ Found source directory: {source_dir.relative_to(ROOT)}")

    # Load the fusion masks
    fusion_dir = source_dir / "carnator" / "cache" / "source_dense" / "fusion_masks"
    masks_json = fusion_dir / "masks.json"
    masks_pt = fusion_dir / "masks.pt"

    if not masks_json.exists() or not masks_pt.exists():
        print("✗ Fusion masks not found")
        return False

    # Load mask metadata
    with open(masks_json, 'r') as f:
        metadata = json.load(f)

    print("\nMask Metadata:")
    print(f"  Format: {metadata.get('format')}")
    print(f"  Latent grid: {metadata.get('latent_grid')}")
    print(f"  Token grid: {metadata.get('token_grid')}")
    print(f"  Patch size: {metadata.get('patch_size')}")

    # Load mask tensors
    masks = torch.load(masks_pt, map_location='cpu')
    print(f"\nLoaded masks:")
    for key, value in masks.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    # Load the first frame from the video
    video_path = source_dir / "source.mp4"
    if not video_path.exists():
        print("✗ Source video not found")
        return False

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("✗ Failed to read video frame")
        return False

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    print(f"\nVideo frame: {w}x{h}")

    # Create output directory
    output_dir = ROOT / "artifacts" / "mask_visualization_carnator"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original frame
    Image.fromarray(frame_rgb).save(output_dir / "original_frame.png")
    print(f"✓ Saved original frame to {output_dir.relative_to(ROOT)}/original_frame.png")

    # Visualize each mask type
    from scipy.ndimage import zoom

    def visualize_mask(mask_tensor, name, color):
        """Visualize a single mask with overlay."""
        # For 5D tensors: [B, T, C, H, W] -> take first batch and first time frame
        if mask_tensor.dim() == 5:
            mask_2d = mask_tensor[0, 0].cpu().numpy()  # [C, H, W] -> take first channel
            if mask_2d.ndim > 2:
                mask_2d = mask_2d[0]  # Take first channel if 3D
        elif mask_tensor.dim() == 4:
            mask_2d = mask_tensor[0, 0].cpu().numpy()  # [B, T, H, W]
        elif mask_tensor.dim() == 3:
            mask_2d = mask_tensor[0].cpu().numpy()  # [T, H, W]
        else:
            mask_2d = mask_tensor.cpu().numpy()

        # Resize to frame size
        mask_resized = zoom(mask_2d.astype(float),
                           (h / mask_2d.shape[0], w / mask_2d.shape[1]),
                           order=0) > 0.5

        # Create colored overlay
        overlay = frame_rgb.copy()
        overlay[mask_resized] = (0.5 * overlay[mask_resized] +
                                 0.5 * np.array(color, dtype=np.uint8))

        result_img = Image.fromarray(overlay)

        # Add label
        draw = ImageDraw.Draw(result_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()

        # Draw background for text
        text = name
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.rectangle([10, 10, 20 + text_w, 20 + text_h], fill=(0, 0, 0, 180))
        draw.text((15, 15), text, fill=color, font=font)

        # Save
        output_path = output_dir / f"{name.lower().replace(' ', '_')}.png"
        result_img.save(output_path)
        print(f"✓ Saved {name} to {output_path.relative_to(ROOT)}")

        # Also save pure mask
        mask_img = Image.fromarray((mask_resized * 255).astype(np.uint8))
        mask_path = output_dir / f"{name.lower().replace(' ', '_')}_only.png"
        mask_img.save(mask_path)

        # Calculate coverage
        coverage = mask_resized.sum() / mask_resized.size * 100
        print(f"  Coverage: {coverage:.2f}%")

        return mask_resized, coverage

    # Visualize different mask types (use actual keys from loaded data)
    mask_types = [
        ('latent_Mbase', 'Base Mask (Mbase)', (255, 255, 0)),  # Yellow
        ('latent_Medit', 'Edit Mask (Medit)', (255, 0, 0)),  # Red
        ('latent_Msee', 'See Mask (Msee)', (0, 0, 255)),  # Blue
    ]

    masks_viz = {}
    for key, name, color in mask_types:
        if key in masks:
            print(f"\nVisualizing {name}:")
            mask_viz, coverage = visualize_mask(masks[key], name, color)
            masks_viz[key] = (mask_viz, coverage)

    # Create combined visualization
    if len(masks_viz) >= 2:
        print("\nCreating combined visualization...")
        combined = frame_rgb.copy()

        # Edit region (red)
        if 'latent_Medit' in masks_viz:
            edit_mask = masks_viz['latent_Medit'][0]
            combined[edit_mask] = (0.6 * combined[edit_mask] +
                                   0.4 * np.array([255, 0, 0], dtype=np.uint8)).astype(np.uint8)

        # See-only region (blue)
        if 'latent_Msee' in masks_viz and 'latent_Medit' in masks_viz:
            see_mask = masks_viz['latent_Msee'][0]
            edit_mask = masks_viz['latent_Medit'][0]
            see_only = see_mask & ~edit_mask
            combined[see_only] = (0.7 * combined[see_only] +
                                  0.3 * np.array([0, 0, 255], dtype=np.uint8)).astype(np.uint8)

        result_img = Image.fromarray(combined)

        # Add legend
        draw = ImageDraw.Draw(result_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        legend_y = h - 100
        draw.rectangle([10, legend_y, 280, legend_y + 90], fill=(255, 255, 255, 230))
        draw.text((20, legend_y + 10), "Red: Edit region (Medit)", fill=(255, 0, 0), font=font)
        draw.text((20, legend_y + 40), "Blue: Context-only (Msee)", fill=(0, 0, 255), font=font)

        combined_path = output_dir / "combined_overlay.png"
        result_img.save(combined_path)
        print(f"✓ Saved combined visualization to {combined_path.relative_to(ROOT)}")

    # Create side-by-side comparison
    print("\nCreating side-by-side comparison...")
    from PIL import Image

    images = []
    labels = []

    # Original
    images.append(Image.fromarray(frame_rgb))
    labels.append("Original Frame")

    # Each mask
    for key, name, color in mask_types:
        path = output_dir / f"{name.lower().replace(' ', '_')}.png"
        if path.exists():
            images.append(Image.open(path))
            labels.append(name)

    # Combined
    combined_path = output_dir / "combined_overlay.png"
    if combined_path.exists():
        images.append(Image.open(combined_path))
        labels.append("Combined")

    # Create grid
    if images:
        cols = min(3, len(images))
        rows = (len(images) + cols - 1) // cols

        cell_w, cell_h = images[0].size
        grid_w = cols * cell_w + (cols + 1) * 20
        grid_h = rows * (cell_h + 50) + 20

        grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except:
            font = ImageFont.load_default()

        for idx, (img, label) in enumerate(zip(images, labels)):
            row = idx // cols
            col = idx % cols
            x = 20 + col * (cell_w + 20)
            y = 20 + row * (cell_h + 50)

            grid.paste(img, (x, y))
            # Draw label below image
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            draw.text((x + (cell_w - text_w) // 2, y + cell_h + 5), label, fill=(0, 0, 0), font=font)

        grid_path = output_dir / "comparison_grid.png"
        grid.save(grid_path)
        print(f"✓ Saved comparison grid to {grid_path.relative_to(ROOT)}")

    print(f"\n{'='*80}")
    print(f"All visualizations saved to: {output_dir.relative_to(ROOT)}")
    print(f"{'='*80}\n")

    return True


def main():
    print("\n" + "="*80)
    print("Carnator Fusion Mask Visualization - Dog to Cat Example")
    print("="*80 + "\n")

    try:
        success = visualize_dog_cat_masks()
        if success:
            print("✓ Visualization complete!")
        else:
            print("✗ Visualization failed")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
