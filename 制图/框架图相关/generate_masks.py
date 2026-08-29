#!/usr/bin/env python3
"""Generate object masks and recolored-car composites with SAM2.

The default boxes target the four images in this directory.  For other images,
pass a box as x1,y1,x2,y2 with ``--box`` or add an entry to DEFAULT_BOXES.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# This job is intentionally pinned to physical GPU 2.  SAM2 sees it as cuda:0
# after CUDA_VISIBLE_DEVICES remaps the selected device.
if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "2"):
    raise RuntimeError("This script is restricted to physical GPU 2; unset CUDA_VISIBLE_DEVICES or set it to 2.")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import torch
from PIL import Image
from scipy import ndimage


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
GROUNDED_SAM_DIR = PROJECT_DIR.parent / "Grounded-SAM-2"
if not GROUNDED_SAM_DIR.exists():
    raise FileNotFoundError(f"Grounded-SAM-2 checkout not found: {GROUNDED_SAM_DIR}")
sys.path.insert(0, str(GROUNDED_SAM_DIR))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# Normalized x1,y1,x2,y2 boxes.  They include a small amount of context so SAM2
# can infer the edge, but do not include the table/ground as the target object.
DEFAULT_BOXES = {
    "apple.png": (0.283, 0.273, 0.708, 0.723),
    "pineapple.png": (0.318, 0.029, 0.684, 0.713),
    "car.jpg": (0.200, 0.473, 0.848, 0.997),
    "car2.png": (0.191, 0.478, 0.821, 0.949),
}


def parse_box(value: str) -> tuple[float, float, float, float]:
    try:
        box = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("box must be x1,y1,x2,y2") from exc
    if len(box) != 4 or not (box[0] < box[2] and box[1] < box[3]):
        raise argparse.ArgumentTypeError("box must satisfy x1<x2 and y1<y2")
    return box


def get_box(filename: str, size: tuple[int, int], override: tuple[float, ...] | None) -> np.ndarray:
    width, height = size
    values = override or DEFAULT_BOXES.get(filename)
    if values is None:
        raise ValueError(f"No default box for {filename}; pass --box x1,y1,x2,y2")
    # Overrides are interpreted as pixels; defaults are normalized coordinates.
    if override is not None:
        return np.asarray(values, dtype=np.float32)
    x1, y1, x2, y2 = values
    return np.asarray((x1 * width, y1 * height, x2 * width, y2 * height), dtype=np.float32)


def segment(predictor: SAM2ImagePredictor, image: np.ndarray, box: np.ndarray) -> np.ndarray:
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    # SAM2's stability-ranked candidate is generally the cleanest one for a box prompt.
    best = int(np.asarray(scores).argmax())
    mask = np.asarray(masks[best], dtype=bool)
    # Remove tiny enclosed pinholes that otherwise show the original car pixels.
    return ndimage.binary_fill_holes(mask)


def save_outputs(image_path: Path, output_dir: Path, mask: np.ndarray, car_color: str) -> None:
    stem = image_path.stem
    mask_image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    mask_image.save(output_dir / f"{stem}_mask.png")

    if stem.startswith("car"):
        image = Image.open(image_path).convert("RGB")
        pixels = np.asarray(image).copy()
        fill_value = 255 if car_color == "white" else 0
        pixels[mask] = fill_value
        Image.fromarray(pixels, mode="RGB").save(output_dir / f"{stem}_{car_color}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "masks")
    parser.add_argument("--only", nargs="+", choices=sorted(DEFAULT_BOXES), default=sorted(DEFAULT_BOXES))
    parser.add_argument("--box", type=parse_box, help="pixel box x1,y1,x2,y2 (for one --only image)")
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path("/home/tangxuebin/models/Grounded-SAM-2-v1.0/checkpoints/sam2.1_hiera_large.pt"),
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--car-color", default="white", choices=["white", "black"])
    args = parser.parse_args()

    if args.box is not None and len(args.only) != 1:
        parser.error("--box can only be used with exactly one --only image")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. The requested workflow requires GPU 2.")
    if not args.sam_checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {args.sam_checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # SAM2's Hydra config is registered relative to its checkout.
    os.chdir(GROUNDED_SAM_DIR)
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        str(args.sam_checkpoint),
        device=args.device,
    )
    predictor = SAM2ImagePredictor(model)

    for filename in args.only:
        image_path = args.input_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = np.asarray(Image.open(image_path).convert("RGB")).copy()
        box = get_box(filename, (image.shape[1], image.shape[0]), args.box)
        with torch.inference_mode():
            mask = segment(predictor, image, box)
        save_outputs(image_path, args.output_dir, mask, args.car_color)
        print(f"{filename}: foreground={mask.mean():.3%}, box={box.round(1).tolist()}")


if __name__ == "__main__":
    main()
