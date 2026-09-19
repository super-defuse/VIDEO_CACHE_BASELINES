#!/usr/bin/env python3
"""Generate and render prompt-aligned Carnator paper figures.

Each displayed prompt is generated as an independent source-only Wan2.2-TI2V-5B
run.  This deliberately follows the static-prompt-defaults workflow: a video
and its fusion ``Mbase`` are derived from the same prompt and source trajectory.
The target panel never receives the source panel's mask.

The two comparisons retain the paper's intended semantics:

* apple on snow versus peach on a meadow: changed background and color;
* football versus barbell: similar placement but incompatible geometry.

The script extracts each run's first frame, overlays that run's real fusion
``Mbase``, and writes four video-frame tiles plus four prompt-specific sparse
self-attention tiles captured at one common denoising step.
No image-generation service is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = Path(
    os.environ.get("CARNATOR_TI2V_5B_MODEL_DIR", str(ROOT / "models" / "Wan2.2-TI2V-5B"))
)
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "paper-figures" / "failure-modes"
FIRST_FRAME_FRACTION = 0.0

MBASE_COLOR = (226, 66, 72)
ATTENTION_COLOR = (21, 129, 144)
ATTENTION_EMPTY = (239, 243, 246)


@dataclass(frozen=True)
class Case:
    """One source-target pair used in the side-by-side paper example."""

    identifier: str
    source_label: str
    target_label: str
    group_title: str
    source_prompt: str
    target_prompt: str
    source_positive: str
    target_positive: str
    seed: int
    trajectory: str | None = None


CASES = (
    Case(
        identifier="background_shift",
        source_label="Apple on snow",
        target_label="Peach on a mountain",
        group_title="Different background and color, matched centered composition",
        source_prompt=(
            "A single glossy red apple resting near the center of a clean white snow field, "
            "softly blurred snow-covered mountains in the distance, blue sky, cold daylight, "
            "the apple occupies about one quarter of the frame height, photorealistic, static camera."
        ),
        target_prompt=(
            "A single ripe peach resting near the center of a clean green alpine meadow, "
            "softly blurred rocky mountains in the distance, warm daylight, the peach occupies "
            "about one quarter of the frame height, photorealistic, static camera."
        ),
        source_positive="apple",
        target_positive="peach",
        seed=1729,
        trajectory=None,
    ),
    Case(
        identifier="geometry_mismatch",
        source_label="American football",
        target_label="Barbell",
        group_title="Similar placement, incompatible geometry",
        source_prompt=(
            "One brown leather American football resting near the center of a plain green stadium "
            "field, softly blurred distant stands, soft overcast daylight, the football spans about "
            "one third of the frame width, photorealistic, static camera."
        ),
        target_prompt=(
            "One black Olympic barbell with two large round weight plates resting near the center of "
            "a plain green stadium field, softly blurred distant stands, soft overcast daylight, the "
            "barbell spans about one third of the frame width, photorealistic, static camera."
        ),
        source_positive="football",
        target_positive="barbell",
        seed=2718,
        trajectory=None,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Figure root. Carnator artifacts, frames, and final PNGs are stored below it.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run Carnator before rendering. Omit to render existing artifacts only.",
    )
    parser.add_argument(
        "--cases",
        choices=tuple(case.identifier for case in CASES),
        nargs="+",
        default=tuple(case.identifier for case in CASES),
        help="Subset of independent source-target pairs to generate.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Exit after generation; useful when independent cases run on separate GPUs.",
    )
    parser.add_argument(
        "--overwrite-runs",
        action="store_true",
        help="Allow Carnator to replace only this script's prompt-specific figure run directories.",
    )
    parser.add_argument("--runtime-python", type=Path, default=Path(os.environ.get("CARNATOR_PYTHON", sys.executable)))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default=os.environ.get("CARNATOR_DEVICE", "cuda:0"))
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--reuse-start-step", type=int, default=10)
    parser.add_argument("--attention-step", type=int, default=20)
    parser.add_argument("--attention-layer", type=int, default=15)
    parser.add_argument(
        "--attention-head-consensus",
        type=float,
        default=0.75,
        help="Show a block when this fraction of attention heads selected it.",
    )
    parser.add_argument(
        "--outer-mask-top-p",
        type=float,
        default=0.75,
        help="Carnator's source block-routing mass retention. Lower values appear sparser.",
    )
    parser.add_argument("--outer-mask-min-top-k", type=float, default=0.10)
    parser.add_argument("--mbase-alpha", type=int, default=142)
    args = parser.parse_args()
    if not 0.0 < args.attention_head_consensus <= 1.0:
        parser.error("--attention-head-consensus must be in (0, 1]")
    if not 0.0 < args.outer_mask_top_p <= 1.0:
        parser.error("--outer-mask-top-p must be in (0, 1]")
    if not 0.0 <= args.outer_mask_min_top_k <= 1.0:
        parser.error("--outer-mask-min-top-k must be in [0, 1]")
    if not 0 <= args.mbase_alpha <= 255:
        parser.error("--mbase-alpha must be in [0, 255]")
    if args.generate_only and not args.generate:
        parser.error("--generate-only requires --generate")
    return args


def paths_for_case(output_root: Path, case: Case) -> dict[str, Path]:
    artifact_run = output_root / "carnator-artifacts" / case.identifier
    result_run = output_root / "carnator-results" / case.identifier
    target_artifact_run = output_root / "prompt-sparse-attention-artifacts" / f"{case.identifier}_{case.target_positive}"
    target_result_run = output_root / "prompt-sparse-attention-results" / f"{case.identifier}_{case.target_positive}"
    frame_dir = output_root / "frames" / case.identifier
    return {
        "artifact_run": artifact_run,
        "result_run": result_run,
        "target_artifact_run": target_artifact_run,
        "target_result_run": target_result_run,
        "source_video": artifact_run / "videos" / "source_dense.mp4",
        "target_video": target_artifact_run / "videos" / "source_dense.mp4",
        "source_mask": artifact_run / "cache" / "source_dense" / "fusion_masks" / "masks.pt",
        "target_mask": target_artifact_run / "cache" / "source_dense" / "fusion_masks" / "masks.pt",
        "attention_cache": artifact_run / "cache" / "source_dense" / "outer_block_masks",
        "target_attention_cache": target_artifact_run / "cache" / "source_dense" / "outer_block_masks",
        "frame_dir": frame_dir,
        "summary": result_run / "summary.json",
    }


def native_command(
    args: argparse.Namespace,
    *,
    artifact_run: Path,
    result_run: Path,
    run_id: str,
    prompt: str,
    positive: str,
    seed: int,
) -> list[str]:
    """Generate one prompt-matched video, fusion mask, and sparse-attention cache."""

    command = [
        str(args.runtime_python),
        "-m",
        "carnator.adapters.wan_ti2v_5b",
        "--no-env-setup",
        "--model-dir",
        str(args.model_dir.resolve()),
        "--artifact-root",
        str(artifact_run.parent),
        "--results-root",
        str(result_run.parent),
        "--run-id",
        run_id,
        "--source-prompt",
        prompt,
        "--target-prompt",
        prompt,
        "--source-positive",
        positive,
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--frames",
        str(args.frames),
        "--fps",
        str(args.fps),
        "--steps",
        str(args.steps),
        "--seed",
        str(seed),
        "--reuse-start-step",
        str(min(args.steps - 1, max(0, args.attention_step - 1))),
        "--latent-injection-mode",
        "hard",
        "--source-only",
        "--capture-outer-block-masks",
        "--no-middle-attention-mask",
        "--sparse-layers",
        "",
        "--outer-attn-backend",
        "dense",
        "--outer-sparse-layers",
        str(args.attention_layer),
        "--ffn-backend",
        "fp",
        "--outer-mask-top-p",
        str(args.outer_mask_top_p),
        "--outer-mask-min-top-k",
        str(args.outer_mask_min_top_k),
        "--negative-prompt",
        "text, watermark, logo, captions, people, extra objects, distorted object",
        "--disable-progress",
    ]
    if args.overwrite_runs:
        command.append("--clean")
    return command


def generate_cases(args: argparse.Namespace, cases: tuple[Case, ...]) -> None:
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Wan2.2-TI2V-5B checkpoint is missing: {args.model_dir}")
    wan_root = ROOT / "external" / "wan2.2"
    if not (wan_root / "wan").is_dir():
        raise FileNotFoundError(f"Wan2.2 submodule is missing: {wan_root}")

    for case in cases:
        paths = paths_for_case(args.output_root, case)
        environment = os.environ.copy()
        environment.update(
            {
                "CARNATOR_ENV_READY": "1",
                "CARNATOR_DEVICE": args.device,
                "CARNATOR_TI2V_5B_MODEL_DIR": str(args.model_dir.resolve()),
                "PYTHONPATH": os.pathsep.join(
                    (str(ROOT), str(wan_root), environment.get("PYTHONPATH", ""))
                ),
            }
        )
        prompt_runs = (
            (
                "source",
                paths["artifact_run"],
                paths["result_run"],
                case.identifier,
                case.source_prompt,
                case.source_positive,
                case.seed,
            ),
            (
                "target",
                paths["target_artifact_run"],
                paths["target_result_run"],
                paths["target_artifact_run"].name,
                case.target_prompt,
                case.target_positive,
                case.seed,
            ),
        )
        for kind, artifact_run, result_run, run_id, prompt, positive, seed in prompt_runs:
            if artifact_run.exists() and not args.overwrite_runs:
                raise FileExistsError(
                    f"{artifact_run} already exists for {case.identifier}/{kind}. "
                    "Use --overwrite-runs to replace this figure run."
                )
            subprocess.run(
                native_command(
                    args,
                    artifact_run=artifact_run,
                    result_run=result_run,
                    run_id=run_id,
                    prompt=prompt,
                    positive=positive,
                    seed=seed,
                ),
                cwd=ROOT,
                env=environment,
                check=True,
            )


def video_frame_at_fraction(video_path: Path, fraction: float) -> tuple[np.ndarray, int, int]:
    """Decode one frame without materializing the full video tensor."""

    if not video_path.is_file():
        raise FileNotFoundError(f"missing Carnator video: {video_path}")
    metadata = iio.immeta(video_path)
    fps = float(metadata.get("fps") or 0.0)
    duration = float(metadata.get("duration") or 0.0)
    estimated_frames = max(1, int(round(fps * duration))) if fps and duration else 1
    index = min(estimated_frames - 1, max(0, int(round((estimated_frames - 1) * fraction))))
    frame = iio.imread(video_path, index=index)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"expected an RGB video frame from {video_path}, got {frame.shape}")
    return frame[..., :3].astype(np.uint8), index, estimated_frames


def load_mbase(mask_path: Path) -> torch.Tensor:
    if not mask_path.is_file():
        raise FileNotFoundError(f"missing fusion mask bundle: {mask_path}")
    payload = torch.load(mask_path, map_location="cpu", weights_only=True)
    mbase = payload.get("Mbase")
    if not isinstance(mbase, torch.Tensor) or mbase.ndim != 3:
        raise ValueError(f"{mask_path} has no [time, height, width] Mbase tensor")
    return mbase.bool()


def expand_token_mask(mask: torch.Tensor, *, frames: int, height: int, width: int) -> np.ndarray:
    """Project Carnator's [token_t, token_h, token_w] mask to RGB video space."""

    token_t, token_h, token_w = mask.shape
    latent_h = height // 16
    latent_w = width // 16
    scale_h = max(1, math.ceil(latent_h / token_h))
    scale_w = max(1, math.ceil(latent_w / token_w))
    latent = mask.repeat_interleave(scale_h, dim=1).repeat_interleave(scale_w, dim=2)
    latent = latent[:, :latent_h, :latent_w]
    latent = torch.nn.functional.interpolate(
        latent[None, None].float(), size=(frames, latent_h, latent_w), mode="nearest"
    )[0, 0]
    pixels = torch.nn.functional.interpolate(
        latent[None, None], size=(frames, height, width), mode="nearest"
    )[0, 0]
    return pixels.bool().numpy()


def overlay_mbase(frame: np.ndarray, mask: np.ndarray, alpha: int) -> Image.Image:
    """Overlay exactly one canonical region: Mbase."""

    base = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", base.size, MBASE_COLOR + (0,))
    alpha_mask = Image.fromarray(mask.astype(np.uint8) * alpha)
    layer.putalpha(alpha_mask)
    return Image.alpha_composite(base, layer).convert("RGB")


def save_panel_frame(
    *,
    video_path: Path,
    mbase: torch.Tensor,
    fraction: float,
    alpha: int,
    output_path: Path,
) -> dict[str, Any]:
    frame, index, frame_count = video_frame_at_fraction(video_path, fraction)
    projected = expand_token_mask(mbase, frames=frame_count, height=frame.shape[0], width=frame.shape[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_mbase(frame, projected[min(index, frame_count - 1)], alpha).save(output_path)
    return {
        "video": str(video_path),
        "frame_index": index,
        "estimated_frame_count": frame_count,
        "output": str(output_path),
    }


def fit_cover(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    scale = max(target_size[0] / image.width, target_size[1] / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - target_size[0]) // 2
    top = (resized.height - target_size[1]) // 2
    return resized.crop((left, top, left + target_size[0], top + target_size[1]))


def render_scene_tiles(args: argparse.Namespace, panel_paths: dict[str, Path]) -> list[Path]:
    """Write the four prompt-matched Mbase-overlaid video images as standalone paper tiles."""

    ordered = (
        (CASES[0], "source"),
        (CASES[0], "target"),
        (CASES[1], "source"),
        (CASES[1], "target"),
    )
    tile_size = (800, 440)
    output_dir = args.output_root / "tiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for case, kind in ordered:
        output = output_dir / f"{case.identifier}_{kind}_mbase.png"
        with Image.open(panel_paths[f"{case.identifier}_{kind}"]) as image:
            fit_cover(image.convert("RGB"), tile_size).save(output, quality=95)
        outputs.append(output)
    return outputs


def attention_entry(cache_root: Path, *, step: int, layer: int, branch: str = "cond") -> tuple[Path, dict[str, Any]]:
    metadata_path = cache_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing persisted outer attention cache: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = metadata.get("entries", [])
    if not entries:
        raise ValueError(f"outer attention cache has no entries: {cache_root}")
    exact = [entry for entry in entries if entry.get("step") == step and entry.get("layer") == layer and entry.get("cfg_branch") == branch]
    candidates = exact or [entry for entry in entries if entry.get("cfg_branch") == branch]
    if not candidates:
        candidates = entries
    selected = min(
        candidates,
        key=lambda entry: (abs(int(entry["step"]) - step), abs(int(entry["layer"]) - layer)),
    )
    path = cache_root / str(selected["path"])
    if not path.is_file():
        raise FileNotFoundError(f"outer attention cache entry is missing: {path}")
    return path, selected


def consensus_attention_map(mask_path: Path, consensus: float) -> np.ndarray:
    mask = torch.load(mask_path, map_location="cpu", weights_only=True)
    if not isinstance(mask, torch.Tensor) or mask.ndim != 3 or mask.dtype != torch.bool:
        raise ValueError(f"{mask_path} is not a [heads, query_blocks, key_blocks] boolean mask")
    return (mask.float().mean(dim=0) >= consensus).numpy()


def render_attention_tile(active: np.ndarray, size: tuple[int, int]) -> Image.Image:
    pixels = np.full((*active.shape, 3), ATTENTION_EMPTY, dtype=np.uint8)
    pixels[active] = ATTENTION_COLOR
    return Image.fromarray(pixels).resize(size, Image.Resampling.NEAREST)


def render_attention_tiles(args: argparse.Namespace, attention_records: list[dict[str, Any]]) -> list[Path]:
    """Write one prompt-specific sparse self-attention map per request."""

    tile_size = (800, 400)
    output_dir = args.output_root / "tiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for record in attention_records:
        tile = render_attention_tile(record["active"], tile_size)
        output = output_dir / f"{record['identifier']}_{record['kind']}_sparse_attention.png"
        tile.save(output, quality=95)
        outputs.append(output)
    return outputs


def render(args: argparse.Namespace) -> dict[str, Any]:
    panel_paths: dict[str, Path] = {}
    frame_records: dict[str, dict[str, Any]] = {}
    attention_records: list[dict[str, Any]] = []

    for case in CASES:
        paths = paths_for_case(args.output_root, case)
        for kind, video_key, mask_key in (
            ("source", "source_video", "source_mask"),
            ("target", "target_video", "target_mask"),
        ):
            panel_path = paths["frame_dir"] / f"{kind}_mbase.png"
            panel_paths[f"{case.identifier}_{kind}"] = panel_path
            frame_record = save_panel_frame(
                video_path=paths[video_key],
                mbase=load_mbase(paths[mask_key]),
                fraction=FIRST_FRAME_FRACTION,
                alpha=args.mbase_alpha,
                output_path=panel_path,
            )
            frame_record["mask"] = str(paths[mask_key])
            frame_records[f"{case.identifier}_{kind}"] = frame_record

        for kind, object_name, cache_key in (
            ("source", case.source_positive, "attention_cache"),
            ("target", case.target_positive, "target_attention_cache"),
        ):
            mask_path, entry = attention_entry(
                paths[cache_key], step=args.attention_step, layer=args.attention_layer
            )
            active = consensus_attention_map(mask_path, args.attention_head_consensus)
            attention_records.append(
                {
                    "identifier": case.identifier,
                    "kind": kind,
                    "object": object_name,
                    "label": case.source_label if kind == "source" else case.target_label,
                    "entry": entry,
                    "mask": str(mask_path),
                    "active": active,
                    "density": float(active.mean()),
                }
            )

    scene_tiles = render_scene_tiles(args, panel_paths)
    attention_tiles = render_attention_tiles(args, attention_records)
    metadata = {
        "description": "Eight standalone prompt-aligned Carnator tiles with Mbase-only overlays and one real sparse self-attention map per prompt.",
        "cases": [asdict(case) for case in CASES],
        "render": {
            "frame_fraction": FIRST_FRAME_FRACTION,
            "mbase_alpha": args.mbase_alpha,
            "attention_step": args.attention_step,
            "attention_layer": args.attention_layer,
            "attention_head_consensus": args.attention_head_consensus,
            "mask_policy": "Each video frame uses fusion Mbase from the same prompt-specific source run.",
        },
        "frames": frame_records,
        "attention": [
            {
                "label": record["label"],
                "kind": record["kind"],
                "object": record["object"],
                "mask": record["mask"],
                "entry": record["entry"],
                "density": record["density"],
            }
            for record in attention_records
        ],
        "outputs": {
            "mbase_tiles": [str(path) for path in scene_tiles],
            "attention_tiles": [str(path) for path in attention_tiles],
        },
    }
    metadata_path = args.output_root / "figure_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    selected_cases = tuple(case for case in CASES if case.identifier in set(args.cases))
    if args.generate:
        generate_cases(args, selected_cases)
    if args.generate_only:
        return 0
    metadata = render(args)
    print(json.dumps(metadata["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
