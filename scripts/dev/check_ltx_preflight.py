#!/usr/bin/env python3
"""P0 preflight for the LTX-Video adapter.

Verifies the three things the LTX integration depends on and that cannot be
assumed from the checkpoint layout alone:

1. diffusers 0.35.x is importable and the LTX classes exist (the user-site
   diffusers 0.40.0 shadows the conda env one and breaks on huggingface_hub,
   so this must run with PYTHONNOUSERSITE=1).
2. ``text_encoder/config.json`` has no sibling weights: the T5-v1.1-XXL
   encoder must be built from the config and loaded from the standalone
   ``models/t5xxl_fp16.safetensors``.  Reports missing/unexpected keys.
3. ``export_to_video`` is importable for the LTX video write-out path.

Optionally (``--smoke``) assembles the full pipeline and runs a 2-step
generation at a tiny resolution to prove the assembled components work.

Run with the screen env used by the LTX baselines::

    PYTHONNOUSERSITE=1 conda run -n wan python scripts/dev/check_ltx_preflight.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "LTX-Video-0.9.7-dev"
T5_WEIGHTS = ROOT / "models" / "t5xxl_fp16.safetensors"


def check_diffusers() -> None:
    import diffusers

    location = Path(diffusers.__file__).resolve()
    if ".local" in location.parts:
        raise SystemExit(
            f"diffusers is being imported from user site-packages ({location}); "
            f"re-run with PYTHONNOUSERSITE=1, as the LTX scripts do."
        )
    from diffusers import (  # noqa: F401
        AutoencoderKLLTXVideo,
        LTXConditionPipeline,
        LTXVideoTransformer3DModel,
    )
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: F401

    print(f"diffusers {diffusers.__version__} at {location}")
    print("  LTXConditionPipeline / LTXVideoTransformer3DModel / AutoencoderKLLTXVideo / "
          "FlowMatchEulerDiscreteScheduler all import")


def check_text_encoder() -> None:
    """Build T5-v1.1-XXL encoder from config + standalone safetensors."""
    import torch
    from safetensors.torch import load_file
    from transformers import T5Config, T5EncoderModel

    subfolder = MODEL_DIR / "text_encoder"
    weights = sorted(subfolder.glob("*.safetensors"))
    if weights:
        raise SystemExit(f"unexpected encoder weights in {subfolder}: {weights}")
    if not T5_WEIGHTS.is_file():
        raise SystemExit(f"standalone T5 weights are missing: {T5_WEIGHTS}")

    config = T5Config.from_pretrained(subfolder)
    model = T5EncoderModel(config)
    state = load_file(str(T5_WEIGHTS))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # ``shared`` and ``encoder.embed_tokens`` are two views of one embedding and
    # T5EncoderModel keeps both; anything else missing means a layout mismatch.
    allowed_missing = {"shared.weight", "encoder.embed_tokens.weight"}
    real_missing = sorted(set(missing) - allowed_missing)
    if real_missing:
        raise SystemExit(f"T5 state dict is missing {len(real_missing)} keys, e.g. {real_missing[:5]}")
    if unexpected:
        raise SystemExit(f"T5 state dict has {len(unexpected)} unexpected keys, e.g. {sorted(unexpected)[:5]}")
    total = sum(item.numel() for item in state.values())
    print(f"T5-v1.1-XXL encoder loads from {T5_WEIGHTS.name}: "
          f"{len(state)} tensors, {total / 1e9:.2f}B params, "
          f"{sorted({str(v.dtype) for v in state.values()})}")
    del model, state
    torch.cuda.empty_cache()


def check_video_export() -> None:
    from diffusers.utils import export_to_video  # noqa: F401

    print("diffusers.utils.export_to_video is importable")


def check_transformer_and_vae() -> None:
    """Load the real weights, not just the configs, and print resolved shapes."""
    import torch
    from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    dtype = torch.bfloat16
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        MODEL_DIR, subfolder="transformer", torch_dtype=dtype
    )
    vae = AutoencoderKLLTXVideo.from_pretrained(MODEL_DIR, subfolder="vae", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler")
    params = sum(item.numel() for item in transformer.parameters())
    print(f"transformer: {len(transformer.transformer_blocks)} blocks, "
          f"{params / 1e9:.2f}B params, dtype {next(transformer.parameters()).dtype}")
    print(f"vae: latent_channels={vae.config.latent_channels}, "
          f"temporal_compression={getattr(vae, 'temporal_compression_ratio', 'n/a')}, "
          f"spatial_compression={getattr(vae, 'spatial_compression_ratio', 'n/a')}")
    print(f"scheduler: {scheduler.__class__.__name__}")
    del transformer, vae
    torch.cuda.empty_cache()


def smoke(device: str, height: int, width: int, frames: int, steps: int, seed: int) -> None:
    """Assemble the pipeline exactly as the adapter will and run a tiny job."""
    import torch
    from diffusers import AutoencoderKLLTXVideo, LTXConditionPipeline, LTXVideoTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from safetensors.torch import load_file
    from transformers import T5Config, T5EncoderModel, T5Tokenizer

    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    text_encoder = T5EncoderModel(T5Config.from_pretrained(MODEL_DIR / "text_encoder"))
    text_encoder.load_state_dict(load_file(str(T5_WEIGHTS)), strict=False)
    text_encoder = text_encoder.to(device=device, dtype=dtype).eval()

    pipe = LTXConditionPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler"),
        text_encoder=text_encoder,
        tokenizer=T5Tokenizer.from_pretrained(MODEL_DIR, subfolder="tokenizer"),
        transformer=LTXVideoTransformer3DModel.from_pretrained(
            MODEL_DIR, subfolder="transformer", torch_dtype=dtype
        ),
        vae=AutoencoderKLLTXVideo.from_pretrained(MODEL_DIR, subfolder="vae", torch_dtype=dtype),
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        output = pipe(
            prompt="a red panda walking through a bamboo forest",
            negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
            height=height,
            width=width,
            num_frames=frames,
            num_inference_steps=steps,
            guidance_scale=3.0,
            generator=generator,
            output_type="pil",
        )
    frame = output.frames[0][0]
    print(f"smoke generation produced {len(output.frames[0])} frames, first frame size {frame.size}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="Assemble the pipeline and run one tiny generation.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260218)
    args = parser.parse_args()

    for path in (MODEL_DIR, T5_WEIGHTS):
        if not path.exists():
            raise SystemExit(f"missing expected path: {path}")

    check_diffusers()
    check_text_encoder()
    check_video_export()
    check_transformer_and_vae()
    if args.smoke:
        smoke(args.device, args.height, args.width, args.frames, args.steps, args.seed)
    print("LTX preflight OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
