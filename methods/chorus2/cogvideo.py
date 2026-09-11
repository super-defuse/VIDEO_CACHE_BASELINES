from __future__ import annotations

from pathlib import Path
from typing import Any

from adapters.cogvideo_attention import CogVideoStepState, track_transformer_steps
from adapters.cogvideo_sparse import (
    CogVideoAttentionProcessors,
    CogVideoBlockMaskCache,
    CogVideoBlockMaskConfig,
    CogVideoFlexExecutor,
    CogVideoMaskCaptureProcessor,
    CogVideoSparseProcessor,
)
from core.types import Generation, Request


FORMAT = "cogvideox15-chorus2-source-routing/v1"


def _config(adapter: Any) -> tuple[dict[str, Any], CogVideoBlockMaskConfig, list[int]]:
    values = adapter.config["chorus2"]
    transformer = adapter._load_delegate().pipe.transformer
    layers = list(range(len(transformer.transformer_blocks)))
    if values.get("layers", "all") != "all":
        layers = sorted({int(item) for item in values["layers"]})
    if layers != list(range(42)):
        raise ValueError("CogVideo Chorus2 requires all 42 joint-attention layers")
    block = CogVideoBlockMaskConfig(
        q_block_size=int(values.get("q_block_size", 128)),
        k_block_size=int(values.get("k_block_size", 128)),
        top_p=float(values.get("top_p", 0.95)),
        min_top_k=float(values.get("min_top_k", 0.1)),
    )
    return values, block, layers


def _source(adapter: Any, request: Request) -> Generation:
    delegate = adapter._load_delegate()
    delegate._load()
    pipe = delegate.pipe
    values, block, layers = _config(adapter)
    steps = int(adapter.config["baseline_steps"])
    first_sparse_step = int(values.get("sparse_start_step", 10)) + 1
    root = Path(adapter.request_cache_dir) / "chorus2_masks"
    cache = CogVideoBlockMaskCache(root, block, layers=layers, start_step=first_sparse_step, create=True)
    state = CogVideoStepState()
    processors = CogVideoAttentionProcessors(
        pipe.transformer,
        lambda layer: CogVideoMaskCaptureProcessor(cache, state, layer),
        layers,
    )
    call = adapter._call_args(request, steps)
    call["output_type"] = "latent"
    with processors, track_transformer_steps(pipe.transformer, state), adapter._profile_transformer(pipe) as profile:
        pipe(**call)
    cache.finalize()
    branches = ("uncond", "cond") if float(adapter.config["guidance_scale"]) > 1.0 else ("cond",)
    cache.validate_coverage(steps=range(first_sparse_step, steps + 1), branches=branches, layers=layers)
    payload = {
        "format": FORMAT,
        "source_prompt": request.prompt,
        "mask_cache": str(root),
        "scheduler": pipe.scheduler.__class__.__name__,
        "steps": steps,
        "first_sparse_step": first_sparse_step,
        "layers": layers,
        "mask_config": block.__dict__,
    }
    return Generation(
        video=None,
        latent=payload,
        latent_step=first_sparse_step - 1,
        executed_steps=steps,
        extra={
            "executed_steps": steps,
            "dit_forward_s": profile[0],
            "scheduler": payload["scheduler"],
            "chorus2": {"mode": "source_routing_capture", **cache.summary()},
        },
    )


def _target(adapter: Any, request: Request, payload: dict[str, Any]) -> Generation:
    delegate = adapter._load_delegate()
    delegate._load()
    pipe = delegate.pipe
    values, block, layers = _config(adapter)
    steps = int(adapter.config["baseline_steps"])
    if int(payload["steps"]) != steps or payload["layers"] != layers:
        raise ValueError("CogVideo Chorus2 source routing does not match the target configuration")
    cache = CogVideoBlockMaskCache(Path(payload["mask_cache"]), block)
    branches = ("uncond", "cond") if float(adapter.config["guidance_scale"]) > 1.0 else ("cond",)
    cache.validate_coverage(
        steps=range(int(payload["first_sparse_step"]), steps + 1),
        branches=branches,
        layers=layers,
    )
    executor = CogVideoFlexExecutor(cache, layer_refresh_size=int(values.get("layer_refresh_size", 8)))
    state = CogVideoStepState()
    processors = CogVideoAttentionProcessors(
        pipe.transformer,
        lambda layer: CogVideoSparseProcessor(executor, state, layer),
        layers,
    )
    with processors, track_transformer_steps(pipe.transformer, state), adapter._profile_transformer(pipe) as profile:
        output = pipe(**adapter._call_args(request, steps))
    expected = len(layers) * (steps - int(payload["first_sparse_step"]) + 1) * len(branches)
    if executor.runs != expected:
        raise RuntimeError(f"CogVideo Chorus2 ran {executor.runs} sparse calls, expected {expected}")
    return Generation(
        video=output.frames[0],
        executed_steps=steps,
        extra={
            "executed_steps": steps,
            "dit_forward_s": profile[0],
            "scheduler": payload["scheduler"],
            "chorus2": {
                "mode": "target_full_schedule",
                "latent_restart": False,
                "first_sparse_step": payload["first_sparse_step"],
                "source_masks": cache.summary(),
                "execution": executor.summary(),
            },
        },
    )


def generate(adapter: Any, request: Request, *, cached_latent: Any, capture: bool) -> Generation:
    if cached_latent is None:
        if not capture:
            raise RuntimeError("CogVideo Chorus2 source generation must capture attention routing")
        return _source(adapter, request)
    if not isinstance(cached_latent, dict) or cached_latent.get("format") != FORMAT:
        raise ValueError("invalid CogVideo Chorus2 source routing payload")
    return _target(adapter, request, cached_latent)
