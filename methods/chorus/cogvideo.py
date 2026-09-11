from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from core.types import Generation, Request

from adapters.cogvideo_attention import (
    CogVideoStepState,
    RegionalAttentionState,
    install_regional_blocks,
    track_transformer_steps,
)
from .cogvideo_mask import build_cogvideo_edit_mask
from .grounded_sam import GroundedSam2Segmenter
from .llm import extract_prompt_diff


FORMAT = "cogvideox15-chorus-trajectory/v1"


def _work_dir(adapter: Any) -> Path:
    path = getattr(adapter, "request_cache_dir", None)
    if path is None:
        raise RuntimeError("CogVideo Chorus requires a request-private cache directory")
    value = Path(path)
    value.mkdir(parents=True, exist_ok=True)
    return value


def _token_indices(tokenizer: Any, prompt: str, entities: list[Any], maximum: int) -> tuple[int, ...]:
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=maximum,
    ).input_ids
    indices: set[int] = set()
    for entity in entities:
        entity_ids = tokenizer(str(entity), add_special_tokens=False).input_ids
        if not entity_ids:
            continue
        width = len(entity_ids)
        for start in range(max(0, len(prompt_ids) - width + 1)):
            if prompt_ids[start:start + width] == entity_ids:
                indices.update(range(start, start + width))
    return tuple(sorted(indices))


def _source(adapter: Any, request: Request) -> Generation:
    delegate = adapter._load_delegate()
    delegate._load()
    pipe = delegate.pipe
    steps = int(adapter.config["baseline_steps"])
    stages = adapter.config["chorus"]["stages"]
    k1, k2 = int(stages["k1"]), int(stages["k2"])
    trajectory: dict[int, Any] = {}

    def capture(_pipe: Any, index: int, _timestep: Any, values: dict[str, Any]) -> dict[str, Any]:
        completed = index + 1
        if k1 <= completed <= k2:
            trajectory[completed] = values["latents"].detach().cpu()
        return values

    call = adapter._call_args(request, steps)
    call.update({
        "callback_on_step_end": capture,
        "callback_on_step_end_tensor_inputs": ["latents"],
    })
    with adapter._profile_transformer(pipe) as profile:
        output = pipe(**call)
    expected = set(range(k1, k2 + 1))
    if set(trajectory) != expected:
        raise RuntimeError("CogVideo Chorus source trajectory is incomplete")
    source_video = _work_dir(adapter) / "source.mp4"
    delegate.save_video(output.frames[0], source_video)
    payload = {
        "format": FORMAT,
        "source_prompt": request.prompt,
        "source_video_path": str(source_video),
        "trajectory": trajectory,
        "latents": trajectory[k1],
        "scheduler": pipe.scheduler.__class__.__name__,
        "timesteps": adapter._scheduler_timesteps(pipe.scheduler, steps, pipe._execution_device),
        "steps": steps,
        "k1": k1,
        "k2": k2,
    }
    return Generation(
        video=output.frames[0],
        latent=payload,
        latent_step=k1,
        executed_steps=steps,
        extra={
            "executed_steps": steps,
            "dit_forward_s": profile[0],
            "scheduler": payload["scheduler"],
            "chorus": {"mode": "source_capture", "k1": k1, "k2": k2, "trajectory_states": len(trajectory)},
        },
    )


def _mask(adapter: Any, cached: dict[str, Any], request: Request, prompt_diff: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    override = adapter.config["chorus"].get("mask_bundle_override")
    if override is not None:
        return override, {"backend": "manual_override"}
    source_entities = prompt_diff.get("source_entities") or prompt_diff.get("removed_tokens") or []
    phrase = " ".join(dict.fromkeys(str(item).strip() for item in source_entities if str(item).strip()))
    if not phrase:
        raise RuntimeError("CogVideo Chorus prompt diff has no source entity for segmentation")
    delegate = adapter._load_delegate()
    transformer = delegate.pipe.transformer
    latent = cached["latents"]
    latent_shape = (int(latent.shape[1]), int(latent.shape[3]), int(latent.shape[4]))
    config = {
        **adapter.config["chorus"].get("grounded_sam2", {}),
        "device": adapter.config["device"],
        "mask_dilation_edit": adapter.config["chorus"].get("mask_dilation_edit", 8),
        "mask_dilation_see": adapter.config["chorus"].get("mask_dilation_see", 14),
    }
    segmenter = getattr(adapter, "_cogvideo_segmenter", None)
    if segmenter is None:
        segmenter = GroundedSam2Segmenter(config)
        adapter._cogvideo_segmenter = segmenter
    patch_t = int(transformer.config.patch_size_t or 1)
    patch = int(transformer.config.patch_size)
    return build_cogvideo_edit_mask(
        Path(cached["source_video_path"]),
        phrase,
        latent_shape=latent_shape,
        patch_size=(patch_t, patch, patch),
        config=config,
        segmenter=segmenter,
    )


def _target(adapter: Any, request: Request, cached: dict[str, Any]) -> Generation:
    import torch

    delegate = adapter._load_delegate()
    delegate._load()
    pipe = delegate.pipe
    config = adapter.config["chorus"]
    steps, k1, k2 = int(cached["steps"]), int(cached["k1"]), int(cached["k2"])
    diff_started = time.perf_counter()
    prompt_diff = config.get("prompt_diff_override") or extract_prompt_diff(cached["source_prompt"], request.prompt)
    diff_seconds = time.perf_counter() - diff_started
    mask_started = time.perf_counter()
    masks, mask_metadata = _mask(adapter, cached, request, prompt_diff)
    mask_seconds = time.perf_counter() - mask_started
    target_entities = prompt_diff.get("target_entities") or prompt_diff.get("added_tokens") or []
    maximum = int(pipe.transformer.config.max_text_seq_length)
    differential = _token_indices(pipe.tokenizer, request.prompt, target_entities, maximum)
    if not differential:
        raise RuntimeError("CogVideo Chorus TGAA could not map target differential tokens")
    region = RegionalAttentionState(
        masks["token_edit"],
        masks["token_see"],
        first_step=k1,
        last_step=k2,
        total_steps=steps,
        differential_tokens=differential,
        gamma_k=float(config.get("gamma_k", 1.5)),
        gamma_o=float(config.get("gamma_o", 1.5)),
    )
    tracker = CogVideoStepState(start_step=k1, on_begin=region.set_step)

    def fuse(_pipe: Any, index: int, _timestep: Any, values: dict[str, Any]) -> dict[str, Any]:
        completed = k1 + index + 1
        if completed <= k2:
            source = cached["trajectory"][completed].to(values["latents"].device, values["latents"].dtype)
            edit = masks["latent_edit"].to(values["latents"].device).bool()[None, :, None]
            values["latents"] = torch.where(edit, values["latents"], source)
        return values

    call = adapter._call_args(request, steps)
    call.update({
        "latents": cached["latents"],
        "callback_on_step_end": fuse,
        "callback_on_step_end_tensor_inputs": ["latents"],
    })
    schedule = adapter._resume_original_schedule(
        pipe.scheduler,
        steps=steps,
        reuse_step=k1,
        expected=[int(item) for item in cached["timesteps"]],
    )
    with (
        schedule,
        install_regional_blocks(pipe.transformer, region),
        track_transformer_steps(pipe.transformer, tracker),
        adapter._profile_transformer(pipe) as profile,
    ):
        output = pipe(**call)
    preprocess = diff_seconds + mask_seconds
    return Generation(
        video=output.frames[0],
        latent_step=k1,
        executed_steps=steps - k1,
        extra={
            "executed_steps": steps - k1,
            "dit_forward_s": profile[0],
            "scheduler": cached["scheduler"],
            "schedule_resume": "original_timestep_suffix",
            "excluded_latency_s": preprocess,
            "chorus_preprocess_s": preprocess,
            "chorus_llm_s": diff_seconds,
            "chorus_sam_s": mask_seconds,
            "chorus_llm": prompt_diff,
            "chorus_mask": mask_metadata,
            "chorus": {
                "mode": "target_reuse",
                "stages": {"stage1": [0, k1], "stage2": [k1, k2], "stage3": [k2, steps]},
                "source_trajectory_window": [k1, k2],
                "regional_blocks": len(pipe.transformer.transformer_blocks),
                "regional_calls": region.calls,
                "tgaa_tokens": list(differential),
                "gamma_k": region.gamma_k,
                "gamma_o": region.gamma_o,
            },
        },
    )


def generate(adapter: Any, request: Request, *, cached_latent: Any, capture: bool) -> Generation:
    if cached_latent is None:
        if not capture:
            raise RuntimeError("CogVideo Chorus source generation must capture its trajectory")
        return _source(adapter, request)
    if not isinstance(cached_latent, dict) or cached_latent.get("format") != FORMAT:
        raise ValueError("invalid CogVideo Chorus source trajectory")
    return _target(adapter, request, cached_latent)
