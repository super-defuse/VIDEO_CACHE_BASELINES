from __future__ import annotations

import hashlib
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from adapters.cogvideo_attention import (
    CogVideoStepState,
    RegionalAttentionState,
    install_regional_blocks,
    track_transformer_steps,
)
from adapters.cogvideo_sparse import (
    CogVideoAttentionProcessors,
    CogVideoBlockMaskCache,
    CogVideoBlockMaskConfig,
    CogVideoFlexExecutor,
    CogVideoMaskCaptureProcessor,
    CogVideoSparseProcessor,
)
from carnator.cogvideo_quantized_ffn import CogVideoQuantizedFFNHooks
from carnator.early_signature import compare_early_signatures, write_early_signature
from core.types import Generation, Request
from methods.chorus.llm import extract_prompt_diff


FORMAT = "cogvideox15-carnator-boundary-b5/v1"


def _parse_layers(value: Any, count: int) -> list[int]:
    if value in (None, "all"):
        return list(range(count))
    if isinstance(value, (list, tuple)):
        result = sorted({int(item) for item in value})
    else:
        result: list[int] = []
        for item in str(value).split(","):
            if "-" in item:
                low, high = item.split("-", 1)
                result.extend(range(int(low), int(high) + 1))
            elif item.strip():
                result.append(int(item))
        result = sorted(set(result))
    if any(layer < 0 or layer >= count for layer in result):
        raise ValueError(f"CogVideo Carnator layer indices must be in [0, {count - 1}]")
    return result


def _configuration(adapter: Any) -> tuple[dict[str, Any], CogVideoBlockMaskConfig, list[int], list[int]]:
    values = adapter.config["carnator_b5"]
    count = len(adapter._load_delegate().pipe.transformer.transformer_blocks)
    region = _parse_layers(values.get("regional_layers", "14-27"), count)
    outer = _parse_layers(values.get("outer_sparse_layers", "0-13,28-41"), count)
    if set(region) & set(outer) or sorted(region + outer) != list(range(count)):
        raise ValueError("CogVideo Carnator regional and outer sparse layers must partition all transformer layers")
    block = CogVideoBlockMaskConfig(
        q_block_size=int(values.get("q_block_size", 128)),
        k_block_size=int(values.get("k_block_size", 128)),
        top_p=float(values.get("top_p", 0.95)),
        min_top_k=float(values.get("min_top_k", 0.1)),
    )
    return values, block, region, outer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _latent_layout(value: Any) -> Any:
    if value.ndim != 5 or value.shape[0] != 1:
        raise ValueError("CogVideo Carnator step latent must have shape [1,T,C,H,W]")
    return value[0].permute(1, 0, 2, 3).contiguous()


def _source(adapter: Any, request: Request) -> Generation:
    import torch
    from carnator.cogvideo_attention_probe import CogVideoAttentionProbe

    delegate = adapter._load_delegate()
    delegate._load()
    pipe = delegate.pipe
    values, block, _region, outer = _configuration(adapter)
    steps = int(adapter.config["baseline_steps"])
    probe_step = int(values.get("probe_step", 5))
    reuse_step = int(values.get("reuse_step", 10))
    if not 0 < probe_step < reuse_step < steps:
        raise ValueError("CogVideo Carnator requires 0 < probe_step < reuse_step < steps")

    # Capture attention at probe_step for fusion mask
    attention_layer = int(values.get("attention_probe_layer", 15))
    attention_probe = CogVideoAttentionProbe(pipe.transformer, attention_layer, target_step=probe_step)
    attention_captured = {"self": None, "cross": None, "text_emb": None, "step": None}

    trajectory: dict[int, Any] = {}
    step5: list[Any] = []

    def capture(_pipe: Any, index: int, _timestep: Any, state: dict[str, Any]) -> dict[str, Any]:
        completed = index + 1
        if completed == probe_step:
            step5.append(state["latents"].detach().cpu())
            # Mark that we're at step 5 for attention capture
            attention_captured["step"] = completed
        if reuse_step <= completed <= steps:
            trajectory[completed] = state["latents"].detach().cpu()
        return state

    root = Path(adapter.request_cache_dir)
    mask_root = root / "carnator_outer_masks"
    mask_cache = CogVideoBlockMaskCache(
        mask_root,
        block,
        layers=outer,
        start_step=reuse_step + 1,
        create=True,
    )
    tracker = CogVideoStepState()
    processors = CogVideoAttentionProcessors(
        pipe.transformer,
        lambda layer: CogVideoMaskCaptureProcessor(mask_cache, tracker, layer),
        outer,
    )
    call = adapter._call_args(request, steps)
    call.update({
        "callback_on_step_end": capture,
        "callback_on_step_end_tensor_inputs": ["latents"],
        "output_type": "latent",
    })

    # Get text embeddings for cross-attention computation
    guidance_scale = float(adapter.config["guidance_scale"])
    attention_captured["text_emb"] = _get_text_embeddings(pipe, request.prompt, guidance_scale)

    # Install attention probe hook
    with attention_probe, processors, track_transformer_steps(pipe.transformer, tracker), adapter._profile_transformer(pipe) as profile:
        output = pipe(**call)

    if len(step5) != 1 or set(trajectory) != set(range(reuse_step, steps + 1)):
        raise RuntimeError("CogVideo Carnator source trajectory is incomplete")
    mask_cache.finalize()
    branches = ("uncond", "cond") if float(adapter.config["guidance_scale"]) > 1.0 else ("cond",)
    mask_cache.validate_coverage(
        steps=range(reuse_step + 1, steps + 1), branches=branches, layers=outer
    )
    anchor_path = root / "carnator_step10.pt"
    torch.save(trajectory[reuse_step], anchor_path)
    torch.cuda.empty_cache()
    source_frames = delegate.decode_video(output.frames)
    source_video = root / "source.mp4"
    delegate.save_video(source_frames, source_video)

    # Generate fusion mask from captured attention during forward pass
    mask_bundle_path = None
    try:
        # Use attention captured during forward pass at step 5
        if 'hidden_states' not in attention_probe.captured:
            raise RuntimeError("Attention probe did not capture hidden states during forward pass")

        mask_bundle_path = _generate_fusion_mask_from_captured(
            adapter=adapter,
            probe=attention_probe,
            request=request,
            root=root,
        )
    except Exception as e:
        print(f"Warning: Fusion mask generation failed: {e}")
        print("Carnator will continue without fusion mask")
        import traceback
        traceback.print_exc()

    torch.cuda.empty_cache()
    decoded_tail = delegate.decode_tail(step5[0])
    payload = {
        "format": FORMAT,
        "source_prompt": request.prompt,
        "source_video_path": str(source_video),
        "step5": step5[0],
        "step5_tail": decoded_tail.cpu(),
        "trajectory": trajectory,
        "latents": trajectory[reuse_step],
        "anchor_path": str(anchor_path),
        "mask_cache": str(mask_root),
        "mask_bundle_path": str(mask_bundle_path) if mask_bundle_path else None,
        "mask_method": "fusion" if mask_bundle_path else None,
        "scheduler": pipe.scheduler.__class__.__name__,
        "timesteps": adapter._scheduler_timesteps(pipe.scheduler, steps, pipe._execution_device),
        "steps": steps,
        "probe_step": probe_step,
        "reuse_step": reuse_step,
        "outer_layers": outer,
        "mask_config": block.__dict__,
    }
    return Generation(
        video=source_frames,
        latent=payload,
        latent_step=reuse_step,
        executed_steps=steps,
        extra={
            "executed_steps": steps,
            "dit_forward_s": profile[0],
            "scheduler": payload["scheduler"],
            "carnator_b5": {
                "mode": "source_capture",
                "probe_step": probe_step,
                "reuse_step": reuse_step,
                "trajectory_states": len(trajectory),
                "outer_masks": mask_cache.summary(),
                "mask_method": payload["mask_method"],
            },
        },
    )


def _get_text_embeddings(pipe: Any, prompt: str, guidance_scale: float) -> Any:
    """Get text embeddings from the prompt."""
    import torch

    # Determine if classifier-free guidance is used based on guidance_scale
    do_classifier_free_guidance = guidance_scale > 1.0

    # Encode prompt using the pipe's text encoder
    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            device=pipe._execution_device,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
    return prompt_embeds


def _generate_fusion_mask_from_captured(
    adapter: Any,
    probe: Any,
    request: Request,
    root: Path,
) -> Path:
    """
    Generate fusion mask from attention scores captured during forward pass.

    Args:
        adapter: CogVideoX adapter
        probe: CogVideoAttentionProbe that captured attention
        request: Request object
        root: Root directory for saving mask bundle

    Returns:
        Path to saved mask bundle
    """
    import torch
    from carnator.cogvideo_attention_probe import (
        compute_self_contrast_score,
        compute_cross_attention_score,
        infer_token_grid_shape,
    )
    from carnator.masks import (
        FusionMaskConfig,
        build_canonical_fusion_mask,
        save_mask_bundle,
    )

    values = adapter.config["carnator_b5"]

    # Get text token count from prompt
    # For CogVideoX, tokenize the prompt to get text token count
    delegate = adapter._load_delegate()
    pipe = delegate.pipe

    # Get text embeddings to determine text token count
    guidance_scale = float(adapter.config["guidance_scale"])
    text_embeddings = _get_text_embeddings(pipe, request.prompt, guidance_scale)
    text_tokens = text_embeddings.shape[1]

    # Extract attention scores from captured hidden states
    self_scores, cross_scores = probe.extract_scores_from_captured(
        text_tokens=text_tokens,
        rotary_emb=None,
    )

    # Infer token grid shape from video tokens
    video_tokens = self_scores.shape[2]  # [B, H, T_video, T_video]
    token_grid_shape = infer_token_grid_shape(video_tokens, frames=49)

    # Compute self-contrast score
    top_p = float(values.get("mask_top_p", 0.02))
    background_p = float(values.get("mask_background_p", 0.02))

    self_contrast = compute_self_contrast_score(
        self_scores=self_scores.cpu(),
        token_grid_shape=token_grid_shape,
        top_p=top_p,
        background_p=background_p,
    )

    # Compute cross-attention score
    cross_attention = compute_cross_attention_score(
        cross_scores=cross_scores.cpu(),
        token_grid_shape=token_grid_shape,
    )

    # Build fusion mask
    fusion_config = FusionMaskConfig(
        eps=float(values.get("fusion_eps", 0.1)),
        gamma=float(values.get("fusion_gamma", 2.0)),
        high_threshold=float(values.get("fusion_high_threshold", 0.7)),
        low_threshold=float(values.get("fusion_low_threshold", 0.3)),
        spatial_medit_radius=int(values.get("mask_spatial_medit_radius", 2)),
        spatial_msee_radius=int(values.get("mask_spatial_msee_radius", 1)),
    )

    fusion_mask = build_canonical_fusion_mask(
        self_contrast=self_contrast,
        cross_attention=cross_attention,
        config=fusion_config,
    )

    # Save mask bundle
    mask_dir = root / "carnator_fusion_mask"
    save_mask_bundle(
        mask_dir=mask_dir,
        fusion_mask=fusion_mask,
        self_contrast=self_contrast,
        cross_attention=cross_attention,
        token_grid_shape=token_grid_shape,
        config=fusion_config,
    )

    return mask_dir


def _generate_fusion_mask_from_latent(
    adapter: Any,
    pipe: Any,
    request: Request,
    step5_latent: Any,
    text_embeddings: Any,
    root: Path,
    attention_layer: int = 15,
) -> Path:
    """
    Generate fusion mask from attention scores computed from step 5 latent.

    Returns:
        Path to saved mask bundle
    """
    import torch
    from carnator.cogvideo_attention_probe import (
        CogVideoAttentionProbe,
        compute_self_contrast_score,
        compute_cross_attention_score,
        infer_token_grid_shape,
    )
    from carnator.masks import (
        FusionMaskConfig,
        build_canonical_fusion_mask,
        save_mask_bundle,
        robust_norm,
    )

    values = adapter.config["carnator_b5"]

    # Infer token grid shape from latent
    token_grid_shape = infer_token_grid_shape(step5_latent.shape)

    # Create attention probe
    probe = CogVideoAttentionProbe(
        transformer=pipe.transformer,
        layer_index=attention_layer,
    )

    # Manually compute attention scores from the captured latent state
    with torch.no_grad():
        # Prepare inputs - step5_latent is [B, C, T, H, W]
        hidden_states = step5_latent.to(pipe._execution_device)

        # For CogVideoX, hidden states need to be in token space
        # The latent is already in the right format [B, C, T, H, W]
        # We need to flatten spatial-temporal dimensions to token sequence
        batch, channels, t, h, w = hidden_states.shape
        # Reshape to [B, T*H*W, C]
        hidden_states_flat = hidden_states.permute(0, 2, 3, 4, 1).reshape(batch, t*h*w, channels)

        # Get text embeddings - these should be [B, T_text, D]
        encoder_hidden_states = text_embeddings.to(pipe._execution_device)

        # Compute attention scores
        self_scores, cross_scores = probe.extract_scores_from_forward(
            hidden_states=hidden_states_flat,
            encoder_hidden_states=encoder_hidden_states,
            rotary_emb=None,  # We don't have access to rotary embeddings here
        )

    # Compute self-contrast score
    top_p = float(values.get("mask_top_p", 0.02))
    background_p = float(values.get("mask_background_p", 0.02))

    self_contrast = compute_self_contrast_score(
        self_scores=self_scores.cpu(),
        token_grid_shape=token_grid_shape,
        top_p=top_p,
        background_p=background_p,
    )

    # Compute cross-attention score
    cross_score = compute_cross_attention_score(
        cross_scores=cross_scores.cpu(),
        token_grid_shape=token_grid_shape,
    )

    # Normalize scores using robust_norm
    self_norm = robust_norm(self_contrast)
    cross_norm = robust_norm(cross_score)

    # Build fusion mask
    config = FusionMaskConfig(
        mask_steps=(int(values.get("probe_step", 5)),),
        completed_step=int(values.get("probe_step", 5)),
        eps=float(values.get("fusion_eps", 0.2)),
        gamma=float(values.get("fusion_gamma", 1.0)),
        high_threshold=float(values.get("fusion_high_threshold", 0.45)),
        low_threshold=float(values.get("fusion_low_threshold", 0.35)),
        spatial_medit_radius=int(values.get("mask_spatial_medit_radius", 2)),
        spatial_msee_radius=int(values.get("mask_spatial_msee_radius", 3)),
        keep_isolated=False,
    )

    mask_bundle = build_canonical_fusion_mask(
        self_score=self_norm,
        cross_score=cross_norm,
        config=config,
    )

    # Save mask bundle
    mask_dir = root / "carnator_fusion_mask"
    save_mask_bundle(mask_dir, mask_bundle, save_cpu=True)

    return mask_dir


def _mask(adapter: Any, payload: dict[str, Any], prompt_diff: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load fusion mask from payload (no external segmentation)."""
    import torch
    from carnator.masks import load_mask_bundle

    override = adapter.config["carnator_b5"].get("mask_bundle_override")
    if override is not None:
        return override, {"backend": "manual_override"}

    # Check if fusion mask was generated
    mask_bundle_path = payload.get("mask_bundle_path")
    if not mask_bundle_path or not Path(mask_bundle_path).exists():
        raise RuntimeError(
            "CogVideo Carnator requires fusion mask bundle. "
            "Source generation should have created it at 'mask_bundle_path'."
        )

    # Load fusion mask bundle
    mask_dir = Path(mask_bundle_path)
    device = adapter.config["device"]
    mask_bundle = load_mask_bundle(mask_dir, device=device)

    # Extract masks in token space
    token_base = mask_bundle["Mbase"]  # [T, H, W] bool tensor
    token_edit = mask_bundle["Medit"]  # [T, H, W] bool tensor
    token_see = mask_bundle["Msee"]    # [T, H, W] bool tensor

    # Convert token masks to latent masks
    # For CogVideoX, the latent space has the same spatial dimensions as token space
    latent_base = token_base  # [T, H, W]
    latent_edit = token_edit  # [T, H, W]
    latent_see = token_see    # [T, H, W]

    masks = {
        "token_base": token_base.cpu(),
        "token_edit": token_edit.cpu(),
        "token_see": token_see.cpu(),
        "latent_base": latent_base.cpu(),
        "latent_edit": latent_edit.cpu(),
        "latent_see": latent_see.cpu(),
    }

    metadata = {
        "backend": "fusion_mask",
        "mask_method": "self_contrast_cross_attention_fusion",
        "mask_variant": mask_bundle.get("mask_variant", "unknown"),
        "source_completed_step": mask_bundle.get("source_completed_step", 5),
        "eps": mask_bundle.get("eps"),
        "gamma": mask_bundle.get("gamma"),
        "token_area": mask_bundle.get("token_area", {}),
        "variation": mask_bundle.get("variation", {}),
    }

    return masks, metadata


def _signature_metadata(adapter: Any, prompt: str, latent: Any, token_grid: Any) -> dict[str, Any]:
    values = adapter.config["carnator_b5"]
    return {
        "prompt": prompt,
        "model": "CogVideoX1.5",
        "scheduler": "CogVideoXDDIMScheduler",
        "scheduler_config": {"total_steps": int(adapter.config["baseline_steps"])},
        "cfg": float(adapter.config["guidance_scale"]),
        "probe_seed": int(adapter.config["seed"]),
        "completed_step": int(values.get("probe_step", 5)),
        "latent_shape": list(_latent_layout(latent).shape),
        "latent_token_grid": list(token_grid.shape),
        "noun_extractor": "cogvideo-prompt-diff-v1",
        "mask_policy": "cogvideo-fusion-mask-v1",  # Updated from grounded-sam2
    }


def _probe(adapter: Any, request: Request, completed_step: int) -> tuple[Any, Any, float]:
    delegate = adapter._load_delegate()
    pipe = delegate.pipe
    captured: list[Any] = []

    def stop(_pipe: Any, index: int, _timestep: Any, values: dict[str, Any]) -> dict[str, Any]:
        if index + 1 == completed_step:
            captured.append(values["latents"].detach().cpu())
            _pipe._interrupt = True
        return values

    probe = Request(request.request_index, request.uuid, request.prompt, int(adapter.config["seed"]))
    call = adapter._call_args(probe, int(adapter.config["baseline_steps"]))
    call.update({
        "callback_on_step_end": stop,
        "callback_on_step_end_tensor_inputs": ["latents"],
        "output_type": "latent",
    })
    with adapter._profile_transformer(pipe) as profile:
        pipe(**call)
    if len(captured) != 1:
        raise RuntimeError("CogVideo Carnator target probe did not capture completed step 5")
    return captured[0], delegate.decode_tail(captured[0]).cpu(), profile[0]


def _boundary_weights(masks: dict[str, Any]) -> tuple[Any, Any, Any]:
    import torch
    import torch.nn.functional as functional

    base = masks["latent_base"].bool()
    edit = masks["latent_edit"].bool() | base
    band = edit & ~base
    frames, height, width = band.shape
    frontier = base.reshape(frames, 1, height, width)
    remaining = band.reshape(frames, 1, height, width)
    distance = torch.zeros_like(remaining, dtype=torch.float32)
    for step in range(1, max(height, width) + 1):
        frontier = functional.max_pool2d(frontier.float(), 3, stride=1, padding=1).bool()
        added = frontier & remaining
        distance[added] = float(step)
        remaining &= ~added
        if not bool(remaining.any()):
            break
    maximum = distance.flatten(1).amax(1).view(frames, 1, 1, 1).clamp_min(1.0)
    normalized = (distance / maximum).clamp(0.0, 1.0)
    alpha = normalized.square() * (3.0 - 2.0 * normalized)
    return base, band, alpha[:, 0]


def _target(adapter: Any, request: Request, payload: dict[str, Any]) -> Generation:
    import torch

    delegate = adapter._load_delegate()
    delegate._load()
    pipe = delegate.pipe
    values, block, region_layers, outer_layers = _configuration(adapter)
    steps, probe_step, reuse_step = int(payload["steps"]), int(payload["probe_step"]), int(payload["reuse_step"])
    preprocess_started = time.perf_counter()
    prompt_diff = values.get("prompt_diff_override") or extract_prompt_diff(payload["source_prompt"], request.prompt)
    masks, mask_metadata = _mask(adapter, payload, prompt_diff)
    source_entities = prompt_diff.get("source_entities") or prompt_diff.get("removed_tokens") or ["source object"]
    target_entities = prompt_diff.get("target_entities") or prompt_diff.get("added_tokens") or ["target object"]
    source_noun, target_noun = str(source_entities[0]), str(target_entities[0])

    # Extract all nouns from source prompt for early signature (aligned with wan version)
    from carnator.retrieval import CarnatorNounExtractor
    from core.retrieval import LexicalTestEntityExtractor
    noun_extractor = CarnatorNounExtractor(LexicalTestEntityExtractor())
    source_request = Request(0, "", payload["source_prompt"], 0)
    all_source_nouns = noun_extractor.for_request(source_request)

    # Build mbase_by_noun with all source nouns using the same mask (simplified approach)
    # In the full implementation, each noun should have its own segmented mask
    mbase_by_noun = {noun: masks["token_base"] for noun in all_source_nouns} if all_source_nouns else {source_noun: masks["token_base"]}

    work_dir = Path(adapter.request_cache_dir)
    source_signature = write_early_signature(
        work_dir / "source_signature",
        step5_latent=_latent_layout(payload["step5"]),
        decoded_tail_frame=payload["step5_tail"],
        metadata=_signature_metadata(adapter, payload["source_prompt"], payload["step5"], masks["token_base"]),
        mbase_by_noun=mbase_by_noun,
        cache={
            "anchor_step": reuse_step,
            "model": "CogVideoX1.5",
            "latent_token_grid": list(masks["token_base"].shape),
            "latent_path": payload["anchor_path"],
            "latent_sha256": _sha256(Path(payload["anchor_path"])),
            "sparse_attention_path": str(Path(payload["mask_cache"]) / "metadata.json"),
            "sparse_attention_sha256": _sha256(Path(payload["mask_cache"]) / "metadata.json"),
        },
    )
    probe_latent, probe_tail, probe_dit = _probe(adapter, request, probe_step)
    target_signature = write_early_signature(
        work_dir / "target_probe",
        step5_latent=_latent_layout(probe_latent),
        decoded_tail_frame=probe_tail,
        metadata=_signature_metadata(adapter, request.prompt, probe_latent, masks["token_base"]),
        mbase_by_noun={target_noun: masks["token_base"]},
    )
    thresholds = adapter.config.get("retrieval", {}).get("carnator", {})
    signature = compare_early_signatures(
        target_signature,
        source_signature,
        query_changed_noun=target_noun,
        candidate_changed_noun=source_noun,
        max_context_color_distance=thresholds.get("max_context_color_distance"),
        max_log_area_ratio=thresholds.get("max_log_area_ratio"),
        max_shape_distance=thresholds.get("max_shape_distance"),
        max_changed_color_distance=thresholds.get("max_changed_color_distance"),
    )
    preprocess_seconds = time.perf_counter() - preprocess_started

    cache = CogVideoBlockMaskCache(Path(payload["mask_cache"]), block)
    branches = ("uncond", "cond") if float(adapter.config["guidance_scale"]) > 1.0 else ("cond",)
    cache.validate_coverage(
        steps=range(reuse_step + 1, steps + 1), branches=branches, layers=outer_layers
    )
    executor = CogVideoFlexExecutor(cache)
    region = RegionalAttentionState(
        masks["token_edit"], masks["token_see"], reuse_step, steps, steps
    )
    tracker = CogVideoStepState(start_step=reuse_step, on_begin=region.set_step)
    base, band, alpha = _boundary_weights(masks)

    def fuse(_pipe: Any, index: int, _timestep: Any, state: dict[str, Any]) -> dict[str, Any]:
        completed = reuse_step + index + 1
        source = payload["trajectory"][completed].to(state["latents"].device, state["latents"].dtype)
        target = state["latents"]
        source_mask = (~masks["latent_edit"].bool()).to(target.device)[None, :, None]
        base_mask = base.to(target.device)[None, :, None]
        band_mask = band.to(target.device)[None, :, None]
        weight = alpha.to(target.device, target.dtype)[None, :, None]
        blended = target * (1.0 - weight) + source * weight
        state["latents"] = torch.where(base_mask, target, torch.where(band_mask, blended, torch.where(source_mask, source, target)))
        return state

    call = adapter._call_args(request, steps)
    call.update({
        "latents": payload["latents"],
        "callback_on_step_end": fuse,
        "callback_on_step_end_tensor_inputs": ["latents"],
        "output_type": "latent",
    })
    schedule = adapter._resume_original_schedule(
        pipe.scheduler,
        steps=steps,
        reuse_step=reuse_step,
        expected=[int(item) for item in payload["timesteps"]],
    )
    sparse_processors = CogVideoAttentionProcessors(
        pipe.transformer,
        lambda layer: CogVideoSparseProcessor(executor, tracker, layer),
        outer_layers,
    )
    with ExitStack() as stack:
        stack.enter_context(schedule)
        stack.enter_context(install_regional_blocks(pipe.transformer, region, layers=region_layers))
        stack.enter_context(sparse_processors)
        quantized = stack.enter_context(CogVideoQuantizedFFNHooks(pipe.transformer, list(range(42))))
        stack.enter_context(track_transformer_steps(pipe.transformer, tracker))
        profile = stack.enter_context(adapter._profile_transformer(pipe))
        output = pipe(**call)
    expected = len(outer_layers) * (steps - reuse_step) * len(branches)
    if executor.runs != expected:
        raise RuntimeError(f"CogVideo Carnator ran {executor.runs} outer sparse calls, expected {expected}")
    torch.cuda.empty_cache()
    target_frames = delegate.decode_video(output.frames)
    target_dit = profile[0]
    return Generation(
        video=target_frames,
        latent_step=reuse_step,
        executed_steps=probe_step + steps - reuse_step,
        extra={
            "executed_steps": probe_step + steps - reuse_step,
            "dit_forward_s": probe_dit + target_dit,
            "scheduler": payload["scheduler"],
            "schedule_resume": "original_timestep_suffix",
            "excluded_latency_s": 0.0,
            "carnator_b5": {
                "mode": "target_probe_and_joint_reuse",
                "fixed_replay_authorization": True,
                "probe_step": probe_step,
                "reuse_step": reuse_step,
                "probe_dit_forward_s": probe_dit,
                "target_dit_forward_s": target_dit,
                "prompt_diff": prompt_diff,
                "mask": mask_metadata,
                "early_signature": signature,
                "early_signature_policy": "audited_without_overriding_fixed_replay",
                "regional_layers": region_layers,
                "regional_calls": region.calls,
                "outer_sparse_layers": outer_layers,
                "outer_sparse": executor.summary(),
                "quantized_ffn": quantized.summary(),
                "preprocess_and_probe_s": preprocess_seconds,
            },
        },
    )


def generate(adapter: Any, request: Request, *, cached_latent: Any, capture: bool) -> Generation:
    if cached_latent is None:
        if not capture:
            raise RuntimeError("CogVideo Carnator source generation must capture B5 artifacts")
        return _source(adapter, request)
    if not isinstance(cached_latent, dict) or cached_latent.get("format") != FORMAT:
        raise ValueError("invalid CogVideo Carnator Boundary B5 source payload")
    return _target(adapter, request, cached_latent)
