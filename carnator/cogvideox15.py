from __future__ import annotations

import hashlib
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from adapters.cogvideo_attention import (
    CogVideoStepState,
    RegionalAttentionState,
    caption_token_indices,
    cogvideo_token_grid,
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
from carnator.cogvideo_attention_probe import CogVideoMultiNounJointAttentionProbe
from carnator.cogvideo_quantized_ffn import CogVideoQuantizedFFNHooks
from carnator.early_signature import (
    compare_early_signatures,
    materialize_per_noun_inference_mask_bundle,
    write_early_signature,
)
from core.types import Generation, Request


FORMAT = "cogvideox15-carnator-boundary-b5/v1"
MASK_POLICY = "cogvideo-native-self-cross-fusion-mbase"


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


def _noun_token_indices(pipe: Any, prompt: str, nouns: Any) -> dict[str, list[int]]:
    """Map every prompt noun onto its caption token positions."""

    maximum = int(pipe.transformer.config.max_text_seq_length)
    index_by_noun: dict[str, list[int]] = {}
    for noun in nouns:
        indices = caption_token_indices(pipe.tokenizer, prompt, [noun], maximum)
        if indices:
            index_by_noun[str(noun)] = list(indices)
    return index_by_noun


def _probe_config(adapter: Any, pipe: Any) -> dict[str, Any]:
    values = adapter.config["carnator_b5"]
    cross_layer = int(values.get("attention_probe_layer", 15))
    token_grid = cogvideo_token_grid(
        pipe,
        frames=int(adapter.config["frames"]),
        height=int(adapter.config["height"]),
        width=int(adapter.config["width"]),
    )
    return {
        "cross_layer": cross_layer,
        "self_layer": int(values.get("attention_self_layer", cross_layer)),
        "token_grid": token_grid,
        "top_p": float(values.get("mask_top_p", 0.02)),
        "background_p": float(values.get("mask_background_p", 0.02)),
    }


def _fusion_mask_config(adapter: Any, probe_step: int) -> Any:
    from carnator.masks import FusionMaskConfig

    values = adapter.config["carnator_b5"]
    return FusionMaskConfig(
        completed_step=int(probe_step),
        eps=float(values.get("fusion_eps", 0.2)),
        gamma=float(values.get("fusion_gamma", 1.0)),
        high_threshold=float(values.get("fusion_high_threshold", 0.45)),
        low_threshold=float(values.get("fusion_low_threshold", 0.35)),
        spatial_medit_radius=int(values.get("mask_spatial_medit_radius", 2)),
        spatial_msee_radius=int(values.get("mask_spatial_msee_radius", 3)),
        keep_isolated=False,
    )


def _mbase_by_noun(adapter: Any, probe: Any, probe_step: int) -> dict[str, Any]:
    """Fuse each noun's cross score and self contrast into its own Mbase.

    Identical to the Wan and LTX Early Signature: cross attention only proposes
    the seed locations, self attention decides where the noun lands.
    """

    from carnator.masks import build_canonical_fusion_mask

    config = _fusion_mask_config(adapter, probe_step)
    return {
        noun: build_canonical_fusion_mask(
            probe.self_records_by_noun[noun]["self_contrast_max"],
            probe.cross_scores[noun],
            config,
            original_mbase=None,
        )["Mbase"].detach().cpu().bool().contiguous()
        for noun in sorted(probe.cross_scores)
    }


def _mask_policy(adapter: Any, probe_step: int) -> str:
    values = adapter.config["carnator_b5"]
    cross_layer = int(values.get("attention_probe_layer", 15))
    return (
        f"{MASK_POLICY}|cross_layer={cross_layer}"
        f"|self_layer={int(values.get('attention_self_layer', cross_layer))}"
        f"|completed_step={int(probe_step)}"
        f"|top_p={float(values.get('mask_top_p', 0.02)):.8f}"
        f"|background_p={float(values.get('mask_background_p', 0.02)):.8f}"
        f"|eps={float(values.get('fusion_eps', 0.2)):.8f}"
        f"|gamma={float(values.get('fusion_gamma', 1.0)):.8f}"
    )


def _source(adapter: Any, request: Request) -> Generation:
    """Capture every artifact the Carnator target reuses, in one dense run.

    The run itself is the dense baseline; the additions are the step-5 joint
    probe, the outer block-mask capture and the per-step trajectory.
    """

    import torch
    from carnator.adapters.wan21_native import extract_request_nouns
    from carnator.masks import build_per_noun_mbase_bundle, save_mask_bundle

    delegate = adapter._load_delegate()
    delegate._load()
    pipe = delegate.pipe
    values, block, _region, outer = _configuration(adapter)
    steps = int(adapter.config["baseline_steps"])
    probe_step = int(values.get("probe_step", 5))
    reuse_step = int(values.get("reuse_step", 10))
    if probe_step != 5:
        # The signature profile is defined at step 5 (EARLY_SIGNATURE_PROFILE_ID)
        # and ``write_early_signature`` rejects any other completed step.
        raise ValueError("carnator_b5.probe_step must be 5; the Early Signature is defined there")
    if not 0 < probe_step < reuse_step < steps:
        raise ValueError("CogVideo Carnator requires 0 < probe_step < reuse_step < steps")

    nouns = extract_request_nouns(request)
    if not nouns:
        raise RuntimeError(f"CogVideo Carnator found no nouns to probe in {request.prompt!r}")
    index_by_noun = _noun_token_indices(pipe, request.prompt, nouns)
    if not index_by_noun:
        raise RuntimeError(f"CogVideo Carnator could not locate any prompt noun of {request.prompt!r} in the caption")
    probe_config = _probe_config(adapter, pipe)
    probe = CogVideoMultiNounJointAttentionProbe(
        pipe.transformer,
        cross_layer=probe_config["cross_layer"],
        self_layer=probe_config["self_layer"],
        positive_indices_by_noun=index_by_noun,
        token_grid=probe_config["token_grid"],
        completed_step=probe_step,
        top_p=probe_config["top_p"],
        background_p=probe_config["background_p"],
    )

    trajectory: dict[int, Any] = {}
    step5: list[Any] = []

    def capture(_pipe: Any, index: int, _timestep: Any, state: dict[str, Any]) -> dict[str, Any]:
        completed = index + 1
        if completed == probe_step:
            step5.append(state["latents"].detach().cpu())
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

    with probe, processors, track_transformer_steps(pipe.transformer, tracker), adapter._profile_transformer(pipe) as profile:
        pipe(**call)

    if len(step5) != 1:
        raise RuntimeError(f"CogVideo Carnator source did not capture completed step {probe_step}")
    if set(trajectory) != set(range(reuse_step, steps + 1)):
        raise RuntimeError("CogVideo Carnator source trajectory is incomplete")
    mask_cache.finalize()
    branches = ("uncond", "cond") if float(adapter.config["guidance_scale"]) > 1.0 else ("cond",)
    mask_cache.validate_coverage(steps=range(reuse_step + 1, steps + 1), branches=branches, layers=outer)

    # A probe that recorded nothing must not silently degrade into an
    # unsegmented mask: the whole point of the B5 path is the noun region.
    probe.validate()
    mbase_by_noun = _mbase_by_noun(adapter, probe, probe_step)
    mask_policy = _mask_policy(adapter, probe_step)

    anchor_path = root / "carnator_step10.pt"
    torch.save(trajectory[reuse_step], anchor_path)
    torch.cuda.empty_cache()
    decoded_tail = delegate.decode_tail(step5[0])

    # The source-side template: one canonical bundle the target's per-noun
    # selection starts from.  ``source_completed_step``/radii are carried into
    # the request-private bundle the target materializes.
    template_noun = sorted(mbase_by_noun)[0]
    mask_dir = root / "carnator_fusion_masks"
    save_mask_bundle(
        mask_dir,
        build_per_noun_mbase_bundle(
            mbase_by_noun[template_noun],
            source_completed_step=probe_step,
            spatial_medit_radius=int(values.get("mask_spatial_medit_radius", 2)),
            spatial_msee_radius=int(values.get("mask_spatial_msee_radius", 3)),
            selection={
                "selected_source_noun": template_noun,
                "selection_reason": "source_template",
                "source_nouns": sorted(mbase_by_noun),
                "target_nouns": sorted(mbase_by_noun),
            },
        ),
        save_cpu=True,
    )
    signature = write_early_signature(
        root / "source_signature",
        step5_latent=_latent_layout(step5[0]),
        decoded_tail_frame=decoded_tail,
        metadata=_signature_metadata(adapter, request.prompt, step5[0], probe_config["token_grid"]),
        mbase_by_noun=mbase_by_noun,
        cache={
            "anchor_step": reuse_step,
            "model": "CogVideoX1.5",
            "latent_token_grid": list(probe_config["token_grid"]),
            "latent_path": str(anchor_path),
            "latent_sha256": _sha256(anchor_path),
            "sparse_attention_path": str(mask_root / "metadata.json"),
            "sparse_attention_sha256": _sha256(mask_root / "metadata.json"),
        },
    )

    payload = {
        "format": FORMAT,
        "source_prompt": request.prompt,
        "step5": step5[0],
        "step5_tail": decoded_tail.cpu(),
        "trajectory": trajectory,
        "latents": trajectory[reuse_step],
        "anchor_path": str(anchor_path),
        "mask_cache": str(mask_root),
        "fusion_mask_dir": str(mask_dir),
        "signature_path": signature["signature_path"],
        "token_grid": list(probe_config["token_grid"]),
        "nouns": sorted(mbase_by_noun),
        "mask_policy": mask_policy,
        "scheduler": pipe.scheduler.__class__.__name__,
        "timesteps": adapter._scheduler_timesteps(pipe.scheduler, steps, pipe._execution_device),
        "steps": steps,
        "probe_step": probe_step,
        "reuse_step": reuse_step,
        "outer_layers": outer,
        "mask_config": block.__dict__,
    }
    return Generation(
        video=None,
        latent=payload,
        latent_step=reuse_step,
        executed_steps=steps,
        extra={
            "carnator_stage": "source_capture",
            "executed_steps": steps,
            "dit_forward_s": profile[0],
            "scheduler": payload["scheduler"],
            # Same audit vocabulary as the Wan2.1/LTX-Video Carnator sources.
            "carnator": {
                "nouns": sorted(mbase_by_noun),
                "probe_step": probe_step,
                "reuse_step": reuse_step,
                "trajectory_states": len(trajectory),
                "outer_masks": mask_cache.summary(),
                "mask_policy": mask_policy,
                "probe": probe.summary(),
            },
        },
    )


def _target(adapter: Any, request: Request, payload: dict[str, Any]) -> Generation:
    """Reuse the source: per-noun mask, soft blend, region+outer sparse, w8a8."""

    import torch
    from carnator.adapters.wan21_native import extract_request_nouns
    from carnator.adapters.wan_ti2v_5b import apply_latent_injection
    from carnator.masks import load_mask_bundle

    delegate = adapter._load_delegate()
    delegate._load()
    pipe = delegate.pipe
    values, block, region_layers, outer_layers = _configuration(adapter)
    steps, probe_step, reuse_step = int(payload["steps"]), int(payload["probe_step"]), int(payload["reuse_step"])
    injection_mode = str(values.get("latent_injection_mode", "boundary-blend"))
    alpha_function = str(values.get("boundary_alpha_function", "smoothstep"))
    if injection_mode != "boundary-blend":
        raise ValueError(f"unexpected Carnator latent injection mode: {injection_mode}")
    ffn_backend = str(values.get("ffn_backend", "w8a8"))
    if ffn_backend != "w8a8":
        raise ValueError(f"unexpected Carnator FFN backend: {ffn_backend}")

    target_nouns = extract_request_nouns(request)
    target_dir = Path(adapter.request_cache_dir) / "carnator_target"
    selection = materialize_per_noun_inference_mask_bundle(
        {"signature_path": payload["signature_path"]},
        target_nouns=target_nouns,
        source_fusion_mask_dir=Path(payload["fusion_mask_dir"]),
        output_dir=target_dir / "inference_masks",
    )
    mask_bundle = load_mask_bundle(Path(selection["mask_bundle_dir"]), None)
    source_noun = str(selection.get("selected_source_noun", ""))

    # Timing parity with the Wan baselines: the target pays the same dense probe
    # prefix, and its DiT time and step count go into the totals below.  It runs
    # before any sparse or quantised hook is installed, so it stays a dense
    # measurement on the plain model.
    probe_latent, probe_tail, probe_dit, probe_probe = _target_probe(adapter, request, pipe, probe_step)
    target_mbase = _mbase_by_noun(adapter, probe_probe, probe_step)
    target_signature = write_early_signature(
        target_dir / "target_probe",
        step5_latent=_latent_layout(probe_latent),
        decoded_tail_frame=probe_tail,
        metadata=_signature_metadata(adapter, request.prompt, probe_latent, probe_probe.token_grid),
        mbase_by_noun=target_mbase,
    )
    comparison: dict[str, Any] | None = None
    target_only = list(selection.get("target_only_nouns") or [])
    source_only = list(selection.get("source_only_nouns") or [])
    try:
        # Audit metadata only: nothing computed here can change which source is
        # reused, because the fixed replay decision and the per-noun mask
        # selection already ran.
        comparison = compare_early_signatures(
            target_signature,
            {"signature_path": payload["signature_path"]},
            query_changed_noun=target_only[0] if len(target_only) == 1 else None,
            candidate_changed_noun=source_only[0] if len(source_only) == 1 else None,
        )
    except Exception as exc:
        comparison = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    cache = CogVideoBlockMaskCache(Path(payload["mask_cache"]), block)
    branches = ("uncond", "cond") if float(adapter.config["guidance_scale"]) > 1.0 else ("cond",)
    cache.validate_coverage(steps=range(reuse_step + 1, steps + 1), branches=branches, layers=outer_layers)
    executor = CogVideoFlexExecutor(cache)
    region = RegionalAttentionState(
        mask_bundle["Medit"], mask_bundle["Msee"], reuse_step, steps, steps
    )
    tracker = CogVideoStepState(start_step=reuse_step, on_begin=region.set_step)

    injections: list[dict[str, Any]] = []

    def fuse(_pipe: Any, index: int, _timestep: Any, state: dict[str, Any]) -> dict[str, Any]:
        completed = reuse_step + index + 1
        source = payload["trajectory"][completed][0].to(state["latents"].device, state["latents"].dtype)
        blended, _elapsed, _source_step, stats = apply_latent_injection(
            state["latents"][0],
            source,
            mask_bundle,
            mode=injection_mode,
            alpha_function=alpha_function,
        )
        state["latents"] = blended.unsqueeze(0)
        if not injections:
            injections.append(stats)
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
        quantized = stack.enter_context(
            CogVideoQuantizedFFNHooks(pipe.transformer, list(range(len(pipe.transformer.transformer_blocks))))
        )
        stack.enter_context(track_transformer_steps(pipe.transformer, tracker))
        profile = stack.enter_context(adapter._profile_transformer(pipe))
        output = pipe(**call)
    expected = len(outer_layers) * (steps - reuse_step) * len(branches)
    if executor.runs != expected:
        raise RuntimeError(f"CogVideo Carnator ran {executor.runs} outer sparse calls, expected {expected}")
    if len(injections) != 1:
        raise RuntimeError(f"CogVideo Carnator latent injection ran {len(injections)} times")
    torch.cuda.empty_cache()
    target_frames = delegate.decode_video(output.frames)
    target_dit = profile[0]
    return Generation(
        video=target_frames,
        latent_step=reuse_step,
        executed_steps=probe_step + steps - reuse_step,
        extra={
            "carnator_stage": "target_reuse",
            "executed_steps": probe_step + steps - reuse_step,
            # denoising_s is the whole executed prefix plus the reuse, so the
            # probe split has to be readable at the top level: the aggregation
            # contract rebuilds probe/target denoising from these two keys.
            "dit_forward_s": probe_dit + target_dit,
            "probe_dit_forward_s": probe_dit,
            "probe_steps": probe_step,
            "scheduler": payload["scheduler"],
            "schedule_resume": "original_timestep_suffix",
            "excluded_latency_s": 0.0,
            # Same audit vocabulary as the Wan2.1/LTX-Video Carnator targets.
            # The CogVideoX target pays no separate GPU prepare step -- its block
            # masks come off disk (already deducted as cache_load_s) and the
            # executor builds indices lazily inside the timed forward -- so
            # cache_prepare_s stays unset at 0.
            "carnator": {
                "inference_mask_selection": {
                    key: value for key, value in selection.items() if key != "mask_bundle_dir"
                },
                "target_nouns": sorted(target_nouns),
                "target_probe": {"completed_steps": probe_step, "dit_forward_s": probe_dit},
                "masked_injection": {
                    "mode": injection_mode,
                    "alpha_function": alpha_function,
                    **injections[0],
                },
                "selected_source_noun": source_noun,
                "probe_step": probe_step,
                "reuse_step": reuse_step,
                "probe_dit_forward_s": probe_dit,
                "target_dit_forward_s": target_dit,
                "early_signature": comparison,
                "early_signature_policy": "audited_without_overriding_fixed_replay",
                "regional_layers": region_layers,
                "regional_calls": region.calls,
                "outer_sparse_layers": outer_layers,
                "outer_sparse": executor.summary(),
                "ffn_backend": ffn_backend,
                "quantized_ffn": quantized.summary(),
            },
        },
    )


def _target_probe(adapter: Any, request: Request, pipe: Any, probe_step: int) -> tuple[Any, Any, float, Any]:
    """The dense probe prefix the Carnator target pays before it reuses.

    Wan2.1 and Wan2.2 both run the first ``probe_step`` steps of the full plan
    densely on the target, with the multi-noun probe armed on the last one, and
    count that DiT time and those steps in the target's totals.  The schedule
    stays the 50-step plan -- the loop is interrupted after the probe step rather
    than re-scaled -- so the probed timesteps are the source's own.
    """

    from carnator.adapters.wan21_native import extract_request_nouns

    delegate = adapter._load_delegate()
    probe_config = _probe_config(adapter, pipe)
    index_by_noun = _noun_token_indices(pipe, request.prompt, extract_request_nouns(request))
    if not index_by_noun:
        raise RuntimeError(f"CogVideo Carnator target probe found no prompt noun in {request.prompt!r}")
    probe = CogVideoMultiNounJointAttentionProbe(
        pipe.transformer,
        cross_layer=probe_config["cross_layer"],
        self_layer=probe_config["self_layer"],
        positive_indices_by_noun=index_by_noun,
        token_grid=probe_config["token_grid"],
        completed_step=probe_step,
        top_p=probe_config["top_p"],
        background_p=probe_config["background_p"],
    )
    captured: list[Any] = []

    def stop(_pipe: Any, index: int, _timestep: Any, values: dict[str, Any]) -> dict[str, Any]:
        if index + 1 == probe_step:
            captured.append(values["latents"].detach().cpu())
            # ``CogVideoXPipeline`` skips the remaining iterations outright, so
            # the probe costs exactly ``probe_step`` DiT forwards.
            _pipe._interrupt = True
        return values

    probe_request = Request(request.request_index, request.uuid, request.prompt, int(adapter.config["seed"]))
    call = adapter._call_args(probe_request, int(adapter.config["baseline_steps"]))
    call.update({
        "callback_on_step_end": stop,
        "callback_on_step_end_tensor_inputs": ["latents"],
        "output_type": "latent",
    })
    with probe, adapter._profile_transformer(pipe) as profile:
        pipe(**call)
    if len(captured) != 1:
        raise RuntimeError(f"CogVideo Carnator target probe did not capture completed step {probe_step}")
    probe.validate()
    return captured[0], delegate.decode_tail(captured[0]), profile[0], probe


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
        "latent_token_grid": [int(item) for item in token_grid],
        "noun_extractor": "lexical-spacy",
        "mask_policy": _mask_policy(adapter, int(values.get("probe_step", 5))),
    }


def generate(adapter: Any, request: Request, *, cached_latent: Any, capture: bool) -> Generation:
    if cached_latent is None:
        if not capture:
            raise RuntimeError("CogVideo Carnator source generation must capture B5 artifacts")
        return _source(adapter, request)
    if not isinstance(cached_latent, dict) or cached_latent.get("format") != FORMAT:
        raise ValueError("invalid CogVideo Carnator Boundary B5 source payload")
    return _target(adapter, request, cached_latent)
