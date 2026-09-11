#!/usr/bin/env python3
"""Wan2.1-T2V-1.3B native pieces for the Chorus2/Carnator replay baselines.

The shared ``carnator`` algorithm modules (outer-sparse masks, fusion masks,
probes, quantized FFN) are model-agnostic: only the code that drives the DiT
differs between Wan2.1 and Wan2.2 (scalar timestep, ``[B,1,C]`` modulation
chunks).  This module supplies those Wan2.1 variants without touching the
Wan2.2 native path in ``wan_ti2v_5b``.
"""

from __future__ import annotations

import sys
import time
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

CARNATOR_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WAN21_ROOT = CARNATOR_ROOT / "external" / "Wan2.1"


def bootstrap_wan21(wan_root: Path = DEFAULT_WAN21_ROOT) -> dict[str, Any]:
    """Import the Wan2.1 package with the contract of ``bootstrap_wan``."""

    wan_root = Path(wan_root).resolve()
    if not (wan_root / "wan" / "__init__.py").is_file():
        raise FileNotFoundError(f"Wan2.1 package not found: {wan_root / 'wan'}")
    if str(wan_root) not in sys.path:
        sys.path.insert(0, str(wan_root))

    import wan.configs as configs
    import wan.modules.model as model_module
    from wan.text2video import WanT2V
    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from wan.utils.utils import masks_like

    return {
        "configs": configs,
        "model_module": model_module,
        "WanT2V": WanT2V,
        "FlowDPMSolverMultistepScheduler": FlowDPMSolverMultistepScheduler,
        "FlowUniPCMultistepScheduler": FlowUniPCMultistepScheduler,
        "get_sampling_sigmas": get_sampling_sigmas,
        "retrieve_timesteps": retrieve_timesteps,
        "masks_like": masks_like,
    }


def wan21_token_grid(pipe: Any, frames: int, height: int, width: int) -> list[int]:
    t = (int(frames) - 1) // pipe.vae_stride[0] + 1
    h = int(height) // pipe.vae_stride[1] // pipe.patch_size[1]
    w = int(width) // pipe.vae_stride[2] // pipe.patch_size[2]
    return [t, h, w]


def build_wan21_runtime(delegate: Any, config: dict[str, Any]):
    """Wrap the chai delegate's WanT2V pipeline in the carnator runtime shape."""

    from carnator.adapters import wan_ti2v_5b as native

    delegate._load()
    pipeline = delegate.pipeline
    if pipeline is None or getattr(pipeline, "model", None) is None:
        raise RuntimeError("Wan2.1 delegate did not load a WanT2V pipeline")
    # Wan2.1's WanT2V records ``self.device`` but never makes it current, while
    # Triton kernels (outer-sparse block masks) launch on the *current* device.
    # Without this they target GPU 0 and reject tensors from the assigned card
    # with "Pointer argument ... cannot be accessed from Triton". The Wan2.2
    # runtime does the equivalent in ``load_runtime``.
    delegate.torch.cuda.set_device(pipeline.device)
    profiler = native.CudaModuleProfiler()
    profiler.attach(pipeline.model)
    runtime = native.NativeRuntime(
        torch=delegate.torch,
        wan=bootstrap_wan21(Path(config.get("wan_source", DEFAULT_WAN21_ROOT))),
        pipe=pipeline,
        cfg=pipeline.config,
        device=pipeline.device,
        profiler=profiler,
    )
    # Wan2.1 signal conventions consumed by the shared native helpers.
    runtime.scalar_timesteps = True
    runtime.expected_token_grid = wan21_token_grid(
        pipeline, config["frames"], config["height"], config["width"]
    )
    return runtime


def build_wan21_args(config: dict[str, Any], *, seed: int, reuse_start_step: int | None = None,
                     extra_flags: list[str] | None = None):
    """Build carnator-compatible args from the Wan2.1 model config.

    Algorithm constants (probe/reuse steps, layer ranges, mask quantiles,
    sparse backends) come from the shared carnator parser defaults, which are
    the values used by the Wan2.2 experiment; only the generation geometry is
    taken from ``configs/models/wan21.yaml``.  ``extra_flags`` carries the
    constants that deviate from the parser defaults (e.g. ``--ffn-backend
    w8a8`` for the Carnator target).
    """

    from carnator.adapters import wan_ti2v_5b as native

    flags = [
        "--no-env-setup",
        "--model-dir", str(config["model_path"]),
        "--device", str(config["device"]),
        "--height", str(int(config["height"])),
        "--width", str(int(config["width"])),
        "--frames", str(int(config["frames"])),
        "--fps", str(int(config.get("fps", 24))),
        "--steps", str(int(config["baseline_steps"])),
        "--seed", str(int(seed)),
        "--scheduler", str(config["solver"]),
        "--flow-shift", str(float(config["shift"])),
        "--guidance-scale", str(float(config["guidance_scale"])),
    ]
    if reuse_start_step is not None:
        flags.extend(["--reuse-start-step", str(int(reuse_start_step))])
    if extra_flags:
        flags.extend(extra_flags)
    return native.build_parser().parse_args(flags)


def wan21_sparse_block_forward(block, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                               query_idx, context_idx, model_module):
    """Wan2.1 modulation layout of the native sparse region block.

    Wan2.1 chunks ``(block.modulation + e)`` on dim 1 into ``[B,1,C]`` pieces
    that broadcast over tokens, instead of the Wan2.2 ``[1,T,C]`` layout.
    """

    import torch

    from carnator.adapters.wan_ti2v_5b import native_sparse_self_attention

    with torch.amp.autocast("cuda", dtype=torch.float32):
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            block.modulation + e
        ).chunk(6, dim=1)

    q_hidden = x.index_select(1, query_idx)
    norm_all = (block.norm1(x).float() * (1 + scale_msa) + shift_msa).type_as(x)
    y = native_sparse_self_attention(
        block.self_attn, norm_all, query_idx, context_idx, grid_sizes, freqs, model_module
    )
    with torch.amp.autocast("cuda", dtype=torch.float32):
        q_hidden = q_hidden + y * gate_msa
    q_hidden = q_hidden + block.cross_attn(block.norm3(q_hidden), context, context_lens)
    y = block.ffn(block.norm2(q_hidden).float() * (1 + c_scale_msa) + c_shift_msa)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        q_hidden = q_hidden + y * c_gate_msa

    out = x.clone()
    out.index_copy_(1, query_idx, q_hidden)
    return out


class Wan21SparseBlockHooks(AbstractContextManager):
    """``NativeSparseBlockHooks`` with the Wan2.1 sparse block forward."""

    def __init__(self, model, state):
        self.model = model
        self.state = state

    def __enter__(self):
        self.state.reset()
        for layer_idx in self.state.layers:
            block = self.model.blocks[layer_idx]
            original_forward = block.forward
            self.state.originals.append((block, original_forward))

            def make_forward(block_ref, original_forward_ref, current_layer_idx: int):
                def sparse_forward(x, e, seq_lens, grid_sizes, freqs, context, context_lens):
                    if not self.state.enabled:
                        self.state.dense_fallback_count += 1
                        return original_forward_ref(x, e, seq_lens, grid_sizes, freqs, context, context_lens)
                    query_idx, context_idx = self.state.masked_token_indices(
                        x,
                        heads=block_ref.self_attn.num_heads,
                        head_dim=block_ref.self_attn.head_dim,
                        layer=current_layer_idx,
                    )
                    started = time.perf_counter()
                    out = wan21_sparse_block_forward(
                        block_ref, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                        query_idx, context_idx, self.state.model_module,
                    )
                    self.state.sparse_call_count += 1
                    self.state.sparse_forward_elapsed_seconds += time.perf_counter() - started
                    key = str(current_layer_idx)
                    self.state.calls_by_layer[key] = self.state.calls_by_layer.get(key, 0) + 1
                    return out

                return sparse_forward

            block.forward = make_forward(block, original_forward, layer_idx)
        return self.state

    def __exit__(self, exc_type, exc, tb):
        for block, original_forward in reversed(self.state.originals):
            block.forward = original_forward
        self.state.originals.clear()
        return False


def extract_request_nouns(request: Any) -> tuple[str, ...]:
    """Raw spaCy ``NOUN`` extraction, matching the Wan2.2 production retriever.

    ``make_carnator_retriever`` builds ``CarnatorNounExtractor`` around
    ``SpacyEntityExtractor`` outside dry runs; the dataset's preset
    ``noun_list`` columns were produced by the same extractor, so source
    signatures and target selections stay pairwise compatible.
    """

    from core.retrieval import SpacyEntityExtractor
    from carnator.retrieval import CarnatorNounExtractor

    return CarnatorNounExtractor(SpacyEntityExtractor("en_core_web_sm")).for_request(request)


def _signature_metadata(args, *, prompt: str, latent, token_grid, probe_seed: int, noun_extractor: str,
                        mask_policy: str) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "model": "Wan2.1-T2V-1.3B",
        "scheduler": args.scheduler,
        "scheduler_config": {
            "flow_shift": float(args.flow_shift),
            "solver_order": int(args.solver_order) if args.scheduler == "unipc" else None,
            "total_steps": int(args.steps),
        },
        "cfg": float(args.guidance_scale),
        "probe_seed": int(probe_seed),
        "completed_step": int(args.mask_probe_step),
        "latent_shape": list(latent.shape),
        "latent_token_grid": list(token_grid),
        "noun_extractor": noun_extractor,
        "mask_policy": mask_policy,
    }


def run_wan21_target_probe(runtime, args, *, prompt: str, nouns, output_dir: Path,
                           source_signature: Any = None, probe_seed: int | None = None,
                           selection: Any = None) -> dict[str, Any]:
    """Dense target probe feeding the Carnator Early Signature audit.

    Runs the same ``mask_probe_step``-step dense prefix as the source -- with
    the multi-noun cross-attention probe enabled on the last step -- and writes
    the *target* Early Signature next to the per-request cache.  It exists so
    the target reports the Wan2.2 step count and denoising breakdown (probe DiT
    time is part of both).

    The replay decision and the per-noun mask selection are fixed before
    generation, so nothing computed here can change which source is reused:
    ``comparison`` is recorded as audit metadata only and never gates.

    The probe runs on ``probe_seed`` (the fixed experiment seed), matching the
    Wan2.2 collector's ``target_probe_seed``, so the probe latent does not
    depend on the request's own seed.
    """

    import copy

    import torch

    from carnator.adapters import wan_ti2v_5b as native
    from carnator.early_signature import compare_early_signatures, write_early_signature

    if probe_seed is not None:
        args = copy.copy(args)
        args.seed = int(probe_seed)
    pipe = runtime.pipe
    model = pipe.model
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    noun_list = tuple(sorted(set(str(noun) for noun in nouns)))
    if not noun_list:
        raise ValueError("Carnator target probe needs at least one prompt noun")

    context, context_null = native.encode_prompts(runtime, args, prompt)
    latent, generator = native.prepare_initial_latent(runtime, args)
    token_grid = native.token_grid_for(latent, pipe.patch_size)
    seq_len = native.seq_len_for(runtime, latent)
    scheduler = native.build_scheduler(runtime, args)
    index_by_noun = {
        noun: native.find_token_indices(pipe.text_encoder, prompt, noun, pipe.config.text_len)
        for noun in noun_list
    }
    probe = native.NativeMultiNounCrossAttentionProbe(
        model, native.parse_int_list(args.mask_layers), index_by_noun, token_grid
    )
    # Wan2.1's WanT2V has no ``init_on_cpu`` (it places the model on the device
    # itself); only Wan2.2's WanTI2V carries that flag.
    if args.offload_model or getattr(pipe, "init_on_cpu", False):
        model.to(runtime.device)
        torch.cuda.empty_cache()

    mask_policy = native.fused_mbase_policy(args)
    dit_forward_seconds = 0.0
    completed_steps = 0
    signature: dict[str, Any] | None = None
    no_sync = getattr(model, "no_sync", nullcontext)
    with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad(), no_sync(), probe:
        for step_index, timestep in enumerate(scheduler.timesteps):
            if step_index + 1 > args.mask_probe_step:
                break
            probe.enabled = (step_index + 1) == args.mask_probe_step
            probe.current_step_index = step_index
            timestep_input = native.timestep_for(runtime, latent, timestep, seq_len)
            torch.cuda.synchronize()
            started = time.perf_counter()
            noise_pred_cond = model([latent], t=timestep_input, context=context, seq_len=seq_len)[0]
            torch.cuda.synchronize()
            dit_forward_seconds += time.perf_counter() - started
            torch.cuda.synchronize()
            started = time.perf_counter()
            noise_pred_uncond = model([latent], t=timestep_input, context=context_null, seq_len=seq_len)[0]
            torch.cuda.synchronize()
            dit_forward_seconds += time.perf_counter() - started
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latent = scheduler.step(
                noise_pred.unsqueeze(0), timestep, latent.unsqueeze(0),
                return_dict=False, generator=generator,
            )[0].squeeze(0)
            completed_steps = step_index + 1

            if completed_steps == args.mask_probe_step:
                mbase_by_noun = native.build_fused_mbase_by_noun(
                    runtime,
                    args,
                    prompt=prompt,
                    token_grid=token_grid,
                    noun_token_indices=index_by_noun,
                    cross_records_by_noun=probe.records_by_noun,
                    self_attention_input=latent,
                    diagnostics_path=output_dir / "mask_diagnostics.json",
                )
                video = pipe.vae.decode([latent])[0]
                if not isinstance(video, torch.Tensor) or video.ndim != 4:
                    raise RuntimeError("Wan2.1 VAE did not return a [C,T,H,W] decoded video")
                signature = write_early_signature(
                    output_dir,
                    step5_latent=latent.detach().float().cpu().contiguous(),
                    decoded_tail_frame=video[:, -1].detach(),
                    metadata=_signature_metadata(
                        args, prompt=prompt, latent=latent, token_grid=token_grid,
                        probe_seed=args.seed, noun_extractor="lexical-spacy", mask_policy=mask_policy,
                    ),
                    mbase_by_noun=mbase_by_noun,
                )

    if completed_steps != args.mask_probe_step or signature is None:
        raise RuntimeError(
            f"Carnator target probe completed {completed_steps} steps without an Early Signature"
        )

    comparison: dict[str, Any] | None = None
    if source_signature is not None:
        selection = selection if isinstance(selection, dict) else {}
        target_only = list(selection.get("target_only_nouns") or [])
        source_only = list(selection.get("source_only_nouns") or [])
        try:
            comparison = compare_early_signatures(
                signature,
                source_signature,
                query_changed_noun=target_only[0] if len(target_only) == 1 else None,
                candidate_changed_noun=source_only[0] if len(source_only) == 1 else None,
            )
        except Exception as exc:  # audit metadata must never gate the reuse
            comparison = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "completed_steps": completed_steps,
        "dit_forward_seconds": dit_forward_seconds,
        "signature": signature,
        "comparison": comparison,
        "output_dir": str(output_dir),
    }


def run_wan21_carnator_source(runtime, args, *, prompt: str, nouns, cache_dir: Path,
                              out_video: Path, outer_sparse_state: Any = None) -> dict[str, Any]:
    """Dense source run producing every artifact the Carnator target reuses.

    Captured per source request: the full post-scheduler latent trajectory,
    per-noun Early Signature (step-5 fused Mbase, decoded tail frame), the
    per-noun fusion-mask template bundle, outer block-attention routing masks
    from completed step ``reuse_start_step + 1``, and the source video.
    """

    import torch

    from carnator.adapters import wan_ti2v_5b as native
    from carnator.early_signature import write_early_signature
    from carnator.masks import build_per_noun_mbase_bundle, save_mask_bundle

    pipe = runtime.pipe
    model = pipe.model
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    noun_list = tuple(sorted(set(str(noun) for noun in nouns)))
    if not noun_list:
        raise ValueError("Carnator source needs at least one prompt noun")

    context, context_null = native.encode_prompts(runtime, args, prompt)
    latent, generator = native.prepare_initial_latent(runtime, args)
    token_grid = native.token_grid_for(latent, pipe.patch_size)
    seq_len = native.seq_len_for(runtime, latent)
    scheduler = native.build_scheduler(runtime, args)
    index_by_noun = {
        noun: native.find_token_indices(pipe.text_encoder, prompt, noun, pipe.config.text_len)
        for noun in noun_list
    }
    probe = native.NativeMultiNounCrossAttentionProbe(
        model, native.parse_int_list(args.mask_layers), index_by_noun, token_grid
    )

    trajectory: dict[int, Path] = {}
    cache_write_seconds = 0.0
    write_started = time.perf_counter()
    native.save_latent(cache_dir / "latents_initial.pt", latent)
    cache_write_seconds += time.perf_counter() - write_started

    # Wan2.1's WanT2V has no ``init_on_cpu`` (it places the model on the device
    # itself); only Wan2.2's WanTI2V carries that flag.
    if args.offload_model or getattr(pipe, "init_on_cpu", False):
        model.to(runtime.device)
        torch.cuda.empty_cache()

    mask_policy = native.fused_mbase_policy(args)
    signature_info: dict[str, Any] | None = None
    mbase_by_noun: dict[str, Any] | None = None
    probe_seconds = 0.0
    dit_forward_seconds = 0.0
    denoise_started = time.perf_counter()
    no_sync = getattr(model, "no_sync", nullcontext)
    with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad(), no_sync(), probe:
        for step_index, timestep in enumerate(scheduler.timesteps):
            probe.enabled = (step_index + 1) == args.mask_probe_step
            probe.current_step_index = step_index
            if outer_sparse_state is not None:
                outer_sparse_state.set_step(step_index + 1)
            timestep_input = native.timestep_for(runtime, latent, timestep, seq_len)
            torch.cuda.synchronize()
            started = time.perf_counter()
            noise_pred_cond = model([latent], t=timestep_input, context=context, seq_len=seq_len)[0]
            torch.cuda.synchronize()
            dit_forward_seconds += time.perf_counter() - started
            torch.cuda.synchronize()
            started = time.perf_counter()
            noise_pred_uncond = model([latent], t=timestep_input, context=context_null, seq_len=seq_len)[0]
            torch.cuda.synchronize()
            dit_forward_seconds += time.perf_counter() - started
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latent = scheduler.step(
                noise_pred.unsqueeze(0), timestep, latent.unsqueeze(0),
                return_dict=False, generator=generator,
            )[0].squeeze(0)
            completed = step_index + 1

            if completed == args.mask_probe_step:
                probe_started = time.perf_counter()
                step5_latent = latent.detach().float().cpu().contiguous()
                mbase_by_noun = native.build_fused_mbase_by_noun(
                    runtime,
                    args,
                    prompt=prompt,
                    token_grid=token_grid,
                    noun_token_indices=index_by_noun,
                    cross_records_by_noun=probe.records_by_noun,
                    self_attention_input=latent,
                    diagnostics_path=cache_dir / "mask_diagnostics.json",
                )
                video = pipe.vae.decode([latent])[0]
                if not isinstance(video, torch.Tensor) or video.ndim != 4:
                    raise RuntimeError("Wan2.1 VAE did not return a [C,T,H,W] decoded video")
                decoded_tail_frame = video[:, -1].detach()
                torch.cuda.synchronize()
                signature_info = write_early_signature(
                    cache_dir,
                    step5_latent=step5_latent,
                    decoded_tail_frame=decoded_tail_frame,
                    metadata=_signature_metadata(
                        args, prompt=prompt, latent=latent, token_grid=token_grid,
                        probe_seed=args.seed, noun_extractor="lexical-spacy", mask_policy=mask_policy,
                    ),
                    mbase_by_noun=mbase_by_noun,
                )
                probe_seconds = time.perf_counter() - probe_started

            write_started = time.perf_counter()
            trajectory[step_index] = Path(
                native.save_latent(cache_dir / f"latents_step_{step_index:03d}.pt", latent)
            )
            cache_write_seconds += time.perf_counter() - write_started

    torch.cuda.synchronize()
    denoising_seconds = time.perf_counter() - denoise_started

    if mbase_by_noun is None or signature_info is None:
        raise RuntimeError("Carnator source completed without a step-5 Early Signature")

    # The per-noun inference path materializes its bundle from the signature;
    # this template only carries the canonical radii and token geometry.
    template_noun = sorted(mbase_by_noun)[0]
    save_mask_bundle(
        cache_dir / "fusion_masks",
        build_per_noun_mbase_bundle(
            mbase_by_noun[template_noun],
            source_completed_step=args.mask_probe_step,
            spatial_medit_radius=args.mask_spatial_medit_radius,
            spatial_msee_radius=args.mask_spatial_msee_radius,
            selection={"selected_source_noun": template_noun,
                       "selection_reason": "source_template",
                       "source_nouns": sorted(mbase_by_noun),
                       "target_nouns": sorted(mbase_by_noun)},
        ),
        save_cpu=True,
    )

    decode_started = time.perf_counter()
    video = pipe.vae.decode([latent])[0]
    native.save_native_video(out_video, video, args.fps)
    decode_seconds = time.perf_counter() - decode_started
    if not out_video.is_file():
        raise RuntimeError(f"Carnator source did not write {out_video}")

    manifest = {
        "format": "wan21-carnator-source-cache/v1",
        "prompt": prompt,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "latent_shape": list(latent.shape),
        "token_grid": list(token_grid),
        "cache_dir": str(cache_dir),
        "source_video": str(out_video),
        "signature": signature_info,
        "trajectory": [
            {"kind": "post_scheduler", "step_index": step, "path": str(path)}
            for step, path in sorted(trajectory.items())
        ],
        "timing": {
            "loop_denoise_seconds": denoising_seconds,
            "dit_forward_seconds": dit_forward_seconds,
            "probe_seconds": probe_seconds,
            "decode_seconds": decode_seconds,
            "cache_write_seconds": cache_write_seconds,
            "end_to_end_seconds": None,
        },
    }
    native.write_json(cache_dir / "manifest.json", manifest)
    manifest["manifest_path"] = str(cache_dir / "manifest.json")
    return manifest
