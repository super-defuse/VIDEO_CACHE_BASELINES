"""LTX-Video 0.9.7-dev baselines, mirroring the Wan2.1 implementations.

The five methods are the same algorithms the Wan2.1 adapter runs; only the
model boundary differs.  Every one of them drives the single sampling loop in
``LtxVideoPipelineDelegate`` rather than a pipeline ``__call__``, because LTX's
released recipe is CFG + STG and the diffusers pipeline only implements CFG.

Where the Wan2.1 code had to hand-roll model arithmetic, this file reuses the
shared, model-agnostic pieces instead:

* Chorus's SRD/TGAA mask comes from ``methods.chorus.mask.build_edit_mask``
  unchanged -- it only needs a latent shape and a patch size, and LTX's patch
  size is 1, so its token grid *is* the latent grid.
* Chorus2's routing capture/replay goes through ``adapters.ltx_sparse``.
* The prompt diff and the entity extraction are ``methods.chorus.llm``.

The pieces that genuinely differ -- the attention hooks, the block patch, the
sparse ``attn1`` processors -- live in ``adapters/ltx_attention.py`` and
``adapters/ltx_sparse.py``.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from core.protocol import ModelAdapter
from core.types import Generation, Request

from .ltx_attention import LtxRegionalState, LtxStepTracker, install_regional_blocks
from .ltx_pipeline import LtxVideoPipelineDelegate
from .ltx_sparse import (
    LtxAttentionProcessors,
    LtxBlockMaskCache,
    LtxBlockMaskConfig,
    LtxMaskCaptureProcessor,
    LtxSparseExecutor,
    LtxSparseProcessor,
)


def _token_summary(indices: list[int]) -> dict[str, Any]:
    """Compact record of a TGAA token set for the run's ``extra``."""
    return {"count": len(indices), "indices": list(indices[:32])}


class LtxVideoAdapter(ModelAdapter):
    """The LTX-Video model boundary shared by all five baselines."""

    model_name = "LtxVideo"

    # Pointer payloads reference an on-disk source cache directory instead of
    # carrying tensors; they serialize as plain JSON.
    POINTER_FORMATS = ("ltxvideo-chorus2-source/v1", "ltxvideo-carnator-source/v1")
    # Chorus's source payload is a full per-step latent trajectory; it needs a
    # sidecar directory, so it gets its own manifest format.
    TRAJECTORY_FORMAT = "ltxvideo-chorus-trajectory/v1"
    CHORUS2_FORMAT = "ltxvideo-chorus2-source/v1"
    CARNATOR_FORMAT = "ltxvideo-carnator-source/v1"

    def __init__(self, config):
        super().__init__(config)
        self._delegate: LtxVideoPipelineDelegate | None = None

    # ------------------------------------------------------------- plumbing

    def _load_delegate(self) -> LtxVideoPipelineDelegate:
        if self._delegate is None:
            self._delegate = LtxVideoPipelineDelegate(self.config)
        return self._delegate

    def prepare(self) -> None:
        self._load_delegate().prepare()

    def _request_cache_root(self) -> Path:
        """Directory the replay runner assigned for this request's cache."""
        root = getattr(self, "request_cache_dir", None)
        if root is None:
            raise RuntimeError("LTX cache methods require an assigned request cache directory")
        return Path(root)

    def baseline_steps(self) -> int:
        return int(self.config["baseline_steps"])

    def negative_prompt(self) -> str:
        return str(self.config.get("negative_prompt", ""))

    @staticmethod
    def _seed(request: Request) -> int:
        """LTX draws its latents from a generator, so the seed must be concrete."""
        import random

        return int(request.seed) if int(request.seed) >= 0 else random.randint(0, 2**63 - 1)

    def _guidance_extra(self, delegate: LtxVideoPipelineDelegate, per_step: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "ltx_stg": delegate.stg.describe(),
            "ltx_guidance": per_step,
        }

    # ----------------------------------------------------------- generation

    def generate(self, request: Request, *, fast=False, cached_latent=None, capture=False,
                 method="baseline"):
        if method == "nirvana_vid":
            return self._generate_nirvana(request, cached_latent=cached_latent, capture=capture)
        if method == "chorus":
            return self._generate_chorus(request, cached_latent=cached_latent, capture=capture)
        if method == "chorus2":
            return self._generate_chorus2(request, cached_latent=cached_latent, capture=capture)
        if method == "carnator_boundary_b5":
            return self._generate_carnator(request, cached_latent=cached_latent, capture=capture)
        if method == "teacache":
            raise ValueError("teacache is not integrated for LTX-Video; see configs/methods/matrix.yaml")
        return self._generate_dense(request)

    def _generate_dense(self, request: Request) -> Generation:
        """The dense baseline: the full CFG + STG schedule, nothing skipped."""
        delegate = self._load_delegate()
        steps = self.baseline_steps()
        seed = self._seed(request)
        latents, video_coords = delegate.make_latents(seed)
        conditioning = delegate.encode(request.prompt, self.negative_prompt())
        latents, _, forward_s, per_step = delegate.denoise(
            latents=latents, conditioning=conditioning, video_coords=video_coords, steps=steps,
        )
        video = delegate.decode(latents, seed=seed)
        return Generation(
            video=video, latent=None, latent_step=None, executed_steps=steps,
            extra={
                "executed_steps": steps,
                "dit_forward_s": forward_s,
                "frame_rate": float(self.config.get("fps", 25)),
                **self._guidance_extra(delegate, per_step),
            },
        )

    # ------------------------------------------------------- nirvana-vid

    def _generate_nirvana(self, request: Request, *, cached_latent: Any, capture: bool) -> Generation:
        """Resume the schedule from a cached latent instead of running step 0.

        The cache holds the *completed step* keyed trajectory, so the replay
        runner can pick any ``reuse_start_step`` out of one source run.  The
        target resumes the LTX schedule at that step with the source's latent,
        which is only well defined because the scheduler's step size comes from
        the per-token timestep rather than from an internal step counter.
        """
        delegate = self._load_delegate()
        steps = self.baseline_steps()
        cfg = dict(self.config.get("nirvana_vid", {}))
        reuse_start = int(cfg.get("reuse_start_step", 5))
        cache_steps = tuple(sorted({int(step) for step in cfg.get("cache_steps", [reuse_start])}))
        if not 0 < reuse_start < steps:
            raise ValueError("nirvana_vid.reuse_start_step must be within the sampling schedule")
        if not cache_steps or any(not 0 < step < steps for step in cache_steps):
            raise ValueError("nirvana_vid.cache_steps must be non-empty steps within the sampling schedule")
        if cached_latent is None and not capture:
            raise RuntimeError("NIRVANA-VID cache miss must be handled before generation")

        seed = self._seed(request)
        conditioning = delegate.encode(request.prompt, self.negative_prompt())
        if cached_latent is None:
            latents, video_coords = delegate.make_latents(seed)
            start_step = 0
        else:
            latents = cached_latent.to(device=delegate.device)
            video_coords = delegate.video_coords(seed)
            start_step = reuse_start

        captured: dict[int, Any] = {}
        latents, _, forward_s, per_step = delegate.denoise(
            latents=latents, conditioning=conditioning, video_coords=video_coords,
            steps=steps, start_step=start_step,
            capture_steps=cache_steps if capture else (), capture_sink=captured,
        )
        video = delegate.decode(latents, seed=seed)
        return Generation(
            video=video,
            latent=captured if capture else None,
            latent_step=max(cache_steps),
            executed_steps=steps - start_step,
            extra={
                "executed_steps": steps - start_step,
                "restart_step": start_step,
                "source_trajectory": cached_latent is not None,
                "cache_steps": list(cache_steps),
                "dit_forward_s": forward_s,
                **self._guidance_extra(delegate, per_step),
            },
        )

    # ------------------------------------------------------------- chorus

    def _generate_chorus(self, request: Request, *, cached_latent: Any, capture: bool) -> Generation:
        """Chorus on LTX: SRD + TGAA over Stage 2, source-latent fusion in-band.

        The stage structure is the Wan2.1 one verbatim: Stage 1 ``[0, K1)`` dense,
        Stage 2 ``[K1, K2)`` runs Medit-queries/Msee-keys self-attention with
        TGAA on the differential caption tokens and fuses the source latent
        outside Medit, Stage 3 ``[K2, N)`` is dense again so it can repair what
        Stage 2's partial model outputs introduced.
        """
        delegate = self._load_delegate()
        chorus_cfg = dict(self.config.get("chorus", {}))
        stages = chorus_cfg.get("stages", {})
        steps = self.baseline_steps()
        k1 = int(stages.get("k1", 5))
        k2 = int(stages.get("k2", 30))
        if not 0 <= k1 < k2 <= steps:
            raise ValueError(f"Invalid Chorus boundaries: K1={k1}, K2={k2}, N={steps}")
        is_source = cached_latent is None
        if is_source and not capture:
            raise RuntimeError("Chorus cache miss must be handled before generation")
        if not is_source and (
            not isinstance(cached_latent, dict)
            or cached_latent.get("format") != self.TRAJECTORY_FORMAT
            or "trajectory_paths" not in cached_latent
        ):
            raise ValueError("Chorus cache is not an LTX per-step trajectory")

        seed = self._seed(request)
        conditioning = delegate.encode(request.prompt, self.negative_prompt())
        latent_frames, latent_height, latent_width = delegate.latent_shape()

        if is_source:
            latents, video_coords = delegate.make_latents(seed)
            initial = latents.detach().to("cpu").clone()
            captured: dict[int, Any] = {}
            latents, _, forward_s, per_step = delegate.denoise(
                latents=latents, conditioning=conditioning, video_coords=video_coords,
                steps=steps, capture_steps=tuple(range(1, steps + 1)), capture_sink=captured,
            )
            video = delegate.decode(latents, seed=seed)
            trajectory = [initial] + [captured[step] for step in range(1, steps + 1)]
            payload = {
                "format": self.TRAJECTORY_FORMAT,
                "trajectory": trajectory,
                "steps": steps,
                "k1": k1,
                "k2": k2,
                "seed": seed,
                "frame_rate": float(self.config.get("fps", 25)),
                "latent_shape": [latent_frames, latent_height, latent_width],
            }
            return Generation(
                video=video, latent=payload, latent_step=k1, executed_steps=steps,
                extra={
                    "chorus_stage": "source_capture",
                    "executed_steps": steps,
                    "dit_forward_s": forward_s,
                    "chorus_stages": {"steps": steps, "k1": k1, "k2": k2,
                                      "stage1": [0, k1], "stage2": [k1, k2], "stage3": [k2, steps]},
                    **self._guidance_extra(delegate, per_step),
                },
            )

        return self._chorus_target(
            delegate, request, cached_latent, chorus_cfg, steps, k1, k2, conditioning, seed
        )

    def _chorus_target(self, delegate, request, cached_latent, chorus_cfg, steps, k1, k2,
                       conditioning, seed) -> Generation:
        import torch

        from methods.chorus.llm import extract_prompt_diff
        from methods.chorus.mask import build_edit_mask

        trajectory_paths = [Path(item) for item in cached_latent["trajectory_paths"]]
        if len(trajectory_paths) != steps + 1 or any(not path.is_file() for path in trajectory_paths):
            raise ValueError(f"Chorus source trajectory is incomplete; expected {steps + 1} per-step files")
        latent_frames, latent_height, latent_width = delegate.latent_shape()

        # --- mask: LLM prompt diff picks the source object, Grounded-SAM-2
        # --- segments it in the source video, nothing here is LTX-specific.
        source_prompt = request.cache_prompt or request.prompt
        llm_started = time.perf_counter()
        llm_diff = extract_prompt_diff(source_prompt, request.prompt)
        llm_s = time.perf_counter() - llm_started
        source_entities = llm_diff.get("source_entities") or llm_diff.get("removed_tokens") or []
        target_entities = llm_diff.get("target_entities") or []
        unique_entities = list(dict.fromkeys(str(item).strip() for item in source_entities if str(item).strip()))
        phrase = " ".join(unique_entities[:4]).strip()
        if not phrase:
            raise RuntimeError("LLM prompt diff did not return a source object for Grounded-SAM-2")
        source_video = getattr(self, "chorus_source_video_path", None)
        if not source_video:
            raise RuntimeError("Chorus source video is unavailable for Grounded-SAM-2")
        sam_started = time.perf_counter()
        mask_bundle, mask_metadata = build_edit_mask(
            Path(source_video),
            phrase,
            latent_shape=(latent_frames, latent_height, latent_width),
            # LTX packs with patch size 1, so the token grid equals the latent
            # grid and the mask's token and latent forms coincide.
            patch_size=(1, 1),
            config={**chorus_cfg.get("grounded_sam2", {}),
                    "mask_dilation_edit": chorus_cfg.get("mask_dilation_edit", 8),
                    "mask_dilation_see": chorus_cfg.get("mask_dilation_see", 14)},
        )
        sam_s = time.perf_counter() - sam_started
        token_edit = mask_bundle["token_edit"]
        token_see = mask_bundle["token_see"]
        latent_edit = mask_bundle["latent_edit"].reshape(-1)

        # --- TGAA: differential caption tokens, mapped in the target prompt.
        differential = self._prompt_token_indices(delegate, request.prompt, target_entities)
        if not differential:
            raise RuntimeError("TGAA could not map the LLM differential tokens to target prompt tokens")

        state = LtxRegionalState(
            video_edit=token_edit,
            video_see=token_see,
            first_step=k1,
            last_step=k2,
            total_steps=steps,
            differential_tokens=tuple(differential),
            gamma_k=float(chorus_cfg.get("gamma_k", 1.5)),
            gamma_o=float(chorus_cfg.get("gamma_o", 1.5)),
            guidance=delegate.guidance,
        )

        def fuse(completed_step: int, sample: Any) -> Any:
            if not k1 <= completed_step - 1 < k2:
                return sample
            source = self._load_chorus_state(trajectory_paths[completed_step], sample)
            return self._fuse_chorus_latent(sample, source, latent_edit)

        latents = self._load_chorus_state(trajectory_paths[k1], None).to(device=delegate.device)
        # The regional state is keyed by the step *index* [K1, K2), the same
        # window the latent fusion below uses, so the hook is a straight pass
        # through -- the CogVideoX Chorus wires ``on_begin=region.set_step`` the
        # same way.
        with install_regional_blocks(delegate.transformer, state):
            latents, _, forward_s, per_step = delegate.denoise(
                latents=latents, conditioning=conditioning,
                video_coords=delegate.video_coords(seed),
                steps=steps, start_step=k1,
                begin_step_hook=state.set_step,
                on_step_end=fuse,
            )
        video = delegate.decode(latents, seed=seed)
        preprocess_s = llm_s + sam_s
        return Generation(
            video=video, latent=None, latent_step=k1, executed_steps=steps - k1,
            extra={
                "chorus_stage": "target_reuse",
                "chorus_stages": {"steps": steps, "k1": k1, "k2": k2,
                                  "stage1": [0, k1], "stage2": [k1, k2], "stage3": [k2, steps]},
                "executed_steps": steps - k1,
                "dit_forward_s": forward_s,
                "chorus_llm": llm_diff,
                "chorus_mask": mask_metadata,
                "chorus_preprocess_s": preprocess_s,
                "chorus_llm_s": llm_s,
                "chorus_sam_s": sam_s,
                "excluded_latency_s": preprocess_s,
                "chorus_tgaa": {
                    "enabled": True,
                    "gamma_k": state.gamma_k,
                    "gamma_o": state.gamma_o,
                    "schedule": "linear_step_decay_approximation",
                    "scope": "stage2",
                    "blocks": "all",
                    "calls": state.tgaa_calls,
                    "differential_tokens": _token_summary(differential),
                },
                "chorus_cache": {
                    "source_trajectory_steps": steps,
                    "format": "per_step_manifest",
                    "stage2_latent_fusion": True,
                    "latent_fusion_from_step": k1,
                    "latent_fusion_through_step": k2 - 1,
                    "latent_fusion_semantics": "source latent fused only during Stage 2; Stage 3 disables latent reuse",
                    "srd": {
                        "enabled": True,
                        "edit_tokens": int(token_edit.sum()),
                        "see_tokens": int(token_see.sum()),
                        "self_attention": "Medit queries / Msee keys-values",
                        "cross_attention": "Medit queries",
                        "ffn": "Medit tokens",
                    },
                },
                "chorus_scheduler": {
                    # FlowMatchEuler is first order: it carries no multistep
                    # derivative history, so Wan's Stage-3 reset has nothing to
                    # clear here.  Recorded so the two models' reports agree.
                    "stage3_history_reset": False,
                    "stage3_history_reset_step": k2,
                    "stage3_history_reset_semantics": "not applicable: FlowMatchEulerDiscreteScheduler is first order",
                },
                **self._guidance_extra(delegate, per_step),
            },
        )

    # ------------------------------------------------------------ chorus2

    def _generate_chorus2(self, request: Request, *, cached_latent: Any, capture: bool) -> Generation:
        """Chorus2 on LTX: dense source records routing, target replays it sparsely.

        Unlike Chorus, no source latent is reused.  The source runs its whole
        schedule densely while recording, per block and per CFG/STG branch, the
        block-level routing its self-attention actually used; the target re-runs
        its own schedule from its own noise and serves every ``attn1`` from the
        recorded routing once the sparse window opens.
        """
        delegate = self._load_delegate()
        cfg = dict(self.config.get("chorus2", {}))
        steps = self.baseline_steps()
        sparse_start = int(cfg.get("sparse_start_step", 10))
        if not 0 <= sparse_start < steps:
            raise ValueError(f"chorus2.sparse_start_step must be in [0, {steps})")
        layer_refresh_size = int(cfg.get("layer_refresh_size", 8))
        mask_config = self._block_mask_config(cfg)
        layers = list(range(delegate.num_layers))
        is_source = cached_latent is None
        if is_source and not capture:
            raise RuntimeError("Chorus2 cache miss must be handled before generation")
        if not is_source and (
            not isinstance(cached_latent, dict) or cached_latent.get("format") != self.CHORUS2_FORMAT
        ):
            raise ValueError("Chorus2 cache payload is not an LTX source mask directory")

        cache_root = self._request_cache_root()
        seed = self._seed(request)
        conditioning = delegate.encode(request.prompt, self.negative_prompt())
        video_coords = None
        tracker = LtxStepTracker()

        if is_source:
            mask_dir = cache_root / "chorus2_source" / "block_masks"
            cache = LtxBlockMaskCache(
                mask_dir, mask_config, layers=layers, start_step=sparse_start + 1, create=True
            )
            latents, video_coords = delegate.make_latents(seed)
            with LtxAttentionProcessors(
                delegate.transformer,
                lambda layer: LtxMaskCaptureProcessor(cache, tracker, layer, delegate.guidance),
                layers,
            ):
                latents, _, forward_s, per_step = delegate.denoise(
                    latents=latents, conditioning=conditioning, video_coords=video_coords,
                    steps=steps, begin_step_hook=tracker.set_step,
                )
            cache.finalize()
            expected = {
                (entry["completed_step"], branch, layer)
                for entry in per_step
                for branch in entry["branches"]
                for layer in layers
                if entry["completed_step"] >= sparse_start + 1
            }
            missing = expected - set(cache.entries)
            if missing:
                raise RuntimeError(
                    f"Chorus2 source mask capture is incomplete: {len(missing)} missing, e.g. {sorted(missing)[0]}"
                )
            video = delegate.decode(latents, seed=seed)
            source_dir = cache_root / "chorus2_source"
            out_video = source_dir / f"{request.uuid}.source.mp4"
            delegate.save_video(video, out_video)
            payload = {
                "format": self.CHORUS2_FORMAT,
                "mask_dir": str(mask_dir),
                "steps": steps,
                "sparse_start_step": sparse_start,
                "seed": seed,
                "layers": layers,
            }
            return Generation(
                video=None, video_path=out_video, latent=payload, latent_step=steps, executed_steps=steps,
                extra={
                    "chorus2_stage": "source_capture",
                    "executed_steps": steps,
                    "dit_forward_s": forward_s,
                    "chorus2": {"layers": layers, "sparse_start_step": sparse_start,
                                "captured_entries": len(cache.entries)},
                    **self._guidance_extra(delegate, per_step),
                },
            )

        mask_dir = Path(str(cached_latent["mask_dir"]))
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Chorus2 source mask cache is missing: {mask_dir}")
        cache = LtxBlockMaskCache(mask_dir, mask_config)
        executor = LtxSparseExecutor(cache, layer_refresh_size=layer_refresh_size)
        cache_prepare = executor.prepare(device=delegate.device)
        latents, video_coords = delegate.make_latents(seed)
        target_dir = cache_root / "chorus2_target"
        out_video = target_dir / f"{request.uuid}.mp4"
        with LtxAttentionProcessors(
            delegate.transformer,
            lambda layer: LtxSparseProcessor(executor, tracker, layer, delegate.guidance),
            layers,
        ):
            latents, _, forward_s, per_step = delegate.denoise(
                latents=latents, conditioning=conditioning, video_coords=video_coords,
                steps=steps, begin_step_hook=tracker.set_step,
            )
        expected_runs = sum(
            len(entry["branches"]) for entry in per_step if entry["completed_step"] >= sparse_start + 1
        ) * len(layers)
        if executor.runs != expected_runs:
            raise RuntimeError(
                f"Chorus2 target sparse coverage is wrong: runs={executor.runs} expected={expected_runs}"
            )
        video = delegate.decode(latents, seed=seed)
        delegate.save_video(video, out_video)
        return Generation(
            video=None, video_path=out_video, latent=None, latent_step=None, executed_steps=steps,
            extra={
                "chorus2_stage": "target_reuse",
                "executed_steps": steps,
                "dit_forward_s": forward_s,
                "cache_prepare_s": cache_prepare["elapsed_seconds"],
                "chorus2": {"layers": layers, "sparse_start_step": sparse_start, **executor.summary()},
                **self._guidance_extra(delegate, per_step),
            },
        )

    # ------------------------------------------------------------ carnator

    def _generate_carnator(self, request: Request, *, cached_latent: Any, capture: bool) -> Generation:
        """Carnator B5 on LTX, algorithm-identical to the Wan2.1 native path.

        Source: dense 50-step run carrying the multi-noun Early Signature probes
        at completed step 5 (cross attention proposes the noun seeds, self
        attention turns them into Mbase), the full latent trajectory, the fusion
        mask template, outer routing masks from completed step 11, and the video.
        Target: per-noun mask selection, resume from the source step-9 latent,
        soft B5 boundary latent injection, region sparse on the middle blocks,
        outer sparse on the rest, and w8a8 FFN.
        """
        from carnator.adapters.wan_ti2v_5b import parse_int_ranges

        delegate = self._load_delegate()
        cfg = dict(self.config.get("carnator_b5", {}))
        steps = self.baseline_steps()
        probe_step = int(cfg.get("probe_step", 5))
        reuse_step = int(cfg.get("reuse_step", 10))
        if probe_step != 5:
            # The signature profile is defined at step 5 (EARLY_SIGNATURE_PROFILE_ID)
            # and ``write_early_signature`` rejects any other completed step.
            raise ValueError("carnator_b5.probe_step must be 5; the Early Signature is defined there")
        if not 0 < reuse_step < steps:
            raise ValueError(f"carnator_b5.reuse_step must be in (0, {steps})")
        regional_layers = parse_int_ranges(str(cfg.get("regional_layers", "")))
        outer_layers = parse_int_ranges(str(cfg.get("outer_sparse_layers", "")))
        if not regional_layers or not outer_layers:
            raise ValueError("carnator_b5 needs both a regional and an outer-sparse layer set")
        if not set(regional_layers).isdisjoint(outer_layers):
            raise ValueError("carnator_b5 regional and outer-sparse layer sets overlap")
        if sorted(regional_layers + outer_layers) != list(range(delegate.num_layers)):
            raise ValueError(
                f"carnator_b5 layer sets must partition the {delegate.num_layers} LTX blocks"
            )
        cross_layer = int(cfg.get("attention_probe_layer", 17))
        self_layer = int(cfg.get("attention_self_layer", cross_layer))
        injection_mode = str(cfg.get("latent_injection_mode", "boundary-blend"))
        alpha_function = str(cfg.get("boundary_alpha_function", "smoothstep"))
        mask_config = self._block_mask_config(cfg)
        if injection_mode != "boundary-blend":
            raise ValueError(f"unexpected Carnator latent injection mode: {injection_mode}")

        is_source = cached_latent is None
        if is_source and not capture:
            raise RuntimeError("Carnator cache miss must be handled before generation")
        if not is_source and (
            not isinstance(cached_latent, dict) or cached_latent.get("format") != self.CARNATOR_FORMAT
        ):
            raise ValueError("Carnator cache payload is not an LTX source cache directory")

        seed = self._seed(request)
        conditioning = delegate.encode(request.prompt, self.negative_prompt())
        cache_root = self._request_cache_root()
        if is_source:
            return self._carnator_source(
                request, delegate, cfg, steps, probe_step, reuse_step, cross_layer, self_layer,
                outer_layers, mask_config, seed, conditioning, cache_root,
            )
        return self._carnator_target(
            request, delegate, cfg, steps, reuse_step, regional_layers, outer_layers, mask_config,
            injection_mode, alpha_function, seed, conditioning, cache_root,
            Path(str(cached_latent["cache_dir"])),
        )

    def _carnator_early_signature(
        self, delegate, cfg, *, request, conditioning, video_coords, steps, probe_step,
        step5_packed, step5_timestep, cross_probe, cross_layer, self_layer, output_dir, seed,
    ):
        """The step-5 Early Signature, built identically for the source and the target.

        Cross attention proposes the per-noun seeds, self attention decides where
        each noun actually lands, and ``build_canonical_fusion_mask`` fuses the
        two into ``Mbase``.  The source writes this next to its cache; the
        target's timing-parity probe writes its own copy in the request's target
        directory, which is the only thing that differs between the two calls.
        """
        from carnator.early_signature import write_early_signature
        from carnator.ltx_attention_probe import LtxMultiNounSelfAttentionContrastProbe
        from carnator.masks import (
            FusionMaskConfig,
            build_canonical_fusion_mask,
            build_cross_attention_seed_scores,
            seed_indices_from_cross_score,
        )

        token_grid = delegate.latent_shape()
        cross_scores = {}
        seed_indices = {}
        for noun, records in cross_probe.records_by_noun.items():
            if not records:
                raise RuntimeError(f"Carnator cross-attention probe recorded nothing for {noun!r}")
            score = build_cross_attention_seed_scores(
                records, completed_step=probe_step
            )["score"].float()
            cross_scores[noun] = score
            seed_indices[noun] = seed_indices_from_cross_score(
                score,
                top_p=float(cfg.get("mask_top_p", 0.02)),
                background_p=float(cfg.get("mask_background_p", 0.02)),
            )
        self_probe = LtxMultiNounSelfAttentionContrastProbe(
            delegate.transformer, self_layer, seed_indices, token_grid
        )
        delegate.probe_forward(
            latents=step5_packed, conditioning=conditioning, video_coords=video_coords,
            timestep=step5_timestep, context=self_probe,
        )
        if set(self_probe.records_by_noun) != set(cross_scores):
            raise RuntimeError("Carnator self-attention probe produced incomplete records")
        fusion_config = FusionMaskConfig(
            completed_step=probe_step,
            eps=float(cfg.get("fusion_eps", 0.2)),
            gamma=float(cfg.get("fusion_gamma", 1.0)),
            high_threshold=float(cfg.get("fusion_high_threshold", 0.45)),
            low_threshold=float(cfg.get("fusion_low_threshold", 0.35)),
            spatial_medit_radius=int(cfg.get("mask_spatial_medit_radius", 2)),
            spatial_msee_radius=int(cfg.get("mask_spatial_msee_radius", 3)),
        )
        mbase_by_noun = {}
        for noun in sorted(cross_scores):
            bundle = build_canonical_fusion_mask(
                self_probe.records_by_noun[noun]["self_contrast_max"],
                cross_scores[noun],
                fusion_config,
                original_mbase=None,
            )
            mbase_by_noun[noun] = bundle["Mbase"].detach().cpu().bool().contiguous()

        mask_policy = (
            "native-self-cross-fusion-mbase"
            f"|cross_layer={cross_layer}|self_layer={self_layer}"
            f"|top_p={float(cfg.get('mask_top_p', 0.02)):.8f}"
            f"|eps={float(cfg.get('fusion_eps', 0.2)):.8f}"
            f"|gamma={float(cfg.get('fusion_gamma', 1.0)):.8f}"
        )
        # ``write_early_signature`` wants a [C,T,H,W] latent and a [T,H,W] token
        # grid.  LTX packs to [B,F*H*W,C], so the signature latent is the
        # *unpacked* step-5 tensor; that is the same tensor the recorded tail
        # frame was decoded from, just in the grid layout the profile assumes.
        step5_grid = delegate.unpack(step5_packed)[0].detach().float().cpu().contiguous()
        signature = write_early_signature(
            output_dir,
            step5_latent=step5_grid,
            decoded_tail_frame=delegate.decode_tail(step5_packed, seed=seed),
            metadata={
                "prompt": request.prompt,
                "model": "LTX-Video-0.9.7-dev",
                "scheduler": "flow-match-euler",
                "scheduler_config": {
                    "total_steps": steps,
                    "frame_rate": float(self.config.get("fps", 25)),
                    "solver_order": None,
                },
                # LTX's scalar stand-in for a CFG weight: the released recipe
                # schedules guidance per step, and the scalar is what a target
                # request compares against.  Both sides read the same config, so
                # the compatibility block stays pairwise equal.
                "cfg": float(self.config.get("guidance_scale", 3.0)),
                "probe_seed": int(seed),
                "completed_step": probe_step,
                "latent_shape": [int(item) for item in step5_grid.shape],
                "latent_token_grid": [int(item) for item in token_grid],
                "noun_extractor": "lexical-spacy",
                "mask_policy": mask_policy,
            },
            mbase_by_noun=mbase_by_noun,
        )
        return signature, mbase_by_noun, mask_policy, self_probe

    def _carnator_source(self, request, delegate, cfg, steps, probe_step, reuse_step, cross_layer,
                         self_layer, outer_layers, mask_config, seed, conditioning,
                         cache_root) -> Generation:
        """Capture every artifact the Carnator target reuses, in one dense run."""
        from carnator.adapters.wan21_native import extract_request_nouns
        from carnator.ltx_attention_probe import LtxMultiNounCrossAttentionProbe
        from carnator.masks import (
            build_per_noun_mbase_bundle,
            save_mask_bundle,
        )

        nouns = extract_request_nouns(request)
        if not nouns:
            raise RuntimeError(f"Carnator found no nouns to probe in {request.prompt!r}")
        index_by_noun = {}
        for noun in nouns:
            indices = self._prompt_token_indices(delegate, request.prompt, [noun])
            if not indices:
                raise RuntimeError(f"Carnator could not locate prompt noun {noun!r} in the T5 encoding")
            index_by_noun[str(noun)] = indices

        source_dir = cache_root / "carnator_source"
        source_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = source_dir / "outer_block_masks"
        mask_cache = LtxBlockMaskCache(
            mask_dir, mask_config, layers=outer_layers, start_step=reuse_step + 1, create=True
        )
        token_grid = delegate.latent_shape()
        tracker = LtxStepTracker()
        cross_probe = LtxMultiNounCrossAttentionProbe(
            delegate.transformer, [cross_layer], index_by_noun, token_grid, guidance=delegate.guidance
        )
        captured: dict[int, Any] = {}

        def arm(index: int) -> None:
            tracker.set_step(index)
            cross_probe.current_step_index = index
            cross_probe.enabled = (index + 1) == probe_step

        latents, video_coords = delegate.make_latents(seed)
        with (
            cross_probe,
            LtxAttentionProcessors(
                delegate.transformer,
                lambda layer: LtxMaskCaptureProcessor(mask_cache, tracker, layer, delegate.guidance),
                outer_layers,
            ),
        ):
            # Every completed step is kept: the Early Signature reads step 5 out
            # of the same dict the target's resume latent and whole per-step
            # trajectory are written from.
            latents, _, forward_s, per_step = delegate.denoise(
                latents=latents, conditioning=conditioning, video_coords=video_coords,
                steps=steps, begin_step_hook=arm,
                capture_steps=tuple(range(1, steps + 1)), capture_sink=captured,
            )
        mask_cache.finalize()
        expected = {
            (entry["completed_step"], branch, layer)
            for entry in per_step
            for branch in entry["branches"]
            for layer in outer_layers
            if entry["completed_step"] >= reuse_step + 1
        }
        missing = expected - set(mask_cache.entries)
        if missing:
            raise RuntimeError(
                f"Carnator source outer-mask capture is incomplete: "
                f"{len(missing)} missing, e.g. {sorted(missing)[0]}"
            )

        # --- Early Signature: cross attention seeds, self attention decides.
        step5_packed = captured[probe_step].to(device=delegate.device, dtype=delegate.dtype)
        signature, mbase_by_noun, mask_policy, self_probe = self._carnator_early_signature(
            delegate, cfg,
            request=request, conditioning=conditioning, video_coords=video_coords,
            steps=steps, probe_step=probe_step, step5_packed=step5_packed,
            step5_timestep=float(per_step[probe_step - 1]["timestep"]),
            cross_probe=cross_probe, cross_layer=cross_layer, self_layer=self_layer,
            output_dir=source_dir, seed=seed,
        )
        template_noun = sorted(mbase_by_noun)[0]
        save_mask_bundle(
            source_dir / "fusion_masks",
            build_per_noun_mbase_bundle(
                mbase_by_noun[template_noun],
                source_completed_step=probe_step,
                spatial_medit_radius=int(cfg.get("mask_spatial_medit_radius", 2)),
                spatial_msee_radius=int(cfg.get("mask_spatial_msee_radius", 3)),
                selection={
                    "selected_source_noun": template_noun,
                    "selection_reason": "source_template",
                    "source_nouns": sorted(mbase_by_noun),
                    "target_nouns": sorted(mbase_by_noun),
                },
            ),
            save_cpu=True,
        )

        trajectory: list[dict[str, Any]] = []
        step_dir = source_dir / "trajectory"
        step_dir.mkdir(parents=True, exist_ok=True)
        for index in range(steps):
            path = step_dir / f"latents_step_{index:03d}.pt"
            delegate.save_latent(captured[index + 1], path)
            trajectory.append({"kind": "post_scheduler", "step_index": index, "path": str(path)})

        out_video = source_dir / f"{request.uuid}.source.mp4"
        video = delegate.decode(latents, seed=seed)
        delegate.save_video(video, out_video)
        manifest = {
            "format": "ltxvideo-carnator-source-cache/v1",
            "prompt": request.prompt,
            "seed": int(seed),
            "steps": steps,
            "probe_step": probe_step,
            "reuse_step": reuse_step,
            "token_grid": list(token_grid),
            "outer_layers": outer_layers,
            "outer_mask_dir": str(mask_dir),
            "source_video": str(out_video),
            "signature": signature,
            "trajectory": trajectory,
        }
        (source_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        payload = {
            "format": self.CARNATOR_FORMAT,
            "cache_dir": str(source_dir),
            "steps": steps,
            "reuse_step": reuse_step,
            "seed": int(seed),
        }
        return Generation(
            video=None, video_path=out_video, latent=payload, latent_step=steps, executed_steps=steps,
            extra={
                "carnator_stage": "source_capture",
                "executed_steps": steps,
                "dit_forward_s": forward_s,
                "carnator": {
                    "nouns": sorted(mbase_by_noun),
                    "source_video": str(out_video),
                    "outer_masks": mask_cache.summary(),
                    "self_probe": self_probe.summary(),
                    "mask_policy": mask_policy,
                },
                **self._guidance_extra(delegate, per_step),
            },
        )

    def _carnator_target_probe(self, request, delegate, cfg, steps, probe_step, cross_layer,
                               self_layer, source_dir, target_dir, selection, seed) -> dict[str, Any]:
        """The dense probe prefix the Carnator target pays before it reuses.

        Wan2.1 and Wan2.2 both run the first ``mask_probe_step`` steps of the
        full plan densely on the target, with the multi-noun cross-attention
        probe armed on the last one, and count that DiT time and those steps in
        the target's totals.  LTX runs the same prefix through the same CFG+STG
        loop the source uses, on the full 50-step schedule, and writes the
        target's own Early Signature beside the per-request cache.

        Nothing computed here can change which source is reused -- the per-noun
        mask selection and the reuse decision are already fixed when this runs,
        exactly as in the Wan probe -- so ``comparison`` is audit metadata only
        and its failure is swallowed.

        The probe runs at the fixed experiment seed rather than the request's,
        matching the Wan target probe's ``target_probe_seed``, so the probe
        latent does not depend on the request's own seed.
        """
        from carnator.adapters.wan21_native import extract_request_nouns
        from carnator.early_signature import compare_early_signatures
        from carnator.ltx_attention_probe import LtxMultiNounCrossAttentionProbe

        nouns = extract_request_nouns(request)
        index_by_noun = {}
        for noun in nouns:
            indices = self._prompt_token_indices(delegate, request.prompt, [noun])
            if indices:
                index_by_noun[str(noun)] = indices
        if not index_by_noun:
            raise RuntimeError(f"Carnator target probe found no prompt nouns in {request.prompt!r}")

        output_dir = Path(target_dir) / "target_probe"
        output_dir.mkdir(parents=True, exist_ok=True)
        conditioning = delegate.encode(request.prompt, self.negative_prompt())
        tracker = LtxStepTracker()
        cross_probe = LtxMultiNounCrossAttentionProbe(
            delegate.transformer, [cross_layer], index_by_noun, delegate.latent_shape(),
            guidance=delegate.guidance,
        )
        captured: dict[int, Any] = {}

        def arm(index: int) -> None:
            tracker.set_step(index)
            cross_probe.current_step_index = index
            cross_probe.enabled = (index + 1) == probe_step

        latents, video_coords = delegate.make_latents(seed)
        with cross_probe:
            # ``schedule_steps`` is what makes this a probe of the *full*
            # schedule's first five steps rather than a re-scaled five-step run:
            # the timesteps and the normalised guidance time both come from the
            # 50-step plan, and the data-dependent guidance schedule LTX uses is
            # indexed by exactly that normalised time.
            latents, _, forward_s, per_step = delegate.denoise(
                latents=latents, conditioning=conditioning, video_coords=video_coords,
                steps=probe_step, schedule_steps=steps, begin_step_hook=arm,
                capture_steps=(probe_step,), capture_sink=captured,
            )
        if probe_step not in captured:
            raise RuntimeError(f"Carnator target probe never captured completed step {probe_step}")

        step5_packed = captured[probe_step].to(device=delegate.device, dtype=delegate.dtype)
        signature, _mbase, _policy, _probe = self._carnator_early_signature(
            delegate, cfg,
            request=request, conditioning=conditioning, video_coords=video_coords,
            steps=steps, probe_step=probe_step, step5_packed=step5_packed,
            step5_timestep=float(per_step[probe_step - 1]["timestep"]),
            cross_probe=cross_probe, cross_layer=cross_layer, self_layer=self_layer,
            output_dir=output_dir, seed=seed,
        )

        comparison: dict[str, Any] | None = None
        source_signature = {"signature_path": str(source_dir / "online_color_signature.json")}
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
            "completed_steps": probe_step,
            "dit_forward_seconds": float(forward_s),
            "signature": signature,
            "comparison": comparison,
            "output_dir": str(output_dir),
        }

    def _carnator_target(self, request, delegate, cfg, steps, reuse_step, regional_layers,
                         outer_layers, mask_config, injection_mode, alpha_function, seed,
                         conditioning, cache_root, source_dir) -> Generation:
        """Reuse the source: per-noun mask, soft blend, region+outer sparse, w8a8."""
        from carnator.adapters.wan21_native import extract_request_nouns
        from carnator.adapters.wan_ti2v_5b import apply_latent_injection
        from carnator.early_signature import materialize_per_noun_inference_mask_bundle
        from carnator.ltx_quantized_ffn import LtxQuantizedFFNHooks
        from carnator.masks import load_mask_bundle

        if not source_dir.is_dir():
            raise FileNotFoundError(f"Carnator source cache is missing: {source_dir}")
        target_nouns = extract_request_nouns(request)
        target_dir = cache_root / "carnator_target"
        out_video = target_dir / f"{request.uuid}.mp4"
        selection = materialize_per_noun_inference_mask_bundle(
            {"signature_path": str(source_dir / "online_color_signature.json")},
            target_nouns=target_nouns,
            source_fusion_mask_dir=source_dir / "fusion_masks",
            output_dir=target_dir / "inference_masks",
        )
        mask_bundle = load_mask_bundle(Path(selection["mask_bundle_dir"]), None)
        manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
        trajectory_paths = [Path(item["path"]) for item in manifest["trajectory"]]
        if len(trajectory_paths) != steps or any(not path.is_file() for path in trajectory_paths):
            raise ValueError(f"Carnator source trajectory is incomplete; expected {steps} per-step files")
        resume_path = trajectory_paths[reuse_step - 1]

        # Timing parity with the Wan baselines: the target pays the same dense
        # probe prefix, and its DiT time and step count go into the totals below.
        # It runs before any sparse/quantised hook is installed, so it stays a
        # dense measurement on the plain model.
        probe_cross_layer = int(cfg.get("attention_probe_layer", 17))
        probe = self._carnator_target_probe(
            request, delegate, cfg, steps, int(cfg.get("probe_step", 5)),
            probe_cross_layer, int(cfg.get("attention_self_layer", probe_cross_layer)),
            source_dir, target_dir, selection, int(self.config.get("seed", 20260218)),
        )
        probe_steps = int(probe["completed_steps"])
        probe_dit_forward_s = float(probe["dit_forward_seconds"])

        tracker = LtxStepTracker()
        state = LtxRegionalState(
            video_edit=mask_bundle["Medit"],
            video_see=mask_bundle["Msee"],
            first_step=reuse_step,
            last_step=steps + 1,
            total_steps=steps,
            # Carnator has no differential-token guidance; TGAA belongs to Chorus.
            differential_tokens=(),
            tgaa_enabled=False,
            guidance=delegate.guidance,
        )
        mask_cache = LtxBlockMaskCache(Path(manifest["outer_mask_dir"]), mask_config)
        executor = LtxSparseExecutor(
            mask_cache, layer_refresh_size=int(cfg.get("layer_refresh_size", 8))
        )
        cache_prepare = executor.prepare(device=delegate.device)
        injection_stats: dict[str, Any] = {}
        injection_calls = 0
        injection_seconds = 0.0

        def fuse(completed_step: int, sample: Any) -> Any:
            nonlocal injection_calls, injection_seconds, injection_stats
            if completed_step < reuse_step + 1:
                return sample
            source_latent = delegate.load_latent(trajectory_paths[completed_step - 1])
            blended, elapsed, source_step, stats = apply_latent_injection(
                delegate.unpack(sample)[0],
                delegate.unpack(source_latent.to(device=delegate.device))[0],
                mask_bundle,
                mode=injection_mode,
                alpha_function=alpha_function,
            )
            injection_calls += 1
            injection_seconds += elapsed
            if not injection_stats:
                injection_stats = {**stats, "source_completed_step": int(source_step)}
            return delegate.pack(blended.unsqueeze(0)).to(dtype=sample.dtype)

        def begin_step(index: int) -> None:
            # Two step-tracked consumers with one hook: the outer-sparse
            # processor reads the tracker's 1-based ``computing_step``, while
            # the regional state's window is the step index [reuse_step, N) --
            # the same window the latent injection below uses.
            tracker.set_step(index)
            state.set_step(index)

        latents = delegate.load_latent(resume_path).to(device=delegate.device, dtype=delegate.dtype)
        with (
            LtxQuantizedFFNHooks(delegate.transformer, list(range(delegate.num_layers))),
            LtxAttentionProcessors(
                delegate.transformer,
                lambda layer: LtxSparseProcessor(executor, tracker, layer, delegate.guidance),
                outer_layers,
            ),
            install_regional_blocks(delegate.transformer, state, layers=regional_layers),
        ):
            latents, _, forward_s, per_step = delegate.denoise(
                latents=latents, conditioning=conditioning,
                video_coords=delegate.video_coords(seed),
                steps=steps, start_step=reuse_step,
                begin_step_hook=begin_step,
                on_step_end=fuse,
            )
        expected_injections = steps - reuse_step
        if injection_calls != expected_injections:
            raise RuntimeError(
                f"Carnator target latent injection is incomplete: {injection_calls} of {expected_injections}"
            )
        expected_runs = sum(
            len(entry["branches"]) for entry in per_step if entry["completed_step"] >= reuse_step + 1
        ) * len(outer_layers)
        if executor.runs != expected_runs:
            raise RuntimeError(
                f"Carnator target outer-sparse coverage is wrong: runs={executor.runs} expected={expected_runs}"
            )
        video = delegate.decode(latents, seed=seed)
        delegate.save_video(video, out_video)
        reuse_steps = steps - reuse_step
        return Generation(
            video=None, video_path=out_video, latent=None, latent_step=None,
            executed_steps=probe_steps + reuse_steps,
            extra={
                "carnator_stage": "target_reuse",
                "executed_steps": probe_steps + reuse_steps,
                "dit_forward_s": forward_s + probe_dit_forward_s,
                "cache_prepare_s": cache_prepare["elapsed_seconds"],
                "probe_dit_forward_s": probe_dit_forward_s,
                "probe_steps": probe_steps,
                "probe_comparison": probe["comparison"],
                "carnator": {
                    "inference_mask_selection": {
                        key: value for key, value in selection.items() if key != "mask_bundle_dir"
                    },
                    "target_nouns": list(target_nouns),
                    "target_probe": {
                        "completed_steps": probe_steps,
                        "dit_forward_s": probe_dit_forward_s,
                    },
                    "masked_injection": {
                        "mode": injection_mode,
                        "alpha_function": alpha_function,
                        "calls": injection_calls,
                        "seconds": injection_seconds,
                        **injection_stats,
                    },
                    "regional_layers": regional_layers,
                    "regional_calls": state.calls,
                    "outer_sparse": {"layers": outer_layers, **executor.summary()},
                    "ffn_backend": "w8a8",
                },
                **self._guidance_extra(delegate, per_step),
            },
        )

    @staticmethod
    def _block_mask_config(cfg: dict[str, Any]) -> LtxBlockMaskConfig:
        return LtxBlockMaskConfig(
            q_block_size=int(cfg.get("q_block_size", 64)),
            k_block_size=int(cfg.get("k_block_size", 128)),
            top_p=float(cfg.get("top_p", 0.95)),
            min_top_k=float(cfg.get("min_top_k", 0.1)),
        )

    @staticmethod
    def _load_chorus_state(path: Path, like: Any) -> Any:
        import torch

        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if like is not None:
            return state.to(device=like.device, dtype=like.dtype)
        return state

    @staticmethod
    def _fuse_chorus_latent(sample: Any, source_state: Any, edit_mask: Any) -> Any:
        """Keep target values inside Medit and take the source outside it.

        LTX latents are packed ``[B, F*H*W, C]`` with patch size 1, so the
        ``(F, H, W)`` edit mask flattens directly onto the token axis.
        """
        import torch

        if sample.shape != source_state.shape:
            raise ValueError(
                "Chorus source/target latent shapes differ: "
                f"sample={tuple(sample.shape)}, source={tuple(source_state.shape)}"
            )
        edit = edit_mask.reshape(1, -1, 1).to(device=sample.device, dtype=torch.bool)
        if edit.shape[1] != sample.shape[1]:
            raise ValueError(
                f"Chorus latent mask has {edit.shape[1]} tokens but the latent has {sample.shape[1]}"
            )
        return torch.where(edit, sample, source_state.to(device=sample.device, dtype=sample.dtype))

    @staticmethod
    def _prompt_token_indices(delegate: Any, prompt: str, entities: list[Any]) -> list[int]:
        """Token positions of ``entities`` in the prompt's T5 encoding.

        Serves both TGAA's differential tokens and Carnator's per-noun seed
        locations, which is why it lives on the adapter rather than in either
        method.  The tokenizer is called with the same padding and length limit
        as ``LTXConditionPipeline.encode_prompt``, so the positions index the
        rows of the conditioning the transformer actually receives.
        """
        tokenizer = delegate.tokenizer
        max_length = int(delegate.config.get("max_sequence_length", 256))
        try:
            encoded = tokenizer(
                prompt, add_special_tokens=True, truncation=True,
                padding="max_length", max_length=max_length, return_tensors="pt",
            )
            ids = encoded.input_ids[0].tolist()
            pieces = [str(tokenizer.decode([token])).strip().lower() for token in ids]
        except Exception:
            return []
        terms = [str(item).strip().lower() for item in entities if str(item).strip()]
        return [index for index, piece in enumerate(pieces)
                if piece and any(term in piece or piece in term for term in terms)]

    # ------------------------------------------------------------ caches I/O

    def save_latent(self, latent: Any, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(latent, dict) and latent.get("format") in self.POINTER_FORMATS:
            path.write_text(json.dumps(latent, indent=2) + "\n", encoding="utf-8")
            return
        if isinstance(latent, dict) and latent.get("format") == self.TRAJECTORY_FORMAT:
            # A trajectory is one tensor per completed step; a sidecar directory
            # keeps the manifest readable and the tensors in their own format.
            delegate = self._load_delegate()
            step_dir = path.with_suffix(".steps")
            if step_dir.exists():
                shutil.rmtree(step_dir)
            step_dir.mkdir(parents=True, exist_ok=True)
            files: list[str] = []
            for index, state in enumerate(latent["trajectory"]):
                name = f"latents_step_{index:03d}.pt"
                delegate.save_latent(state, step_dir / name)
                files.append(name)
            manifest = {key: value for key, value in latent.items() if key != "trajectory"}
            manifest["trajectory_files"] = files
            path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            return
        self._load_delegate().save_latent(latent, path)

    def load_latent(self, path: Path) -> Any:
        path = Path(path)
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return self._load_delegate().load_latent(path)
        if manifest.get("format") in self.POINTER_FORMATS:
            return manifest
        if manifest.get("format") != self.TRAJECTORY_FORMAT:
            return self._load_delegate().load_latent(path)
        step_dir = path.with_suffix(".steps")
        names = list(manifest.get("trajectory_files", []))
        paths = [step_dir / name for name in names]
        if len(paths) != int(manifest.get("steps", -1)) + 1 or any(not item.is_file() for item in paths):
            raise ValueError(f"Chorus trajectory sidecar is incomplete for {path}")
        return {**manifest, "trajectory_paths": paths}

    def save_video(self, video: Any, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # ``export_to_video`` takes a sequence of frames; a decoded ``[F,H,W,C]``
        # array (or a packed torch tensor) is split into per-frame arrays here.
        if hasattr(video, "ndim") and video.ndim == 4:
            video = list(video)
        self._load_delegate().save_video(video, path)
