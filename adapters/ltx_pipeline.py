"""LTX-Video 0.9.7-dev model boundary.

The checkpoint is diffusers-layout, so ``LTXConditionPipeline`` supplies the
components and the packing/latent helpers.  Its ``__call__`` is *not* used:
0.9.7-dev is a non-distilled checkpoint whose released recipe is CFG + STG, and
the stock pipeline implements CFG only.  Every baseline therefore drives the
same hand-rolled loop here, which keeps the dense baseline and the cache
methods on one sampling implementation.

With ``stg.enabled: false`` the loop reproduces
``LTXConditionPipeline.__call__`` step for step (same ``linear_quadratic``
timestep schedule, same scheduler call, same CFG arithmetic), so the two can be
cross-checked directly.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

# The diffusion-time frame rate.  ``LTXConditionPipeline.__call__`` takes
# ``frame_rate: int = 25`` and divides the first ``video_coords`` axis by it
# before the transformer sees it, so this is a *model* constant, not a
# container choice: 24 here would rotate the RoPE time coordinates by 25/24
# relative to the stock pipeline.  The same value is used to write the video so
# the two cannot drift apart.
DEFAULT_FPS = 25


def _resolve_dtype(torch: Any, name: str) -> Any:
    return torch.bfloat16 if str(name) == "bfloat16" else torch.float16


class StgSchedule:
    """Per-step CFG / STG weights from the released piecewise schedule.

    The released ``configs/ltxv-13b-0.9.7-dev.yaml`` describes a multi-scale
    30-step pipeline; this experiment runs a single-stage schedule whose length
    is set by ``baseline_steps``.  The segments are therefore selected by
    normalised time through ``guidance_timesteps`` -- the same lookup the
    reference implementation uses -- which makes the mapping independent of the
    step count.  That alignment is an engineering choice, not a bit-exact
    reproduction of the released run, and every request records it in ``extra``.
    """

    def __init__(self, values: dict[str, Any]) -> None:
        self.enabled = bool(values.get("enabled", False))
        self.mode = str(values.get("mode", "attention_values"))
        self.cfg_star_rescale = bool(values.get("cfg_star_rescale", False))
        self.boundaries = [float(item) for item in values.get("guidance_timesteps", [1.0])]
        self.guidance_scale = [float(item) for item in values.get("guidance_scale", [1.0])]
        self.stg_scale = [float(item) for item in values.get("stg_scale", [0.0])]
        self.rescaling_scale = [float(item) for item in values.get("rescaling_scale", [1.0])]
        self.skip_block_list = [list(item) for item in values.get("skip_block_list", [[]])]
        sizes = {len(self.guidance_scale), len(self.stg_scale), len(self.skip_block_list)}
        if self.enabled and sizes != {len(self.boundaries)}:
            raise ValueError(f"STG schedule segments must all have {len(self.boundaries)} entries, got {sorted(sizes)}")
        if self.mode != "attention_values":
            raise ValueError(f"unsupported STG mode {self.mode!r}; only 'attention_values' is implemented")

    def segment(self, normalized_time: float) -> int:
        """Index of the segment covering ``normalized_time`` (1.0 -> 0.0)."""
        for index, boundary in enumerate(self.boundaries):
            if boundary <= normalized_time:
                return index
        return len(self.boundaries) - 1

    def for_step(self, normalized_time: float, default_guidance: float) -> "StgStep":
        index = self.segment(normalized_time)
        if not self.enabled:
            return StgStep(default_guidance, 0.0, 1.0, (), self.cfg_star_rescale, index, enabled=False)
        return StgStep(
            self.guidance_scale[index],
            self.stg_scale[index],
            self.rescaling_scale[index] if index < len(self.rescaling_scale) else 1.0,
            tuple(self.skip_block_list[index]),
            self.cfg_star_rescale,
            index,
            enabled=True,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "cfg_star_rescale": self.cfg_star_rescale,
            "guidance_timesteps": self.boundaries,
            "guidance_scale": self.guidance_scale,
            "stg_scale": self.stg_scale,
            "skip_block_list": self.skip_block_list,
            "alignment": "released first_pass segments mapped by normalised time onto baseline_steps",
        }


class StgStep:
    """Resolved guidance weights for one sampling step."""

    __slots__ = ("guidance_scale", "stg_scale", "rescaling_scale", "skip_blocks", "cfg_star_rescale",
                 "segment", "enabled")

    def __init__(self, guidance_scale: float, stg_scale: float, rescaling_scale: float,
                 skip_blocks: tuple[int, ...], cfg_star_rescale: bool, segment: int, *, enabled: bool) -> None:
        self.guidance_scale = guidance_scale
        self.stg_scale = stg_scale
        self.rescaling_scale = rescaling_scale
        self.skip_blocks = skip_blocks
        self.cfg_star_rescale = cfg_star_rescale
        self.segment = segment
        self.enabled = enabled

    @property
    def perturbed(self) -> bool:
        return self.stg_scale > 0.0 and bool(self.skip_blocks)


class GuidanceState:
    """STG perturbation state for the step currently being denoised.

    The reference implementation does not run a second forward for the
    perturbed branch: it concatenates the branches into one batch
    (``[negative, prompt, prompt]``) and hands ``attn1`` a
    ``[num_layers, num_conds]`` skip mask whose perturbation column is 0 for the
    blocks named in ``skip_block_list``.  Where that mask is 0 the attention
    output is replaced by the raw value projection.

    The mask is published here rather than threaded through call arguments
    because the hooks that need it -- the regional block patch and the sparse
    ``attn1`` processors -- are installed on the transformer, not called by this
    loop.  It is reset to empty outside the forward, so a leaked hook can only
    ever see "no perturbation".
    """

    # The branch batch is always a slice of this order, which is why the
    # perturbed branch is the last column of the skip mask.
    LABELS = ("uncond", "cond", "cond_perturbed")

    def __init__(self) -> None:
        self.mask: Any | None = None
        self.skip_blocks: frozenset[int] = frozenset()
        # True branch identity for the current step.  The branch count alone
        # does not identify the rows: with STG on and CFG off the batch is
        # [cond, cond_perturbed], not [uncond, cond], and every consumer that
        # keys on a branch name has to agree with the source run.
        self.branches: tuple[str, ...] | None = None

    def set(self, mask: Any | None, skip_blocks: Sequence[int], branches: Sequence[str]) -> None:
        self.mask = mask
        self.skip_blocks = frozenset(int(item) for item in skip_blocks)
        self.branches = tuple(str(item) for item in branches)

    def reset(self) -> None:
        self.mask = None
        self.skip_blocks = frozenset()
        self.branches = None

    def layer(self, layer: int, batch: int) -> Any | None:
        """The perturbation column for one layer, or ``None`` if not skipped."""
        if self.mask is None or layer not in self.skip_blocks:
            return None
        row = self.mask[layer]
        if row.numel() != batch:
            raise ValueError(
                f"LTX STG skip mask has batch {row.numel()} but the block sees {batch}; "
                f"the mask must be rebuilt whenever the branch count changes"
            )
        return row

    def branch_labels(self, batch: int) -> tuple[str, ...]:
        """Names for the ``batch`` rows of the current forward."""
        if self.branches is not None and len(self.branches) == batch:
            return self.branches
        if batch == 1:
            return ("cond",)
        if batch == 2:
            return ("uncond", "cond")
        if batch == 3:
            return self.LABELS
        raise ValueError(f"unsupported LTX branch count: {batch}")


class LtxVideoPipelineDelegate:
    """Loads LTX-Video 0.9.7-dev and runs the shared CFG(+STG) sampling loop."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.pipe: Any | None = None
        self.torch: Any | None = None
        self.stg = StgSchedule(config.get("stg", {}))
        # Read by the baselines' attention hooks on every block forward.
        self.guidance = GuidanceState()
        self._forward_seconds = 0.0

    # ------------------------------------------------------------------ load

    def _load(self) -> None:
        if self.pipe is not None:
            return
        import torch
        from diffusers import AutoencoderKLLTXVideo, LTXConditionPipeline, LTXVideoTransformer3DModel
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from safetensors.torch import load_file
        from transformers import T5Config, T5EncoderModel, T5Tokenizer

        device = str(self.config["device"])
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
        dtype = _resolve_dtype(torch, self.config["dtype"])
        root = Path(self.config["model_path"])
        weights = Path(self.config["t5_weights"])

        # The checkpoint ships text_encoder/config.json without weights; the
        # T5-v1.1-XXL encoder is loaded from the standalone safetensors file.
        text_encoder = T5EncoderModel(T5Config.from_pretrained(root / "text_encoder"))
        missing, unexpected = text_encoder.load_state_dict(load_file(str(weights)), strict=False)
        unexpected = [key for key in unexpected if not key.startswith("decoder.")]
        if unexpected:
            raise RuntimeError(f"unexpected T5 weights: {sorted(unexpected)[:5]}")
        text_encoder = text_encoder.to(dtype=dtype).eval()

        pipe = LTXConditionPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(root, subfolder="scheduler"),
            text_encoder=text_encoder,
            tokenizer=T5Tokenizer.from_pretrained(root, subfolder="tokenizer"),
            transformer=LTXVideoTransformer3DModel.from_pretrained(
                root, subfolder="transformer", torch_dtype=dtype
            ),
            vae=AutoencoderKLLTXVideo.from_pretrained(root, subfolder="vae", torch_dtype=dtype),
        )
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)
        for enable in ("enable_tiling", "enable_slicing"):
            try:
                getattr(pipe.vae, enable)()
            except (AttributeError, NotImplementedError):
                pass
        self.torch = torch
        self.pipe = pipe

    def prepare(self) -> None:
        self._load()

    @property
    def transformer(self) -> Any:
        self._load()
        return self.pipe.transformer

    @property
    def tokenizer(self) -> Any:
        self._load()
        return self.pipe.tokenizer

    @property
    def num_layers(self) -> int:
        self._load()
        return len(self.pipe.transformer.transformer_blocks)

    @property
    def device(self) -> Any:
        self._load()
        return self.pipe._execution_device

    @property
    def dtype(self) -> Any:
        self._load()
        return next(self.pipe.transformer.parameters()).dtype

    def latent_shape(self) -> tuple[int, int, int]:
        """``(latent_frames, latent_height, latent_width)`` for the configured video."""
        self._load()
        frames = int(self.config["frames"])
        height = int(self.config["height"])
        width = int(self.config["width"])
        return (
            (frames - 1) // self.pipe.vae_temporal_compression_ratio + 1,
            height // self.pipe.vae_spatial_compression_ratio,
            width // self.pipe.vae_spatial_compression_ratio,
        )

    def token_count(self, steps: int | None = None) -> int:
        frames, height, width = self.latent_shape()
        return frames * height * width

    # -------------------------------------------------------------- prompting

    def video_coords(self, seed: int = 0) -> Any:
        """Positional coordinates alone, for runs that resume a cached latent.

        ``prepare_latents`` produces the coordinates as a side effect of drawing
        noise; with a fixed seed the draw is deterministic and discarded, which
        is cheaper than reimplementing the coordinate construction here.
        """
        return self.make_latents(seed)[1]

    def encode(self, prompt: str, negative_prompt: str | None = None) -> dict[str, Any]:
        """Text conditioning, concatenated for CFG as the stock pipeline does."""
        self._load()
        torch = self.torch
        with torch.no_grad():
            embeds, mask, negative_embeds, negative_mask = self.pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=True,
                max_sequence_length=int(self.config.get("max_sequence_length", 256)),
                device=self.device,
                dtype=self.dtype,
            )
        return {
            "cond_embeds": embeds,
            "cond_mask": mask,
            "uncond_embeds": negative_embeds,
            "uncond_mask": negative_mask,
            # The reference builds the branch batch once, in the order
            # [negative, prompt, prompt], and slices it per step; keeping the
            # triple here is what makes the per-step slicing below faithful.
            "triple_embeds": torch.cat([negative_embeds, embeds, embeds]),
            "triple_mask": torch.cat([negative_mask, mask, mask]),
        }

    @staticmethod
    def branch_slice(do_cfg: bool, do_stg: bool) -> tuple[int, int]:
        """The ``[start, stop)`` slice of the triple batch for this step's branches.

        Straight from the reference: cfg+stg -> ``(0, 3)``, cfg -> ``(0, 2)``,
        stg -> ``(1, 3)``, neither -> ``(1, 2)``.  The perturbation branch is
        always last, which is what makes ``ptb_index = num_conds - 1`` valid.
        """
        if do_cfg and do_stg:
            return 0, 3
        if do_cfg:
            return 0, 2
        if do_stg:
            return 1, 3
        return 1, 2

    def make_latents(self, seed: int) -> tuple[Any, Any]:
        """Fresh noise latents plus the positional ``video_coords``.

        Drawn in float32 and kept there, because that is what
        ``LTXConditionPipeline.__call__`` passes to ``prepare_latents``: a bf16
        or fp16 ``randn_tensor`` consumes the generator differently and would
        not reproduce the stock pipeline's latents for the same seed.
        """
        self._load()
        torch = self.torch
        device = self.device
        generator = torch.Generator(device=device).manual_seed(int(seed))
        latents, _mask, video_coords, extra = self.pipe.prepare_latents(
            conditions=[],
            condition_strength=[],
            condition_frame_index=[],
            batch_size=1,
            num_channels_latents=int(self.pipe.transformer.config.in_channels),
            height=int(self.config["height"]),
            width=int(self.config["width"]),
            num_frames=int(self.config["frames"]),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        if extra:
            raise RuntimeError("LTX baseline sampling does not support conditioning frames")
        # Time coordinates are normalised by the frame rate before the
        # transformer's RoPE, exactly as the stock pipeline does.
        video_coords = video_coords.float()
        video_coords[:, 0] = video_coords[:, 0] * (1.0 / float(self.config.get("fps", DEFAULT_FPS)))
        return latents, video_coords

    # ---------------------------------------------------------------- sampler

    def timesteps(self, steps: int) -> Any:
        """The stock LTX timestep schedule (``linear_quadratic`` sigmas * 1000)."""
        self._load()
        from diffusers.pipelines.ltx.pipeline_ltx_condition import (
            linear_quadratic_schedule,
            retrieve_timesteps,
        )

        timesteps = linear_quadratic_schedule(steps) * 1000
        timesteps, _ = retrieve_timesteps(self.pipe.scheduler, steps, self.device, timesteps)
        return timesteps

    def set_timesteps(self, steps: int) -> Any:
        """Initialise the scheduler on the stock schedule and return it."""
        self._load()
        from diffusers.pipelines.ltx.pipeline_ltx_condition import (
            linear_quadratic_schedule,
            retrieve_timesteps,
        )

        timesteps = retrieve_timesteps(
            self.pipe.scheduler, steps, self.device, linear_quadratic_schedule(steps) * 1000
        )[0]
        return timesteps

    def _forward(self, hidden_states: Any, embeds: Any, timestep: Any, mask: Any, video_coords: Any) -> Any:
        """One CUDA-synchronised transformer forward.

        ``no_grad`` is not an optimisation here: the sampler feeds each step's
        output back in, so without it the autograd graph chains across the whole
        schedule and every step's activations are retained -- 50 steps of a 13B
        model over 6144 tokens exhausts an 80 GB card by step 2.  The Wan2.1
        baseline wraps its loop the same way.
        """
        torch = self.torch
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with torch.no_grad():
                return self.pipe.transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=embeds,
                    timestep=timestep,
                    encoder_attention_mask=mask,
                    video_coords=video_coords,
                    return_dict=False,
                )[0]
        finally:
            torch.cuda.synchronize()
            self._forward_seconds += time.perf_counter() - started

    def stg_skip_mask(self, skip_blocks: Sequence[int], num_conds: int) -> Any:
        """``[num_layers, num_conds]`` mask, 0 on the perturbation column.

        ``create_skip_layer_mask`` in the reference writes
        ``mask[block, ptb_index::num_conds] = 0`` with ``ptb_index = num_conds - 1``;
        with a single sample per branch that is exactly ``[block, -1] = 0``, but
        the stride is kept so the two agree if the batch ever grows.
        """
        torch = self.torch
        layers = len(self.pipe.transformer.transformer_blocks)
        mask = torch.ones(layers, num_conds, dtype=torch.float32, device=self.device)
        for block in skip_blocks:
            if not 0 <= int(block) < layers:
                raise ValueError(f"STG skip_block_list names block {block}, but LTX has {layers}")
            mask[int(block), num_conds - 1::num_conds] = 0.0
        return mask

    def denoise(
        self,
        *,
        latents: Any,
        conditioning: dict[str, Any],
        video_coords: Any,
        steps: int,
        start_step: int = 0,
        schedule_steps: int | None = None,
        begin_step_hook: Callable[[int], None] | None = None,
        on_step_end: Callable[[int, Any], Any] | None = None,
        capture_steps: Sequence[int] = (),
        capture_sink: dict[int, Any] | None = None,
    ) -> tuple[Any, dict[int, Any], float, list[dict[str, Any]]]:
        """Run the shared CFG(+STG) loop.

        One transformer forward per step, batched over the active branches
        ``[negative, prompt, prompt_perturbed]`` -- the reference's layout, not a
        second pass for the perturbed branch.

        ``begin_step_hook`` receives the 0-based index *before* the forward, so
        attention hooks can be armed for the step (they cannot count forwards:
        STG changes the branch count, not the forward count, but a resumed run
        starts mid-schedule).  ``on_step_end`` sees each completed step's latent
        and may replace it (Chorus/Carnator latent fusion).  ``capture_steps``
        snapshots completed steps into ``capture_sink``.

        ``steps`` is how many steps to execute; ``schedule_steps`` is how long
        the schedule those steps come from is.  They differ only for a *prefix*
        run: the Carnator target probe executes the first 5 steps of the
        50-step plan, so it needs the 50-step timesteps and the 50-step
        normalised time on step 0..4, not a re-scaled 5-step schedule.  Wan's
        target probe does the same thing by enumerating the full plan's
        timesteps and breaking out at ``mask_probe_step``.

        Returns ``(latents, captured, dit_forward_seconds, per_step_guidance)``.
        """
        self._load()
        torch = self.torch
        scheduler = self.pipe.scheduler
        total = int(steps) if schedule_steps is None else int(schedule_steps)
        if steps > total:
            raise ValueError(f"steps must not exceed schedule_steps; got steps={steps}, schedule={total}")
        timesteps = self.set_timesteps(total)
        if not 0 <= start_step < steps:
            raise ValueError(f"start_step must be in [0, {steps}), got {start_step}")

        captured: dict[int, Any] = capture_sink if capture_sink is not None else {}
        per_step: list[dict[str, Any]] = []
        self._forward_seconds = 0.0

        try:
            for index in range(start_step, steps):
                timestep_value = timesteps[index]
                normalized_time = 1.0 - float(index) / float(max(1, total - 1))
                guidance = self.stg.for_step(
                    normalized_time, float(self.config["guidance_scale"])
                )
                if begin_step_hook is not None:
                    begin_step_hook(index)

                do_cfg = guidance.guidance_scale > 1.0
                do_stg = guidance.stg_scale > 0.0 and bool(guidance.skip_blocks)
                start, stop = self.branch_slice(do_cfg, do_stg)
                num_conds = stop - start
                embeds = conditioning["triple_embeds"][start:stop]
                embed_mask = conditioning["triple_mask"][start:stop]

                # Labels come from the slice, not the count: see GuidanceState.
                self.guidance.set(
                    self.stg_skip_mask(guidance.skip_blocks, num_conds) if do_stg else None,
                    guidance.skip_blocks if do_stg else (),
                    GuidanceState.LABELS[start:stop],
                )
                try:
                    latent_model_input = torch.cat([latents] * num_conds).to(embeds.dtype)
                    noise_pred = self._forward(
                        latent_model_input,
                        embeds,
                        timestep_value.expand(num_conds).unsqueeze(-1).float(),
                        embed_mask,
                        torch.cat([video_coords] * num_conds, dim=0),
                    )
                finally:
                    self.guidance.reset()

                parts = noise_pred.chunk(num_conds)
                text_pred = parts[-2] if do_stg else parts[1] if do_cfg else parts[0]
                noise_pred = text_pred
                if do_cfg:
                    uncond_pred, text_pred = parts[0], parts[1]
                    if guidance.cfg_star_rescale:
                        # alpha = <eps_text, eps_uncond> / ||eps_uncond||^2
                        text_flat = text_pred.reshape(text_pred.size(0), -1)
                        uncond_flat = uncond_pred.reshape(uncond_pred.size(0), -1)
                        alpha = (text_flat * uncond_flat).sum(dim=1, keepdim=True) / (
                            (uncond_flat**2).sum(dim=1, keepdim=True) + 1e-8
                        )
                        uncond_pred = (alpha * uncond_flat).reshape(uncond_pred.shape)
                    noise_pred = uncond_pred + guidance.guidance_scale * (text_pred - uncond_pred)

                if do_stg:
                    perturbed = parts[-1]
                    noise_pred = noise_pred + guidance.stg_scale * (text_pred - perturbed)
                    if guidance.rescaling_scale != 1.0:
                        std_text = text_pred.reshape(text_pred.size(0), -1).std(dim=1, keepdim=True)
                        std_pred = noise_pred.reshape(noise_pred.size(0), -1).std(dim=1, keepdim=True)
                        factor = guidance.rescaling_scale * (std_text / std_pred) + (
                            1.0 - guidance.rescaling_scale
                        )
                        noise_pred = noise_pred * factor.reshape(-1, 1, 1)

                # The reference hands the scheduler a single shared timestep, not
                # the branch-expanded one; a [num_conds, 1] tensor here would
                # broadcast the sample back up to num_conds on the way out.
                latents = scheduler.step(
                    -noise_pred,
                    timestep_value,
                    latents,
                    per_token_timesteps=timestep_value.reshape(1, 1).float(),
                    return_dict=False,
                )[0]

                completed = index + 1
                if on_step_end is not None:
                    latents = on_step_end(completed, latents)
                if capture_steps and completed in capture_steps:
                    captured[completed] = latents.detach().to("cpu").clone()
                per_step.append({
                    "completed_step": completed,
                    "timestep": float(timestep_value),
                    "normalized_time": normalized_time,
                    "segment": guidance.segment,
                    "guidance_scale": guidance.guidance_scale,
                    "stg_scale": guidance.stg_scale,
                    "skip_blocks": list(guidance.skip_blocks),
                    # Recorded because the branch count varies across the
                    # schedule, so anything keyed by branch name needs to know
                    # which names a step actually produced.
                    "num_conds": num_conds,
                    "branches": list(GuidanceState.LABELS[start:stop]),
                })
        finally:
            self.guidance.reset()

        return latents, captured, self._forward_seconds, per_step

    def probe_forward(
        self,
        *,
        latents: Any,
        conditioning: dict[str, Any],
        video_coords: Any,
        timestep: float,
        context: Any,
    ) -> Any:
        """One conditioning-only forward, for the Carnator Early Signature probe.

        Mirrors ``run_wan21_carnator_source``'s probe call: the *prompt* branch
        alone, at the timestep the probe step used, on that step's latent.  No
        CFG and no STG, because the probe measures where the prompt's nouns land
        in the self-attention -- averaging in an uncond row or an STG-substituted
        row would move that measurement rather than the guidance.
        """
        self._load()
        torch = self.torch
        embeds = conditioning["cond_embeds"]
        state = self.guidance
        state.set(None, (), ("cond",))
        try:
            with context:
                return self._forward(
                    latents.to(embeds.dtype),
                    embeds,
                    torch.tensor([[float(timestep)]], device=self.device),
                    conditioning["cond_mask"],
                    video_coords,
                )
        finally:
            state.reset()

    # ------------------------------------------------- packed <-> grid layout

    def unpack(self, latents: Any) -> Any:
        """Packed ``[B, S, C]`` -> video-grid ``[B, C, F, H, W]``.

        The spatial baselines (boundary blend, Carnator's per-token masks) work
        on the grid layout, while everything the sampler touches is packed, so
        the conversions are named here rather than re-derived at each call site.
        """
        self._load()
        frames, height, width = self.latent_shape()
        return self.pipe._unpack_latents(
            latents,
            frames,
            height,
            width,
            self.pipe.transformer_spatial_patch_size,
            self.pipe.transformer_temporal_patch_size,
        )

    def pack(self, latents: Any) -> Any:
        """Video-grid ``[B, C, F, H, W]`` -> packed ``[B, S, C]``."""
        self._load()
        return self.pipe._pack_latents(
            latents,
            self.pipe.transformer_spatial_patch_size,
            self.pipe.transformer_temporal_patch_size,
        )

    # ----------------------------------------------------------------- decode

    def _decode_tensor(self, latents: Any, *, seed: int | None = None) -> Any:
        """Unpack, denormalise and VAE-decode a packed latent to ``[B, C, F, H, W]``."""
        self._load()
        torch = self.torch
        latent_frames, latent_height, latent_width = self.latent_shape()
        unpacked = self.pipe._unpack_latents(
            latents,
            latent_frames,
            latent_height,
            latent_width,
            self.pipe.transformer_spatial_patch_size,
            self.pipe.transformer_temporal_patch_size,
        )
        unpacked = self.pipe._denormalize_latents(
            unpacked, self.pipe.vae.latents_mean, self.pipe.vae.latents_std, self.pipe.vae.config.scaling_factor
        )
        unpacked = unpacked.to(self.dtype)
        timestep = None
        if self.pipe.vae.config.timestep_conditioning:
            decode_timestep = float(self.config.get("decode_timestep", 0.0))
            scale = float(self.config.get("decode_noise_scale", decode_timestep))
            timestep = torch.tensor([decode_timestep], device=self.device, dtype=unpacked.dtype)
            if scale > 0.0:
                generator = torch.Generator(device=self.device).manual_seed(int(seed or 0))
                noise = torch.randn(
                    unpacked.shape, generator=generator, device=self.device, dtype=unpacked.dtype
                )
                unpacked = (1.0 - scale) * unpacked + scale * noise
        with torch.inference_mode():
            return self.pipe.vae.decode(unpacked, timestep, return_dict=False)[0]

    def decode(self, latents: Any, *, seed: int | None = None) -> list[Any]:
        """Decode a packed latent into PIL frames."""
        video = self._decode_tensor(latents, seed=seed)
        return self.pipe.video_processor.postprocess_video(video, output_type="pil")[0]

    def decode_video(self, latents: Any, *, seed: int | None = None) -> Any:
        """Decode a packed latent to a ``[F, H, W, C]`` uint8 array."""
        video = self._decode_tensor(latents, seed=seed)
        return self.pipe.video_processor.postprocess_video(video, output_type="np")[0]

    def decode_tail(self, latents: Any, *, seed: int | None = None) -> Any:
        """Final decoded frame as a CPU tensor; the Early Signature tail.

        Mirrors the CogVideoX delegate, which also compares a raw decoded frame
        rather than a postprocessed image, so the two stay exchangeable.
        """
        video = self._decode_tensor(latents, seed=seed)
        return video[0, :, -1].detach().float().cpu()

    # ------------------------------------------------------------------- I/O

    def save_latent(self, latent: Any, path: Path) -> None:
        self._load()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(latent, path)

    def load_latent(self, path: Path) -> Any:
        self._load()
        try:
            return self.torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return self.torch.load(path, map_location="cpu")

    def save_video(self, video: Any, path: Path) -> None:
        from diffusers.utils import export_to_video

        path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(video, output_video_path=str(path), fps=int(self.config.get("fps", DEFAULT_FPS)))
