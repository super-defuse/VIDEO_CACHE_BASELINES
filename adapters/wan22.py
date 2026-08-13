from __future__ import annotations

from pathlib import Path
from typing import Any

from core.protocol import ModelAdapter
from core.types import Generation, Request
from ._chai_source import load_chai_source
from .teacache import TeaCacheController


class Wan22Adapter(ModelAdapter):
    model_name = "Wan2.2"

    def __init__(self, config):
        super().__init__(config)
        self._delegate = None

    def _load_delegate(self):
        if self._delegate is None:
            load_chai_source()
            from chai.adapters.wan_t2v import WanT2VAdapter
            self._delegate = WanT2VAdapter(self.config)
        return self._delegate

    def prepare(self):
        self._load_delegate().prepare()

    def reset_hooks(self):
        delegate = self._load_delegate()
        delegate.state.set_cached_tokens(None, False)

    def generate(self, request: Request, *, fast=False, cached_latent=None, capture=False, method="baseline"):
        if method == "teacache":
            return self._generate_teacache(request)
        if method == "nirvana_vid":
            return self._generate_nirvana(request, cached_latent=cached_latent, capture=capture)
        delegate = self._load_delegate()
        result = delegate.generate(request.prompt, request.seed, fast=fast, cached_latent=cached_latent, capture=capture)
        return Generation(result.video, result.latent, result.latent_step, result.extra.get("executed_steps"), extra=result.extra)

    def _generate_teacache(self, request: Request) -> Generation:
        delegate = self._load_delegate()
        delegate._load()
        config = self.config["teacache"]
        steps = int(self.config["baseline_steps"])
        controller = TeaCacheController(
            steps=steps,
            threshold=float(config["relative_l1_threshold"]),
            max_skip_steps=int(config["max_skip_steps"]),
        )
        restore = controller.patch(delegate.pipeline.model, branches=2)
        try:
            video = delegate.pipeline.t2v(
                input_prompt=request.prompt,
                size=(int(self.config["width"]), int(self.config["height"])),
                frame_num=int(self.config["frames"]),
                shift=float(self.config["shift"]),
                sample_solver=self.config["solver"],
                sampling_steps=steps,
                guide_scale=float(self.config["guidance_scale"]),
                seed=request.seed,
                offload_model=False,
            )
        finally:
            restore()
        return Generation(
            video=video,
            executed_steps=len(controller.executed_steps),
            extra={"executed_steps": len(controller.executed_steps), "skipped_steps": steps - len(controller.executed_steps)},
        )

    def _generate_nirvana(self, request: Request, *, cached_latent: Any, capture: bool) -> Generation:
        """Resume the native Wan schedule from a cached diffusion latent."""
        delegate = self._load_delegate()
        delegate._load()
        pipeline = delegate.pipeline
        assert pipeline is not None
        if self.config["solver"] != "unipc":
            raise ValueError("NIRVANA-VID Wan adapter currently requires solver=unipc")

        import math
        import random
        from contextlib import contextmanager

        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from wan.utils.utils import masks_like

        torch = delegate.torch
        steps = int(self.config["baseline_steps"])
        reuse_start = int(self.config["nirvana_vid"]["reuse_start_step"])
        if not 0 < reuse_start < steps:
            raise ValueError("nirvana_vid.reuse_start_step must be within the sampling schedule")
        start = reuse_start if cached_latent is not None else 0
        frame_num = int(self.config["frames"])
        size = (int(self.config["width"]), int(self.config["height"]))
        target_shape = (
            pipeline.vae.model.z_dim,
            (frame_num - 1) // pipeline.vae_stride[0] + 1,
            size[1] // pipeline.vae_stride[1],
            size[0] // pipeline.vae_stride[2],
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) /
            (pipeline.patch_size[1] * pipeline.patch_size[2]) * target_shape[1] / pipeline.sp_size
        ) * pipeline.sp_size
        seed = request.seed if request.seed >= 0 else random.randint(0, 2**63 - 1)
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        if not pipeline.t5_cpu:
            pipeline.text_encoder.model.to(pipeline.device)
            context = pipeline.text_encoder([request.prompt], pipeline.device)
            context_null = pipeline.text_encoder([pipeline.sample_neg_prompt], pipeline.device)
        else:
            context = [item.to(pipeline.device) for item in pipeline.text_encoder([request.prompt], torch.device("cpu"))]
            context_null = [item.to(pipeline.device) for item in pipeline.text_encoder([pipeline.sample_neg_prompt], torch.device("cpu"))]
        latents = [cached_latent.to(pipeline.device)] if cached_latent is not None else [
            torch.randn(*target_shape, dtype=torch.float32, device=pipeline.device, generator=generator)
        ]
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=pipeline.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        scheduler.set_timesteps(steps, device=pipeline.device, shift=float(self.config["shift"]))
        mask1, mask2 = masks_like(latents, zero=False)
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}
        captured = None

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(pipeline.model, "no_sync", noop_no_sync)
        with torch.amp.autocast("cuda", dtype=pipeline.param_dtype), torch.no_grad(), no_sync():
            for index, timestep_value in enumerate(scheduler.timesteps[start:], start=start):
                timestep = torch.stack([timestep_value])
                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                timestep = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep]).unsqueeze(0)
                cond = pipeline.model(latents, t=timestep, **arg_c)[0]
                uncond = pipeline.model(latents, t=timestep, **arg_null)[0]
                noise_pred = uncond + float(self.config["guidance_scale"]) * (cond - uncond)
                sample = scheduler.step(
                    noise_pred.unsqueeze(0), timestep_value, latents[0].unsqueeze(0), return_dict=False, generator=generator
                )[0]
                latents = [sample.squeeze(0)]
                if capture and index + 1 == reuse_start:
                    captured = latents[0].detach().cpu()
            video = pipeline.vae.decode(latents)[0]
        return Generation(
            video=video,
            latent=captured,
            latent_step=reuse_start,
            executed_steps=steps - start,
            extra={"executed_steps": steps - start, "restart_step": start, "source_trajectory": cached_latent is not None},
        )

    def save_latent(self, latent, path: Path):
        self._load_delegate().save_latent(latent, path)

    def load_latent(self, path: Path):
        return self._load_delegate().load_latent(path)

    def save_video(self, video, path: Path):
        self._load_delegate().save_video(video, path)
