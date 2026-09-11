from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from core.protocol import ModelAdapter
from core.types import Generation, Request
from .teacache import TeaCacheController


class CogVideoX15Adapter(ModelAdapter):
    model_name = "CogVideoX1.5"

    def __init__(self, config):
        super().__init__(config)
        self._delegate = None

    def _load_delegate(self):
        if self._delegate is None:
            from .cogvideo_pipeline import CogVideoPipelineDelegate
            self._delegate = CogVideoPipelineDelegate(self.config)
        return self._delegate

    def prepare(self):
        self._load_delegate().prepare()

    def reset_hooks(self):
        pass

    def generate(self, request: Request, *, fast=False, cached_latent=None, capture=False, method="baseline"):
        if method == "teacache":
            return self._generate_teacache(request)
        if method == "nirvana_vid":
            return self._generate_nirvana(request, cached_latent=cached_latent, capture=capture)
        if method == "chorus":
            from methods.chorus.cogvideo import generate

            return generate(self, request, cached_latent=cached_latent, capture=capture)
        if method == "chorus2":
            from methods.chorus2.cogvideo import generate

            return generate(self, request, cached_latent=cached_latent, capture=capture)
        if method in {"carnator", "carnator_boundary_b5"}:
            from carnator.cogvideox15 import generate

            return generate(self, request, cached_latent=cached_latent, capture=capture)
        delegate = self._load_delegate()
        delegate._load()
        with self._profile_transformer(delegate.pipe if delegate.pipe is not None else None) as profile:
            result = delegate.generate(request.prompt, request.seed)
        result.extra.setdefault("dit_forward_s", profile[0])
        return Generation(result.video, result.latent, result.latent_step, result.extra.get("executed_steps"), extra=result.extra)

    def _call_args(self, request: Request, steps: int) -> dict:
        delegate = self._load_delegate()
        return {
            "prompt": request.prompt,
            "height": int(self.config["height"]),
            "width": int(self.config["width"]),
            "num_frames": int(self.config["frames"]),
            "num_inference_steps": steps,
            "guidance_scale": float(self.config["guidance_scale"]),
            "use_dynamic_cfg": True,
            "generator": delegate.torch.Generator(device=self.config["device"]).manual_seed(request.seed),
        }

    @contextlib.contextmanager
    def _profile_transformer(self, pipe: Any | None):
        elapsed = [0.0]
        if pipe is None:
            yield elapsed
            return
        import time

        transformer = pipe.transformer
        original = transformer.forward

        def forward(*args: Any, **kwargs: Any) -> Any:
            value = args[0] if args else kwargs.get("hidden_states")
            TeaCacheController._synchronize_cuda(value)
            started = time.perf_counter()
            output = original(*args, **kwargs)
            TeaCacheController._synchronize_cuda(value)
            elapsed[0] += time.perf_counter() - started
            return output

        transformer.forward = forward
        try:
            yield elapsed
        finally:
            transformer.forward = original

    @contextlib.contextmanager
    def _resume_original_schedule(self, scheduler: Any, *, steps: int, reuse_step: int, expected: list[int]):
        if scheduler.__class__.__name__ != "CogVideoXDDIMScheduler":
            raise RuntimeError(
                f"CogVideo exact resume requires CogVideoXDDIMScheduler, got {scheduler.__class__.__name__}"
            )
        original = scheduler.set_timesteps

        def set_timesteps(num_inference_steps: int, device: Any = None) -> None:
            if int(num_inference_steps) != steps:
                raise ValueError("CogVideo target must initialize the original full scheduler")
            original(steps, device=device)
            actual = [int(item) for item in scheduler.timesteps.detach().cpu().tolist()]
            if actual != expected:
                raise ValueError("CogVideo cached timestep sequence does not match the current scheduler")
            scheduler.timesteps = scheduler.timesteps[reuse_step:]

        scheduler.set_timesteps = set_timesteps
        try:
            yield
        finally:
            scheduler.set_timesteps = original

    @staticmethod
    def _scheduler_timesteps(scheduler: Any, steps: int, device: Any) -> list[int]:
        scheduler.set_timesteps(steps, device=device)
        return [int(item) for item in scheduler.timesteps.detach().cpu().tolist()]

    def _generate_teacache(self, request: Request) -> Generation:
        delegate = self._load_delegate()
        delegate._load()
        steps = int(self.config["baseline_steps"])
        config = self.config["teacache"]
        controller = TeaCacheController(
            steps=steps,
            threshold=float(config["relative_l1_threshold"]),
            max_skip_steps=int(config["max_skip_steps"]),
        )
        restore = controller.patch(delegate.pipe.transformer, branches=1)
        try:
            output = delegate.pipe(**self._call_args(request, steps))
        finally:
            restore()
        if controller.forward_calls != steps:
            raise RuntimeError(f"TeaCache expected {steps} transformer calls, observed {controller.forward_calls}")
        return Generation(
            video=output.frames[0],
            executed_steps=len(controller.executed_steps),
            extra={
                "executed_steps": len(controller.executed_steps),
                "skipped_steps": steps - len(controller.executed_steps),
                "dit_forward_s": controller.forward_seconds,
                "teacache": {
                    "scheduler_steps": steps,
                    "transformer_calls": controller.forward_calls,
                    "cfg_layout": "single_forward_cond_uncond_batch",
                    "executed_step_indices": sorted(controller.executed_steps),
                },
            },
        )

    def _generate_nirvana(self, request: Request, *, cached_latent, capture: bool) -> Generation:
        delegate = self._load_delegate()
        delegate._load()
        steps = int(self.config["baseline_steps"])
        reuse_start = int(getattr(self, "nirvana_reuse_start_step", self.config["nirvana_vid"]["reuse_start_step"]))
        if not 0 < reuse_start < steps:
            raise ValueError("nirvana_vid.reuse_start_step must be within the sampling schedule")
        cache_steps = tuple(sorted({int(step) for step in self.config["nirvana_vid"].get("cache_steps", [reuse_start])}))
        if not cache_steps or any(not 0 < step < steps for step in cache_steps):
            raise ValueError("nirvana_vid.cache_steps must be non-empty steps within the sampling schedule")
        captured: dict[int, Any] = {}

        scheduler = delegate.pipe.scheduler
        full_timesteps = self._scheduler_timesteps(scheduler, steps, delegate.pipe._execution_device)

        def capture_latent(_pipe, index, _timestep, values):
            completed_step = index + 1
            if capture and completed_step in cache_steps:
                captured[completed_step] = {
                    "latents": values["latents"].detach().cpu(),
                    "scheduler": scheduler.__class__.__name__,
                    "timesteps": full_timesteps,
                    "completed_step": completed_step,
                }
            if capture and completed_step == max(cache_steps):
                _pipe._interrupt = True
            return values

        call = self._call_args(request, steps)
        if cached_latent is not None:
            if not isinstance(cached_latent, dict):
                raise TypeError("NIRVANA CogVideo cache must include latent and scheduler state")
            if int(cached_latent.get("completed_step", -1)) != reuse_start:
                raise ValueError("NIRVANA cache completed_step does not match reuse_start_step")
            if cached_latent.get("scheduler") != scheduler.__class__.__name__:
                raise ValueError("NIRVANA cache scheduler does not match the current scheduler")
            cached_timesteps = [int(item) for item in cached_latent.get("timesteps", [])]
            if len(cached_timesteps) != steps:
                raise ValueError("NIRVANA cache must contain the original full timestep sequence")
            call["latents"] = cached_latent["latents"]
        if capture:
            call["callback_on_step_end"] = capture_latent
            call["callback_on_step_end_tensor_inputs"] = ["latents"]
            call["output_type"] = "latent"
        schedule = (
            self._resume_original_schedule(
                scheduler,
                steps=steps,
                reuse_step=reuse_start,
                expected=cached_timesteps,
            )
            if cached_latent is not None
            else contextlib.nullcontext()
        )
        with schedule, self._profile_transformer(delegate.pipe) as profile:
            output = delegate.pipe(**call)
        executed_steps = steps - reuse_start if cached_latent is not None else (max(cache_steps) if capture else steps)
        return Generation(
            video=output.frames[0],
            latent=captured if capture else None,
            latent_step=max(cache_steps),
            executed_steps=executed_steps,
            extra={
                "executed_steps": executed_steps,
                "restart_step": reuse_start if cached_latent is not None else 0,
                "source_trajectory": cached_latent is not None,
                "cache_steps": list(cache_steps),
                "scheduler": scheduler.__class__.__name__,
                "scheduler_steps": steps,
                "scheduler_timesteps": full_timesteps,
                "schedule_resume": "original_timestep_suffix" if cached_latent is not None else None,
                "dit_forward_s": profile[0],
            },
        )

    def save_latent(self, latent, path: Path):
        self._load_delegate().save_latent(latent, path)

    def load_latent(self, path: Path):
        return self._load_delegate().load_latent(path)

    def save_video(self, video, path: Path):
        self._load_delegate().save_video(video, path)
