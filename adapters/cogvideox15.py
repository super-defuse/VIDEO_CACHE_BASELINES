from __future__ import annotations

from pathlib import Path

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
            from methods.chai.adapters.cogvideo_t2v import CogVideoT2VAdapter
            self._delegate = CogVideoT2VAdapter(self.config)
        return self._delegate

    def prepare(self):
        self._load_delegate().prepare()

    def reset_hooks(self):
        self._load_delegate().state.set_cached_tokens(None, False)

    def generate(self, request: Request, *, fast=False, cached_latent=None, capture=False, method="baseline"):
        if method == "teacache":
            return self._generate_teacache(request)
        if method == "nirvana_vid":
            return self._generate_nirvana(request, cached_latent=cached_latent, capture=capture)
        if method == "carnator":
            return self._generate_carnator(request, cached_latent=cached_latent, capture=capture)
        delegate = self._load_delegate()
        result = delegate.generate(request.prompt, request.seed, fast=fast, cached_latent=cached_latent, capture=capture)
        return Generation(result.video, result.latent, result.latent_step, result.extra.get("executed_steps"), extra=result.extra)

    def _generate_carnator(self, request: Request, *, cached_latent, capture: bool) -> Generation:
        """CogVideoX source-trajectory reuse used by the Carnator adapter.

        CogVideoX1.5 has no upstream Carnator C0/B0--B5 implementation.  The
        compatible part is the source latent hand-off: capture the source
        trajectory at the configured reuse step, then restart the target
        schedule from that latent.  The result records this explicitly.
        """
        delegate = self._load_delegate()
        delegate._load()
        steps = int(self.config["baseline_steps"])
        reuse_start = int(self.config["nirvana_vid"]["reuse_start_step"])
        if not 0 < reuse_start < steps:
            raise ValueError("Carnator reuse_start_step must be within the sampling schedule")
        start = reuse_start if cached_latent is not None else 0
        captured = []

        def capture_latent(_pipe, index, _timestep, values):
            if capture and index + 1 == reuse_start:
                captured.append(values["latents"].detach().cpu())
            return values

        call = self._call_args(request, steps - start)
        if cached_latent is not None:
            call["latents"] = cached_latent.to(delegate.pipe.device)
        if capture:
            call["callback_on_step_end"] = capture_latent
            call["callback_on_step_end_tensor_inputs"] = ["latents"]
        output = delegate.pipe(**call)
        return Generation(
            video=output.frames[0],
            latent=captured[-1] if captured else None,
            latent_step=reuse_start,
            executed_steps=steps - start,
            extra={
                "executed_steps": steps - start,
                "restart_step": start,
                "source_trajectory": cached_latent is not None,
                "carnator_variant": "cogvideox15_source_latent_restart",
            },
        )

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
        return Generation(
            video=output.frames[0],
            executed_steps=len(controller.executed_steps),
            extra={"executed_steps": len(controller.executed_steps), "skipped_steps": steps - len(controller.executed_steps)},
        )

    def _generate_nirvana(self, request: Request, *, cached_latent, capture: bool) -> Generation:
        delegate = self._load_delegate()
        delegate._load()
        steps = int(self.config["baseline_steps"])
        reuse_start = int(self.config["nirvana_vid"]["reuse_start_step"])
        if not 0 < reuse_start < steps:
            raise ValueError("nirvana_vid.reuse_start_step must be within the sampling schedule")
        start = reuse_start if cached_latent is not None else 0
        captured = []

        def capture_latent(_pipe, index, _timestep, values):
            if capture and index + 1 == reuse_start:
                captured.append(values["latents"].detach().cpu())
            return values

        call = self._call_args(request, steps - start)
        if cached_latent is not None:
            # Diffusers starts a fresh, shorter scheduler at this retained
            # diffusion state; the restart is recorded in the result manifest.
            call["latents"] = cached_latent
        if capture:
            call["callback_on_step_end"] = capture_latent
            call["callback_on_step_end_tensor_inputs"] = ["latents"]
        output = delegate.pipe(**call)
        return Generation(
            video=output.frames[0],
            latent=captured[-1] if captured else None,
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
