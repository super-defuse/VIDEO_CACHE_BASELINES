from __future__ import annotations

from pathlib import Path
from typing import Any

from core.types import Generation


class CogVideoPipelineDelegate:
    """Minimal Diffusers CogVideoX1.5 model boundary."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.pipe: Any | None = None

    def _load(self) -> None:
        if self.pipe is not None:
            return
        import torch
        from diffusers import CogVideoXPipeline

        if str(self.config["device"]).startswith("cuda"):
            torch.cuda.set_device(self.config["device"])
        self.torch = torch
        dtype = torch.bfloat16 if self.config["dtype"] == "bfloat16" else torch.float16
        self.pipe = CogVideoXPipeline.from_pretrained(self.config["model_path"], torch_dtype=dtype)
        self.pipe.to(self.config["device"])
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

    def prepare(self) -> None:
        self._load()

    def generate(self, prompt: str, seed: int) -> Generation:
        self._load()
        assert self.pipe is not None
        steps = int(self.config["baseline_steps"])
        output = self.pipe(
            prompt=prompt,
            height=int(self.config["height"]),
            width=int(self.config["width"]),
            num_frames=int(self.config["frames"]),
            num_inference_steps=steps,
            guidance_scale=float(self.config["guidance_scale"]),
            use_dynamic_cfg=True,
            generator=self.torch.Generator(device=self.config["device"]).manual_seed(seed),
        )
        return Generation(video=output.frames[0], executed_steps=steps, extra={"executed_steps": steps})

    def save_latent(self, latent: Any, path: Path) -> None:
        self._load()
        self.torch.save(latent, path)

    def load_latent(self, path: Path) -> Any:
        self._load()
        try:
            return self.torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return self.torch.load(path, map_location="cpu")

    def save_video(self, video: Any, path: Path) -> None:
        from diffusers.utils import export_to_video

        export_to_video(video, output_video_path=str(path), fps=int(self.config.get("fps", 16)))

    def decode_video(self, latents: Any) -> Any:
        self._load()
        assert self.pipe is not None
        latent_frames = (int(self.config["frames"]) - 1) // self.pipe.vae_scale_factor_temporal + 1
        value = latents[:, -latent_frames:].to(self.config["device"])
        with self.torch.inference_mode():
            frames = self.pipe.decode_latents(value)
            return self.pipe.video_processor.postprocess_video(video=frames, output_type="np")[0]

    def decode_tail(self, latents: Any) -> Any:
        self._load()
        assert self.pipe is not None
        latent_frames = (int(self.config["frames"]) - 1) // self.pipe.vae_scale_factor_temporal + 1
        value = latents[:, -latent_frames:].to(self.config["device"])
        with self.torch.inference_mode():
            frames = self.pipe.decode_latents(value)
            return frames[0, :, -1].cpu()
