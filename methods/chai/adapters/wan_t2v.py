from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .base import T2VAdapter
from ..attention.cache_attention import CacheAttentionState, cache_tokens_like
from core.external import external_path
from core.types import Generation


class WanT2VAdapter(T2VAdapter):
    """Wan2.2 TI2V-5B text-only path with latent capture and Cache Attention."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.pipeline: Any | None = None
        self.state = CacheAttentionState()
        self._patched: list[tuple[Any, Any]] = []
        self._captures: dict[int, Any] = {}
        self._capture_enabled = False
        self._capture_step: int | None = None
        self._active_cache_attention_steps: set[int] = set()

    def _source_cache_steps(self) -> tuple[int, ...]:
        return tuple(int(step) for step in self.config["source_cache_steps"])

    @staticmethod
    def _cached_tokens_for_step(cached_latent: Any, step: int) -> Any | None:
        if cached_latent is None:
            return None
        if not isinstance(cached_latent, dict):
            raise TypeError("CHAI cache payload must be a per-step token dictionary")
        return cached_latent.get(step, cached_latent.get(str(step)))

    def _load(self) -> None:
        if self.pipeline is not None:
            return
        upstream = Path(self.config["wan_source"]).resolve() if self.config.get("wan_source") else external_path("wan22")
        if str(upstream) not in sys.path:
            sys.path.insert(0, str(upstream))
        import torch
        import wan
        from wan.configs import WAN_CONFIGS

        self.torch = torch
        self.pipeline = wan.WanTI2V(
            config=WAN_CONFIGS["ti2v-5B"],
            checkpoint_dir=self.config["model_path"],
            device_id=int(str(self.config["device"]).split(":")[-1]),
            init_on_cpu=False,
            convert_model_dtype=True,
        )

    def prepare(self) -> None:
        self._load()

    def _install_attention_patches(self) -> None:
        assert self.pipeline is not None
        from wan.modules.attention import flash_attention
        from wan.modules.model import rope_apply

        blocks = set(self.config["cache_attention"]["blocks"])
        for index, block in enumerate(self.pipeline.model.blocks):
            if index not in blocks:
                continue
            attention = block.self_attn
            original = attention.forward

            def patched(x: Any, seq_lens: Any, grid_sizes: Any, freqs: Any, *, _attention: Any = attention, _original: Any = original) -> Any:
                if self._capture_enabled and self._capture_step is not None and self._capture_step not in self._captures:
                    self._captures[self._capture_step] = x.detach().cpu()
                if not self.state.active:
                    return _original(x, seq_lens, grid_sizes, freqs)
                cached = cache_tokens_like(x, self.state.cached_tokens)
                batch, tokens, heads, head_dim = x.shape[0], x.shape[1], _attention.num_heads, _attention.head_dim
                query = _attention.norm_q(_attention.q(x)).view(batch, tokens, heads, head_dim)
                key = _attention.norm_k(_attention.k(x)).view(batch, tokens, heads, head_dim)
                value = _attention.v(x).view(batch, tokens, heads, head_dim)
                cached_key = _attention.norm_k(_attention.k(cached)).view(batch, -1, heads, head_dim)
                cached_value = _attention.v(cached).view(batch, -1, heads, head_dim)
                query = rope_apply(query, grid_sizes, freqs)
                key = rope_apply(key, grid_sizes, freqs)
                cached_key = rope_apply(cached_key, grid_sizes, freqs)
                output = flash_attention(
                    query,
                    self.torch.cat((key, cached_key), dim=1),
                    self.torch.cat((value, cached_value), dim=1),
                )
                return _attention.o(output.flatten(2))

            attention.forward = patched
            self._patched.append((attention, original))

    def generate(
        self,
        prompt: str,
        seed: int,
        *,
        fast: bool = False,
        cached_latent: Any | None = None,
        capture: bool = False,
    ) -> Generation:
        self._load()
        assert self.pipeline is not None
        steps = self.config["fast_steps"] if fast else self.config["baseline_steps"]
        source_cache_steps = self._source_cache_steps()
        target_cache_steps = set(int(step) for step in self.config["cache_attention"]["steps"])
        if not fast and not capture:
            video = self.pipeline.t2v(
                input_prompt=prompt,
                size=(int(self.config["width"]), int(self.config["height"])),
                frame_num=int(self.config["frames"]),
                shift=float(self.config["shift"]),
                sample_solver=self.config["solver"],
                sampling_steps=int(steps),
                guide_scale=float(self.config["guidance_scale"]),
                seed=seed,
                offload_model=False,
            )
            return Generation(video=video, latent=None, latent_step=max(source_cache_steps), extra={"executed_steps": int(steps)})
        if not self._patched:
            self._install_attention_patches()
        self._captures.clear()
        self._capture_step = None
        self._active_cache_attention_steps.clear()
        self.state.set_cached_tokens(None, False)
        # The original Wan public API does not expose the latent denoising state.
        # Capture a calibrated transformer input; it is the cache source used by
        # Cache Attention on later requests.
        original_forward = self.pipeline.model.forward
        has_cached_tokens = cached_latent is not None and fast
        try:
            calls = 0

            def scheduled_forward(*args: Any, **kwargs: Any) -> Any:
                nonlocal calls
                # Wan evaluates conditional then unconditional predictions per
                # denoising step. Capture only the conditional pass.
                denoising_step = calls // 2
                capture_at_step = capture and not fast and denoising_step in source_cache_steps
                self._capture_enabled = capture_at_step and calls % 2 == 0
                self._capture_step = denoising_step if self._capture_enabled else None
                read_at_step = (
                    self.chai_manager.should_cache_read_at_t(denoising_step)
                    if self.chai_manager is not None
                    else denoising_step in target_cache_steps
                )
                selected_cached = self._cached_tokens_for_step(cached_latent, denoising_step) if has_cached_tokens else None
                if has_cached_tokens and read_at_step and selected_cached is None:
                    raise KeyError(f"CHAI cache payload has no tokens for target step {denoising_step}")
                self.state.set_cached_tokens(selected_cached, has_cached_tokens and read_at_step)
                if self.state.active:
                    self._active_cache_attention_steps.add(denoising_step)
                try:
                    return original_forward(*args, **kwargs)
                finally:
                    self._capture_enabled = False
                    self._capture_step = None
                    calls += 1

            self.pipeline.model.forward = scheduled_forward
            video = self.pipeline.t2v(
                input_prompt=prompt,
                size=(int(self.config["width"]), int(self.config["height"])),
                frame_num=int(self.config["frames"]),
                shift=float(self.config["shift"]),
                sample_solver=self.config["solver"],
                sampling_steps=int(steps),
                guide_scale=float(self.config["guidance_scale"]),
                seed=seed,
                offload_model=False,
            )
        finally:
            self.pipeline.model.forward = original_forward
        self.state.set_cached_tokens(None, False)
        if capture and set(self._captures) != set(source_cache_steps):
            missing = sorted(set(source_cache_steps) - set(self._captures))
            raise RuntimeError(f"CHAI source run did not capture block-0 tokens at steps {missing}")
        latent = dict(self._captures) if capture else None
        return Generation(
            video=video,
            latent=latent,
            latent_step=max(source_cache_steps),
            extra={
                "executed_steps": int(steps),
                "cache_attention_steps": sorted(self._active_cache_attention_steps),
            },
        )

    def save_latent(self, latent: Any, path: Path) -> None:
        self.torch.save(latent, path)

    def load_latent(self, path: Path) -> Any:
        self._load()
        try:
            return self.torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return self.torch.load(path, map_location="cpu")

    def save_video(self, video: Any, path: Path) -> None:
        self._load()
        from wan.utils.utils import save_video

        # Wan returns [C, T, H, W]; its writer expects a batch dimension.
        save_video(video[None], str(path), fps=24, nrow=1, normalize=True, value_range=(-1, 1))
