from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from .base import T2VAdapter
from ..attention.cache_attention import CacheAttentionState, cache_tokens_like
from core.types import Generation


class CogVideoT2VAdapter(T2VAdapter):
    """CogVideoX1.5 T2V adapter using Diffusers attention processors."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.pipe: Any | None = None
        self.state = CacheAttentionState()
        self._step = -1
        self._capture_enabled = False
        self._capture_requested = False
        self._captures: dict[int, Any] = {}
        self._active_cache_attention_steps: set[int] = set()
        self._capture_step: int | None = None

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
        if self.pipe is not None:
            return
        import torch
        from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline

        self.torch = torch
        dtype = torch.bfloat16 if self.config["dtype"] == "bfloat16" else torch.float16
        self.pipe = CogVideoXPipeline.from_pretrained(self.config["model_path"], torch_dtype=dtype)
        self.pipe.scheduler = CogVideoXDPMScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipe.to(self.config["device"])
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

    def prepare(self) -> None:
        self._load()

    def _step_hook(self, _module: Any, _inputs: tuple[Any, ...]) -> None:
        self._step += 1
        active = (
            self.chai_manager.should_cache_read_at_t(self._step)
            if self.chai_manager is not None
            else self._step in set(self.config["cache_attention"]["steps"])
        )
        selected_cached = self._cached_tokens_for_step(self._cached_latent, self._step)
        if self._fast and active and selected_cached is None:
            raise KeyError(f"CHAI cache payload has no tokens for target step {self._step}")
        self.state.set_cached_tokens(selected_cached, self._fast and active)
        if self.state.active:
            self._active_cache_attention_steps.add(self._step)
        capture_at_step = self._capture_enabled and not self._fast and self._step in self._source_cache_steps()
        self._capture_requested = self._capture_enabled and capture_at_step
        self._capture_step = self._step if self._capture_requested else None

    def _capture_hook(self, _module: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden_states = inputs[0] if inputs else kwargs.get("hidden_states")
        if self._capture_requested and self._capture_step is not None and self._capture_step not in self._captures and hidden_states is not None:
            self._captures[self._capture_step] = hidden_states.detach().cpu()
        self._capture_requested = False

    @contextlib.contextmanager
    def _run_state(self, fast: bool, cached_latent: Any | None, capture: bool):
        assert self.pipe is not None
        transformer = self.pipe.transformer
        self._step = -1
        self._captures.clear()
        self._active_cache_attention_steps.clear()
        self._capture_enabled = capture
        self._capture_step = None
        self._capture_requested = False
        self._cached_latent = cached_latent
        self._fast = fast
        self.state.set_cached_tokens(None, False)
        step_handle = transformer.register_forward_pre_hook(self._step_hook)
        blocks = [transformer.transformer_blocks[index] for index in self.config["cache_attention"]["blocks"]]
        capture_handles = (
            [block.attn1.register_forward_pre_hook(self._capture_hook, with_kwargs=True) for block in blocks]
            if capture
            else []
        )
        originals: dict[int, Any] = {}
        if fast and cached_latent is not None:
            processor = self._cache_processor()
            for block in blocks:
                originals[id(block)] = block.attn1.get_processor()
                block.attn1.set_processor(processor)
        try:
            yield
        finally:
            for block in blocks:
                if id(block) in originals:
                    block.attn1.set_processor(originals[id(block)])
            for handle in capture_handles:
                handle.remove()
            step_handle.remove()
            self._capture_enabled = False
            self._capture_step = None
            self._cached_latent = None
            self._fast = False
            self.state.set_cached_tokens(None, False)

    def _cache_processor(self) -> Any:
        """Build a processor compatible with the pinned Diffusers CogVideoX API."""
        import torch.nn.functional as functional
        from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
        from diffusers.models.embeddings import apply_rotary_emb

        state = self.state

        class CacheProcessor(CogVideoXAttnProcessor2_0):
            def __call__(
                self,
                attn: Any,
                hidden_states: Any,
                encoder_hidden_states: Any,
                attention_mask: Any = None,
                image_rotary_emb: Any = None,
            ) -> tuple[Any, Any]:
                image_length = hidden_states.size(1)
                text_length = encoder_hidden_states.size(1)
                combined = self._combine(hidden_states, encoder_hidden_states)
                batch_size = combined.size(0)
                query = attn.to_q(combined)
                key = attn.to_k(combined)
                value = attn.to_v(combined)
                head_dim = key.shape[-1] // attn.heads
                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                if attn.norm_q is not None:
                    query = attn.norm_q(query)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)
                if image_rotary_emb is not None:
                    query[:, :, text_length:] = apply_rotary_emb(query[:, :, text_length:], image_rotary_emb)
                    if not attn.is_cross_attention:
                        key[:, :, text_length:] = apply_rotary_emb(key[:, :, text_length:], image_rotary_emb)
                if state.active:
                    cached = cache_tokens_like(hidden_states, state.cached_tokens)
                    cached_key = attn.to_k(cached).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    cached_value = attn.to_v(cached).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    if attn.norm_k is not None:
                        cached_key = attn.norm_k(cached_key)
                    if image_rotary_emb is not None:
                        cached_key = apply_rotary_emb(cached_key, image_rotary_emb)
                    key = self._append(key, cached_key)
                    value = self._append(value, cached_value)
                output = functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                output = output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                output = attn.to_out[1](attn.to_out[0](output))
                encoder_output, image_output = output.split([text_length, image_length], dim=1)
                return image_output, encoder_output

            @staticmethod
            def _combine(hidden_states: Any, encoder_hidden_states: Any) -> Any:
                import torch

                return torch.cat([encoder_hidden_states, hidden_states], dim=1)

            @staticmethod
            def _append(left: Any, right: Any) -> Any:
                import torch

                return torch.cat([left, right], dim=-2)

        return CacheProcessor()

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
        assert self.pipe is not None
        steps = int(self.config["fast_steps"] if fast else self.config["baseline_steps"])
        generator = self.torch.Generator(device=self.config["device"]).manual_seed(seed)
        call = dict(
            prompt=prompt,
            height=int(self.config["height"]),
            width=int(self.config["width"]),
            num_frames=int(self.config["frames"]),
            num_inference_steps=steps,
            guidance_scale=float(self.config["guidance_scale"]),
            use_dynamic_cfg=True,
            generator=generator,
        )
        if not fast and not capture:
            output = self.pipe(**call)
        else:
            with self._run_state(fast, cached_latent, capture):
                output = self.pipe(**call)
        source_cache_steps = self._source_cache_steps()
        if capture and set(self._captures) != set(source_cache_steps):
            missing = sorted(set(source_cache_steps) - set(self._captures))
            raise RuntimeError(f"CHAI source run did not capture attention tokens at steps {missing}")
        latent = dict(self._captures) if capture else None
        return Generation(
            video=output.frames[0],
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
        from diffusers.utils import export_to_video

        export_to_video(video, output_video_path=str(path), fps=16)
