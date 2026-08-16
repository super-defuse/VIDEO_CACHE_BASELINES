"""CARNATOR adapter for LightX2V's Wan2.2-14B distilled T2V runtime.

The adapter intentionally owns only the hooks around LightX2V.  Checkpoint
loading, prompt encoding, sampling and video decoding remain upstream code.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import types
import uuid
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

from carnator.masks import (
    FusionMaskConfig,
    build_self_contrast_fusion_masks,
    load_mask_bundle,
    minmax_norm,
    robust_norm,
    save_mask_bundle,
    seed_indices_from_cross_score,
    token_mask_to_latent_mask,
)
from carnator.outer_sparse import BlockMaskCacheError, BlockMaskConfig, OuterSparseExecutor, SourceBlockMaskCache
from carnator.quantized_ffn import _quantize_activation_per_token, _quantize_weight, _w8a8_linear, w8a8_kernel_available
from carnator.runtime import dependency_versions, read_json, utc_now, write_json


ROOT = Path(__file__).resolve().parents[2]
LIGHTX2V_ROOT = ROOT / "external" / "lightx2v"
STATIC_CONFIG = ROOT / "configs" / "lightx2v_wan22_t2v_distill.json"
STEPS = 4
TOKEN_SHAPE = (21, 30, 52)  # 81 x 480 x 832 after Wan VAE and (1, 2, 2) patching.


def parse_layers(value: str) -> list[int]:
    values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lower, upper = (int(part) for part in item.split("-", 1))
            if upper < lower:
                raise ValueError(f"invalid layer range: {item}")
            values.extend(range(lower, upper + 1))
        else:
            values.append(int(item))
    if not values or any(value < 0 or value >= 40 for value in values):
        raise ValueError("Wan2.2-14B layer indices must be in [0, 39]")
    return sorted(set(values))


def phase_for_step(step: int) -> str:
    if step not in range(1, STEPS + 1):
        raise ValueError(f"step must be in [1, {STEPS}], got {step}")
    return "high_noise" if step <= 2 else "low_noise"


@dataclass(frozen=True)
class T2VRequest:
    prompt: str
    seed: int
    output: Path | None = None
    negative_prompt: str = ""


class LightX2V14BBackend:
    """Fixed four-step, no-CFG T2V backend with per-step callbacks."""

    def __init__(self, checkpoint: Path, *, dtype: str = "bf16", cpu_offload: bool = False) -> None:
        self.checkpoint = checkpoint.expanduser().resolve()
        self.dtype_name = dtype
        self.dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        self.cpu_offload = bool(cpu_offload)
        self.pipe: Any | None = None
        self.runtime_config: Path | None = None

    @property
    def device(self) -> torch.device:
        return torch.device("cuda:0")

    def preflight(self) -> dict[str, Any]:
        required = (
            LIGHTX2V_ROOT / "lightx2v",
            STATIC_CONFIG,
            self.checkpoint / "distill_models" / "high_noise_model" / "distill_model.safetensors",
            self.checkpoint / "distill_models" / "low_noise_model" / "distill_model.safetensors",
            self.checkpoint / "Wan2.1_VAE.pth",
            self.checkpoint / "models_t5_umt5-xxl-enc-bf16.pth",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("14B T2V preflight failed; missing: " + ", ".join(missing))
        return {
            "model": "Wan2.2-14B-T2V-distilled",
            "checkpoint": str(self.checkpoint),
            "runtime": dependency_versions(),
            "fixed_defaults": {"steps": STEPS, "enable_cfg": False, "token_shape": list(TOKEN_SHAPE)},
        }

    def load(self, work_dir: Path) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Wan2.2-14B-T2V-distilled requires CUDA")
        runtime = self.preflight()
        if str(LIGHTX2V_ROOT) not in sys.path:
            sys.path.insert(0, str(LIGHTX2V_ROOT))
        os.environ["DTYPE"] = "BF16" if self.dtype is torch.bfloat16 else "FP16"
        try:
            import flash_attn.cute  # type: ignore # LightX2V imports this optional namespace.
        except (AttributeError, ImportError):
            stub = types.ModuleType("flash_attn.cute")
            stub.flash_attn_func = None
            sys.modules["flash_attn.cute"] = stub
        from lightx2v import LightX2VPipeline

        config = read_json(STATIC_CONFIG)
        config.update(
            {
                "high_noise_original_ckpt": str(
                    self.checkpoint / "distill_models" / "high_noise_model" / "distill_model.safetensors"
                ),
                "low_noise_original_ckpt": str(
                    self.checkpoint / "distill_models" / "low_noise_model" / "distill_model.safetensors"
                ),
                "cpu_offload": self.cpu_offload,
            }
        )
        self.runtime_config = work_dir / "lightx2v_runtime.json"
        write_json(self.runtime_config, config)
        self.pipe = LightX2VPipeline(
            model_path=str(self.checkpoint), model_cls="wan2.2_moe_distill", task="t2v"
        )
        self.pipe.create_generator(config_json=str(self.runtime_config))
        actual = self.pipe.runner.config
        expected = {"infer_steps": STEPS, "enable_cfg": False, "target_video_length": 81, "target_height": 480, "target_width": 832}
        mismatch = {key: (actual.get(key), value) for key, value in expected.items() if actual.get(key) != value}
        if mismatch:
            raise RuntimeError(f"LightX2V did not retain the fixed 14B T2V profile: {mismatch}")
        runtime["runtime_config"] = str(self.runtime_config)

    def _runner(self) -> Any:
        if self.pipe is None:
            raise RuntimeError("14B backend has not been loaded")
        return self.pipe.runner

    def models(self) -> dict[str, Any]:
        models = getattr(self._runner().model, "model", None)
        if not isinstance(models, list) or len(models) != 2 or any(model is None for model in models):
            raise RuntimeError("14B CARNATOR hooks require both LightX2V denoisers to be resident; disable CPU offload")
        return {"high_noise": models[0], "low_noise": models[1]}

    @contextmanager
    def _step_hook(
        self,
        before_step: Callable[[str, int, Any], None] | None,
        after_step: Callable[[str, int, torch.Tensor, Any], None] | None,
    ) -> Iterator[None]:
        runner = self._runner()
        original = runner.run_segment

        def run_segment(owner: Any, segment_idx: int = 0) -> torch.Tensor:
            scheduler = owner.model.scheduler
            for step_index in range(int(scheduler.infer_steps)):
                step = step_index + 1
                if owner.video_segment_num == 1:
                    owner.check_stop()
                if before_step is not None:
                    before_step(phase_for_step(step), step, scheduler)
                scheduler.step_pre(step_index=step_index)
                owner.model.infer(owner.inputs)
                scheduler.step_post()
                if after_step is not None:
                    after_step(phase_for_step(step), step, scheduler.latents, scheduler)
                if owner.progress_callback:
                    owner.progress_callback((step / scheduler.infer_steps) * 100, 100)
            if segment_idx == owner.video_segment_num - 1:
                del owner.inputs
            return scheduler.latents

        runner.run_segment = types.MethodType(run_segment, runner)
        try:
            yield
        finally:
            runner.run_segment = original

    def run(
        self,
        request: T2VRequest,
        *,
        before_step: Callable[[str, int, Any], None] | None = None,
        after_step: Callable[[str, int, torch.Tensor, Any], None] | None = None,
    ) -> Any:
        with self._step_hook(before_step, after_step):
            return self.pipe.generate(
                seed=int(request.seed),
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                save_result_path=str(request.output) if request.output else None,
                return_result_tensor=request.output is None,
            )

    def unload(self) -> None:
        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _modules(model: Any) -> Iterator[tuple[int, Any]]:
    weights = model.transformer_weights
    phases = weights.iter_self_attention_phases() if hasattr(weights, "iter_self_attention_phases") else (block.compute_phases[0] for block in weights.blocks)
    seen: set[int] = set()
    for phase in phases:
        attention = phase.self_attn_1
        if id(attention) not in seen:
            seen.add(id(attention))
            yield int(getattr(phase, "block_index", 0)), attention


def _track_layers(model: Any) -> Any:
    infer = model.transformer_infer
    original = infer.infer_self_attn

    def tracked(owner: Any, phase: Any, *args: Any, **kwargs: Any) -> Any:
        phase.self_attn_1._carnator_layer = int(getattr(owner, "block_idx", getattr(phase, "block_index", 0)))
        return original(phase, *args, **kwargs)

    infer.infer_self_attn = types.MethodType(tracked, infer)
    return original


class CrossAttentionProbe(AbstractContextManager):
    """Selected-text QK salience at a fixed denoising step."""

    def __init__(self, model: Any, layers: list[int], token_indices: list[int], target_step: int = 2) -> None:
        self.model, self.layers, self.token_indices, self.target_step = model, set(layers), token_indices, target_step
        self.step = 0
        self.records: list[torch.Tensor] = []
        self.original: Any | None = None

    def set_step(self, step: int) -> None:
        self.step = step

    def __enter__(self) -> "CrossAttentionProbe":
        infer = self.model.transformer_infer
        self.original = infer.infer_cross_attn

        def wrapped(owner: Any, phase: Any, x: torch.Tensor, context: torch.Tensor, y_out: torch.Tensor, gate: torch.Tensor) -> Any:
            if owner.block_idx in self.layers and self.step == self.target_step:
                with torch.no_grad():
                    source_x = x + y_out * gate.squeeze()
                    heads, dim = owner.num_heads, owner.head_dim
                    q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(phase.norm3.apply(source_x))).view(-1, heads, dim)
                    indices = self.token_indices or list(range(context.shape[0]))
                    indices = [index for index in indices if 0 <= index < context.shape[0]]
                    if indices:
                        idx = torch.tensor(indices, dtype=torch.long, device=context.device)
                        k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context.index_select(0, idx))).view(-1, heads, dim)
                        score = torch.empty(q.shape[0], device=q.device, dtype=torch.float32)
                        for start in range(0, q.shape[0], 2048):
                            logits = torch.einsum("qhd,khd->qhk", q[start : start + 2048].float(), k.float()) * (dim**-0.5)
                            score[start : start + 2048] = logits.mean(dim=(1, 2))
                        if score.numel() == int(torch.tensor(TOKEN_SHAPE).prod()):
                            self.records.append(score.reshape(TOKEN_SHAPE).cpu())
            return self.original(phase, x, context, y_out, gate)

        infer.infer_cross_attn = types.MethodType(wrapped, infer)
        return self

    def score(self) -> torch.Tensor:
        if not self.records:
            raise RuntimeError("no 14B T2V cross-attention probe was captured at step 2")
        return torch.stack(self.records).mean(dim=0)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self.original is not None:
            self.model.transformer_infer.infer_cross_attn = self.original
        return False


class SelfContrastProbe(AbstractContextManager):
    def __init__(self, model: Any, layer: int, noun: torch.Tensor, background: torch.Tensor) -> None:
        self.model, self.layer = model, layer
        self.noun, self.background = noun.cpu().long(), background.cpu().long()
        self.step = 0
        self.score_value: torch.Tensor | None = None
        self.original: Any | None = None

    def set_step(self, step: int) -> None:
        self.step = step

    def __enter__(self) -> "SelfContrastProbe":
        infer = self.model.transformer_infer
        self.original = infer.infer_self_attn

        def wrapped(owner: Any, phase: Any, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, grid_sizes: Any = None) -> Any:
            if self.score_value is None and owner.block_idx == self.layer and self.step == 2:
                with torch.no_grad():
                    norm = owner.modulate_func(phase.norm1.apply(x), scale=scale, shift=shift).squeeze()
                    heads, dim = owner.num_heads, owner.head_dim
                    q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm)).view(-1, heads, dim)
                    k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm)).view(-1, heads, dim)
                    q, k = phase.rope.apply(q, k, owner.cos_sin) if owner.rope_positions is None else phase.rope.apply(q, k, owner.cos_sin, positions=owner.rope_positions)

                    def max_seed(indices: torch.Tensor) -> torch.Tensor:
                        key = k.index_select(0, indices.to(k.device))
                        out = torch.empty(q.shape[0], dtype=torch.float32, device=q.device)
                        for start in range(0, q.shape[0], 512):
                            logits = torch.einsum("qhd,khd->qhk", q[start : start + 512].float(), key.float()) * (dim**-0.5)
                            out[start : start + 512] = logits.max(dim=2).values.mean(dim=1)
                        return out

                    self.score_value = (max_seed(self.noun) - max_seed(self.background)).reshape(TOKEN_SHAPE).cpu()
            return self.original(phase, x, shift, scale, grid_sizes)

        infer.infer_self_attn = types.MethodType(wrapped, infer)
        return self

    def score(self) -> torch.Tensor:
        if self.score_value is None:
            raise RuntimeError("no 14B T2V self-attention contrast was captured at step 2")
        return self.score_value

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self.original is not None:
            self.model.transformer_infer.infer_self_attn = self.original
        return False


class RegionAdapter(AbstractContextManager):
    """Evaluate the planned region only; everything outside comes from source latents."""

    def __init__(self, model: Any, layers: list[int]) -> None:
        self.model, self.layers = model, set(layers)
        self.medit: torch.Tensor | None = None
        self.msee: torch.Tensor | None = None
        self.original: dict[str, Any] = {}

    def set_masks(self, medit: torch.Tensor, msee: torch.Tensor) -> None:
        device = torch.device("cuda:0")
        self.medit, self.msee = medit.to(device=device, dtype=torch.bool).flatten(), msee.to(device=device, dtype=torch.bool).flatten()

    def _active(self, owner: Any) -> bool:
        return owner.block_idx in self.layers and self.medit is not None and self.msee is not None

    def __enter__(self) -> "RegionAdapter":
        infer = self.model.transformer_infer
        names = ("infer_self_attn", "infer_cross_attn", "infer_ffn", "post_process")
        self.original = {name: getattr(infer, name) for name in names}

        def modulate(owner: Any, phase: Any, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, norm: str) -> torch.Tensor:
            return owner.modulate_func(getattr(phase, norm).apply(x), scale=scale, shift=shift).squeeze()

        def self_attn(owner: Any, phase: Any, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, grid_sizes: Any = None) -> torch.Tensor:
            if not self._active(owner):
                return self.original["infer_self_attn"](phase, x, shift, scale, grid_sizes)
            query_idx, context_idx = self.medit.nonzero().flatten(), self.msee.nonzero().flatten()  # type: ignore[union-attr]
            if not query_idx.numel() or not context_idx.numel():
                return torch.zeros_like(x)
            norm = modulate(owner, phase, x, scale, shift, "norm1")
            heads, dim = owner.num_heads, owner.head_dim
            q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm)).view(-1, heads, dim)
            k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm)).view(-1, heads, dim)
            v = phase.self_attn_v.apply(norm).view(-1, heads, dim)
            q, k = phase.rope.apply(q, k, owner.cos_sin) if owner.rope_positions is None else phase.rope.apply(q, k, owner.cos_sin, positions=owner.rope_positions)
            q, k, v = q.index_select(0, query_idx), k.index_select(0, context_idx), v.index_select(0, context_idx)
            cu_q, cu_k = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device), torch.tensor([0, k.shape[0]], dtype=torch.int32, device=q.device)
            out = phase.self_attn_1.apply(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_kv=cu_k, max_seqlen_q=q.shape[0], max_seqlen_kv=k.shape[0], block_idx=owner.block_idx, scheduler=owner.scheduler, grid_sizes=grid_sizes)
            result = torch.zeros_like(x)
            result.index_copy_(0, query_idx, phase.self_attn_o.apply(out))
            return result

        def cross_attn(owner: Any, phase: Any, x: torch.Tensor, context: torch.Tensor, y_out: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if not self._active(owner):
                return self.original["infer_cross_attn"](phase, x, context, y_out, gate)
            x = x + y_out * gate.squeeze()
            indices = self.medit.nonzero().flatten()  # type: ignore[union-attr]
            if not indices.numel():
                return x, torch.zeros_like(x)
            heads, dim = owner.num_heads, owner.head_dim
            q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(phase.norm3.apply(x.index_select(0, indices)))).view(-1, heads, dim)
            k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, heads, dim)
            v = phase.cross_attn_v.apply(context).view(-1, heads, dim)
            cu_q, cu_k = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device), torch.tensor([0, k.shape[0]], dtype=torch.int32, device=q.device)
            out = phase.cross_attn_1.apply(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_kv=cu_k, max_seqlen_q=q.shape[0], max_seqlen_kv=k.shape[0])
            result = torch.zeros_like(x)
            result.index_copy_(0, indices, phase.cross_attn_o.apply(out))
            return x, result

        def ffn(owner: Any, phase: Any, x: torch.Tensor, attn: torch.Tensor, c_shift: torch.Tensor, c_scale: torch.Tensor, c_gate: torch.Tensor | None = None) -> torch.Tensor:
            if not self._active(owner):
                return self.original["infer_ffn"](phase, x, attn, c_shift, c_scale, c_gate)
            x.add_(attn)
            indices = self.medit.nonzero().flatten()  # type: ignore[union-attr]
            result = torch.zeros_like(x)
            if indices.numel():
                norm = modulate(owner, phase, x.index_select(0, indices), c_scale, c_shift, "norm2")
                result.index_copy_(0, indices, phase.ffn_2.apply(torch.nn.functional.gelu(phase.ffn_0.apply(norm), approximate="tanh")))
            return result

        def post(owner: Any, x: torch.Tensor, y: torch.Tensor | None, c_gate: torch.Tensor, pre: Any = None) -> torch.Tensor:
            if not self._active(owner) or y is None:
                return self.original["post_process"](x, y, c_gate, pre)
            return x.add_(y * c_gate.squeeze())

        infer.infer_self_attn, infer.infer_cross_attn = types.MethodType(self_attn, infer), types.MethodType(cross_attn, infer)
        infer.infer_ffn, infer.post_process = types.MethodType(ffn, infer), types.MethodType(post, infer)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        for name, value in self.original.items():
            setattr(self.model.transformer_infer, name, value)
        return False


class OuterCapture(AbstractContextManager):
    def __init__(self, model: Any, cache: SourceBlockMaskCache, layers: list[int]) -> None:
        self.model, self.cache, self.layers = model, cache, set(layers)
        self.step = 1
        self.originals: list[tuple[Any, Any]] = []
        self.original_infer: Any | None = None

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def __enter__(self) -> "OuterCapture":
        self.original_infer = _track_layers(self.model)
        for layer, attn in _modules(self.model):
            if layer not in self.layers:
                continue
            original = attn.apply

            def wrapped(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, _attn: Any = attn, _original: Any = original, _layer: int = layer, **kwargs: Any) -> torch.Tensor:
                self.cache.capture(self.step, "cond", int(getattr(_attn, "_carnator_layer", _layer)), q, k)
                return _original(q, k, v, **kwargs)

            attn.apply = wrapped
            self.originals.append((attn, original))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        for attn, original in reversed(self.originals):
            attn.apply = original
        if self.original_infer is not None:
            self.model.transformer_infer.infer_self_attn = self.original_infer
        return False


class OuterAdapter(AbstractContextManager):
    def __init__(self, model: Any, cache: SourceBlockMaskCache, layers: list[int], backend: str, **options: Any) -> None:
        self.model, self.layers = model, set(layers)
        self.executor = OuterSparseExecutor(cache, backend, **options)
        self.step = 1
        self.originals: list[tuple[Any, Any]] = []
        self.original_infer: Any | None = None
        self.fallbacks: dict[str, int] = {}

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def __enter__(self) -> "OuterAdapter":
        self.original_infer = _track_layers(self.model)
        for layer, attn in _modules(self.model):
            if layer not in self.layers:
                continue
            original = attn.apply

            def wrapped(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, _attn: Any = attn, _original: Any = original, _layer: int = layer, **kwargs: Any) -> torch.Tensor:
                try:
                    return self.executor.run(q, k, v, step=self.step, branch="cond", layer=int(getattr(_attn, "_carnator_layer", _layer))).flatten(1)
                except (BlockMaskCacheError, RuntimeError, ValueError) as exc:
                    name = type(exc).__name__
                    self.fallbacks[name] = self.fallbacks.get(name, 0) + 1
                    return _original(q, k, v, **kwargs)

            attn.apply = wrapped
            self.originals.append((attn, original))
        return self

    def summary(self) -> dict[str, Any]:
        return {**self.executor.summary(), "layers": sorted(self.layers), "dense_fallbacks": self.fallbacks}

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        for attn, original in reversed(self.originals):
            attn.apply = original
        if self.original_infer is not None:
            self.model.transformer_infer.infer_self_attn = self.original_infer
        return False


class FFNQuantAdapter(AbstractContextManager):
    """W8A8 replacement for LightX2V's effective FFN linear matrices."""

    def __init__(self, model: Any, layers: list[int]) -> None:
        self.model, self.layers = model, set(layers)
        self.originals: list[tuple[Any, Any]] = []
        self.enabled: list[int] = []

    def __enter__(self) -> "FFNQuantAdapter":
        if not w8a8_kernel_available():
            raise RuntimeError("14B W8A8 requires CUDA Triton")
        for block in self.model.transformer_weights.blocks:
            layer = int(block.block_index)
            if layer not in self.layers:
                continue
            phase = block.compute_phases[2]
            try:
                matrices = []
                for module in (phase.ffn_0, phase.ffn_2):
                    weight = module._get_actual_weight().t().contiguous()
                    bias = module._get_actual_bias() if getattr(module, "bias", None) is not None else None
                    qweight, scale, _ = _quantize_weight(weight)
                    matrices.append((qweight, scale, bias, weight.dtype))
            except (AttributeError, RuntimeError) as exc:
                raise RuntimeError(f"cannot access resident 14B FFN weights for W8A8 layer {layer}") from exc

            def make_apply(module: Any, matrix: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.dtype]) -> Callable[[torch.Tensor], torch.Tensor]:
                def apply(value: torch.Tensor) -> torch.Tensor:
                    qweight, scale, bias, dtype = matrix
                    activation, activation_scale = _quantize_activation_per_token(value)
                    result = _w8a8_linear(activation, qweight, activation_scale, scale, bias, dtype).reshape(*value.shape[:-1], qweight.shape[0])
                    return result + module.apply_lora(value) if getattr(module, "has_lora_branch", False) else result

                return apply

            originals = (phase.ffn_0.apply, phase.ffn_2.apply)
            phase.ffn_0.apply, phase.ffn_2.apply = make_apply(phase.ffn_0, matrices[0]), make_apply(phase.ffn_2, matrices[1])
            self.originals.extend(((phase.ffn_0, originals[0]), (phase.ffn_2, originals[1])))
            self.enabled.append(layer)
        return self

    def summary(self) -> dict[str, Any]:
        return {"backend": "w8a8", "layers": self.enabled, "activation": "int8", "weight": "int8", "accumulator": "int32"}

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        for module, original in reversed(self.originals):
            module.apply = original
        return False


class T2V14BEngine:
    """Owns a source cache and runs C0/B0--B5 against that exact cache."""

    def __init__(
        self,
        backend: LightX2V14BBackend,
        source_root: Path,
        *,
        region_layers: list[int],
        outer_layers: list[int],
        mask_self_layer: int,
        mask_config: BlockMaskConfig,
        outer_options: dict[str, Any],
    ) -> None:
        if mask_self_layer not in region_layers:
            raise ValueError("mask_self_layer must be included in region_layers")
        if set(region_layers) & set(outer_layers):
            raise ValueError("region and outer sparse layers must not overlap")
        self.backend, self.source_root = backend, source_root
        self.region_layers, self.outer_layers = region_layers, outer_layers
        self.mask_self_layer, self.mask_config, self.outer_options = mask_self_layer, mask_config, outer_options
        self.cache: SourceBlockMaskCache | None = None
        self.bundle: dict[str, Any] | None = None

    @property
    def source_manifest_path(self) -> Path:
        return self.source_root / "manifest.json"

    def capture_source(self, request: T2VRequest, positive_indices: list[int]) -> dict[str, Any]:
        started = time.perf_counter()
        self.source_root.mkdir(parents=True, exist_ok=True)
        source_id = str(uuid.uuid4())
        self.cache = SourceBlockMaskCache(
            self.source_root / "outer_masks", self.mask_config, source_run_id=source_id, cfg_branch_count=1, create=True
        )
        models = self.backend.models()
        cross = CrossAttentionProbe(models["high_noise"], self.region_layers, positive_indices)
        captures = {phase: OuterCapture(model, self.cache, self.outer_layers) for phase, model in models.items()}
        latent_paths: dict[str, str] = {}
        initial_path = self.source_root / "latents_initial.pt"

        def before(_phase: str, step: int, scheduler: Any) -> None:
            if step == 1:
                torch.save(scheduler.latents.detach().cpu(), initial_path)
            cross.set_step(step)
            captures[_phase].set_step(step)

        def after(_phase: str, step: int, latents: torch.Tensor, _scheduler: Any) -> None:
            path = self.source_root / "latents" / f"step{step:03d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(latents.detach().cpu(), path)
            latent_paths[str(step)] = str(path)

        source_output = self.source_root / "source.mp4"
        with ExitStack() as stack:
            stack.enter_context(cross)
            for capture in captures.values():
                stack.enter_context(capture)
            self.backend.run(T2VRequest(request.prompt, request.seed, source_output, request.negative_prompt), before_step=before, after_step=after)

        cross_score = minmax_norm(cross.score())
        noun, background = seed_indices_from_cross_score(cross_score)
        self_probe = SelfContrastProbe(models["high_noise"], self.mask_self_layer, noun, background)

        def replay_before(_phase: str, step: int, _scheduler: Any) -> None:
            self_probe.set_step(step)

        replay_start = time.perf_counter()
        with self_probe:
            self.backend.run(T2VRequest(request.prompt, request.seed, self.source_root / "source_replay.mp4", request.negative_prompt), before_step=replay_before)
        self.bundle = build_self_contrast_fusion_masks(
            robust_norm(self_probe.score()), cross_score, FusionMaskConfig(mask_steps=(1, 2, 3, 4), completed_step=2)
        )
        self.bundle["source_completed_step"] = 2
        save_mask_bundle(self.source_root / "masks", self.bundle)
        manifest = {
            "status": "completed",
            "kind": "t2v_14b_source_cache",
            "created_at": utc_now(),
            "source_id": source_id,
            "profile": "t2v-14b-distilled",
            "prompt": request.prompt,
            "seed": request.seed,
            "negative_prompt": request.negative_prompt,
            "configuration": {"enable_cfg": False, "steps": STEPS, "token_shape": list(TOKEN_SHAPE)},
            "initial_latent": str(initial_path),
            "latents": latent_paths,
            "masks": str(self.source_root / "masks"),
            "outer_masks": self.cache.summary(),
            "mask_probe": {"positive_token_indices": positive_indices, "self_attention_layer": self.mask_self_layer, "replay_seconds": time.perf_counter() - replay_start},
            "timing": {"source_capture_seconds": time.perf_counter() - started},
        }
        write_json(self.source_manifest_path, manifest)
        return manifest

    def load_source(self) -> dict[str, Any]:
        manifest = read_json(self.source_manifest_path)
        if manifest.get("kind") != "t2v_14b_source_cache" or manifest.get("profile") != "t2v-14b-distilled":
            raise BlockMaskCacheError(f"not a compatible 14B T2V source cache: {self.source_manifest_path}")
        if manifest.get("configuration", {}).get("enable_cfg") is not False:
            raise BlockMaskCacheError("14B source cache unexpectedly enables CFG")
        self.cache = SourceBlockMaskCache(self.source_root / "outer_masks", self.mask_config)
        self.cache.expect_cfg_branches(1)
        self.bundle = load_mask_bundle(self.source_root / "masks")
        return manifest

    def run_baseline(
        self,
        *,
        baseline: str,
        outer_backend: str,
        ffn_backend: str,
        region_mode: str,
        output_root: Path,
        request: T2VRequest,
    ) -> dict[str, Any]:
        if baseline != "C0" and (self.cache is None or self.bundle is None):
            self.load_source()
        models = self.backend.models()
        output_root.mkdir(parents=True, exist_ok=True)
        output = output_root / f"{baseline}.mp4"
        start = time.perf_counter()
        denoising_started: float | None = None
        denoising_finished: float | None = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sparse: dict[str, OuterAdapter] = {}
        quant: dict[str, FFNQuantAdapter] = {}
        regions: dict[str, RegionAdapter] = {}
        with ExitStack() as stack:
            if region_mode == "carnator":
                regions = {phase: stack.enter_context(RegionAdapter(model, self.region_layers)) for phase, model in models.items()}
            if outer_backend != "dense":
                assert self.cache is not None
                sparse = {
                    phase: stack.enter_context(OuterAdapter(model, self.cache, self.outer_layers, outer_backend, **self.outer_options))
                    for phase, model in models.items()
                }
            if ffn_backend == "w8a8":
                quant = {phase: stack.enter_context(FFNQuantAdapter(model, list(range(40)))) for phase, model in models.items()}

            def before(phase: str, step: int, _scheduler: Any) -> None:
                nonlocal denoising_started
                if step == 1:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    denoising_started = time.perf_counter()
                if phase in regions:
                    assert self.bundle is not None
                    regions[phase].set_masks(self.bundle[f"step{step:03d}_Medit"], self.bundle[f"step{step:03d}_Msee"])
                if phase in sparse:
                    sparse[phase].set_step(step)

            def after(_phase: str, step: int, latents: torch.Tensor, _scheduler: Any) -> None:
                nonlocal denoising_finished
                if region_mode != "carnator":
                    if step == STEPS:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        denoising_finished = time.perf_counter()
                    return
                assert self.bundle is not None
                source = torch.load(self.source_root / "latents" / f"step{step:03d}.pt", map_location="cpu", weights_only=True).to(latents.device, latents.dtype)
                mask = token_mask_to_latent_mask(self.bundle[f"step{step:03d}_Medit"], latents.unsqueeze(0)).squeeze(0)
                latents.copy_(torch.where(mask, latents, source))
                if step == STEPS:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    denoising_finished = time.perf_counter()

            self.backend.run(T2VRequest(request.prompt, request.seed, output, request.negative_prompt), before_step=before, after_step=after)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        denoising = (
            denoising_finished - denoising_started
            if denoising_started is not None and denoising_finished is not None
            else elapsed
        )
        summary = {
            "status": "completed",
            "baseline": baseline,
            "profile": "t2v-14b-distilled",
            "created_at": utc_now(),
            "output": str(output),
            "prompt": request.prompt,
            "seed": request.seed,
            "configuration": {"enable_cfg": False, "steps": STEPS, "outer_attention": outer_backend, "ffn": ffn_backend, "region_mode": region_mode},
            "source_manifest": str(self.source_manifest_path) if region_mode == "carnator" else None,
            "timing": {"denoising_seconds": denoising, "end_to_end_seconds": elapsed},
            "outer_sparse": {phase: adapter.summary() for phase, adapter in sparse.items()} or {"backend": "dense"},
            "ffn": {phase: adapter.summary() for phase, adapter in quant.items()} or {"backend": "fp"},
        }
        write_json(output_root / f"{baseline}.json", summary)
        return summary


def positive_token_indices(backend: LightX2V14BBackend, prompt: str, phrase: str) -> list[int]:
    """Map an optional phrase to the native T5 tokenizer; empty means all text tokens."""

    if not phrase:
        return []
    if backend.pipe is None:
        raise RuntimeError("backend must be loaded before resolving source-positive")
    encoder = backend.pipe.runner.text_encoders[0]
    tokenizer = getattr(getattr(encoder, "tokenizer", None), "tokenizer", getattr(encoder, "tokenizer", None))
    if tokenizer is None:
        raise RuntimeError("LightX2V text encoder does not expose a tokenizer")
    start = prompt.lower().find(phrase.lower())
    if start < 0:
        raise ValueError(f"source-positive phrase {phrase!r} is not in source prompt")
    encoded = tokenizer(prompt, padding="max_length", max_length=512, truncation=True, return_offsets_mapping=True, return_attention_mask=True)
    end = start + len(phrase)
    return [index for index, ((lo, hi), valid) in enumerate(zip(encoded["offset_mapping"], encoded["attention_mask"])) if valid and hi > lo and lo < end and hi > start]
