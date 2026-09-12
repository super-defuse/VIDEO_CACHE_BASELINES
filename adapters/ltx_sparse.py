"""LTX-Video block-sparse self-attention: source routing capture and replay.

Chorus2 and Carnator both learn a per-layer block routing at the source and
replay it at the target.  Only ``attn1`` is routed: it is the quadratic
video-self-attention (16x16x24 = 6144 tokens at the configured budget), while
``attn2`` reads 256 caption tokens and stays dense.

The capture is the reference ``block_attention_mass`` / ``block_mass_to_mask``
pair from ``carnator.outer_sparse`` applied to the RoPE-applied Q/K, so the
routing matches what the Wan baselines record.  Replay goes through the same
backend the Wan baselines execute: SpargeAttn's ``block_sparse_sage2_attn_cuda``
(INT8 QK, FP8 PV).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ltx_attention import ltx_project, ltx_projection, ltx_qkv, stg_mix


@dataclass(frozen=True)
class LtxBlockMaskConfig:
    q_block_size: int = 64
    k_block_size: int = 128
    top_p: float = 0.95
    min_top_k: float = 0.1

    def __post_init__(self) -> None:
        if self.q_block_size <= 0 or self.k_block_size <= 0:
            raise ValueError("LTX sparse block sizes must be positive")
        if not 0.0 < self.top_p <= 1.0 or not 0.0 <= self.min_top_k <= 1.0:
            raise ValueError("LTX sparse thresholds are outside their valid range")


def branches_for(batch: int) -> tuple[str, ...]:
    """Fallback branch labels when no guidance state is available.

    This is a guess from the batch size alone and is only correct for a run with
    STG off (``[uncond, cond]`` or ``[cond]``).  With STG on the batch can be
    ``[cond, cond_perturbed]`` -- the same size, different rows -- so callers
    must prefer ``GuidanceState.branch_labels``.
    """
    if batch == 1:
        return ("cond",)
    if batch == 2:
        return ("uncond", "cond")
    if batch == 3:
        return ("uncond", "cond", "cond_perturbed")
    raise ValueError(f"LTX block-sparse execution supports one sample with optional CFG/STG, got batch {batch}")


def branch_labels(guidance: Any, batch: int) -> tuple[str, ...]:
    """Names for the current forward's branch rows, from the sampler if possible."""
    if guidance is not None:
        return guidance.branch_labels(batch)
    return branches_for(batch)


class LtxBlockMaskCache:
    """Per-(step, branch, layer) routing masks, bit-packed on disk."""

    FORMAT = "ltxvideo-self-block-mask/v1"

    def __init__(
        self,
        root: Path,
        config: LtxBlockMaskConfig,
        *,
        layers: list[int] | None = None,
        start_step: int | None = None,
        create: bool = False,
    ) -> None:
        self.root = Path(root)
        self.config = config
        self.entries: dict[tuple[int, str, int], dict[str, Any]] = {}
        self.metadata_path = self.root / "metadata.json"
        self.capture_seconds = 0.0
        self.write_seconds = 0.0
        # Decoded masks, memoised the way ``SourceBlockMaskCache`` does it: the
        # target stage asks for each mask once per step, branch and layer, and
        # re-reading plus re-hashing the file on every one of those calls costs
        # more than the sparse kernel saves.
        self.prepared: dict[tuple[int, str, int], Any] = {}
        self.prepare_seconds = 0.0
        if create:
            if layers is None or start_step is None:
                raise ValueError("layers and start_step are required for a new LTX block-mask cache")
            self.root.mkdir(parents=True, exist_ok=True)
            self.metadata: dict[str, Any] = {
                "format": self.FORMAT,
                "model": "LTX-Video-0.9.7-dev",
                "layers": list(layers),
                "start_step": int(start_step),
                "mask": asdict(config),
                "entries": [],
            }
        else:
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if self.metadata.get("format") != self.FORMAT or self.metadata.get("mask") != asdict(config):
                raise ValueError("LTX source block-mask cache metadata does not match the target")
            self.entries = {
                (int(item["step"]), str(item["branch"]), int(item["layer"])): item
                for item in self.metadata.get("entries", [])
            }

    def capture(self, step: int, branch: str, layer: int, query: Any, key: Any) -> None:
        import torch

        from carnator.outer_sparse import block_attention_mass, block_mass_to_mask

        started = time.perf_counter()
        mass = block_attention_mass(query, key, self.config.q_block_size, self.config.k_block_size)
        mask, coverage = block_mass_to_mask(mass, self.config.top_p, self.config.min_top_k)
        if query.is_cuda:
            torch.cuda.synchronize(query.device)
        self.capture_seconds += time.perf_counter() - started

        packed = np.packbits(mask.detach().cpu().numpy().reshape(-1), bitorder="little")
        relative = Path("masks") / f"step{step:03d}_{branch}_layer{layer:02d}.bin"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        write_started = time.perf_counter()
        path.write_bytes(packed.tobytes())
        self.write_seconds += time.perf_counter() - write_started
        density = mask.float().mean(dim=(1, 2))
        self.entries[(int(step), str(branch), int(layer))] = {
            "step": int(step),
            "branch": str(branch),
            "layer": int(layer),
            "path": str(relative),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "qkv_shape": [int(item) for item in query.shape],
            "mask_shape": [int(item) for item in mask.shape],
            "density": float(mask.float().mean().item()),
            "coverage": float(coverage),
            "head_density_min": float(density.min().item()),
            "head_density_max": float(density.max().item()),
        }

    def finalize(self) -> None:
        self.metadata["entries"] = [self.entries[key] for key in sorted(self.entries)]
        self.metadata["capture_seconds"] = self.capture_seconds
        self.metadata["write_seconds"] = self.write_seconds
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8")

    def validate_coverage(self, *, steps: range, branches: tuple[str, ...], layers: list[int]) -> None:
        missing = [
            (step, branch, layer)
            for step in steps
            for branch in branches
            for layer in layers
            if (step, branch, layer) not in self.entries
        ]
        if missing:
            raise ValueError(f"LTX source block-mask cache is incomplete; first missing entry is {missing[0]}")

    def _decode(self, entry: dict[str, Any]) -> tuple[Any, int]:
        import torch

        path = self.root / entry["path"]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"LTX source mask checksum mismatch: {path}")
        shape = tuple(int(item) for item in entry["mask_shape"])
        values = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little", count=math.prod(shape))
        return torch.from_numpy(values.reshape(shape).copy()).bool(), len(payload)

    def prepare(self, *, device: Any = None) -> dict[str, Any]:
        """Decode every source mask once, optionally onto the target device.

        Mirrors ``SourceBlockMaskCache.prepare``.  ``get`` alone would decode
        lazily on first use, which is still inside the timed denoise loop; doing
        all of them here is what keeps the whole cost out of it.
        """
        import torch

        target = None if device is None else torch.device(device)
        started = time.perf_counter()
        file_bytes = 0
        decoded = 0
        for key in sorted(self.entries):
            if key in self.prepared:
                continue
            mask, payload_bytes = self._decode(self.entries[key])
            if target is not None:
                mask = mask.to(device=target, non_blocking=True)
            self.prepared[key] = mask
            file_bytes += payload_bytes
            decoded += 1
        if target is not None and target.type == "cuda":
            torch.cuda.synchronize(target)
        self.prepare_seconds = time.perf_counter() - started
        return {
            "entries": len(self.entries),
            "decoded_entries": decoded,
            "file_bytes": file_bytes,
            "device": None if target is None else str(target),
            "elapsed_seconds": self.prepare_seconds,
        }

    def get(self, step: int, branch: str, layer: int, query: Any) -> tuple[Any, dict[str, Any]]:
        entry = self.entries.get((int(step), str(branch), int(layer)))
        if entry is None:
            raise KeyError(f"missing LTX source mask for step={step}, branch={branch}, layer={layer}")
        # Captured as [heads, q_tokens, head_dim] after the transpose, so the
        # query's [1, 0, 2] axes are the shape that must match the target.
        expected = [int(query.shape[1]), int(query.shape[0]), int(query.shape[2])]
        if entry["qkv_shape"] != expected:
            raise ValueError(f"LTX sparse QKV shape {expected} does not match source {entry['qkv_shape']}")
        key = (int(step), str(branch), int(layer))
        mask = self.prepared.get(key)
        if mask is None:
            mask, _bytes = self._decode(entry)
            self.prepared[key] = mask
        # A no-op when ``prepare`` already placed it here, which is the point.
        return mask.to(device=query.device, non_blocking=True), entry

    def summary(self) -> dict[str, Any]:
        entries = list(self.entries.values())
        return {
            "format": self.FORMAT,
            "path": str(self.root),
            "entries": len(entries),
            "layers": self.metadata.get("layers"),
            "start_step": self.metadata.get("start_step"),
            "average_density": (
                sum(float(item["density"]) for item in entries) / len(entries) if entries else None
            ),
            "average_coverage": (
                sum(float(item["coverage"]) for item in entries) / len(entries) if entries else None
            ),
            "capture_seconds": self.metadata.get("capture_seconds", self.capture_seconds),
            "write_seconds": self.metadata.get("write_seconds", self.write_seconds),
            "prepared_entries": len(self.prepared),
            "prepare_seconds": self.prepare_seconds,
        }


def dense_self_attention(
    attention: Any, hidden_states: Any, image_rotary_emb: Any, guidance: Any = None, layer: int = 0
) -> Any:
    """The stock ``attn1`` path, used before the sparse window opens.

    The STG substitution is applied here as well, so a sparse-method run is
    perturbed on exactly the same blocks as the dense baseline it is compared
    against; skipping it would move the guidance, not just the attention.
    """
    import torch.nn.functional as functional

    query, key, value = ltx_qkv(attention, hidden_states, image_rotary_emb)
    output = functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    flat = output.transpose(1, 2).flatten(2, 3)
    if guidance is not None:
        flat = stg_mix(attention, hidden_states, flat, guidance.layer(layer, flat.size(0)))
    return ltx_project(attention, flat)


class LtxMaskCaptureProcessor:
    """Replaces ``attn1``'s processor: dense attention, routing recorded."""

    def __init__(self, cache: LtxBlockMaskCache, state: Any, layer: int, guidance: Any = None) -> None:
        self.cache = cache
        self.state = state
        self.layer = int(layer)
        self.guidance = guidance

    def __call__(
        self,
        attention: Any,
        hidden_states: Any,
        encoder_hidden_states: Any = None,
        attention_mask: Any = None,
        image_rotary_emb: Any = None,
    ) -> Any:
        import torch
        import torch.nn.functional as functional

        if self.state.computing_step < int(self.cache.metadata["start_step"]):
            return dense_self_attention(
                attention, hidden_states, image_rotary_emb, self.guidance, self.layer
            )
        query, key, value = ltx_qkv(attention, hidden_states, image_rotary_emb)
        output = functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        for index, branch in enumerate(branch_labels(self.guidance, query.shape[0])):
            self.cache.capture(
                self.state.computing_step, branch, self.layer,
                query[index].transpose(0, 1).to(dtype=torch.float16),
                key[index].transpose(0, 1).to(dtype=torch.float16),
            )
        flat = output.transpose(1, 2).flatten(2, 3)
        if self.guidance is not None:
            flat = stg_mix(attention, hidden_states, flat, self.guidance.layer(self.layer, flat.size(0)))
        return ltx_project(attention, flat)


class LtxSparseExecutor:
    """Serves ``attn1`` from recorded routing through SpargeAttn.

    This is the same kernel the Wan2.1 and Wan2.2 baselines run: SpargeAttn's
    SageAttention2 block-sparse CUDA kernel, INT8 QK with FP8 PV, sm90 only.
    There is deliberately no second backend here.  The Triton
    ``flex_attention`` path this replaced was FP16-only (so it got none of the
    quantisation speedup), rebuilt a ``BlockMask`` per call, and -- because its
    BLOCK_M/BLOCK_N are hardcoded to 128 -- could not even run at the 64/128
    geometry SpargeAttn requires, which means it could not share a mask cache
    with it either.  Nothing in the LTX experiment used it.

    Q/K/V arrive as ``[heads, sequence, head_dim]`` for a single branch:
    ``ltx_qkv`` produces ``[B, heads, S, head_dim]`` and the processor indexes
    one branch at a time.  SpargeAttn wants ``[B, heads, S, head_dim]`` under
    ``tensor_layout="HND"``, so the batch axis is the only one added.
    """

    def __init__(
        self,
        cache: LtxBlockMaskCache,
        *,
        layer_refresh_size: int = 0,
        execution_dtype: Any = None,
        sparge_smooth_k: bool = True,
        sparge_pv_threshold: float = 50.0,
    ) -> None:
        import torch

        self.cache = cache
        self.layer_refresh_size = int(layer_refresh_size)
        self.execution_dtype = torch.float16 if execution_dtype is None else execution_dtype
        if self.execution_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"LTX sparse execution dtype must be FP16 or BF16, got {self.execution_dtype}"
            )
        self.sparge_smooth_k = bool(sparge_smooth_k)
        self.sparge_pv_threshold = float(sparge_pv_threshold)
        self.runs = 0
        self.density_sum = 0.0
        self.refresh_events = 0
        self.prepare_seconds = 0.0
        self._visited: dict[tuple[int, str, int], Any] = {}

    def prepare(self, *, device: Any = None) -> dict[str, Any]:
        """Pay the backend's one-time cost outside the timed denoise loop.

        SpargeAttn is a pre-compiled CUDA extension, so its warm-up is an
        import plus the device and block-geometry checks that decide whether it
        can run at all.  The mask cache is prepared here too, so the per-call
        decode, checksum and device copy all land outside the timed loop.
        """
        started = time.perf_counter()
        kernel = self._sparge_kernel(device)
        masks = self.cache.prepare(device=device)
        self.prepare_seconds = time.perf_counter() - started
        return {
            "backend": "sparge-sage2",
            "elapsed_seconds": self.prepare_seconds,
            "device": None if device is None else str(device),
            "masks": masks,
            "kernel": getattr(kernel, "__name__", str(kernel)),
        }

    def _sparge_kernel(self, device: Any = None) -> Any:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("sparge-sage2 requires CUDA")
        capability = torch.cuda.get_device_capability(device)
        if capability != (9, 0):
            raise RuntimeError(f"sparge-sage2 requires sm90, got capability={capability}")
        if (self.cache.config.q_block_size, self.cache.config.k_block_size) != (64, 128):
            raise ValueError("sparge-sage2 requires Q=64 and K=128 block sizes")
        try:
            from spas_sage_attn import block_sparse_sage2_attn_cuda
        except ImportError as exc:
            raise RuntimeError("sparge-sage2 requested but SpargeAttn is unavailable") from exc
        return block_sparse_sage2_attn_cuda

    def _refresh(self, mask: Any, *, step: int, branch: str, layer: int) -> Any:
        """Union the masks of each ``layer_refresh_size`` group onto its last layer.

        Recomputing every layer's mask is not what the reference does: a group of
        layers shares one routing, refreshed when the group's last layer runs.
        """
        import torch

        if self.layer_refresh_size <= 0:
            return mask
        layers = list(self.cache.metadata["layers"])
        position = layers.index(layer)
        group_start = position // self.layer_refresh_size * self.layer_refresh_size
        group = layers[group_start:group_start + self.layer_refresh_size]
        visited = self._visited.setdefault((step, branch, group_start), torch.zeros_like(mask))
        if layer == group[-1]:
            mask = mask | ~visited
            self.refresh_events += 1
        visited |= mask
        return mask

    def run(self, query: Any, key: Any, value: Any, *, step: int, branch: str, layer: int) -> Any:
        """SpargeAttn's block-sparse SageAttention2 kernel, over ``[heads, S, D]``."""
        import torch

        output_dtype = query.dtype
        query, key, value = (
            item.to(dtype=self.execution_dtype) for item in (query, key, value)
        )
        mask, _entry = self.cache.get(step, branch, layer, query)
        mask = self._refresh(mask, step=step, branch=branch, layer=layer)
        output = self._sparge_kernel(query.device)(
            query.unsqueeze(0).contiguous(),
            key.unsqueeze(0).contiguous(),
            value.unsqueeze(0).contiguous(),
            mask_id=mask.to(device=query.device, dtype=torch.bool, non_blocking=True).unsqueeze(0).contiguous(),
            smooth_k=self.sparge_smooth_k,
            pvthreshd=self.sparge_pv_threshold,
            tensor_layout="HND",
        )[0].contiguous()
        self.runs += 1
        self.density_sum += float(mask.float().mean().item())
        return output.to(dtype=output_dtype)

    def summary(self) -> dict[str, Any]:
        return {
            "backend": "sparge-sage2",
            "kernel": "block-sparse-sage2-attn-cuda",
            "execution_dtype": str(self.execution_dtype),
            "quantization": "QK INT8 and PV FP8",
            "sparge_smooth_k": self.sparge_smooth_k,
            "sparge_pvthreshd": self.sparge_pv_threshold,
            "runs": self.runs,
            "average_runtime_density": self.density_sum / self.runs if self.runs else None,
            "layer_refresh_size": self.layer_refresh_size,
            "refresh_events": self.refresh_events,
            "prepare_seconds": self.prepare_seconds,
            "mask_cache": self.cache.summary(),
        }


class LtxSparseProcessor:
    """Replaces ``attn1``'s processor: attention served from recorded routing."""

    def __init__(self, executor: LtxSparseExecutor, state: Any, layer: int, guidance: Any = None) -> None:
        self.executor = executor
        self.state = state
        self.layer = int(layer)
        self.guidance = guidance

    def __call__(
        self,
        attention: Any,
        hidden_states: Any,
        encoder_hidden_states: Any = None,
        attention_mask: Any = None,
        image_rotary_emb: Any = None,
    ) -> Any:
        import torch

        if self.state.computing_step < int(self.executor.cache.metadata["start_step"]):
            return dense_self_attention(
                attention, hidden_states, image_rotary_emb, self.guidance, self.layer
            )
        query, key, value = ltx_qkv(attention, hidden_states, image_rotary_emb)
        outputs = [
            self.executor.run(
                query[index], key[index], value[index],
                step=self.state.computing_step, branch=branch, layer=self.layer,
            )
            for index, branch in enumerate(branch_labels(self.guidance, query.shape[0]))
        ]
        flat = torch.stack(outputs).transpose(1, 2).flatten(2, 3)
        if self.guidance is not None:
            flat = stg_mix(attention, hidden_states, flat, self.guidance.layer(self.layer, flat.size(0)))
        return ltx_project(attention, flat)


class LtxAttentionProcessors:
    """Swaps ``attn1``'s processor across a layer list for the enclosed scope."""

    def __init__(self, transformer: Any, factory: Any, layers: list[int]) -> None:
        self.transformer = transformer
        # ``factory(layer)`` builds a processor.  A caller that wants the STG
        # mask applied closes over the guidance state when building the factory,
        # so the processors all observe the same live per-step mask.
        self.factory = factory
        self.layers = list(layers)
        self.originals: list[tuple[Any, Any]] = []

    def __enter__(self) -> "LtxAttentionProcessors":
        for layer in self.layers:
            attention = self.transformer.transformer_blocks[layer].attn1
            self.originals.append((attention, attention.get_processor()))
            attention.set_processor(self.factory(layer))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        for attention, processor in reversed(self.originals):
            attention.set_processor(processor)
        self.originals.clear()
        return False
