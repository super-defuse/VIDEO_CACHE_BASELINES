from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cogvideo_attention import dense_joint_attention, joint_qkv, project_joint


@dataclass(frozen=True)
class CogVideoBlockMaskConfig:
    q_block_size: int = 128
    k_block_size: int = 128
    top_p: float = 0.95
    min_top_k: float = 0.1

    def __post_init__(self) -> None:
        if self.q_block_size <= 0 or self.k_block_size <= 0:
            raise ValueError("CogVideo sparse block sizes must be positive")
        if not 0.0 < self.top_p <= 1.0 or not 0.0 <= self.min_top_k <= 1.0:
            raise ValueError("CogVideo sparse thresholds are outside their valid range")


class CogVideoBlockMaskCache:
    FORMAT = "cogvideox15-joint-block-mask/v1"

    def __init__(
        self,
        root: Path,
        config: CogVideoBlockMaskConfig,
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
        if create:
            if layers is None or start_step is None:
                raise ValueError("layers and start_step are required for a new CogVideo block-mask cache")
            self.root.mkdir(parents=True, exist_ok=True)
            self.metadata = {
                "format": self.FORMAT,
                "model": "CogVideoX1.5",
                "layers": list(layers),
                "start_step": int(start_step),
                "mask": asdict(config),
                "entries": [],
            }
        else:
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if self.metadata.get("format") != self.FORMAT or self.metadata.get("mask") != asdict(config):
                raise ValueError("CogVideo source block-mask cache metadata does not match the target")
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
        entry = {
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
        self.entries[(int(step), str(branch), int(layer))] = entry

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
            raise ValueError(f"CogVideo source block-mask cache is incomplete; first missing entry is {missing[0]}")

    def get(self, step: int, branch: str, layer: int, query: Any) -> tuple[Any, dict[str, Any]]:
        import torch

        entry = self.entries.get((int(step), str(branch), int(layer)))
        if entry is None:
            raise KeyError(f"missing CogVideo source mask for step={step}, branch={branch}, layer={layer}")
        expected = [int(query.shape[1]), int(query.shape[0]), int(query.shape[2])]
        if entry["qkv_shape"] != expected:
            raise ValueError(f"CogVideo sparse QKV shape {expected} does not match source {entry['qkv_shape']}")
        path = self.root / entry["path"]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"CogVideo source mask checksum mismatch: {path}")
        shape = tuple(int(item) for item in entry["mask_shape"])
        values = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little", count=math.prod(shape))
        mask = torch.from_numpy(values.reshape(shape).copy()).bool().to(query.device, non_blocking=True)
        return mask, entry

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
        }


class CogVideoMaskCaptureProcessor:
    def __init__(self, cache: CogVideoBlockMaskCache, state: Any, layer: int) -> None:
        self.cache = cache
        self.state = state
        self.layer = int(layer)

    def __call__(
        self,
        attention: Any,
        hidden_states: Any,
        encoder_hidden_states: Any,
        attention_mask: Any = None,
        image_rotary_emb: Any = None,
    ) -> tuple[Any, Any]:
        import torch

        if self.state.current_step + 1 < int(self.cache.metadata["start_step"]):
            return dense_joint_attention(attention, hidden_states, encoder_hidden_states, image_rotary_emb)
        query, key, value, text_tokens = joint_qkv(
            attention, hidden_states, encoder_hidden_states, image_rotary_emb
        )
        import torch.nn.functional as functional

        output = functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        branches = ("cond",) if query.shape[0] == 1 else ("uncond", "cond")
        if len(branches) != query.shape[0]:
            raise ValueError("CogVideo sparse capture supports one sample with optional CFG")
        for batch_index, branch in enumerate(branches):
            self.cache.capture(
                self.state.current_step + 1,
                branch,
                self.layer,
                query[batch_index].transpose(0, 1).to(dtype=torch.float16),
                key[batch_index].transpose(0, 1).to(dtype=torch.float16),
            )
        return project_joint(attention, output, text_tokens)


class CogVideoFlexExecutor:
    def __init__(self, cache: CogVideoBlockMaskCache, *, layer_refresh_size: int = 0) -> None:
        import torch
        from torch.nn.attention.flex_attention import flex_attention

        self.cache = cache
        self.layer_refresh_size = int(layer_refresh_size)
        self._kernel = torch.compile(flex_attention, dynamic=False)
        self.runs = 0
        self.density_sum = 0.0
        self.refresh_events = 0
        self._visited: dict[tuple[int, str, int], Any] = {}

    def _refresh(self, mask: Any, *, step: int, branch: str, layer: int) -> Any:
        import torch

        if self.layer_refresh_size <= 0:
            return mask
        layers = list(self.cache.metadata["layers"])
        position = layers.index(layer)
        group_start = position // self.layer_refresh_size * self.layer_refresh_size
        group = layers[group_start:group_start + self.layer_refresh_size]
        key = (step, branch, group_start)
        visited = self._visited.setdefault(key, torch.zeros_like(mask))
        if layer == group[-1]:
            mask = mask | ~visited
            self.refresh_events += 1
        visited |= mask
        return mask

    @staticmethod
    def _block_mask(mask: Any, sequence: int, block_size: tuple[int, int]) -> Any:
        import torch
        from torch.nn.attention.flex_attention import BlockMask

        counts = mask.sum(dim=-1, dtype=torch.int32).unsqueeze(0)
        indices = mask.to(dtype=torch.int8).argsort(dim=-1, descending=True).to(dtype=torch.int32).unsqueeze(0)
        return BlockMask.from_kv_blocks(
            counts,
            indices,
            BLOCK_SIZE=block_size,
            seq_lengths=(sequence, sequence),
        )

    def run(self, query: Any, key: Any, value: Any, *, step: int, branch: str, layer: int) -> Any:
        import torch

        output_dtype = query.dtype
        query, key, value = (item.to(dtype=torch.float16) for item in (query, key, value))
        mask, entry = self.cache.get(step, branch, layer, query)
        mask = self._refresh(mask, step=step, branch=branch, layer=layer)
        block = self._block_mask(
            mask,
            query.shape[1],
            (self.cache.config.q_block_size, self.cache.config.k_block_size),
        )
        output = self._kernel(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), block_mask=block)[0]
        self.runs += 1
        self.density_sum += float(mask.float().mean().item())
        return output.to(dtype=output_dtype)

    def summary(self) -> dict[str, Any]:
        return {
            "backend": "torch-compiled-flex-attention",
            "kernel": "block-sparse-flex-attention",
            "runs": self.runs,
            "average_runtime_density": self.density_sum / self.runs if self.runs else None,
            "layer_refresh_size": self.layer_refresh_size,
            "refresh_events": self.refresh_events,
        }


class CogVideoSparseProcessor:
    def __init__(self, executor: CogVideoFlexExecutor, state: Any, layer: int) -> None:
        self.executor = executor
        self.state = state
        self.layer = int(layer)

    def __call__(
        self,
        attention: Any,
        hidden_states: Any,
        encoder_hidden_states: Any,
        attention_mask: Any = None,
        image_rotary_emb: Any = None,
    ) -> tuple[Any, Any]:
        import torch

        if self.state.current_step + 1 < int(self.executor.cache.metadata["start_step"]):
            return dense_joint_attention(attention, hidden_states, encoder_hidden_states, image_rotary_emb)
        query, key, value, text_tokens = joint_qkv(
            attention, hidden_states, encoder_hidden_states, image_rotary_emb
        )
        branches = ("cond",) if query.shape[0] == 1 else ("uncond", "cond")
        if len(branches) != query.shape[0]:
            raise ValueError("CogVideo sparse execution supports one sample with optional CFG")
        outputs = [
            self.executor.run(
                query[index], key[index], value[index],
                step=self.state.current_step + 1,
                branch=branch,
                layer=self.layer,
            )
            for index, branch in enumerate(branches)
        ]
        return project_joint(attention, torch.stack(outputs), text_tokens)


class CogVideoAttentionProcessors:
    def __init__(self, transformer: Any, factory: Any, layers: list[int]) -> None:
        self.transformer = transformer
        self.factory = factory
        self.layers = list(layers)
        self.originals: list[tuple[Any, Any]] = []

    def __enter__(self) -> "CogVideoAttentionProcessors":
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
