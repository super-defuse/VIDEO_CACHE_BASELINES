"""Reusable source block masks and runtime outer self-attention backends."""

from __future__ import annotations

import json
import hashlib
import math
import os
import sys
import time
import types
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


class BlockMaskCacheError(RuntimeError):
    """A cached source mask cannot safely be applied to the target attention."""


@dataclass(frozen=True)
class BlockMaskConfig:
    q_block_size: int = 64
    k_block_size: int = 128
    top_p: float = 0.95
    min_top_k: float = 0.1

    def __post_init__(self) -> None:
        if self.q_block_size <= 0 or self.k_block_size <= 0:
            raise ValueError("block sizes must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 0.0 <= self.min_top_k <= 1.0:
            raise ValueError("min_top_k must be in [0, 1]")


def build_flashinfer_packed_bsr(
    block_mask: torch.Tensor,
    seq_len: int,
    q_block_size: int,
    k_block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pack independent per-head masks into FlashInfer's single BSR pattern.

    ``BlockSparseAttentionWrapper`` accepts one BSR matrix.  Concatenating the
    original heads along sequence space preserves each head's mask while
    retaining block-level indices.  This is the A14B FlashInfer layout used by
    Chorus2.
    """

    if block_mask.ndim != 3 or block_mask.dtype != torch.bool:
        raise ValueError("FlashInfer packed BSR expects a [heads, q_blocks, k_blocks] bool mask")
    if seq_len % q_block_size or seq_len % k_block_size:
        raise ValueError(
            "FlashInfer packed BSR requires seq_len divisible by both block sizes; "
            f"got seq_len={seq_len}, Q_block_size={q_block_size}, K_block_size={k_block_size}"
        )

    num_heads, q_blocks, k_blocks = block_mask.shape
    expected_q_blocks = seq_len // q_block_size
    expected_k_blocks = seq_len // k_block_size
    if (q_blocks, k_blocks) != (expected_q_blocks, expected_k_blocks):
        raise ValueError(
            "cached mask geometry does not match FlashInfer packed BSR: "
            f"mask=({q_blocks}, {k_blocks}), expected=({expected_q_blocks}, {expected_k_blocks})"
        )

    row_counts = block_mask.sum(dim=-1, dtype=torch.int32).reshape(-1)
    indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=block_mask.device),
            torch.cumsum(row_counts, dim=0, dtype=torch.int32),
        ]
    )
    head_idx, _q_idx, key_block_idx = block_mask.nonzero(as_tuple=True)
    indices = key_block_idx.to(torch.int32)
    indices.add_(head_idx.to(torch.int32) * k_blocks)
    return (
        indptr.to(device=device, non_blocking=True),
        indices.to(device=device, non_blocking=True),
        num_heads * seq_len,
    )


def flashinfer_block_layout(config: BlockMaskConfig, seq_len: int) -> tuple[int, int]:
    if seq_len % config.q_block_size == 0 and seq_len % config.k_block_size == 0:
        return config.q_block_size, config.k_block_size
    if (
        seq_len % 16 == 0
        and config.q_block_size % 16 == 0
        and config.k_block_size % 16 == 0
    ):
        return 16, 16
    return config.q_block_size, config.k_block_size


def refine_block_mask(
    block_mask: torch.Tensor,
    seq_len: int,
    source_q_block_size: int,
    source_k_block_size: int,
    target_q_block_size: int,
    target_k_block_size: int,
) -> torch.Tensor:
    if (source_q_block_size, source_k_block_size) == (target_q_block_size, target_k_block_size):
        return block_mask
    if source_q_block_size % target_q_block_size or source_k_block_size % target_k_block_size:
        raise ValueError("FlashInfer runtime blocks must evenly refine cached source blocks")
    q_repeat = source_q_block_size // target_q_block_size
    k_repeat = source_k_block_size // target_k_block_size
    expected = (math.ceil(seq_len / source_q_block_size), math.ceil(seq_len / source_k_block_size))
    if tuple(block_mask.shape[1:]) != expected:
        raise ValueError(f"cached block mask shape {tuple(block_mask.shape[1:])} != expected {expected}")
    return block_mask.repeat_interleave(q_repeat, dim=1).repeat_interleave(k_repeat, dim=2)[
        :, : seq_len // target_q_block_size, : seq_len // target_k_block_size
    ]


def select_region_tokens_from_block_mask(
    block_mask: torch.Tensor,
    medit: torch.Tensor,
    msee: torch.Tensor,
    *,
    q_block_size: int,
    k_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
    """Restrict a canonical region update to source-attended token blocks.

    ``block_mask`` is source attention routing in ``[head, q_block, k_block]``
    form.  The native region path uses a shared token set across heads, so the
    per-head routing is conservatively unioned only after it is intersected
    with ``Msee``.  A Medit query token is retained only when its query block
    has an allowed Msee key block; the resulting context is then limited to
    Msee keys reachable from retained query blocks.
    """

    if block_mask.ndim != 3 or block_mask.dtype != torch.bool:
        raise ValueError("block_mask must be a [heads, q_blocks, k_blocks] boolean tensor")
    if medit.ndim != 1 or msee.ndim != 1 or medit.shape != msee.shape:
        raise ValueError("Medit and Msee must be matching one-dimensional token masks")
    if q_block_size <= 0 or k_block_size <= 0:
        raise ValueError("block sizes must be positive")

    token_count = int(medit.numel())
    q_blocks = math.ceil(token_count / q_block_size)
    k_blocks = math.ceil(token_count / k_block_size)
    if tuple(block_mask.shape[1:]) != (q_blocks, k_blocks):
        raise ValueError(
            "block mask geometry does not match token masks: "
            f"mask={tuple(block_mask.shape[1:])}, expected={(q_blocks, k_blocks)}"
        )

    device = medit.device
    positions = torch.arange(token_count, device=device)
    query_blocks = torch.div(positions, q_block_size, rounding_mode="floor")
    key_blocks = torch.div(positions, k_block_size, rounding_mode="floor")
    msee_key_blocks = torch.zeros(k_blocks, dtype=torch.bool, device=device)
    msee_key_blocks[key_blocks[msee]] = True

    # A query block can only be updated when at least one source-selected key
    # block contains a region context token.  This is the Medit/block-mask
    # intersection that lets the rest of the middle block retain its latent.
    reachable_queries = (block_mask & msee_key_blocks.view(1, 1, -1)).any(dim=(0, 2))
    query_mask = medit & reachable_queries.index_select(0, query_blocks)
    active_query_blocks = torch.zeros(q_blocks, dtype=torch.bool, device=device)
    active_query_blocks[query_blocks[query_mask]] = True
    reachable_keys = (block_mask & active_query_blocks.view(1, -1, 1)).any(dim=(0, 1))
    context_mask = msee & reachable_keys.index_select(0, key_blocks)

    query_count = int(query_mask.sum().item())
    context_count = int(context_mask.sum().item())
    if query_count and not context_count:
        raise ValueError("block-mask intersection left Medit queries without an Msee context")
    return query_mask, context_mask, {
        "input_medit_count": int(medit.sum().item()),
        "input_msee_count": int(msee.sum().item()),
        "query_count": query_count,
        "context_count": context_count,
        "active_query_blocks": int(active_query_blocks.sum().item()),
        "active_key_blocks": int(reachable_keys.sum().item()),
        "query_ratio": float(query_count / token_count) if token_count else 0.0,
        "context_ratio": float(context_count / token_count) if token_count else 0.0,
    }


def cfg_branches(guidance_scale: float) -> int:
    return 1 if float(guidance_scale) == 1.0 else 2


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _prepare_flashinfer_build_tools() -> None:
    """Expose the active Conda toolchain to FlashInfer's lazy JIT planner."""

    def prepend_paths(name: str, paths: list[Path]) -> None:
        existing = os.environ.get(name, "")
        values = [str(path) for path in paths if path.exists()]
        if existing:
            values.extend(value for value in existing.split(os.pathsep) if value)
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value not in seen:
                unique.append(value)
                seen.add(value)
        if unique:
            os.environ[name] = os.pathsep.join(unique)

    binary_dir = Path(sys.executable).resolve().parent
    current_path = os.environ.get("PATH", "")
    if str(binary_dir) not in current_path.split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join([str(binary_dir), current_path]) if current_path else str(binary_dir)
    nvcc = binary_dir / "nvcc"
    if nvcc.exists():
        os.environ.setdefault("CUDA_HOME", str(binary_dir.parent))
        os.environ.setdefault("CUDA_PATH", str(binary_dir.parent))
        os.environ.setdefault("CUDACXX", str(nvcc))
    nvidia_root = binary_dir.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia"
    includes = [path for path in sorted(nvidia_root.glob("*/include")) if path.exists()]
    for name in ("CPATH", "CPLUS_INCLUDE_PATH"):
        prepend_paths(name, includes)
    libraries = [binary_dir.parent / "lib", binary_dir.parent / "targets" / "x86_64-linux" / "lib"]
    libraries.extend(path for path in sorted(nvidia_root.glob("*/lib")) if path.exists())
    for name in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        prepend_paths(name, libraries)


_TRITON_BLOCK_MASS: Any = None


def _triton_block_attention_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    q_block_size: int,
    k_block_size: int,
) -> torch.Tensor:
    """Exact block mass with online softmax, avoiding a full attention matrix."""

    global _TRITON_BLOCK_MASS
    if _TRITON_BLOCK_MASS is None:
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(
            q_ptr,
            k_ptr,
            out_ptr,
            q_stride_s: tl.constexpr,
            q_stride_h: tl.constexpr,
            q_stride_d: tl.constexpr,
            k_stride_s: tl.constexpr,
            k_stride_h: tl.constexpr,
            k_stride_d: tl.constexpr,
            out_stride_h: tl.constexpr,
            out_stride_q: tl.constexpr,
            out_stride_k: tl.constexpr,
            seq_len: tl.constexpr,
            head_dim: tl.constexpr,
            q_block_size: tl.constexpr,
            k_block_size: tl.constexpr,
            k_blocks: tl.constexpr,
            block_dim: tl.constexpr,
            softmax_scale: tl.constexpr,
        ):
            q_block = tl.program_id(0)
            head = tl.program_id(1)
            q_start = q_block * q_block_size
            q_offsets = q_start + tl.arange(0, q_block_size)
            d_offsets = tl.arange(0, block_dim)
            q_valid = q_offsets < seq_len
            q_values = tl.load(
                q_ptr + q_offsets[:, None] * q_stride_s + head * q_stride_h + d_offsets[None, :] * q_stride_d,
                mask=q_valid[:, None] & (d_offsets[None, :] < head_dim),
                other=0.0,
            )
            row_max = tl.where(q_valid, -float("inf"), 0.0)
            row_sum = tl.where(q_valid, 0.0, 1.0)
            for k_block in tl.range(0, k_blocks):
                k_offsets = k_block * k_block_size + tl.arange(0, k_block_size)
                k_values = tl.load(
                    k_ptr + k_offsets[:, None] * k_stride_s + head * k_stride_h + d_offsets[None, :] * k_stride_d,
                    mask=(k_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim),
                    other=0.0,
                )
                scores = tl.dot(q_values, tl.trans(k_values)) * softmax_scale
                valid = q_valid[:, None] & (k_offsets[None, :] < seq_len)
                scores = tl.where(valid, scores, -float("inf"))
                block_max = tl.max(scores, axis=1)
                next_max = tl.maximum(row_max, block_max)
                next_sum = row_sum * tl.exp(row_max - next_max) + tl.sum(tl.exp(scores - next_max[:, None]), axis=1)
                row_max = tl.where(q_valid, next_max, 0.0)
                row_sum = tl.where(q_valid, next_sum, 1.0)

            q_count = tl.sum(tl.where(q_valid, 1.0, 0.0), axis=0)
            for k_block in tl.range(0, k_blocks):
                k_offsets = k_block * k_block_size + tl.arange(0, k_block_size)
                k_values = tl.load(
                    k_ptr + k_offsets[:, None] * k_stride_s + head * k_stride_h + d_offsets[None, :] * k_stride_d,
                    mask=(k_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim),
                    other=0.0,
                )
                scores = tl.dot(q_values, tl.trans(k_values)) * softmax_scale
                valid = q_valid[:, None] & (k_offsets[None, :] < seq_len)
                probs = tl.exp(scores - row_max[:, None]) / row_sum[:, None]
                probs = tl.where(valid, probs, 0.0)
                mass = tl.sum(tl.sum(probs, axis=1), axis=0) / q_count
                tl.store(out_ptr + head * out_stride_h + q_block * out_stride_q + k_block * out_stride_k, mass)

        _TRITON_BLOCK_MASS = (triton, kernel)

    triton, kernel = _TRITON_BLOCK_MASS
    seq_len, heads, head_dim = q.shape
    q_blocks = math.ceil(seq_len / q_block_size)
    k_blocks = math.ceil(seq_len / k_block_size)
    block_dim = 1 << (head_dim - 1).bit_length()
    output = torch.empty((heads, q_blocks, k_blocks), dtype=torch.float32, device=q.device)
    kernel[(q_blocks, heads)](
        q,
        k,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        seq_len,
        head_dim,
        q_block_size,
        k_block_size,
        k_blocks,
        block_dim,
        head_dim**-0.5,
        num_warps=8,
        num_stages=3,
    )
    return output


def block_attention_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    q_block_size: int,
    k_block_size: int,
) -> torch.Tensor:
    """Return source attention mass by ``[head, q_block, k_block]``.

    Q and K are RoPE-applied tensors in NHD layout.  The implementation is
    deliberately chunked by query block: it is exact, has a small peak memory
    footprint, and remains useful as a correctness reference for accelerator
    capture kernels.
    """

    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("q and k must have NHD layout")
    if q.shape != k.shape:
        raise ValueError(f"q shape {tuple(q.shape)} != k shape {tuple(k.shape)}")
    seq_len, heads, head_dim = q.shape
    if not seq_len:
        raise ValueError("attention sequence must be non-empty")

    use_triton = (
        q.is_cuda
        and q_block_size >= 16
        and k_block_size >= 16
        and not q_block_size & (q_block_size - 1)
        and not k_block_size & (k_block_size - 1)
    )
    if use_triton:
        return _triton_block_attention_mass(q.contiguous(), k.contiguous(), q_block_size, k_block_size)

    qh = q.float().transpose(0, 1).contiguous()
    kh = k.float().transpose(0, 1).contiguous()
    q_blocks = math.ceil(seq_len / q_block_size)
    k_blocks = math.ceil(seq_len / k_block_size)
    result = torch.empty((heads, q_blocks, k_blocks), dtype=torch.float32, device=q.device)
    scale = head_dim**-0.5
    for q_block in range(q_blocks):
        q_start = q_block * q_block_size
        q_stop = min(seq_len, q_start + q_block_size)
        scores = torch.matmul(qh[:, q_start:q_stop], kh.transpose(-1, -2)).mul_(scale)
        probs = scores.softmax(dim=-1)
        for k_block in range(k_blocks):
            k_start = k_block * k_block_size
            k_stop = min(seq_len, k_start + k_block_size)
            result[:, q_block, k_block] = probs[:, :, k_start:k_stop].sum(dim=-1).mean(dim=-1)
    return result


def block_mass_to_mask(
    block_mass: torch.Tensor,
    top_p: float,
    min_top_k: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """Keep the smallest per-head block set covering ``top_p`` mass."""

    if block_mass.ndim != 3:
        raise ValueError("block_mass must have shape [heads, q_blocks, k_blocks]")
    heads, q_blocks, k_blocks = block_mass.shape
    if not heads or not q_blocks or not k_blocks:
        raise ValueError("block_mass dimensions must be non-empty")
    sorted_mass, sorted_idx = torch.sort(block_mass, dim=-1, descending=True)
    floor = max(1, min(k_blocks, math.ceil(k_blocks * min_top_k)))
    keep_counts = (sorted_mass.cumsum(dim=-1) < top_p).sum(dim=-1) + 1
    keep_counts = keep_counts.clamp(min=floor, max=k_blocks)
    ranks = torch.arange(k_blocks, device=block_mass.device).view(1, 1, -1)
    mask = torch.zeros_like(block_mass, dtype=torch.bool)
    mask.scatter_(-1, sorted_idx, ranks < keep_counts.unsqueeze(-1))
    total_mass = block_mass.sum()
    coverage = (block_mass * mask).sum() / total_mass if total_mass.item() else total_mass
    return mask, float(coverage.item())


def dense_block_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    q_block_size: int,
    k_block_size: int,
) -> torch.Tensor:
    """Dense masked reference for probes and small-shape tests, using NHD layout."""

    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError("q, k and v must share NHD shape")
    seq_len, heads, head_dim = q.shape
    expected = (heads, math.ceil(seq_len / q_block_size), math.ceil(seq_len / k_block_size))
    if tuple(block_mask.shape) != expected:
        raise ValueError(f"block mask {tuple(block_mask.shape)} != expected {expected}")
    token_mask = torch.zeros((heads, seq_len, seq_len), dtype=torch.bool, device=q.device)
    for q_block in range(expected[1]):
        q_start, q_stop = q_block * q_block_size, min(seq_len, (q_block + 1) * q_block_size)
        for k_block in range(expected[2]):
            k_start, k_stop = k_block * k_block_size, min(seq_len, (k_block + 1) * k_block_size)
            token_mask[:, q_start:q_stop, k_start:k_stop] = block_mask[:, q_block, k_block].view(-1, 1, 1)
    qh, kh, vh = (tensor.float().transpose(0, 1) for tensor in (q, k, v))
    scores = torch.matmul(qh, kh.transpose(-1, -2)).mul_(head_dim**-0.5)
    probs = scores.masked_fill(~token_mask, float("-inf")).softmax(dim=-1)
    return torch.matmul(probs, vh).transpose(0, 1).to(dtype=q.dtype)


def pack_bool_mask(mask: torch.Tensor) -> tuple[bytes, int]:
    """Bit-pack a canonical attention mask into the v1 on-disk layout.

    Elements are packed LSB-first: flat element ``k`` lands in byte ``k // 8``
    at bit ``k % 8``, which is exactly what :func:`unpack_bool_mask` and
    ``SourceBlockMaskCache._decode`` invert.  A ``torch.bool`` tensor spends a
    whole byte per element, so this is an 8x reduction with no information loss.
    """

    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 3:
        raise ValueError("canonical attention mask must be bool[heads,q_blocks,k_blocks]")
    flat = mask.detach().cpu().contiguous().reshape(-1).to(torch.uint8)
    element_count = int(flat.numel())
    padding = (-element_count) % 8
    if padding:
        flat = torch.cat((flat, torch.zeros(padding, dtype=torch.uint8)))
    # int16 accumulation keeps every byte unambiguously in 0..255.
    weights = (1 << torch.arange(8, dtype=torch.uint8)).view(1, 8)
    packed = (flat.view(-1, 8) * weights).sum(dim=1, dtype=torch.int16).to(torch.uint8)
    return packed.numpy().tobytes(), element_count


def unpack_bool_mask(payload: bytes, shape: list[int] | tuple[int, ...]) -> torch.Tensor:
    """Invert :func:`pack_bool_mask` back into a ``torch.bool`` mask."""

    if len(shape) != 3 or any(int(value) < 1 for value in shape):
        raise ValueError("mask shape must be three positive dimensions")
    element_count = math.prod(int(value) for value in shape)
    if len(payload) != math.ceil(element_count / 8):
        raise ValueError("bit-packed mask byte size does not match metadata")
    raw = torch.frombuffer(memoryview(bytearray(payload)), dtype=torch.uint8)
    shifts = torch.arange(8, dtype=torch.uint8)
    unpacked = ((raw.unsqueeze(1) >> shifts) & 1).reshape(-1)[:element_count]
    return unpacked.to(torch.bool).reshape(tuple(int(value) for value in shape)).contiguous()


class SourceBlockMaskCache:
    """Disk-backed source masks keyed by denoising step, CFG branch and layer."""

    FORMAT = "carnator-source-block-mask-v1"
    PACKED_FORMAT = "carnator-source-block-mask-bitpacked-v1"
    CHORUS2_PACKED_FORMAT = "chorus2-normal-sparse-attention-mask-bitpacked-v1"

    def __init__(
        self,
        root: Path,
        config: BlockMaskConfig,
        *,
        source_run_id: str | None = None,
        cfg_branch_count: int | None = None,
        capture_block_mass: bool = False,
        create: bool = False,
        pack_format: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = config
        self.metadata_path = self.root / "metadata.json"
        self.entries: dict[tuple[int, str, int], dict[str, Any]] = {}
        self.metadata: dict[str, Any] = {}
        self.capture_seconds = 0.0
        self.write_seconds = 0.0
        self.configuration_error: str | None = None
        self._bitpacked = False
        self._created = False
        self._prepared_masks: dict[tuple[int, str, int], torch.Tensor] = {}
        self.capture_block_mass = bool(capture_block_mass)
        if create:
            if source_run_id is None or cfg_branch_count is None:
                raise ValueError("source_run_id and cfg_branch_count are required to create a cache")
            if pack_format is not None and pack_format not in {self.PACKED_FORMAT, self.CHORUS2_PACKED_FORMAT}:
                raise ValueError(f"unsupported packed source block-mask format: {pack_format}")
            self.root.mkdir(parents=True, exist_ok=True)
            # Opt-in: callers that do not ask for a packed format keep writing
            # the original bool layout, so already-measured runs stay comparable.
            # Reading is format-agnostic (_load_metadata detects it), so only the
            # source side needs to choose.
            self._bitpacked = pack_format is not None
            self._created = True
            self.metadata = {
                "format": pack_format or self.FORMAT,
                "source_run_id": source_run_id,
                "cfg_branches": int(cfg_branch_count),
                "mask": asdict(config),
                "block_mass": {"enabled": self.capture_block_mass},
                "entries": [],
            }
            self._write_metadata()
        else:
            self._load_metadata()

    def _load_metadata(self) -> None:
        if not self.metadata_path.exists():
            raise BlockMaskCacheError(f"missing source block-mask metadata: {self.metadata_path}")
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        cache_format = self.metadata.get("format")
        if cache_format not in {self.FORMAT, self.PACKED_FORMAT, self.CHORUS2_PACKED_FORMAT}:
            raise BlockMaskCacheError("unsupported source block-mask cache format")
        self._bitpacked = cache_format in {self.PACKED_FORMAT, self.CHORUS2_PACKED_FORMAT}
        self.capture_block_mass = bool(self.metadata.get("block_mass", {}).get("enabled", False))
        cached_config = self.metadata.get("mask", {})
        if cached_config != asdict(self.config):
            self.configuration_error = "source block-mask parameters do not match this run"
        for entry in self.metadata.get("entries", []):
            self.entries[(int(entry["step"]), str(entry["cfg_branch"]), int(entry["layer"]))] = entry

    def expect_cfg_branches(self, expected: int) -> None:
        if int(self.metadata.get("cfg_branches", 0)) != int(expected):
            self.configuration_error = (
                f"source cache cfg_branches={self.metadata.get('cfg_branches')} does not match target={expected}"
            )

    def _write_metadata(self) -> None:
        self.metadata["entries"] = [
            self.entries[key] for key in sorted(self.entries, key=lambda value: (value[0], value[1], value[2]))
        ]
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8")

    def capture(self, step: int, branch: str, layer: int, q: torch.Tensor, k: torch.Tensor) -> None:
        started = time.perf_counter()
        mass = block_attention_mass(q, k, self.config.q_block_size, self.config.k_block_size)
        mask, coverage = block_mass_to_mask(mass, self.config.top_p, self.config.min_top_k)
        # The device-to-host copy below is itself the stream synchronization
        # point for the kernels launched above, so no explicit
        # torch.cuda.synchronize is needed -- draining the whole device per
        # capture only stalled the pipeline the mask is meant to observe.
        mask_cpu = mask.cpu()
        self.capture_seconds += time.perf_counter() - started

        seq_len, heads, head_dim = q.shape
        key = (int(step), str(branch), int(layer))
        rel_path = Path("masks") / f"step{step:03d}_{branch}_layer{layer:02d}.pt"
        write_started = time.perf_counter()
        (self.root / rel_path).parent.mkdir(parents=True, exist_ok=True)
        packed_fields: dict[str, Any] = {}
        if self._bitpacked:
            payload, element_count = pack_bool_mask(mask_cpu)
            (self.root / rel_path).write_bytes(payload)
            packed_fields = {
                "mask_shape": [int(value) for value in mask_cpu.shape],
                "packed_elements": element_count,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        else:
            torch.save(mask_cpu, self.root / rel_path)
        self.write_seconds += time.perf_counter() - write_started
        mass_path: Path | None = None
        if self.capture_block_mass:
            mass_path = Path("block_mass") / f"step{step:03d}_{branch}_layer{layer:02d}.pt"
            (self.root / mass_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(mass.cpu(), self.root / mass_path)
        # One float materialization feeds both density readings.  Reductions stay
        # on the originating device and keep the original accumulation order so
        # the reported numbers are bit-identical to the two-copy version.
        float_mask = mask.float()
        head_densities = float_mask.mean(dim=(1, 2))
        entry = {
            "step": int(step),
            "cfg_branch": str(branch),
            "layer": int(layer),
            "path": str(rel_path),
            "shape": [int(seq_len), int(heads), int(head_dim)],
            "dtype": str(q.dtype),
            "q_blocks": int(mask.shape[1]),
            "k_blocks": int(mask.shape[2]),
            "density": float(float_mask.mean().item()),
            "coverage": coverage,
            "head_density_min": float(head_densities.min().item()),
            "head_density_max": float(head_densities.max().item()),
            "head_density_cv": float(
                (head_densities.std(unbiased=False) / head_densities.mean()).item()
                if head_densities.mean().item()
                else 0.0
            ),
        }
        entry.update(packed_fields)
        if mass_path is not None:
            entry["block_mass_path"] = str(mass_path)
            entry["block_mass_dtype"] = str(mass.dtype)
        self.entries[key] = entry
        # metadata.json is deliberately not rewritten per capture: a source run
        # performs thousands of captures and the file grows to ~600 KB, so
        # re-serializing it every time burned gigabytes of I/O for a record that
        # is only read after the stage finishes.  Entries accumulate in memory
        # and flush() persists them once.

    def flush(self) -> None:
        """Persist accumulated entries. No-op for a cache opened for reading."""
        if self._created:
            self._write_metadata()

    def _decode(self, entry: dict[str, Any]) -> tuple[torch.Tensor, int, float]:
        started = time.perf_counter()
        path = self.root / entry["path"]
        file_bytes = 0
        if self._bitpacked:
            try:
                payload = path.read_bytes()
            except FileNotFoundError as exc:
                raise BlockMaskCacheError(f"missing source block-mask file: {path}") from exc
            file_bytes = len(payload)
            expected_digest = entry.get("sha256")
            if expected_digest and hashlib.sha256(payload).hexdigest() != expected_digest:
                raise BlockMaskCacheError(f"source block-mask checksum mismatch: {path}")
            shape = tuple(int(value) for value in entry.get("mask_shape", ()))
            expected_elements = math.prod(shape) if len(shape) == 3 else 0
            if expected_elements < 1 or int(entry.get("packed_elements", 0)) != expected_elements:
                raise BlockMaskCacheError(f"invalid bit-packed source block-mask metadata: {path}")
            if len(payload) != math.ceil(expected_elements / 8):
                raise BlockMaskCacheError(f"bit-packed source block-mask byte size mismatch: {path}")
            mask = unpack_bool_mask(payload, shape)
        else:
            try:
                file_bytes = path.stat().st_size
                mask = torch.load(path, map_location="cpu", weights_only=True)
            except FileNotFoundError as exc:
                raise BlockMaskCacheError(f"missing source block-mask file: {path}") from exc
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise BlockMaskCacheError(f"source block mask is not a boolean tensor: {path}")
        return mask, file_bytes, time.perf_counter() - started

    def get(self, step: int, branch: str, layer: int, q: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any], float]:
        if self.configuration_error is not None:
            raise BlockMaskCacheError(self.configuration_error)
        if int(self.metadata.get("cfg_branches", 0)) == 1 and branch != "cond":
            raise BlockMaskCacheError("single-branch source cache cannot serve an unconditional branch")
        key = (int(step), str(branch), int(layer))
        entry = self.entries.get(key)
        if entry is None:
            raise BlockMaskCacheError(f"missing source mask for step={step}, branch={branch}, layer={layer}")
        expected_shape = [int(value) for value in q.shape]
        if entry["shape"] != expected_shape or entry["dtype"] != str(q.dtype):
            raise BlockMaskCacheError(
                f"mask metadata mismatch for step={step}, branch={branch}, layer={layer}: "
                f"cached shape/dtype={entry['shape']}/{entry['dtype']}, target={expected_shape}/{q.dtype}"
            )
        mask = self._prepared_masks.get(key)
        if mask is None:
            mask, _, elapsed = self._decode(entry)
            self._prepared_masks[key] = mask
        else:
            elapsed = 0.0
        return mask.to(device=q.device, non_blocking=True), entry, elapsed

    def prepare(self, *, device: torch.device | str | None = None) -> dict[str, Any]:
        """Decode all source masks once, optionally placing them on the target device."""
        if self.configuration_error is not None:
            raise BlockMaskCacheError(self.configuration_error)
        target_device = torch.device(device) if device is not None else None
        started = time.perf_counter()
        file_bytes = 0
        decoded_entries = 0
        for key in sorted(self.entries, key=lambda value: (value[0], value[1], value[2])):
            if key in self._prepared_masks:
                continue
            mask, read_bytes, _ = self._decode(self.entries[key])
            if target_device is not None:
                mask = mask.to(device=target_device, non_blocking=True)
            self._prepared_masks[key] = mask
            file_bytes += read_bytes
            decoded_entries += 1
        if target_device is not None and target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        return {
            "entries": len(self.entries),
            "decoded_entries": decoded_entries,
            "file_bytes": file_bytes,
            "device": str(target_device) if target_device is not None else "cpu",
            "elapsed_seconds": time.perf_counter() - started,
        }

    def prewarm(self, *, chunk_size: int = 8 * 1024 * 1024) -> dict[str, Any]:
        """Compatibility alias for eager mask decoding."""
        if chunk_size <= 0:
            raise ValueError("prewarm chunk_size must be positive")
        return self.prepare()

    def summary(self) -> dict[str, Any]:
        self.flush()
        entries = list(self.entries.values())
        return {
            "source_mask_cache_id": self.metadata.get("source_run_id"),
            "path": str(self.root),
            "format": self.metadata.get("format"),
            "cfg_branches": self.metadata.get("cfg_branches"),
            "entries": len(entries),
            "capture_seconds": self.capture_seconds,
            "write_seconds": self.write_seconds,
            "avg_density": _mean([float(entry["density"]) for entry in entries]),
            "avg_coverage": _mean([float(entry["coverage"]) for entry in entries]),
        }


class OuterSparseState:
    """Shared step/branch position and statistics for capture or target execution."""

    def __init__(self, *, cfg_branch_count: int, start_completed_step: int = 1) -> None:
        self.cfg_branch_count = int(cfg_branch_count)
        self.start_completed_step = int(start_completed_step)
        self.enabled = False
        self.current_completed_step = 0
        self._forward_index = -1
        self.branch = "cond"
        self._summary_callback: Any = None

    def reset(self) -> None:
        self._forward_index = -1
        self.branch = "cond"

    def set_step(self, completed_step: int) -> None:
        self.current_completed_step = int(completed_step)
        self.enabled = self.current_completed_step >= self.start_completed_step

    def begin_model_forward(self) -> None:
        self._forward_index += 1
        self.branch = "cond" if self.cfg_branch_count == 1 or self._forward_index % 2 == 0 else "uncond"

    def summary(self) -> dict[str, Any]:
        if self._summary_callback is not None:
            return self._summary_callback()
        return {"enabled": False}


class OuterSparseExecutor:
    """Apply source block masks to target Q/K/V using an actual sparse backend."""

    def __init__(
        self,
        cache: SourceBlockMaskCache,
        backend: str,
        *,
        flashinfer_wrapper: str = "auto",
        workspace_mib: int = 256,
        execution_dtype: torch.dtype = torch.float16,
        use_fp16_qk_reduction: bool = False,
        flashinfer_plan_cache: str = "layer",
        layer_refresh_size: int = 0,
        sparge_smooth_k: bool = True,
        sparge_pv_threshold: float = 50.0,
    ) -> None:
        if backend not in {"flashinfer", "sparge-sage2"}:
            raise ValueError(f"unsupported outer attention backend: {backend}")
        self.cache = cache
        self.backend = backend
        self.workspace_mib = int(workspace_mib)
        if execution_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"outer sparse execution dtype must be FP16 or BF16, got {execution_dtype}")
        if flashinfer_plan_cache not in {"layer", "all", "none"}:
            raise ValueError(f"unsupported FlashInfer plan cache mode: {flashinfer_plan_cache}")
        if flashinfer_wrapper not in {"auto", "block", "variable"}:
            raise ValueError(f"unsupported FlashInfer wrapper policy: {flashinfer_wrapper}")
        if layer_refresh_size < 0:
            raise ValueError("layer_refresh_size must be non-negative")
        self.execution_dtype = execution_dtype
        self.flashinfer_wrapper = flashinfer_wrapper
        self.use_fp16_qk_reduction = bool(use_fp16_qk_reduction)
        self.flashinfer_plan_cache = flashinfer_plan_cache
        self.layer_refresh_size = int(layer_refresh_size)
        self.sparge_smooth_k = bool(sparge_smooth_k)
        self.sparge_pv_threshold = float(sparge_pv_threshold)
        self.workspace: torch.Tensor | None = None
        self.wrappers: dict[tuple[Any, ...], Any] = {}
        self._refresh_masks: dict[tuple[int, str, int], torch.Tensor] = {}
        self.flashinfer_execution_modes: set[str] = set()
        self.stats: dict[str, float | int] = {
            "runs": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "planned_masks": 0,
            "plan_seconds": 0.0,
            "mask_load_seconds": 0.0,
            "density_sum": 0.0,
            "source_density_sum": 0.0,
            "source_coverage_sum": 0.0,
            "head_density_min_sum": 0.0,
            "head_density_max_sum": 0.0,
            "head_density_cv_sum": 0.0,
            "refresh_events": 0,
            "refresh_added_blocks": 0,
        }

    def reset_request(self) -> None:
        """Drop per-request layer-group coverage before target denoising."""

        self._refresh_masks.clear()

    def _apply_layer_refresh(
        self,
        mask: torch.Tensor,
        *,
        step: int,
        branch: str,
        layer: int,
    ) -> tuple[torch.Tensor, int]:
        """Ensure the final layer in each group covers blocks missed earlier.

        This is Chorus2's original layer refresh: masks remain unchanged until
        the final layer of an eight-layer group, which receives all block
        positions not selected by preceding layers in that group.
        """

        if self.layer_refresh_size == 0:
            return mask, 0
        group_start = layer // self.layer_refresh_size * self.layer_refresh_size
        max_layer = max((key[2] for key in self.cache.entries), default=layer)
        group_end = min(group_start + self.layer_refresh_size - 1, max_layer)
        key = (int(step), str(branch), group_start)
        visited = self._refresh_masks.get(key)
        if layer == group_start or visited is None:
            visited = torch.zeros_like(mask, dtype=torch.bool)
            self._refresh_masks[key] = visited
        effective_mask = mask
        added_blocks = 0
        if layer == group_end:
            missing = ~visited
            effective_mask = mask | missing
            added_blocks = int((missing & ~mask).sum().item())
            self.stats["refresh_events"] = int(self.stats["refresh_events"]) + 1
            self.stats["refresh_added_blocks"] = int(self.stats["refresh_added_blocks"]) + added_blocks
        visited |= effective_mask
        return effective_mask, added_blocks

    def _workspace(self, device: torch.device) -> torch.Tensor:
        if self.workspace is None or self.workspace.device != device:
            self.workspace = torch.empty(self.workspace_mib * 1024 * 1024, dtype=torch.uint8, device=device)
        return self.workspace

    @staticmethod
    def _block_sizes(
        seq_len: int, blocks: int, block_size: int, num_heads: int, device: torch.device
    ) -> torch.Tensor:
        tail = seq_len - (blocks - 1) * block_size
        if blocks < 1 or tail < 1 or tail > block_size:
            raise ValueError(
                "cached block geometry is incompatible with the runtime sequence length: "
                f"seq_len={seq_len}, blocks={blocks}, block_size={block_size}"
            )
        sizes = torch.full((blocks,), block_size, dtype=torch.int32, device=device)
        sizes[-1] = tail
        return sizes.unsqueeze(0).expand(num_heads, -1).contiguous()

    def _flashinfer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        *,
        step: int,
        branch: str,
        layer: int,
    ) -> torch.Tensor:
        try:
            _prepare_flashinfer_build_tools()
            from flashinfer import sparse
        except ImportError as exc:
            raise RuntimeError("flashinfer backend requested but flashinfer is unavailable") from exc
        seq_len, heads, head_dim = q.shape
        if self.flashinfer_wrapper == "variable":
            q_block_size, k_block_size = self.cache.config.q_block_size, self.cache.config.k_block_size
            runtime_mask = mask
            aligned = False
        elif self.flashinfer_wrapper == "block":
            q_block_size, k_block_size = self.cache.config.q_block_size, self.cache.config.k_block_size
            if seq_len % q_block_size or seq_len % k_block_size:
                raise ValueError(
                    "FlashInfer block wrapper requires a sequence length divisible by the cached block sizes; "
                    f"got seq_len={seq_len}, Q_block_size={q_block_size}, K_block_size={k_block_size}"
                )
            runtime_mask = mask
            aligned = True
        else:
            q_block_size, k_block_size = flashinfer_block_layout(self.cache.config, seq_len)
            runtime_mask = refine_block_mask(
                mask,
                seq_len,
                self.cache.config.q_block_size,
                self.cache.config.k_block_size,
                q_block_size,
                k_block_size,
            )
            aligned = seq_len % q_block_size == 0 and seq_len % k_block_size == 0
        full_key = (
            "flashinfer",
            step,
            branch,
            layer,
            seq_len,
            heads,
            head_dim,
            q_block_size,
            k_block_size,
            str(q.device),
            str(self.execution_dtype),
        )
        if self.flashinfer_plan_cache == "all":
            wrapper_key: tuple[Any, ...] | None = full_key
        elif self.flashinfer_plan_cache == "layer":
            wrapper_key = (
                "flashinfer",
                layer,
                seq_len,
                heads,
                head_dim,
                q_block_size,
                k_block_size,
                str(q.device),
                str(self.execution_dtype),
            )
        else:
            wrapper_key = None
        wrapper = self.wrappers.get(wrapper_key) if wrapper_key is not None else None
        # ``layer`` reuses the allocated wrapper/workspace, but each source
        # mask still needs a fresh plan.  FlashInfer binds the sparse layout
        # in ``plan``; skipping it would silently execute a stale mask.  This
        # mirrors Chorus2's wrapper reuse semantics.
        plan_required = wrapper is None or self.flashinfer_plan_cache != "all"
        self.flashinfer_execution_modes.add("packed_bsr" if aligned else "variable_block")
        if wrapper is None:
            wrapper = (
                sparse.BlockSparseAttentionWrapper(self._workspace(q.device))
                if aligned
                else sparse.VariableBlockSparseAttentionWrapper(self._workspace(q.device))
            )
        if plan_required:
            started = time.perf_counter()
            if aligned:
                indptr, indices, packed_seq_len = build_flashinfer_packed_bsr(
                    runtime_mask,
                    seq_len,
                    q_block_size,
                    k_block_size,
                    q.device,
                )
                wrapper.plan(
                    indptr,
                    indices,
                    packed_seq_len,
                    packed_seq_len,
                    q_block_size,
                    k_block_size,
                    1,
                    1,
                    head_dim,
                    sm_scale=head_dim**-0.5,
                    q_data_type=self.execution_dtype,
                    kv_data_type=self.execution_dtype,
                    o_data_type=self.execution_dtype,
                    use_fp16_qk_reduction=self.use_fp16_qk_reduction,
                )
            else:
                runtime_mask = runtime_mask.to(device=q.device, dtype=torch.bool, non_blocking=True)
                wrapper.plan(
                    runtime_mask,
                    self._block_sizes(seq_len, runtime_mask.size(1), q_block_size, heads, q.device),
                    self._block_sizes(seq_len, runtime_mask.size(2), k_block_size, heads, q.device),
                    heads,
                    heads,
                    head_dim,
                    sm_scale=head_dim**-0.5,
                    q_data_type=self.execution_dtype,
                    kv_data_type=self.execution_dtype,
                    use_fp16_qk_reduction=self.use_fp16_qk_reduction,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize(q.device)
            self.stats["plan_seconds"] = float(self.stats["plan_seconds"]) + time.perf_counter() - started
            self.stats["planned_masks"] = int(self.stats["planned_masks"]) + 1
            if wrapper_key is not None:
                self.wrappers[wrapper_key] = wrapper
        if k.shape != q.shape or v.shape != q.shape:
            raise ValueError("FlashInfer packed BSR expects Q/K/V with the same [sequence, heads, head_dim] shape")
        if not aligned:
            return wrapper.run(
                q.transpose(0, 1).contiguous(),
                k.transpose(0, 1).contiguous(),
                v.transpose(0, 1).contiguous(),
            ).transpose(0, 1).contiguous()
        packed_seq_len = seq_len * heads
        q_packed = q.transpose(0, 1).contiguous().reshape(packed_seq_len, 1, head_dim)
        k_packed = k.transpose(0, 1).contiguous().reshape(packed_seq_len, 1, head_dim)
        v_packed = v.transpose(0, 1).contiguous().reshape(packed_seq_len, 1, head_dim)
        output = wrapper.run(q_packed, k_packed, v_packed)
        return output.reshape(heads, seq_len, head_dim).transpose(0, 1).contiguous()

    def _sparge(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError("sparge-sage2 backend requires CUDA")
        capability = torch.cuda.get_device_capability(q.device)
        if capability != (9, 0):
            raise RuntimeError(f"sparge-sage2 requires sm90, got capability={capability}")
        if (self.cache.config.q_block_size, self.cache.config.k_block_size) != (64, 128):
            raise ValueError("sparge-sage2 requires Q=64 and K=128 block sizes")
        try:
            from spas_sage_attn import block_sparse_sage2_attn_cuda
        except ImportError as exc:
            raise RuntimeError("sparge-sage2 backend requested but SpargeAttn is unavailable") from exc
        mask = mask.to(device=q.device, dtype=torch.bool, non_blocking=True)
        return block_sparse_sage2_attn_cuda(
            q.transpose(0, 1).unsqueeze(0).contiguous(),
            k.transpose(0, 1).unsqueeze(0).contiguous(),
            v.transpose(0, 1).unsqueeze(0).contiguous(),
            mask_id=mask.unsqueeze(0).contiguous(),
            smooth_k=self.sparge_smooth_k,
            pvthreshd=self.sparge_pv_threshold,
            tensor_layout="HND",
        )[0].transpose(0, 1).contiguous()

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, step: int, branch: str, layer: int) -> torch.Tensor:
        output_dtype = q.dtype
        q = q.to(dtype=self.execution_dtype)
        k = k.to(dtype=self.execution_dtype)
        v = v.to(dtype=self.execution_dtype)
        try:
            mask, entry, load_seconds = self.cache.get(step, branch, layer, q)
        except BlockMaskCacheError:
            self.stats["cache_misses"] = int(self.stats["cache_misses"]) + 1
            raise
        self.stats["cache_hits"] = int(self.stats["cache_hits"]) + 1
        self.stats["mask_load_seconds"] = float(self.stats["mask_load_seconds"]) + load_seconds
        mask, added_blocks = self._apply_layer_refresh(mask, step=step, branch=branch, layer=layer)
        out = (
            self._flashinfer(q, k, v, mask, step=step, branch=branch, layer=layer)
            if self.backend == "flashinfer"
            else self._sparge(q, k, v, mask)
        )
        self.stats["runs"] = int(self.stats["runs"]) + 1
        source_density = float(entry["density"])
        density = source_density + added_blocks / mask.numel()
        for stat, field in (
            ("density_sum", "runtime_density"),
            ("source_density_sum", "density"),
            ("source_coverage_sum", "coverage"),
            ("head_density_min_sum", "head_density_min"),
            ("head_density_max_sum", "head_density_max"),
            ("head_density_cv_sum", "head_density_cv"),
        ):
            if field == "runtime_density":
                value = density
            else:
                default = source_density if field.startswith("head_density_") and field != "head_density_cv" else 0.0
                value = float(entry.get(field, default))
            self.stats[stat] = float(self.stats[stat]) + value
        return out.to(dtype=output_dtype)

    def summary(self) -> dict[str, Any]:
        runs = int(self.stats["runs"])
        return {
            "backend": self.backend,
            "execution_dtype": str(self.execution_dtype),
            "source_mask_cache_id": self.cache.metadata.get("source_run_id"),
            "source_mask_cache_path": str(self.cache.root),
            "quantization": "QK INT8 and PV FP8" if self.backend == "sparge-sage2" else None,
            "runs": runs,
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "planned_masks": self.stats["planned_masks"],
            "plan_seconds": self.stats["plan_seconds"],
            "mask_load_seconds": self.stats["mask_load_seconds"],
            "routing_time_ms": 0.0,
            "avg_runtime_density": float(self.stats["density_sum"]) / runs if runs else None,
            "avg_source_per_head_density": float(self.stats["source_density_sum"]) / runs if runs else None,
            "avg_source_gt_coverage": float(self.stats["source_coverage_sum"]) / runs if runs else None,
            "avg_head_density_min": float(self.stats["head_density_min_sum"]) / runs if runs else None,
            "avg_head_density_max": float(self.stats["head_density_max_sum"]) / runs if runs else None,
            "avg_head_density_cv": float(self.stats["head_density_cv_sum"]) / runs if runs else None,
            "flashinfer_plan_cache": self.flashinfer_plan_cache if self.backend == "flashinfer" else None,
            "flashinfer_wrapper_policy": self.flashinfer_wrapper if self.backend == "flashinfer" else None,
            "flashinfer_execution_modes": sorted(self.flashinfer_execution_modes) if self.backend == "flashinfer" else [],
            "layer_refresh_size": self.layer_refresh_size,
            "refresh_events": self.stats["refresh_events"],
            "refresh_added_blocks": self.stats["refresh_added_blocks"],
            "use_fp16_qk_reduction": self.use_fp16_qk_reduction if self.backend == "flashinfer" else None,
            "sparge_smooth_k": self.sparge_smooth_k if self.backend == "sparge-sage2" else None,
            "sparge_pvthreshd": self.sparge_pv_threshold if self.backend == "sparge-sage2" else None,
        }


class OuterSparseCaptureHooks(AbstractContextManager):
    """Capture source RoPE Q/K masks while retaining the native dense output."""

    def __init__(
        self,
        model: Any,
        model_module: Any,
        state: OuterSparseState,
        cache: SourceBlockMaskCache,
        layers: list[int],
        *,
        execution_dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.model_module = model_module
        self.state = state
        self.cache = cache
        self.layers = list(layers)
        self.execution_dtype = execution_dtype
        self.originals: list[tuple[Any, Any]] = []
        self.position_handle: Any = None

    def __enter__(self):
        self.state.reset()
        self.state._summary_callback = lambda: {
            "enabled": True,
            "mode": "source_capture",
            "layers": self.layers,
            **self.cache.summary(),
        }
        self.position_handle = self.model.blocks[0].self_attn.register_forward_pre_hook(
            lambda _module, _args: self.state.begin_model_forward()
        )
        for layer in self.layers:
            attn = self.model.blocks[layer].self_attn
            original = attn.forward
            self.originals.append((attn, original))

            def capture_forward(attn_self, x, seq_lens, grid_sizes, freqs, *, layer_index=layer, original_forward=original):
                if not self.state.enabled:
                    return original_forward(x, seq_lens, grid_sizes, freqs)
                batch, seq_len = x.shape[:2]
                if batch != 1 or int(seq_lens[0].item()) != seq_len:
                    raise ValueError("source block-mask capture requires an unpadded batch of one")
                heads, head_dim = attn_self.num_heads, attn_self.head_dim
                q = attn_self.norm_q(attn_self.q(x)).view(batch, seq_len, heads, head_dim)
                k = attn_self.norm_k(attn_self.k(x)).view(batch, seq_len, heads, head_dim)
                v = attn_self.v(x).view(batch, seq_len, heads, head_dim)
                q = self.model_module.rope_apply(q, grid_sizes, freqs)
                k = self.model_module.rope_apply(k, grid_sizes, freqs)
                out = self.model_module.flash_attention(q=q, k=k, v=v, k_lens=seq_lens, window_size=attn_self.window_size)
                self.cache.capture(
                    self.state.current_completed_step,
                    self.state.branch,
                    layer_index,
                    q[0].to(dtype=self.execution_dtype),
                    k[0].to(dtype=self.execution_dtype),
                )
                return attn_self.o(out.flatten(2))

            attn.forward = types.MethodType(capture_forward, attn)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.position_handle is not None:
            self.position_handle.remove()
            self.position_handle = None
        for attn, original in reversed(self.originals):
            attn.forward = original
        self.originals.clear()
        # Every production capture goes through this context manager, so this is
        # the single point that persists metadata.json (see capture()).
        self.cache.flush()
        return False


class OuterSparseAttentionHooks(AbstractContextManager):
    """Replace only configured outer Wan self-attention layers during a target run."""

    def __init__(self, model: Any, model_module: Any, state: OuterSparseState, executor: OuterSparseExecutor, layers: list[int]) -> None:
        self.model = model
        self.model_module = model_module
        self.state = state
        self.executor = executor
        self.layers = list(layers)
        self.originals: list[tuple[Any, Any]] = []
        self.position_handle: Any = None
        self.dense_fallbacks = 0

    def __enter__(self):
        self.state.reset()
        self.executor.reset_request()
        self.state._summary_callback = self.summary
        self.position_handle = self.model.blocks[0].self_attn.register_forward_pre_hook(
            lambda _module, _args: self.state.begin_model_forward()
        )
        for layer in self.layers:
            attn = self.model.blocks[layer].self_attn
            original = attn.forward
            self.originals.append((attn, original))

            def sparse_forward(attn_self, x, seq_lens, grid_sizes, freqs, *, layer_index=layer, original_forward=original):
                if not self.state.enabled:
                    return original_forward(x, seq_lens, grid_sizes, freqs)
                batch, seq_len = x.shape[:2]
                if batch != 1 or int(seq_lens[0].item()) != seq_len:
                    raise ValueError("outer sparse attention requires an unpadded batch of one")
                heads, head_dim = attn_self.num_heads, attn_self.head_dim
                q = attn_self.norm_q(attn_self.q(x)).view(batch, seq_len, heads, head_dim)
                k = attn_self.norm_k(attn_self.k(x)).view(batch, seq_len, heads, head_dim)
                v = attn_self.v(x).view(batch, seq_len, heads, head_dim)
                q = self.model_module.rope_apply(q, grid_sizes, freqs)[0].contiguous()
                k = self.model_module.rope_apply(k, grid_sizes, freqs)[0].contiguous()
                try:
                    out = self.executor.run(
                        q,
                        k,
                        v[0].contiguous(),
                        step=self.state.current_completed_step,
                        branch=self.state.branch,
                        layer=layer_index,
                    )
                except BlockMaskCacheError:
                    self.dense_fallbacks += 1
                    return original_forward(x, seq_lens, grid_sizes, freqs)
                return attn_self.o(out.unsqueeze(0).flatten(2))

            attn.forward = types.MethodType(sparse_forward, attn)
        return self

    def summary(self) -> dict[str, Any]:
        payload = self.executor.summary()
        payload.update({"enabled": True, "layers": self.layers, "dense_fallback_count": self.dense_fallbacks})
        return payload

    def __exit__(self, exc_type, exc, tb):
        if self.position_handle is not None:
            self.position_handle.remove()
            self.position_handle = None
        for attn, original in reversed(self.originals):
            attn.forward = original
        self.originals.clear()
        return False


def dependency_status(backend: str, device: torch.device | None = None) -> dict[str, Any]:
    """Report optional sparse backend availability without importing it at module load."""

    status: dict[str, Any] = {"backend": backend, "available": False}
    try:
        if backend == "flashinfer":
            import flashinfer

            status.update({"available": True, "version": getattr(flashinfer, "__version__", "unknown")})
        elif backend == "sparge-sage2":
            from spas_sage_attn import block_sparse_sage2_attn_cuda  # noqa: F401

            status["available"] = True
            if torch.cuda.is_available():
                status["capability"] = list(torch.cuda.get_device_capability(device))
                status["sm90"] = tuple(status["capability"]) == (9, 0)
        elif backend == "dense":
            status["available"] = True
        else:
            status["error"] = f"unsupported backend: {backend}"
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status
