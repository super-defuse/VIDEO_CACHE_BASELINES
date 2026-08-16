from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CacheAttentionState:
    """State shared by model-specific attention patches during one denoising step."""

    cached_tokens: Any | None = None
    active: bool = False

    def set_cached_tokens(self, tokens: Any | None, active: bool) -> None:
        self.cached_tokens = tokens
        self.active = active and tokens is not None


def cached_attention(query: Any, key: Any, value: Any, cached_key: Any, cached_value: Any) -> Any:
    """Attend query tokens to the concatenated current and cached K/V tensors.

    Tensors are shaped ``[batch, heads, tokens, head_dim]``.  The cache is
    intentionally only a K/V source: prompt-specific tokens remain the queries.
    """
    import torch
    import torch.nn.functional as functional

    keys = torch.cat((key, cached_key), dim=-2)
    values = torch.cat((value, cached_value), dim=-2)
    return functional.scaled_dot_product_attention(query, keys, values, dropout_p=0.0, is_causal=False)


def head_view(tensor: Any, heads: int) -> Any:
    batch, tokens, channels = tensor.shape
    if channels % heads:
        raise ValueError(f"channels ({channels}) must divide heads ({heads})")
    return tensor.view(batch, tokens, heads, channels // heads).transpose(1, 2)


def merge_heads(tensor: Any) -> Any:
    return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], -1)


def cache_tokens_like(current_tokens: Any, cached_tokens: Any) -> Any:
    """Validate cached activation shape before entering a model attention block."""
    if cached_tokens.ndim != 3 or cached_tokens.shape[0] != current_tokens.shape[0]:
        raise ValueError("cached tokens must be [batch, tokens, channels]")
    if cached_tokens.shape[-1] != current_tokens.shape[-1]:
        raise ValueError("cached and current token channels differ")
    return cached_tokens.to(device=current_tokens.device, dtype=current_tokens.dtype)
