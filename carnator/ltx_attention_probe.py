"""LTX-Video multi-noun cross-attention probe for the Carnator fusion mask.

This is ``NativeMultiNounCrossAttentionProbe`` (``carnator/adapters/wan_ti2v_5b.py``)
transposed onto LTX's attention modules.  The algorithm is unchanged: for each
canonical noun, take the raw pre-softmax dot product between every video query
and that noun's caption keys, scale by ``head_dim ** -0.5``, average over
batches, heads and the noun's key tokens, and keep the resulting ``[T, H, W]``
volume.  There is no softmax and no attention mask, so the caption padding mask
is irrelevant -- only the rows named by ``positive_indices_by_noun`` are read.

Two things differ from Wan and are worth naming:

* Wan's cross-attention is a separate ``cross_attn`` module with ``(x, context)``
  inputs; LTX's is ``attn2`` with ``hidden_states`` positional and
  ``encoder_hidden_states`` as a keyword, so the hook is a keyword-aware
  pre-hook.
* BLTX runs a batched CFG/STG forward, so ``hidden_states`` can carry three
  branch rows.  The probe deliberately reads only the plain ``cond`` row: the
  uncond row has different tokens and the perturbed row has been through the
  STG skip substitution, and averaging them would smear the mask.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Mapping, Sequence


class LtxMultiNounCrossAttentionProbe(AbstractContextManager):
    """Capture one cross-attention score volume per canonical noun in one pass."""

    def __init__(
        self,
        transformer: Any,
        layers: Sequence[int],
        positive_indices_by_noun: Mapping[str, Sequence[int]],
        token_grid: tuple[int, int, int],
        *,
        guidance: Any = None,
    ) -> None:
        if not positive_indices_by_noun:
            raise ValueError("multi-noun cross-attention probe needs at least one noun")
        self.transformer = transformer
        self.layers = [int(item) for item in layers]
        self.positive_indices_by_noun = {
            str(noun): list(indices) for noun, indices in positive_indices_by_noun.items()
        }
        self.token_grid = (int(token_grid[0]), int(token_grid[1]), int(token_grid[2]))
        # The delegate's per-step branch identity; without it the probe falls
        # back to the conventional [uncond, cond] ordering.
        self.guidance = guidance
        self.enabled = False
        self.current_step_index = -1
        self.records_by_noun: dict[str, list[dict[str, Any]]] = {
            noun: [] for noun in self.positive_indices_by_noun
        }
        self.handles: list[Any] = []

    def __enter__(self) -> "LtxMultiNounCrossAttentionProbe":
        for layer_index in self.layers:
            attention = self.transformer.transformer_blocks[layer_index].attn2
            self.handles.append(
                attention.register_forward_pre_hook(self._make_hook(layer_index), with_kwargs=True)
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def _conditioning_row(self, batch: int) -> int:
        """Which batch row holds the plain conditional branch."""
        labels = (
            tuple(self.guidance.branches)
            if self.guidance is not None and self.guidance.branches is not None
            else ("uncond", "cond") if batch >= 2 else ("cond",)
        )
        if len(labels) != batch:
            labels = ("uncond", "cond") if batch >= 2 else ("cond",)
        try:
            return labels.index("cond")
        except ValueError:
            return 0

    def _make_hook(self, layer_index: int):
        def hook(attention: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if not self.enabled:
                return
            import torch

            hidden_states = args[0] if args else kwargs.get("hidden_states")
            text = kwargs.get("encoder_hidden_states")
            if hidden_states is None or text is None:
                return
            row = self._conditioning_row(hidden_states.size(0))
            hidden_states = hidden_states[row: row + 1]
            text = text[row: row + 1]

            batch, length = hidden_states.size(0), hidden_states.size(1)
            heads = int(attention.heads)
            head_dim = hidden_states.size(-1) // heads
            # LTX normalises across all heads on the flat tensor, so the split
            # has to happen after norm_q/norm_k, exactly as the native
            # LTXVideoAttnProcessor does it.
            query = attention.norm_q(attention.to_q(hidden_states))
            key = attention.norm_k(attention.to_k(text))
            query = query.view(batch, length, heads, head_dim)
            key = key.view(text.size(0), text.size(1), heads, head_dim)

            grid = self.token_grid
            for noun, indices in self.positive_indices_by_noun.items():
                positive = torch.tensor(indices, device=key.device, dtype=torch.long)
                positive_keys = key.index_select(1, positive)
                score = (
                    torch.einsum("bqhd,bkhd->bhqk", query, positive_keys)
                    .mul(head_dim ** -0.5)
                    .mean(dim=(0, 1, 3))
                )
                self.records_by_noun[noun].append({
                    "step_index": int(self.current_step_index),
                    "layer_index": int(layer_index),
                    "positive": score[: grid[0] * grid[1] * grid[2]]
                    .reshape(*grid).detach().float().cpu(),
                })

        return hook


class LtxMultiNounSelfAttentionContrastProbe(AbstractContextManager):
    """Capture the self-attention noun/background contrast at one layer.

    ``NativeMultiNounSelfAttentionContrastProbe`` (``carnator/adapters/wan_ti2v_5b.py``)
    ported to LTX ``attn1``.  Cross attention only proposes seed locations; the
    persisted Mbase comes from this contrast, so the two probes are always used
    as a pair.

    The probe is single-shot: it records on the first forward it sees and stays
    quiet afterwards, because the caller runs one dedicated conditioning-only
    forward for it rather than reusing a sampling step (the sampled forward is
    batched over CFG/STG branches and carries the STG value substitution on the
    skipped blocks, either of which would contaminate the map).
    """

    def __init__(
        self,
        transformer: Any,
        layer: int,
        seed_indices_by_noun: Mapping[str, Any],
        token_grid: tuple[int, int, int],
        *,
        chunk_size: int = 512,
    ) -> None:
        if not seed_indices_by_noun:
            raise ValueError("multi-noun self-attention probe needs at least one noun")
        self.transformer = transformer
        self.layer = int(layer)
        self.seed_indices_by_noun = dict(seed_indices_by_noun)
        self.token_grid = (int(token_grid[0]), int(token_grid[1]), int(token_grid[2]))
        self.token_count = self.token_grid[0] * self.token_grid[1] * self.token_grid[2]
        self.chunk_size = int(chunk_size)
        self.records_by_noun: dict[str, dict[str, Any]] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> "LtxMultiNounSelfAttentionContrastProbe":
        attention = self.transformer.transformer_blocks[self.layer].attn1
        self.handles.append(
            attention.register_forward_pre_hook(self._hook, with_kwargs=True)
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def _hook(self, attention: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if self.records_by_noun:
            return
        import torch

        from adapters.ltx_attention import ltx_qkv

        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            raise RuntimeError("LTX self-attention contrast probe saw no hidden states")
        # The dedicated probe forward is conditioning-only, so there is one row.
        query, key, _value = ltx_qkv(
            attention, hidden_states[:1], kwargs.get("image_rotary_emb")
        )
        head_dim = key.shape[-1]
        scale = head_dim ** -0.5
        for noun, (noun_indices, background_indices) in self.seed_indices_by_noun.items():
            noun_key = key.index_select(2, noun_indices.to(device=key.device, dtype=torch.long))
            background_key = key.index_select(
                2, background_indices.to(device=key.device, dtype=torch.long)
            )
            noun_parts: list[Any] = []
            background_parts: list[Any] = []
            # Chunked over queries: the key set is the noun's 2% seed, so a chunk
            # is [1, heads, chunk, seeds] and never the full token-by-token map.
            for start in range(0, self.token_count, self.chunk_size):
                stop = min(self.token_count, start + self.chunk_size)
                query_chunk = query[:, :, start:stop]
                noun_logits = torch.einsum("bhsd,bhkd->bhsk", query_chunk, noun_key).mul(scale)
                background_logits = torch.einsum(
                    "bhsd,bhkd->bhsk", query_chunk, background_key
                ).mul(scale)
                noun_parts.append(
                    noun_logits.max(dim=3).values.mean(dim=(0, 1)).detach().float().cpu()
                )
                background_parts.append(
                    background_logits.max(dim=3).values.mean(dim=(0, 1)).detach().float().cpu()
                )
                del noun_logits, background_logits
            noun_max = torch.cat(noun_parts).reshape(*self.token_grid)
            background_max = torch.cat(background_parts).reshape(*self.token_grid)
            self.records_by_noun[noun] = {
                "self_noun_max": noun_max,
                "self_bg_max": background_max,
                "self_contrast_max": noun_max - background_max,
            }

    def summary(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "token_grid": list(self.token_grid),
            "nouns": {
                noun: {
                    "contrast_min": float(record["self_contrast_max"].min().item()),
                    "contrast_max": float(record["self_contrast_max"].max().item()),
                    "contrast_mean": float(record["self_contrast_max"].mean().item()),
                }
                for noun, record in self.records_by_noun.items()
            },
        }
