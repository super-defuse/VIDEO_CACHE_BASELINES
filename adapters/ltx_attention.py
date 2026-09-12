"""LTX-Video attention hooks: regional execution (SRD) and TGAA.

LTX differs from Wan and CogVideoX in one way that matters here: it has two
separate attention modules per block.  ``attn1`` is video self-attention with
3D RoPE, ``attn2`` is cross-attention to the T5 caption and carries the
padding mask.  Neither is a joint text+video attention, so the CogVideoX
``joint_qkv``/``project_joint`` shape does not apply and the Wan2.1 native
block is the closer reference (``Wan21Adapter._native_srd_block``).

The patched block reproduces the stock LTX block exactly, restricting the
self-attention queries to Medit and its keys/values to Msee, running the
cross-attention and the FFN on the Medit rows only, and writing those rows
back into the untouched hidden state.  Positions outside Medit therefore stay
bit-identical to a dense step, which is what lets Stage 3 repair the Stage 2
discontinuities.
"""

from __future__ import annotations

import contextlib
import types
from dataclasses import dataclass, field
from typing import Any, Callable

# The reference implementation's TGAA constants live on the state, not the
# config, because gamma_k/gamma_o are already resolved per method variant.
DEFAULT_GAMMA_K = 1.5
DEFAULT_GAMMA_O = 1.5


def ltx_qkv(attention: Any, hidden: Any, rotary: Any = None) -> tuple[Any, Any, Any]:
    """Q/K/V as ``[B, heads, S, head_dim]``, in the native norm/ROPE order.

    ``norm_q``/``norm_k`` are RMSNorm across *all* heads jointly and are applied
    to the flat ``[B, S, inner_dim]`` tensor before RoPE and before the head
    split, exactly as ``LTXVideoAttnProcessor`` does.  Doing it in another order
    would change the result.
    """
    from diffusers.models.transformers.transformer_ltx import apply_rotary_emb

    query = attention.to_q(hidden)
    key = attention.to_k(hidden)
    value = attention.to_v(hidden)
    query = attention.norm_q(query)
    key = attention.norm_k(key)
    if rotary is not None:
        query = apply_rotary_emb(query, rotary)
        key = apply_rotary_emb(key, rotary)
    query = query.unflatten(2, (attention.heads, -1)).transpose(1, 2)
    key = key.unflatten(2, (attention.heads, -1)).transpose(1, 2)
    value = value.unflatten(2, (attention.heads, -1)).transpose(1, 2)
    return query, key, value


def ltx_project(attention: Any, flat: Any) -> Any:
    """``[B, S, heads * head_dim]`` through ``to_out`` into ``[B, S, dim]``."""
    return attention.to_out[1](attention.to_out[0](flat))


def ltx_projection(attention: Any, output: Any) -> Any:
    """``[B, heads, S, head_dim]`` back through ``to_out`` into ``[B, S, dim]``.

    ``dispatch_attention_fn`` returns ``[B, S, heads, head_dim]`` and the native
    processor flattens from there; the explicit transpose here is the same
    permutation of the same values.
    """
    return ltx_project(attention, output.transpose(1, 2).flatten(2, 3))


def stg_mix(attention: Any, hidden: Any, attended: Any, mask: Any) -> Any:
    """Apply the STG ``attention_values`` substitution to one block.

    For every batch row whose ``mask`` entry is 0 -- the perturbation branch --
    the self-attention output is discarded and the *raw* value projection takes
    its place, **before** ``to_out``.  That is exactly what the reference does
    for the blocks named in ``skip_block_list``::

        hidden_states = hidden_states_a * skip_layer_mask + value_for_stg * (1 - skip_layer_mask)
        hidden_states = attn.to_out[0](hidden_states)

    ``attended`` is the flattened ``[B, S, inner]`` attention output and
    ``hidden`` the states the value projection is taken from -- for the regional
    path those are the Medit rows only, so the substitution is taken over the
    same token set the attention ran on.  Returns ``attended`` untouched when the
    block is not being skipped.
    """
    if mask is None:
        return attended
    value = attention.to_v(hidden)
    weight = mask.reshape(-1, 1, 1).to(dtype=attended.dtype, device=attended.device)
    return attended * weight + value.to(attended.dtype) * (1.0 - weight)


def ltx_cross_attention(
    attention: Any,
    hidden: Any,
    text: Any,
    attention_mask: Any,
    state: "LtxRegionalState | None" = None,
) -> Any:
    """``attn2`` with optional TGAA key/output rescaling.

    Mirrors ``LTXVideoAttnProcessor.__call__`` for the cross-attention case
    (no RoPE, mask present).  TGAA is folded in here rather than installed as a
    second patch because it is active on exactly the same steps as SRD -- both
    are Stage 2 only -- which is how the Wan2.1 adapter drives them too.
    """
    import torch
    import torch.nn.functional as functional

    batch_size, sequence_length, _ = text.shape
    mask = attention_mask
    if mask is not None:
        mask = attention.prepare_attention_mask(mask, sequence_length, batch_size)
        mask = mask.view(batch_size, attention.heads, -1, mask.shape[-1])

    query = attention.norm_q(attention.to_q(hidden))
    key = attention.norm_k(attention.to_k(text))
    value = attention.to_v(text)

    output_scale = 1.0
    if state is not None and state.tgaa_active:
        scale_k, scale_o = state.scales()
        # ``to_k`` returns a fresh tensor, so scaling in place cannot leak into
        # later blocks the way an in-place scale on a shared context would.
        valid = [index for index in state.differential_tokens if 0 <= index < key.shape[1]]
        if valid:
            key[:, valid] *= scale_k
        output_scale = scale_o

    query = query.unflatten(2, (attention.heads, -1)).transpose(1, 2)
    key = key.unflatten(2, (attention.heads, -1)).transpose(1, 2)
    value = value.unflatten(2, (attention.heads, -1)).transpose(1, 2)
    output = functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    output = ltx_projection(attention, output)
    return output * output_scale


@dataclass
class LtxStepTracker:
    """The sampler step an attention hook is currently executing in.

    Driven by ``LtxVideoPipelineDelegate.denoise(begin_step_hook=...)`` rather
    than by counting transformer forwards: forwards are one per step regardless
    of how many CFG/STG branches they carry, and a resumed run (Nirvana Vid)
    starts mid-schedule, so neither a counter nor a call count is the step index.
    """

    start_step: int = 0
    current_step: int = 0

    def set_step(self, index: int) -> None:
        self.current_step = int(index)

    @property
    def computing_step(self) -> int:
        """1-based index of the step being denoised, matching the cached keys."""
        return self.current_step + 1


@dataclass
class LtxRegionalState:
    """Medit/Msee token regions, the TGAA token set, and the SRD step window."""

    video_edit: Any
    video_see: Any
    first_step: int
    last_step: int
    total_steps: int
    differential_tokens: tuple[int, ...] = ()
    gamma_k: float = DEFAULT_GAMMA_K
    gamma_o: float = DEFAULT_GAMMA_O
    tgaa_enabled: bool = True
    active: bool = False
    tgaa_active: bool = False
    current_step: int = 0
    calls: int = 0
    tgaa_calls: int = 0
    # The delegate's per-step STG state; ``None`` for a run with STG off.
    guidance: Any = None
    layer: int = 0
    _indices: dict[tuple[str, int], tuple[Any, Any]] = field(default_factory=dict)

    def set_step(self, step: int) -> None:
        self.current_step = int(step)
        self.active = self.first_step <= self.current_step < self.last_step
        self.tgaa_active = self.active and self.tgaa_enabled

    def indices(self, device: Any, video_tokens: int) -> tuple[Any, Any]:
        """Cached ``(Medit, Msee)`` flat token indices, with Msee ⊇ Medit."""
        import torch

        key = (str(device), int(video_tokens))
        if key not in self._indices:
            edit = self.video_edit.flatten().bool()
            see = self.video_see.flatten().bool() | edit
            if edit.numel() != video_tokens or see.numel() != video_tokens:
                raise ValueError(
                    f"LTX region mask has {edit.numel()} tokens but the transformer has {video_tokens}"
                )
            edit_indices = torch.nonzero(edit, as_tuple=False).flatten().to(device)
            see_indices = torch.nonzero(see, as_tuple=False).flatten().to(device)
            if edit_indices.numel() == 0 or see_indices.numel() == 0:
                raise ValueError("LTX regional attention requires non-empty Medit and Msee")
            self._indices[key] = edit_indices, see_indices
        return self._indices[key]

    def scales(self) -> tuple[float, float]:
        """Linear step decay for F_k/F_o from the reference implementation."""
        denominator = max(1, self.total_steps - 1)
        progress = min(1.0, max(0.0, self.current_step / denominator))
        return (
            1.0 + (self.gamma_k - 1.0) * (1.0 - progress),
            1.0 + (self.gamma_o - 1.0) * (1.0 - progress),
        )


def ltx_regional_block(
    block: Any,
    hidden_states: Any,
    encoder_hidden_states: Any,
    temb: Any,
    image_rotary_emb: Any,
    encoder_attention_mask: Any,
    state: LtxRegionalState,
) -> Any:
    """One LTX block computed on the Medit rows only.

    The AdaRMS modulation, attention arithmetic and residual order are copied
    from ``LTXVideoTransformerBlock.forward``; only the token set differs, and
    the non-Medit rows are copied through unchanged.
    """
    import torch.nn.functional as functional

    batch_size = hidden_states.size(0)
    ada_values = block.scale_shift_table[None, None].to(temb.device) + temb.reshape(
        batch_size, temb.size(1), block.scale_shift_table.shape[0], -1
    )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
    norm_hidden = block.norm1(hidden_states) * (1 + scale_msa) + shift_msa

    edit_indices, see_indices = state.indices(hidden_states.device, hidden_states.shape[1])
    query, key, value = ltx_qkv(block.attn1, norm_hidden, image_rotary_emb)
    attended = functional.scaled_dot_product_attention(
        query.index_select(2, edit_indices),
        key.index_select(2, see_indices),
        value.index_select(2, see_indices),
        dropout_p=0.0,
        is_causal=False,
    )
    attended = attended.transpose(1, 2).flatten(2, 3)
    if state.guidance is not None:
        # STG's perturbation branch runs the regional path too and only differs
        # in the skipped blocks, where its attention output is replaced by the
        # raw value projection over the same Medit rows.
        attended = stg_mix(
            block.attn1,
            norm_hidden.index_select(1, edit_indices),
            attended,
            state.guidance.layer(state.layer, batch_size),
        )
    edited = hidden_states.index_select(1, edit_indices) + ltx_project(block.attn1, attended) * gate_msa

    edited = edited + ltx_cross_attention(
        block.attn2, edited, encoder_hidden_states, encoder_attention_mask, state
    )
    state.calls += 1
    state.tgaa_calls += 1 if state.tgaa_active else 0

    norm_edited = block.norm2(edited) * (1 + scale_mlp) + shift_mlp
    edited = edited + block.ff(norm_edited) * gate_mlp

    output = hidden_states.clone()
    output.index_copy_(1, edit_indices, edited.to(dtype=output.dtype))
    return output


@contextlib.contextmanager
def install_regional_blocks(
    transformer: Any,
    state: LtxRegionalState,
    *,
    layers: list[int] | None = None,
):
    """Patch LTX blocks to run the Medit/Msee regional path while active."""
    selected = set(range(len(transformer.transformer_blocks)) if layers is None else layers)
    originals: list[tuple[Any, Callable[..., Any]]] = []
    for index, block in enumerate(transformer.transformer_blocks):
        if index not in selected:
            continue
        original = block.forward
        originals.append((block, original))

        def forward(
            block_self: Any,
            hidden_states: Any,
            encoder_hidden_states: Any,
            temb: Any,
            image_rotary_emb: Any = None,
            encoder_attention_mask: Any = None,
            *,
            _original: Callable[..., Any] = original,
        ) -> Any:
            if not state.active:
                return _original(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                )
            # Blocks run in order within a forward, so a single slot on the
            # shared state is enough to tell STG-mix which block it is in.
            state.layer = index
            return ltx_regional_block(
                block_self, hidden_states, encoder_hidden_states, temb,
                image_rotary_emb, encoder_attention_mask, state,
            )

        block.forward = types.MethodType(forward, block)
    try:
        yield state
    finally:
        for block, original in reversed(originals):
            block.forward = original
