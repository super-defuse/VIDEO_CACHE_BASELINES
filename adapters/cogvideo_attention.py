from __future__ import annotations

import contextlib
import types
from dataclasses import dataclass, field
from typing import Any, Callable


def joint_qkv(attention: Any, hidden: Any, text: Any, rotary: Any) -> tuple[Any, Any, Any, int]:
    import torch
    from diffusers.models.embeddings import apply_rotary_emb

    text_tokens = int(text.shape[1])
    states = torch.cat((text, hidden), dim=1)
    batch, tokens = states.shape[:2]
    query = attention.to_q(states)
    key = attention.to_k(states)
    value = attention.to_v(states)
    head_dim = query.shape[-1] // attention.heads
    shape = (batch, tokens, attention.heads, head_dim)
    query = query.view(shape).transpose(1, 2)
    key = key.view(shape).transpose(1, 2)
    value = value.view(shape).transpose(1, 2)
    if attention.norm_q is not None:
        query = attention.norm_q(query)
    if attention.norm_k is not None:
        key = attention.norm_k(key)
    if rotary is not None:
        query[:, :, text_tokens:] = apply_rotary_emb(query[:, :, text_tokens:], rotary)
        key[:, :, text_tokens:] = apply_rotary_emb(key[:, :, text_tokens:], rotary)
    return query, key, value, text_tokens


def project_joint(attention: Any, output: Any, text_tokens: int) -> tuple[Any, Any]:
    output = output.transpose(1, 2).reshape(output.shape[0], output.shape[2], -1)
    output = attention.to_out[1](attention.to_out[0](output))
    text, video = output.split((text_tokens, output.shape[1] - text_tokens), dim=1)
    return video, text


def dense_joint_attention(attention: Any, hidden: Any, text: Any, rotary: Any) -> tuple[Any, Any]:
    import torch.nn.functional as functional

    query, key, value, text_tokens = joint_qkv(attention, hidden, text, rotary)
    output = functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    return project_joint(attention, output, text_tokens)


@dataclass
class CogVideoStepState:
    start_step: int = 0
    current_step: int = 0
    calls: int = 0
    on_begin: Callable[[int], None] | None = None

    def begin(self) -> None:
        self.current_step = self.start_step + self.calls
        self.calls += 1
        if self.on_begin is not None:
            self.on_begin(self.current_step)


@contextlib.contextmanager
def track_transformer_steps(transformer: Any, state: CogVideoStepState):
    original = transformer.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        state.begin()
        return original(*args, **kwargs)

    transformer.forward = forward
    try:
        yield state
    finally:
        transformer.forward = original


@dataclass
class RegionalAttentionState:
    video_edit: Any
    video_see: Any
    first_step: int
    last_step: int
    total_steps: int
    differential_tokens: tuple[int, ...] = ()
    gamma_k: float = 1.0
    gamma_o: float = 1.0
    active: bool = False
    current_step: int = 0
    calls: int = 0
    _indices: dict[tuple[str, int], tuple[Any, Any]] = field(default_factory=dict)

    def set_step(self, step: int) -> None:
        self.current_step = int(step)
        self.active = self.first_step <= self.current_step < self.last_step

    def indices(self, device: Any, video_tokens: int) -> tuple[Any, Any]:
        import torch

        key = (str(device), video_tokens)
        if key not in self._indices:
            edit = self.video_edit.flatten().bool()
            see = self.video_see.flatten().bool() | edit
            if edit.numel() != video_tokens or see.numel() != video_tokens:
                raise ValueError(
                    f"CogVideo region mask has {edit.numel()} tokens, transformer has {video_tokens}"
                )
            edit_indices = torch.nonzero(edit, as_tuple=False).flatten().to(device)
            see_indices = torch.nonzero(see, as_tuple=False).flatten().to(device)
            if edit_indices.numel() == 0 or see_indices.numel() == 0:
                raise ValueError("CogVideo regional attention requires non-empty Medit and Msee")
            self._indices[key] = edit_indices, see_indices
        return self._indices[key]

    def scales(self) -> tuple[float, float]:
        denominator = max(1, self.total_steps - 1)
        progress = min(1.0, max(0.0, self.current_step / denominator))
        return (
            1.0 + (self.gamma_k - 1.0) * (1.0 - progress),
            1.0 + (self.gamma_o - 1.0) * (1.0 - progress),
        )


def regional_joint_attention(
    attention: Any,
    hidden: Any,
    text: Any,
    rotary: Any,
    state: RegionalAttentionState,
) -> tuple[Any, Any, Any]:
    import torch
    import torch.nn.functional as functional

    query, key, value, text_tokens = joint_qkv(attention, hidden, text, rotary)
    edit, see = state.indices(hidden.device, hidden.shape[1])
    text_indices = torch.arange(text_tokens, device=hidden.device)
    query_indices = torch.cat((text_indices, edit + text_tokens))
    context_indices = torch.cat((text_indices, see + text_tokens))
    selected_query = query.index_select(2, query_indices)
    selected_key = key.index_select(2, context_indices).clone()
    selected_value = value.index_select(2, context_indices)
    scale_k, scale_o = state.scales()
    valid_tokens = [index for index in state.differential_tokens if 0 <= index < text_tokens]
    if valid_tokens:
        selected_key[:, :, valid_tokens] *= scale_k
    output = functional.scaled_dot_product_attention(
        selected_query, selected_key, selected_value, dropout_p=0.0, is_causal=False
    )
    output[:, :, text_tokens:] *= scale_o
    output = output.transpose(1, 2).reshape(output.shape[0], output.shape[2], -1)
    output = attention.to_out[1](attention.to_out[0](output))
    state.calls += 1
    return output[:, text_tokens:], output[:, :text_tokens], edit


@contextlib.contextmanager
def install_regional_blocks(
    transformer: Any,
    state: RegionalAttentionState,
    *,
    layers: list[int] | None = None,
):
    import torch

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
            attention_kwargs: Any = None,
            *,
            _original: Callable[..., Any] = original,
        ) -> tuple[Any, Any]:
            if not state.active:
                return _original(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    attention_kwargs,
                )
            norm_hidden, norm_text, gate_msa, text_gate_msa = block_self.norm1(
                hidden_states, encoder_hidden_states, temb
            )
            attention_hidden, attention_text, edit = regional_joint_attention(
                block_self.attn1, norm_hidden, norm_text, image_rotary_emb, state
            )
            edited_hidden = hidden_states.index_select(1, edit) + gate_msa * attention_hidden
            encoder_hidden_states = encoder_hidden_states + text_gate_msa * attention_text
            norm_hidden, norm_text, gate_ff, text_gate_ff = block_self.norm2(
                edited_hidden, encoder_hidden_states, temb
            )
            text_tokens = encoder_hidden_states.shape[1]
            feed_forward = block_self.ff(torch.cat((norm_text, norm_hidden), dim=1))
            edited_hidden = edited_hidden + gate_ff * feed_forward[:, text_tokens:]
            encoder_hidden_states = encoder_hidden_states + text_gate_ff * feed_forward[:, :text_tokens]
            output = hidden_states.clone()
            output.index_copy_(1, edit, edited_hidden)
            return output, encoder_hidden_states

        block.forward = types.MethodType(forward, block)
    try:
        yield state
    finally:
        for block, original in reversed(originals):
            block.forward = original
