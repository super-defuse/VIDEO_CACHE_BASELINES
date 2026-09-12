from __future__ import annotations

import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as functional

from adapters import ltx_attention
from adapters.ltx_attention import (
    LtxRegionalState,
    LtxStepTracker,
    install_regional_blocks,
    ltx_cross_attention,
    ltx_projection,
    ltx_project,
    ltx_qkv,
    ltx_regional_block,
    stg_mix,
)

DIM = 4
HEADS = 2
TOKENS = 6
BATCH = 1


class _RecordingNorm:
    """RMSNorm stand-in that remembers the layout it was handed."""

    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.shapes.append(tuple(value.shape))
        return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-6)


class _Offset(nn.Module):
    """Stands in for a dropout: an order-sensitive, non-linear second ``to_out``."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.value


class _FakeAttention(nn.Module):
    def __init__(self, *, dim: int = DIM, heads: int = HEADS, seed: int = 3,
                 output: nn.Module | None = None) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)

        def linear() -> nn.Linear:
            layer = nn.Linear(dim, dim, bias=False)
            with torch.no_grad():
                layer.weight.copy_(torch.randn((dim, dim), generator=generator))
            return layer

        self.to_q, self.to_k, self.to_v = linear(), linear(), linear()
        self.to_out = nn.ModuleList([linear(), output if output is not None else nn.Identity()])
        self.heads = heads
        self.norm_q = _RecordingNorm()
        self.norm_k = _RecordingNorm()
        self.prepared: list[tuple[int, int]] = []

    def prepare_attention_mask(self, mask, sequence_length, batch_size):
        """"Mirrors ``Attention.prepare_attention_mask``: a 4D additive mask."""
        self.prepared.append((sequence_length, batch_size))
        return mask[:, None, None, :].to(torch.float32).expand(
            batch_size, self.heads, 1, sequence_length
        ).contiguous()


class _FakeBlock:
    """The parts of ``LTXVideoTransformerBlock`` the regional path touches."""

    def __init__(self) -> None:
        # gate_msa = gate_mlp = 1, every other modulation zero, so the block is
        # its stock residual shape without needing a real temb.
        self.scale_shift_table = torch.zeros(6, DIM)
        self.scale_shift_table[2] = 1.0
        self.scale_shift_table[5] = 1.0
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.attn1 = _FakeAttention(seed=5)
        self.attn2 = _FakeAttention(seed=9)
        self.ff = nn.Linear(DIM, DIM, bias=False)


def _state(*, first_step: int = 0, last_step: int = 50, total_steps: int = 50, **kwargs) -> LtxRegionalState:
    return LtxRegionalState(
        video_edit=torch.zeros(1, TOKENS), video_see=torch.zeros(1, TOKENS),
        first_step=first_step, last_step=last_step, total_steps=total_steps, **kwargs,
    )


def _region(edit: list[int], see: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    edit_mask = torch.zeros(1, TOKENS, dtype=torch.bool)
    see_mask = torch.zeros(1, TOKENS, dtype=torch.bool)
    edit_mask[0, edit] = True
    see_mask[0, see] = True
    return edit_mask, see_mask


class _SdpaRecorder:
    def __init__(self, fill: float | None = None) -> None:
        self.calls: list[dict] = []
        self.fill = fill

    def __call__(self, query, key, value, **kwargs):
        self.calls.append({"q": query, "k": key, "v": value, **kwargs})
        if self.fill is None:
            return torch.zeros_like(query)
        return torch.full_like(query, self.fill)


# ------------------------------------------------------------------ step index


def test_step_tracker_reports_a_one_based_computing_step():
    """Cached keys are 1-based; the hook is armed with the 0-based index."""
    tracker = LtxStepTracker()
    assert (tracker.current_step, tracker.computing_step) == (0, 1)
    tracker.set_step(17)
    assert (tracker.current_step, tracker.computing_step) == (17, 18)


def test_regional_window_is_half_open_on_the_configured_step_range():
    state = _state(first_step=5, last_step=8)
    for step, expected in [(4, False), (5, True), (7, True), (8, False), (9, False)]:
        state.set_step(step)
        assert state.active is expected, step
        assert state.tgaa_active is expected


def test_srd_can_be_active_with_tgaa_switched_off():
    state = _state(first_step=5, last_step=8, tgaa_enabled=False)
    state.set_step(6)
    assert state.active is True
    assert state.tgaa_active is False


def test_tgaa_scales_decay_linearly_and_reach_one_on_the_last_step():
    state = _state(total_steps=50, gamma_k=1.5, gamma_o=2.0)
    state.set_step(0)
    assert state.scales() == (1.5, 2.0)
    state.set_step(49)
    assert state.scales() == (1.0, 1.0)
    state.set_step(24)
    assert state.scales() == pytest.approx((1.5 - 0.5 * 24 / 49, 2.0 - 1.0 * 24 / 49))
    # Out-of-range indices clamp rather than extrapolate past gamma or below 1.
    state.set_step(100)
    assert state.scales() == (1.0, 1.0)
    state.set_step(-5)
    assert state.scales() == (1.5, 2.0)


def test_tgaa_scales_do_not_divide_by_zero_on_a_single_step_schedule():
    state = _state(total_steps=1)
    state.set_step(0)
    assert state.scales() == (1.5, 1.5)


# --------------------------------------------------------------- token regions


def test_indices_unions_medit_into_msee_and_caches_per_device():
    edit_mask, see_mask = _region([1, 4], [0, 5])
    state = _state()
    state.video_edit, state.video_see = edit_mask, see_mask

    first = state.indices("cpu", TOKENS)
    edit, see = first
    assert edit.tolist() == [1, 4]
    # Msee is the union: a key the editor attends to is never dropped because
    # the source mask left it out.
    assert see.tolist() == [0, 1, 4, 5]
    assert state.indices("cpu", TOKENS) is first


def test_indices_reject_a_region_mask_of_the_wrong_length():
    edit_mask, see_mask = _region([1], [1])
    state = _state()
    state.video_edit, state.video_see = edit_mask, see_mask
    with pytest.raises(ValueError, match="has 6 tokens but the transformer has 8"):
        state.indices("cpu", 8)


def test_indices_require_both_regions_to_be_non_empty():
    state = _state()
    state.video_edit, state.video_see = _region([], [])
    with pytest.raises(ValueError, match="non-empty Medit and Msee"):
        state.indices("cpu", TOKENS)


# ----------------------------------------------------------- model arithmetic


def test_ltx_qkv_norms_the_flat_tensor_before_splitting_the_heads():
    attention = _FakeAttention()
    hidden = torch.randn(2, 5, DIM)
    query, key, value = ltx_qkv(attention, hidden)

    assert tuple(query.shape) == (2, HEADS, 5, DIM // HEADS)
    # norm_q/norm_k are RMSNorm across all heads jointly: they have to see
    # [B, S, inner_dim], never the head-split layout.
    assert attention.norm_q.shapes == [(2, 5, DIM)]
    assert attention.norm_k.shapes == [(2, 5, DIM)]
    expected = (attention.norm_q(attention.to_q(hidden))
                .unflatten(2, (HEADS, -1)).transpose(1, 2))
    assert torch.equal(query, expected)
    assert torch.equal(value, attention.to_v(hidden).unflatten(2, (HEADS, -1)).transpose(1, 2))


def test_ltx_project_runs_the_output_projection_before_the_dropout_slot():
    attention = _FakeAttention(output=_Offset(0.5))
    flat = torch.randn(1, 3, DIM)
    expected = attention.to_out[1](attention.to_out[0](flat))
    assert torch.equal(ltx_project(attention, flat), expected)
    # The reverse order would be a different tensor, so this is not vacuous.
    assert not torch.equal(expected, attention.to_out[0](attention.to_out[1](flat)))


def test_ltx_projection_agrees_with_the_flattening_call_site():
    attention = _FakeAttention()
    output = torch.randn(2, HEADS, 5, DIM // HEADS)  # [B, heads, S, head_dim]
    assert torch.equal(
        ltx_projection(attention, output),
        ltx_project(attention, output.transpose(1, 2).flatten(2, 3)),
    )


# ------------------------------------------------------------------- STG mix


def test_stg_mix_is_a_no_op_without_a_skip_mask():
    attention = _FakeAttention()
    attended = torch.randn(1, 3, DIM)
    assert stg_mix(attention, torch.randn(1, 3, DIM), attended, None) is attended


def test_stg_mix_replaces_only_the_perturbed_rows_before_to_out():
    attention = _FakeAttention()
    hidden = torch.randn(2, 3, DIM)
    attended = torch.randn(2, 3, DIM)
    mask = torch.tensor([1.0, 0.0])
    mixed = stg_mix(attention, hidden, attended, mask)

    # Row 0 is the guidance row: bit-identical attention output.
    assert torch.equal(mixed[0], attended[0])
    # Row 1 is the perturbation branch: the raw value projection, pre-to_out.
    assert torch.equal(mixed[1], attention.to_v(hidden)[1])


def test_stg_mix_keeps_the_attention_dtype_across_the_substitution():
    attention = _FakeAttention().to(torch.bfloat16)
    hidden = torch.randn(1, 3, DIM, dtype=torch.bfloat16)
    attended = torch.randn(1, 3, DIM, dtype=torch.bfloat16)
    mixed = stg_mix(attention, hidden, attended, torch.zeros([1]))
    assert mixed.dtype is torch.bfloat16


# ---------------------------------------------------------------------- TGAA


def test_cross_attention_scales_differential_keys_and_the_output(monkeypatch):
    attention = _FakeAttention()
    state = _state(total_steps=50, gamma_k=2.0, gamma_o=3.0)
    state.differential_tokens = (1,)
    state.set_step(0)  # the first step: both scales are at full gamma
    recorder = _SdpaRecorder(fill=1.0)
    monkeypatch.setattr(functional, "scaled_dot_product_attention", recorder)

    hidden = torch.randn(1, 3, DIM)
    text = torch.randn(1, 3, DIM)
    plain = ltx_cross_attention(attention, hidden, text, None, None)
    scaled = ltx_cross_attention(attention, hidden, text, None, state)

    reference_key = attention.norm_k(attention.to_k(text)).unflatten(2, (HEADS, -1)).transpose(1, 2)
    unscaled_key, scaled_key = recorder.calls[0]["k"], recorder.calls[1]["k"]
    # Only the differential token row is amplified; the rest are untouched.
    assert torch.equal(unscaled_key, reference_key)
    assert torch.equal(scaled_key[:, :, 1], reference_key[:, :, 1] * 2.0)
    assert torch.equal(scaled_key[:, :, [0, 2]], reference_key[:, :, [0, 2]])
    # And the whole output carries F_o, which is what makes TGAA a pair.
    assert torch.allclose(scaled, plain * 3.0)


def test_cross_attention_ignores_differential_indices_the_caption_does_not_have(monkeypatch):
    attention = _FakeAttention()
    state = _state(total_steps=50, gamma_k=2.0, gamma_o=3.0)
    state.differential_tokens = (0, 99)  # 99 is past the caption's three tokens
    state.set_step(0)
    recorder = _SdpaRecorder()
    monkeypatch.setattr(functional, "scaled_dot_product_attention", recorder)

    ltx_cross_attention(attention, torch.randn(1, 3, DIM), torch.randn(1, 3, DIM), None, state)
    assert recorder.calls[0]["k"].shape[2] == 3


def test_cross_attention_leaves_the_keys_alone_when_tgaa_is_inactive(monkeypatch):
    attention = _FakeAttention()
    state = _state(total_steps=50, gamma_k=2.0, gamma_o=3.0, tgaa_enabled=False)
    state.differential_tokens = (1,)
    state.set_step(0)
    recorder = _SdpaRecorder(fill=1.0)
    monkeypatch.setattr(functional, "scaled_dot_product_attention", recorder)

    hidden, text = torch.randn(1, 3, DIM), torch.randn(1, 3, DIM)
    plain = ltx_cross_attention(attention, hidden, text, None, None)
    skipped = ltx_cross_attention(attention, hidden, text, None, state)

    assert torch.equal(recorder.calls[0]["k"], recorder.calls[1]["k"])
    assert torch.equal(plain, skipped)


def test_cross_attention_prepares_the_padding_mask_for_the_head_layout():
    attention = _FakeAttention()
    hidden = torch.randn(1, 3, DIM)
    text = torch.randn(1, 4, DIM)
    padding = torch.ones(1, 4, dtype=torch.bool)

    output = ltx_cross_attention(attention, hidden, text, padding, None)
    assert attention.prepared == [(4, 1)]
    assert output.shape == (1, 3, DIM)


# ------------------------------------------------------------- regional block


def test_regional_block_restricts_self_attention_to_medit_queries_and_msee_keys(monkeypatch):
    block = _FakeBlock()
    state = _state()
    state.video_edit, state.video_see = _region([1, 4], [0, 5])
    recorder = _SdpaRecorder()
    monkeypatch.setattr(functional, "scaled_dot_product_attention", recorder)

    hidden = torch.randn(BATCH, TOKENS, DIM)
    ltx_regional_block(
        block, hidden, torch.randn(BATCH, 3, DIM), torch.zeros(BATCH, 1, 6 * DIM),
        None, None, state,
    )

    # SDPA sees [B, heads, sequence, head_dim].
    self_attention, cross_attention = recorder.calls
    assert (self_attention["q"].shape[2], self_attention["k"].shape[2]) == (2, 4)
    # Values follow the keys, not the queries: Msee is what is read from.
    assert self_attention["v"].shape[2] == 4
    assert state.calls == 1
    assert state.tgaa_calls == 0
    # The cross-attention then runs on the Medit rows only.
    assert cross_attention["q"].shape[2] == 2


def test_regional_block_leaves_the_non_medit_rows_bit_identical():
    block = _FakeBlock()
    state = _state()
    state.video_edit, state.video_see = _region([1, 4], [0, 1, 4, 5])

    hidden = torch.randn(BATCH, TOKENS, DIM)
    text = torch.randn(BATCH, 3, DIM)
    temb = torch.zeros(BATCH, 1, 6 * DIM)
    output = ltx_regional_block(block, hidden, text, temb, None, None, state)

    assert output.shape == hidden.shape
    outside = [0, 2, 3, 5]
    assert torch.equal(output[:, outside], hidden[:, outside])
    # And the edited rows really were recomputed, not copied through.
    assert not torch.equal(output[:, [1, 4]], hidden[:, [1, 4]])
    assert state.calls == 1


def test_regional_block_counts_tgaa_calls_separately(monkeypatch):
    block = _FakeBlock()
    state = _state(first_step=0, last_step=50, tgaa_enabled=False)
    state.video_edit, state.video_see = _region([1], [1, 2])
    hidden = torch.randn(BATCH, TOKENS, DIM)
    temb = torch.zeros(BATCH, 1, 6 * DIM)

    state.set_step(3)
    ltx_regional_block(block, hidden, torch.randn(BATCH, 3, DIM), temb, None, None, state)
    assert (state.calls, state.tgaa_calls) == (1, 0)

    state.tgaa_enabled = True
    state.set_step(3)
    ltx_regional_block(block, hidden, torch.randn(BATCH, 3, DIM), temb, None, None, state)
    assert (state.calls, state.tgaa_calls) == (2, 1)


def test_regional_block_mixes_the_guidance_branch_through_the_stg_mask():
    block = _FakeBlock()
    state = _state()
    state.video_edit, state.video_see = _region([1, 4], [0, 1, 4, 5])
    state.guidance = SimpleNamespace(
        layer=lambda layer, batch: torch.tensor([1.0, 0.0])
    )
    hidden = torch.randn(2, TOKENS, DIM)
    temb = torch.zeros(2, 1, 6 * DIM)
    output = ltx_regional_block(
        block, hidden, torch.randn(2, 3, DIM), temb, None, None, state
    )

    # Both rows were recomputed on their own Medit set, so neither is copied.
    assert not torch.equal(output[:, [1, 4]], hidden[:, [1, 4]])


# ----------------------------------------------------------- block installation


def _fake_transformer(layers: int = 3) -> SimpleNamespace:
    def dense(index: int):
        # The stock signature: the patch forwards these by keyword when inactive.
        def forward(hidden_states=None, encoder_hidden_states=None, temb=None,
                    image_rotary_emb=None, encoder_attention_mask=None):
            return ("dense", index)

        return forward

    blocks = [SimpleNamespace(forward=dense(index)) for index in range(layers)]
    return SimpleNamespace(transformer_blocks=blocks)


def _call(block) -> object:
    """Call a block the way the LTX transformer does."""
    return block.forward(hidden_states="hidden", encoder_hidden_states="text", temb="temb")


def test_install_regional_blocks_patches_only_the_selected_layers(monkeypatch):
    transformer = _fake_transformer()
    originals = [block.forward for block in transformer.transformer_blocks]
    state = _state()
    seen: list[int] = []

    def fake_regional_block(block, hidden_states, encoder_hidden_states, temb,
                            image_rotary_emb, encoder_attention_mask, regional_state):
        seen.append(regional_state.layer)
        return "regional"

    monkeypatch.setattr(ltx_attention, "ltx_regional_block", fake_regional_block)
    state.set_step(7)

    with install_regional_blocks(transformer, state, layers=[1, 2]):
        assert _call(transformer.transformer_blocks[0]) == ("dense", 0)
        assert _call(transformer.transformer_blocks[2]) == "regional"
        # The shared state slot tells STG-mix which block it is inside.
        assert seen == [2]

    assert [block.forward for block in transformer.transformer_blocks] == originals


def test_install_regional_blocks_defaults_to_every_layer_and_falls_through_when_inactive(monkeypatch):
    transformer = _fake_transformer()
    originals = [block.forward for block in transformer.transformer_blocks]
    state = _state()
    calls: list[str] = []
    monkeypatch.setattr(
        ltx_attention, "ltx_regional_block",
        lambda *args, **kwargs: calls.append("regional") or "regional",
    )

    state.set_step(999)  # outside [first_step, last_step)
    with install_regional_blocks(transformer, state):
        for index, block in enumerate(transformer.transformer_blocks):
            assert _call(block) == ("dense", index)
    assert calls == []
    assert [block.forward for block in transformer.transformer_blocks] == originals


def test_install_regional_blocks_restores_the_bound_method_on_error():
    transformer = _fake_transformer()
    originals = [block.forward for block in transformer.transformer_blocks]
    state = _state()

    with pytest.raises(RuntimeError, match="boom"):
        with install_regional_blocks(transformer, state, layers=[0]):
            raise RuntimeError("boom")

    assert [block.forward for block in transformer.transformer_blocks] == originals


def test_installed_forward_accepts_the_stock_keyword_call_shape(monkeypatch):
    """The patch has to be callable the way the transformer calls a block."""
    transformer = _fake_transformer(layers=1)
    state = _state()
    captured: dict = {}

    def fake_regional_block(block, hidden_states, encoder_hidden_states, temb,
                            image_rotary_emb, encoder_attention_mask, regional_state):
        captured.update(hidden_states=hidden_states, mask=encoder_attention_mask)
        return "regional"

    monkeypatch.setattr(ltx_attention, "ltx_regional_block", fake_regional_block)
    state.set_step(7)
    with install_regional_blocks(transformer, state):
        block = transformer.transformer_blocks[0]
        assert isinstance(block.forward, types.MethodType)
        result = block.forward(
            hidden_states="hidden", encoder_hidden_states="text", temb="temb",
            image_rotary_emb=None, encoder_attention_mask="mask",
        )

    assert result == "regional"
    assert captured == {"hidden_states": "hidden", "mask": "mask"}
