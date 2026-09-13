"""CogVideoX joint-attention probe for the Carnator fusion mask.

CogVideoX has one attention module per block: ``attn1`` concatenates the caption
and the video into a single joint sequence, so one hook yields both halves of
the Wan/LTX probe pair -- the cross part (video queries against one noun's
caption keys) that proposes the seeds, and the self part (video queries against
the noun and background seed keys) that decides where the noun actually lands.
Wan and LTX need two probe classes, and LTX needs a dedicated extra forward for
the contrast; here the step-5 forward the source is already running is enough.

The QKV math is ``adapters.cogvideo_attention.joint_qkv`` -- the same helper the
Chorus2 and Carnator sparse processors substitute for diffusers'
``CogVideoXAttnProcessor2_0`` -- so the probe cannot drift from the attention it
is measuring, RoPE included.

Only the ``cond`` row of a CFG batch is read.  The ``uncond`` row carries the
negative prompt, where the noun's caption token indices name nothing at all, and
averaging the two would smear the mask across the frame.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, Mapping, Sequence

# Queries are scored in chunks: the key set is one noun's 2% seed, which is still
# ~900 keys on CogVideoX's 44880-token grid, so a full [heads, tokens, seeds]
# logit map would be several gigabytes.
DEFAULT_CHUNK_SIZE = 512


class CogVideoMultiNounJointAttentionProbe(contextlib.AbstractContextManager):
    """Capture the per-noun cross score and self contrast in one step-5 pass.

    ``cross_layer`` proposes the seeds (``cross_scores``, ``seed_indices``) and
    ``self_layer`` turns them into a per-noun contrast volume
    (``self_records_by_noun``), mirroring
    ``LtxMultiNounCrossAttentionProbe`` + ``LtxMultiNounSelfAttentionContrastProbe``
    on a single joint-attention module.  The caller fuses the pair with
    ``build_canonical_fusion_mask`` exactly as the Wan and LTX paths do.
    """

    def __init__(
        self,
        transformer: Any,
        *,
        cross_layer: int,
        self_layer: int,
        positive_indices_by_noun: Mapping[str, Sequence[int]],
        token_grid: tuple[int, int, int],
        completed_step: int,
        top_p: float = 0.02,
        background_p: float = 0.02,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        if not positive_indices_by_noun:
            raise ValueError("multi-noun joint-attention probe needs at least one noun")
        self.transformer = transformer
        self.cross_layer = int(cross_layer)
        self.self_layer = int(self_layer)
        if self.self_layer < self.cross_layer:
            raise ValueError(
                "attention_self_layer must not precede attention_probe_layer: the contrast reads "
                "the seeds the cross score produces earlier in the same transformer forward"
            )
        self.positive_indices_by_noun = {
            str(noun): [int(item) for item in indices] for noun, indices in positive_indices_by_noun.items()
        }
        self.token_grid = (int(token_grid[0]), int(token_grid[1]), int(token_grid[2]))
        self.token_count = math.prod(self.token_grid)
        self.completed_step = int(completed_step)
        if self.completed_step < 1:
            raise ValueError("completed_step is 1-based and must be positive")
        self.top_p = float(top_p)
        self.background_p = float(background_p)
        self.chunk_size = max(1, int(chunk_size))
        self.calls: dict[int, int] = {}
        self.records_by_noun: dict[str, list[dict[str, Any]]] = {
            noun: [] for noun in self.positive_indices_by_noun
        }
        self.cross_scores: dict[str, Any] = {}
        self.seed_indices: dict[str, tuple[Any, Any]] = {}
        self.self_records_by_noun: dict[str, dict[str, Any]] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> "CogVideoMultiNounJointAttentionProbe":
        for layer_index in sorted({self.cross_layer, self.self_layer}):
            blocks = self.transformer.transformer_blocks
            if not 0 <= layer_index < len(blocks):
                raise ValueError(f"CogVideo probe layer {layer_index} is outside the {len(blocks)}-block transformer")
            attention = blocks[layer_index].attn1
            self.handles.append(
                attention.register_forward_pre_hook(self._make_hook(layer_index), with_kwargs=True)
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def validate(self) -> None:
        """Fail closed: a probe that recorded nothing must not yield a mask."""

        expected = set(self.positive_indices_by_noun)
        missing_cross = sorted(noun for noun in expected if not self.records_by_noun.get(noun))
        if missing_cross:
            raise RuntimeError(f"CogVideo joint-attention probe recorded no cross score for {missing_cross}")
        missing_self = sorted(expected - set(self.self_records_by_noun))
        if missing_self:
            raise RuntimeError(f"CogVideo joint-attention probe recorded no self contrast for {missing_self}")

    def summary(self) -> dict[str, Any]:
        return {
            "cross_layer": self.cross_layer,
            "self_layer": self.self_layer,
            "completed_step": self.completed_step,
            "token_grid": list(self.token_grid),
            "nouns": {
                noun: {
                    "caption_tokens": len(self.positive_indices_by_noun[noun]),
                    "noun_seeds": int(self.seed_indices[noun][0].numel()),
                    "background_seeds": int(self.seed_indices[noun][1].numel()),
                    "contrast_min": float(record["self_contrast_max"].min().item()),
                    "contrast_max": float(self.self_records_by_noun[noun]["self_contrast_max"].max().item()),
                    "contrast_mean": float(self.self_records_by_noun[noun]["self_contrast_max"].mean().item()),
                }
                for noun, record in sorted(self.self_records_by_noun.items())
            },
        }

    # ------------------------------------------------------------- internals

    def _conditioning_row(self, batch: int) -> int:
        """Which batch row holds the plain conditional branch.

        ``CogVideoXPipeline`` feeds ``torch.cat([latents] * 2)`` with
        ``prompt_embeds = cat([negative, positive])``, so the conditional row is
        the last one; a single-row batch (no CFG) is conditional by definition.
        """

        return batch - 1 if batch >= 2 else 0

    def _make_hook(self, layer_index: int):
        def hook(attention: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            # Each block calls attn1 exactly once per transformer forward, so the
            # per-layer call count is the 1-based completed step.  Counting here
            # rather than reading a shared tracker keeps the probe independent of
            # whatever step bookkeeping the enclosing run installs.
            count = self.calls.get(layer_index, 0) + 1
            self.calls[layer_index] = count
            if count != self.completed_step:
                return
            self._capture(attention, layer_index, args, kwargs)

        return hook

    def _capture(self, attention: Any, layer_index: int, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        import torch

        from adapters.cogvideo_attention import joint_qkv

        with torch.no_grad():
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            text = kwargs.get("encoder_hidden_states")
            if hidden_states is None or text is None:
                raise RuntimeError("CogVideo joint-attention probe saw no video or caption states")
            row = self._conditioning_row(int(hidden_states.shape[0]))
            query, key, _value, text_tokens = joint_qkv(
                attention, hidden_states[row: row + 1], text[row: row + 1], kwargs.get("image_rotary_emb")
            )
            video_tokens = int(query.shape[2]) - text_tokens
            if video_tokens != self.token_count:
                raise RuntimeError(
                    f"CogVideo joint-attention probe saw {video_tokens} video tokens, but the "
                    f"{self.token_grid} grid has {self.token_count}; the latent geometry changed"
                )
            video_query = query[:, :, text_tokens:]
            scale = float(key.shape[-1]) ** -0.5
            if layer_index == self.cross_layer:
                self._capture_cross(video_query, key[:, :, :text_tokens], scale, layer_index)
            if layer_index == self.self_layer:
                self._capture_contrast(video_query, key[:, :, text_tokens:], scale)

    def _capture_cross(self, video_query: Any, caption_key: Any, scale: float, layer_index: int) -> None:
        """One score volume per noun: every video query against that noun's keys.

        Raw pre-softmax logits, averaged over heads and the noun's own key
        tokens, exactly as the Wan and LTX cross probes do.  There is no
        softmax and no padding mask, so the caption's padding rows are simply
        never indexed.
        """

        import torch

        from carnator.masks import build_cross_attention_seed_scores, seed_indices_from_cross_score

        caption_tokens = int(caption_key.shape[2])
        for noun, indices in self.positive_indices_by_noun.items():
            positive = torch.tensor(indices, device=caption_key.device, dtype=torch.long)
            if int(positive.max().item()) >= caption_tokens:
                raise RuntimeError(
                    f"noun {noun!r} maps to caption token {int(positive.max().item())}, "
                    f"beyond the {caption_tokens}-token caption"
                )
            score = (
                torch.einsum("bhqd,bhkd->bhqk", video_query, caption_key.index_select(2, positive))
                .mul(scale)
                .mean(dim=(0, 1, 3))
            )
            self.records_by_noun[noun].append({
                "step_index": self.completed_step - 1,
                "layer_index": int(layer_index),
                "positive": score.reshape(*self.token_grid).detach().float().cpu(),
            })
        self.cross_scores = {
            noun: build_cross_attention_seed_scores(records, completed_step=self.completed_step)["score"].float()
            for noun, records in self.records_by_noun.items()
        }
        self.seed_indices = {
            noun: seed_indices_from_cross_score(score, top_p=self.top_p, background_p=self.background_p)
            for noun, score in self.cross_scores.items()
        }

    def _capture_contrast(self, video_query: Any, video_key: Any, scale: float) -> None:
        """Per-noun max-logit contrast between the noun and background seeds.

        Cross attention only proposes seed locations; this contrast is what the
        persisted ``Mbase`` is actually built from, so the two always run as a
        pair.  The key set is only the seeds, so the token-by-token logit map is
        still chunked over queries.
        """

        import torch

        if set(self.seed_indices) != set(self.positive_indices_by_noun):
            raise RuntimeError(
                "CogVideo self-contrast needs the cross-score seeds; set attention_self_layer "
                "to the cross layer or to a later one"
            )
        for noun, (noun_indices, background_indices) in self.seed_indices.items():
            noun_key = video_key.index_select(2, noun_indices.to(device=video_key.device, dtype=torch.long))
            background_key = video_key.index_select(
                2, background_indices.to(device=video_key.device, dtype=torch.long)
            )
            noun_parts: list[Any] = []
            background_parts: list[Any] = []
            for start in range(0, self.token_count, self.chunk_size):
                stop = min(self.token_count, start + self.chunk_size)
                query_chunk = video_query[:, :, start:stop]
                noun_logits = torch.einsum("bhqd,bhkd->bhqk", query_chunk, noun_key).mul(scale)
                background_logits = torch.einsum("bhqd,bhkd->bhqk", query_chunk, background_key).mul(scale)
                noun_parts.append(noun_logits.max(dim=3).values.mean(dim=(0, 1)).detach().float().cpu())
                background_parts.append(
                    background_logits.max(dim=3).values.mean(dim=(0, 1)).detach().float().cpu()
                )
                del noun_logits, background_logits
            noun_max = torch.cat(noun_parts).reshape(*self.token_grid)
            background_max = torch.cat(background_parts).reshape(*self.token_grid)
            self.self_records_by_noun[noun] = {
                "self_noun_max": noun_max,
                "self_bg_max": background_max,
                "self_contrast_max": noun_max - background_max,
            }
