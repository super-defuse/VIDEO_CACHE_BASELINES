from __future__ import annotations

import json

import pytest
import torch

from adapters.cogvideo_pipeline import CogVideoPipelineDelegate
from adapters.cogvideo_source_cache import create_manifest, validate_directory, write_manifest
from adapters.cogvideox15 import CogVideoX15Adapter
from adapters.synthetic import SyntheticAdapter
from core.cogvideo_runner import CogVideoReplayRunner
from core.registry import make_method
from core.types import Request


def config():
    return {
        "model": "cogvideox15",
        "dry_run": True,
        "seed": 11,
        "dtype": "bfloat16",
        "width": 1360,
        "height": 768,
        "frames": 81,
        "baseline_steps": 50,
        "nirvana_vid": {"reuse_start_step": 5, "cache_steps": [5]},
    }


def test_source_cache_schema_fails_closed_on_model_shape_config(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "latent.pt").write_bytes(b"latent")
    manifest = create_manifest(config(), source_uuid="source", scheduler="CogVideoXDDIMScheduler",
                               reuse_step=5, latent=b"latent")
    write_manifest(cache / "manifest.json", manifest)

    validate_directory(cache, config=config(), source_uuid="source")
    incompatible = {**config(), "frames": 49}
    with pytest.raises(ValueError, match="frames"):
        validate_directory(cache, config=incompatible, source_uuid="source")


class _Timesteps(list):
    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self)


class CogVideoXDDIMScheduler:
    def __init__(self):
        self.timesteps = _Timesteps()

    def set_timesteps(self, steps, device=None):
        del device
        self.timesteps = _Timesteps(range(steps, 0, -1))


def test_nirvana_resume_uses_suffix_of_original_50_step_schedule():
    adapter = CogVideoX15Adapter(config())
    scheduler = CogVideoXDDIMScheduler()
    expected = list(range(50, 0, -1))

    with adapter._resume_original_schedule(scheduler, steps=50, reuse_step=5, expected=expected):
        scheduler.set_timesteps(50, device="cuda:0")
        assert list(scheduler.timesteps) == expected[5:]

    scheduler.set_timesteps(50)
    assert list(scheduler.timesteps) == expected


def test_nirvana_resume_rejects_a_short_scheduler_request():
    adapter = CogVideoX15Adapter(config())
    scheduler = CogVideoXDDIMScheduler()

    with adapter._resume_original_schedule(scheduler, steps=50, reuse_step=5,
                                           expected=list(range(50, 0, -1))):
        with pytest.raises(ValueError, match="original full scheduler"):
            scheduler.set_timesteps(45)


def test_cogvideo_decode_helpers_remove_temporal_padding():
    class VideoProcessor:
        @staticmethod
        def postprocess_video(*, video, output_type):
            assert output_type == "np"
            return video.permute(0, 2, 3, 4, 1).numpy()

    class Pipe:
        vae_scale_factor_temporal = 4
        _execution_device = "cpu"
        video_processor = VideoProcessor()

        @staticmethod
        def decode_latents(value):
            return value.permute(0, 2, 1, 3, 4)

    delegate = CogVideoPipelineDelegate({"frames": 81, "device": "cpu"})
    delegate.pipe = Pipe()
    delegate.torch = torch
    latents = torch.arange(22).reshape(1, 22, 1, 1, 1).float()

    video = delegate.decode_video(latents)
    tail = delegate.decode_tail(latents)

    assert video.shape == (21, 1, 1, 1)
    assert video[:, 0, 0, 0].tolist() == list(range(1, 22))
    assert tail.item() == 21


def test_replay_runner_separates_source_time_and_always_cleans_tmp_cache(tmp_path):
    source_seeds = []

    class RecordingSynthetic(SyntheticAdapter):
        def generate(self, request, **kwargs):
            if kwargs.get("capture"):
                source_seeds.append(request.seed)
            return super().generate(request, **kwargs)

    adapter = RecordingSynthetic(config())
    method = make_method("nirvana_vid", config())
    method.adapter = adapter
    replay = {
        0: {"request_index": 0, "uuid": "target", "prompt": "target prompt", "decision": "hit",
            "source_uuid": "source", "legacy_status": "ok"},
        1: {"request_index": 1, "uuid": "miss", "prompt": "miss prompt", "decision": "miss",
            "source_uuid": None, "legacy_status": "cache_miss"},
    }
    runner = CogVideoReplayRunner(
        adapter,
        method,
        output_dir=tmp_path,
        replay=replay,
        source_prompts={"source": {"request_index": 4, "uuid": "source", "prompt": "source prompt"}},
        config=config(),
    )

    hit = runner.process(Request(0, "target", "target prompt", 11))
    miss = runner.process(Request(1, "miss", "miss prompt", 12))

    assert hit.status == "ok"
    assert source_seeds == [11]
    assert hit.cache_match.source_uuid == "source"
    assert hit.extra["source_prepare_s"] >= 0
    assert hit.extra["timing"]["target_e2e_s"] == hit.latency_s
    assert hit.extra["timing"]["save_video_s"] >= 0
    assert hit.extra["timing"]["hash_s"] >= 0
    assert hit.extra["cleanup"]["tmp_cache_removed"] is True
    assert list((tmp_path / "tmp_cache" / "nirvana_vid").iterdir()) == []
    assert miss.status == "cache_miss"
    assert miss.executed_steps == 0


def test_replay_runner_preserves_hit_when_source_execution_errors(tmp_path):
    class BrokenSynthetic(SyntheticAdapter):
        def generate(self, *args, **kwargs):
            raise RuntimeError("source failed")

    adapter = BrokenSynthetic(config())
    method = make_method("nirvana_vid", config())
    method.adapter = adapter
    replay = {0: {"request_index": 0, "uuid": "target", "prompt": "target prompt", "decision": "hit",
                  "source_uuid": "source", "legacy_status": "error"}}
    runner = CogVideoReplayRunner(adapter, method, output_dir=tmp_path, replay=replay,
                                  source_prompts={"source": {"request_index": 4, "prompt": "source prompt"}},
                                  config=config())

    result = runner.process(Request(0, "target", "target prompt", 11))

    assert result.status == "error"
    assert result.cache_match.hit is True
    assert result.cache_match.source_uuid == "source"
    assert "source failed" in result.error
    assert result.extra["cleanup"]["tmp_cache_removed"] is True


# --------------------------------------------------------------- fusion mask
#
# The Carnator CogVideoX mask path is fully offline-testable: the probe only
# needs a module with to_q/to_k/to_v, and everything downstream of it is plain
# tensor work.  These tests pin the three things the previous implementation got
# wrong -- the token grid, the per-noun mask, and the failure policy.


def _toy_attention(dim=8, heads=2):
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = heads
            self.to_q = torch.nn.Linear(dim, dim, bias=False)
            self.to_k = torch.nn.Linear(dim, dim, bias=False)
            self.to_v = torch.nn.Linear(dim, dim, bias=False)
            self.norm_q = None
            self.norm_k = None

        def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # The probe reads the call the block makes, not the module's output.
            return hidden_states, encoder_hidden_states

    torch.manual_seed(0)
    return Attention()


def _toy_transformer(blocks=2, dim=8, heads=2):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = _toy_attention(dim, heads)

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([Block() for _ in range(blocks)])

    return Transformer()


def _run_block(transformer, layer, *, video_tokens, text_tokens, dim=8, batch=1, step_calls=1):
    attention = transformer.transformer_blocks[layer].attn1
    for _ in range(step_calls):
        attention(
            torch.randn(batch, video_tokens, dim),
            encoder_hidden_states=torch.randn(batch, text_tokens, dim),
            image_rotary_emb=None,
        )


def test_probe_records_per_noun_cross_and_self_scores_at_the_completed_step():
    from carnator.cogvideo_attention_probe import CogVideoMultiNounJointAttentionProbe

    transformer = _toy_transformer()
    probe = CogVideoMultiNounJointAttentionProbe(
        transformer,
        cross_layer=0,
        self_layer=1,
        positive_indices_by_noun={"cat": [1], "dog": [2]},
        token_grid=(2, 2, 2),
        completed_step=2,
    )
    with probe:
        # Step 1 must be ignored, step 2 recorded.
        for _ in range(2):
            for layer in (0, 1):
                _run_block(transformer, layer, video_tokens=8, text_tokens=3)
    probe.validate()

    assert set(probe.cross_scores) == {"cat", "dog"}
    assert set(probe.self_records_by_noun) == {"cat", "dog"}
    for noun in ("cat", "dog"):
        assert probe.cross_scores[noun].shape == (2, 2, 2)
        assert probe.records_by_noun[noun][0]["step_index"] == 1
        assert probe.self_records_by_noun[noun]["self_contrast_max"].shape == (2, 2, 2)
        noun_seeds, background_seeds = probe.seed_indices[noun]
        assert noun_seeds.numel() == max(1, int(round(8 * 0.02)))
        assert background_seeds.numel() == max(1, int(round(8 * 0.02)))
    assert probe.summary()["token_grid"] == [2, 2, 2]


def test_probe_reads_only_the_conditional_row_of_a_cfg_batch():
    from carnator.cogvideo_attention_probe import CogVideoMultiNounJointAttentionProbe

    # The uncond row carries a different prompt, so reading it would smear the
    # score.  The recorded score must match a single-row recomputation of the
    # conditional row alone, taken straight from joint_qkv.
    torch.manual_seed(7)
    video = torch.randn(2, 8, 8)
    text = torch.randn(2, 3, 8)
    probe = CogVideoMultiNounJointAttentionProbe(
        _toy_transformer(),
        cross_layer=0,
        self_layer=0,
        positive_indices_by_noun={"cat": [1]},
        token_grid=(2, 2, 2),
        completed_step=1,
    )
    with probe:
        probe.transformer.transformer_blocks[0].attn1(video, encoder_hidden_states=text, image_rotary_emb=None)
    probe.validate()

    recorded = probe.records_by_noun["cat"][0]["positive"]
    assert torch.allclose(recorded, _reference_cross_score(probe, video[1:2], text[1:2]))
    # ... and it is not the uncond row.
    assert not torch.allclose(recorded, _reference_cross_score(probe, video[:1], text[:1]))


def _reference_cross_score(probe, video, text):
    """Recompute one noun's cross score straight from joint_qkv."""

    import torch as _torch

    from adapters.cogvideo_attention import joint_qkv

    attention = probe.transformer.transformer_blocks[probe.cross_layer].attn1
    query, key, _value, text_tokens = joint_qkv(attention, video, text, None)
    video_query = query[:, :, text_tokens:]
    caption_key = key[:, :, :text_tokens]
    scale = float(key.shape[-1]) ** -0.5
    indices = _torch.tensor(probe.positive_indices_by_noun["cat"], dtype=_torch.long)
    return (
        _torch.einsum("bhqd,bhkd->bhqk", video_query, caption_key.index_select(2, indices))
        .mul(scale)
        .mean(dim=(0, 1, 3))
        .reshape(*probe.token_grid)
    )


def test_probe_fails_closed_on_geometry_and_layer_order():
    from carnator.cogvideo_attention_probe import CogVideoMultiNounJointAttentionProbe

    with pytest.raises(ValueError, match="self_layer"):
        CogVideoMultiNounJointAttentionProbe(
            _toy_transformer(), cross_layer=1, self_layer=0,
            positive_indices_by_noun={"cat": [1]}, token_grid=(2, 2, 2), completed_step=1,
        )
    with pytest.raises(ValueError, match="at least one noun"):
        CogVideoMultiNounJointAttentionProbe(
            _toy_transformer(), cross_layer=0, self_layer=0,
            positive_indices_by_noun={}, token_grid=(2, 2, 2), completed_step=1,
        )

    probe = CogVideoMultiNounJointAttentionProbe(
        _toy_transformer(), cross_layer=0, self_layer=0,
        positive_indices_by_noun={"cat": [1]}, token_grid=(2, 2, 3), completed_step=1,
    )
    with pytest.raises(RuntimeError, match="video tokens"):
        with probe:
            _run_block(probe.transformer, 0, video_tokens=8, text_tokens=3)

    probe = CogVideoMultiNounJointAttentionProbe(
        _toy_transformer(), cross_layer=0, self_layer=0,
        positive_indices_by_noun={"cat": [9]}, token_grid=(2, 2, 2), completed_step=1,
    )
    with pytest.raises(RuntimeError, match="caption token"):
        with probe:
            _run_block(probe.transformer, 0, video_tokens=8, text_tokens=3)

    # Nothing recorded at all: validate() must refuse, not return an empty mask.
    probe = CogVideoMultiNounJointAttentionProbe(
        _toy_transformer(), cross_layer=0, self_layer=1,
        positive_indices_by_noun={"cat": [1]}, token_grid=(2, 2, 2), completed_step=3,
    )
    with probe:
        _run_block(probe.transformer, 0, video_tokens=8, text_tokens=3)
    with pytest.raises(RuntimeError, match="no cross score"):
        probe.validate()


def test_token_grid_matches_the_padded_cogvideox15_latent_grid():
    from adapters.cogvideo_attention import cogvideo_token_grid

    class Config:
        patch_size = 2
        patch_size_t = 2

    class Transformer:
        config = Config()

    class Pipe:
        transformer = Transformer()
        vae_scale_factor_temporal = 4
        vae_scale_factor_spatial = 8

    # 81 pixel frames -> 21 latent frames -> padded to 22 -> 11 temporal tokens.
    assert cogvideo_token_grid(Pipe(), frames=81, height=768, width=1360) == (11, 48, 85)
    # 77 pixel frames -> 20 latent frames, already even, so no padding.
    assert cogvideo_token_grid(Pipe(), frames=77, height=768, width=1360) == (10, 48, 85)


def test_caption_token_indices_are_padding_free_positions():
    from adapters.cogvideo_attention import caption_token_indices

    class Tokenizer:
        """Stand-in whose ids depend only on the word, so positions are exact."""

        vocabulary = {"a": 9, "cat": 7, "and": 10, "dog": 8}

        def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
            del truncation, max_length
            ids = [1] if add_special_tokens else []
            ids.extend(self.vocabulary.get(word, 11) for word in text.split())
            return type("Encoding", (), {"input_ids": ids})()

    tokenizer = Tokenizer()
    assert caption_token_indices(tokenizer, "a cat and a dog", ["cat"], 226) == (2,)
    assert caption_token_indices(tokenizer, "a cat and a dog", ["dog"], 226) == (5,)
    assert caption_token_indices(tokenizer, "a cat", ["bird"], 226) == ()


def test_token_masks_expand_onto_the_padded_latent_frames():
    from carnator.masks import token_mask_to_latent_mask

    latent = torch.zeros(1, 4, 22, 8, 10)
    grid = (11, 4, 5)
    mask = torch.zeros(grid, dtype=torch.bool)
    mask[10, :, :] = True          # the last temporal token

    expanded = token_mask_to_latent_mask(mask, latent)

    assert expanded.shape == latent.shape
    assert expanded[0, 0, 20].all() and expanded[0, 0, 21].all()
    # The first latent frame is the 1.5 padding the pipeline drops on decode.
    assert not expanded[0, 0, 0].any()


def test_per_noun_mbase_reaches_the_targets_mask_selection(tmp_path):
    """The exact source -> signature -> target-mask chain, offline."""

    from carnator.early_signature import materialize_per_noun_inference_mask_bundle
    from carnator.masks import build_per_noun_mbase_bundle, load_mask_bundle, save_mask_bundle

    grid = (2, 3, 3)
    mbase_by_noun = {}
    for noun, (row, column) in (("cat", (0, 0)), ("sofa", (2, 2))):
        mask = torch.zeros(grid, dtype=torch.bool)
        # The signature is read off the tail temporal token, so every noun mask
        # has to be supported there.
        mask[:, row, column] = True
        mbase_by_noun[noun] = mask

    assert "cat" in mbase_by_noun and "sofa" in mbase_by_noun
    template = build_per_noun_mbase_bundle(
        mbase_by_noun["cat"],
        source_completed_step=5,
        spatial_medit_radius=1,
        spatial_msee_radius=1,
        selection={"selected_source_noun": "cat", "selection_reason": "source_template"},
    )
    save_mask_bundle(tmp_path / "fusion_masks", template, save_cpu=True)

    _write_signature(tmp_path, mbase_by_noun)

    selection = materialize_per_noun_inference_mask_bundle(
        {"signature_path": str(tmp_path / "signature" / "online_color_signature.json")},
        target_nouns=("dog", "sofa"),
        source_fusion_mask_dir=tmp_path / "fusion_masks",
        output_dir=tmp_path / "inference_masks",
    )
    bundle = load_mask_bundle(tmp_path / "inference_masks", None)

    # One noun substitution selects the source-only noun, exactly as the Wan and
    # LTX targets do.
    assert selection["selected_source_noun"] == "cat"
    assert selection["target_only_nouns"] == ["dog"]
    assert bundle["Medit"].shape == grid
    assert bundle["Medit"].any()
    assert bundle["token_shape"] == list(grid)


def _write_signature(root, mbase_by_noun):
    from carnator.early_signature import write_early_signature

    return write_early_signature(
        root / "signature",
        step5_latent=torch.zeros(1, 4, 3, 3),
        decoded_tail_frame=torch.zeros(3, 3, 3),
        metadata={
            "prompt": "a cat on a sofa",
            "model": "CogVideoX1.5",
            "scheduler": "CogVideoXDDIMScheduler",
            "scheduler_config": {"total_steps": 50},
            "cfg": 6.0,
            "probe_seed": 11,
            "completed_step": 5,
            "latent_shape": [1, 4, 3, 3],
            "latent_token_grid": [2, 3, 3],
            "noun_extractor": "lexical-spacy",
            "mask_policy": "test-policy",
        },
        mbase_by_noun=mbase_by_noun,
    )
