from __future__ import annotations

import numpy as np
import pytest
import torch

from adapters.cogvideo_attention import CogVideoStepState, RegionalAttentionState
from adapters.cogvideo_sparse import CogVideoBlockMaskCache, CogVideoBlockMaskConfig
from adapters.synthetic import SyntheticAdapter
from core.cogvideo_runner import CogVideoReplayRunner
from core.registry import make_method
from core.types import Request
from methods.chorus.cogvideo_mask import build_cogvideo_edit_mask


def cogvideo_config() -> dict:
    return {
        "model": "cogvideox15",
        "dry_run": True,
        "seed": 20260218,
        "dtype": "bfloat16",
        "width": 1360,
        "height": 768,
        "frames": 81,
        "baseline_steps": 50,
        "guidance_scale": 6.0,
        "nirvana_vid": {"reuse_start_step": 5, "cache_steps": [5]},
    }


@pytest.mark.parametrize("name", ["chorus", "chorus2", "carnator_boundary_b5"])
def test_advanced_cogvideo_methods_are_registered(name):
    method = make_method(name, cogvideo_config())

    assert method.name == name
    assert method.capabilities.cross_request_cache
    assert method.capabilities.source_trajectory


def test_cogvideo_mask_projects_temporal_and_spatial_patches(tmp_path):
    class Segmenter:
        def segment(self, video_path, phrase):
            assert video_path == tmp_path / "source.mp4"
            assert phrase == "dog"
            mask = np.zeros((9, 32, 48), dtype=np.uint8)
            mask[:, 8:24, 12:36] = 1
            return mask, {"backend": "test"}

    masks, metadata = build_cogvideo_edit_mask(
        tmp_path / "source.mp4",
        "dog",
        latent_shape=(4, 8, 12),
        patch_size=(2, 2, 2),
        config={"mask_dilation_edit": 1, "mask_dilation_see": 2},
        segmenter=Segmenter(),
    )

    assert masks["token_base"].shape == (2, 4, 6)
    assert masks["latent_edit"].shape == (4, 8, 12)
    assert torch.all(masks["token_see"] | ~masks["token_edit"])
    assert metadata["patch_size"] == [2, 2, 2]


def test_cogvideo_block_masks_are_bitpacked_and_checked(tmp_path):
    config = CogVideoBlockMaskConfig(q_block_size=2, k_block_size=2, top_p=0.8, min_top_k=0.25)
    cache = CogVideoBlockMaskCache(tmp_path, config, layers=[0], start_step=1, create=True)
    generator = torch.Generator().manual_seed(4)
    query = torch.randn((8, 2, 4), generator=generator)
    key = torch.randn((8, 2, 4), generator=generator)
    cache.capture(1, "cond", 0, query, key)
    cache.finalize()

    loaded = CogVideoBlockMaskCache(tmp_path, config)
    loaded.validate_coverage(steps=range(1, 2), branches=("cond",), layers=[0])
    mask, entry = loaded.get(1, "cond", 0, query.transpose(0, 1))

    assert mask.dtype == torch.bool
    assert tuple(mask.shape) == tuple(entry["mask_shape"])
    assert (tmp_path / entry["path"]).stat().st_size == (mask.numel() + 7) // 8


def test_regional_state_tracks_schedule_and_exact_token_grid():
    active = torch.tensor([[[True, False], [False, False]]])
    visible = torch.ones_like(active)
    region = RegionalAttentionState(active, visible, 5, 30, 50, gamma_k=1.5, gamma_o=2.0)
    tracker = CogVideoStepState(start_step=5, on_begin=region.set_step)

    tracker.begin()
    edit, see = region.indices(torch.device("cpu"), 4)
    scale_k, scale_o = region.scales()

    assert region.active
    assert edit.tolist() == [0]
    assert see.tolist() == [0, 1, 2, 3]
    assert scale_k > 1.0
    assert scale_o > scale_k
    with pytest.raises(ValueError, match="region mask"):
        region.indices(torch.device("cpu"), 5)


@pytest.mark.parametrize("method_name", ["chorus", "chorus2", "carnator_boundary_b5"])
def test_advanced_replay_hit_uses_private_cache_and_cleans_it(tmp_path, method_name):
    config = cogvideo_config()
    adapter = SyntheticAdapter(config)
    method = make_method(method_name, config)
    method.adapter = adapter
    replay = {
        0: {
            "request_index": 0,
            "uuid": "target",
            "prompt": "target prompt",
            "decision": "hit",
            "source_uuid": "source",
            "legacy_status": "ok",
        }
    }
    runner = CogVideoReplayRunner(
        adapter,
        method,
        output_dir=tmp_path,
        replay=replay,
        source_prompts={"source": {"request_index": 7, "prompt": "source prompt"}},
        config=config,
    )

    result = runner.process(Request(0, "target", "target prompt", 9))

    assert result.status == "ok"
    assert result.cache_match.hit
    assert result.extra["cleanup"]["tmp_cache_removed"] is True
    assert list((tmp_path / "tmp_cache" / method_name).iterdir()) == []
