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
