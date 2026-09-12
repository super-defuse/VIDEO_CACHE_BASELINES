from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from adapters import make_adapter
from adapters.ltx_sparse import LtxBlockMaskConfig
from adapters.ltxvideo import LtxVideoAdapter
from adapters.synthetic import SyntheticAdapter
from core.registry import REAL_SUPPORTED, ensure_supported

# The five baselines the LTX-Video experiment runs, on top of the dense one.
LTX_METHODS = ("baseline", "nirvana_vid", "chorus", "chorus2", "carnator_boundary_b5")


def config(**overrides) -> dict:
    values = {
        "model": "ltxvideo",
        "dry_run": True,
        "device": "cpu",
        "dtype": "bfloat16",
        "baseline_steps": 50,
        "guidance_scale": 3.0,
        "fps": 25,
        "negative_prompt": "worst quality",
        "stg": {"enabled": False},
    }
    values.update(overrides)
    return values


class _StubDelegate:
    """The pieces of ``LtxVideoPipelineDelegate`` the adapter's I/O touches."""

    def __init__(self) -> None:
        self.saved: list = []

    def save_latent(self, latent, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latent, path)

    def load_latent(self, path):
        return torch.load(path, map_location="cpu")


def adapter(**overrides) -> LtxVideoAdapter:
    instance = LtxVideoAdapter(config(**overrides))
    instance._delegate = _StubDelegate()
    return instance


class _FakeScheduler:
    def step(self, model_output, timestep, sample, per_token_timesteps=None, return_dict=False):
        del model_output, timestep, per_token_timesteps, return_dict
        return (sample + 1.0,)


# ------------------------------------------------------------------ registry


def test_make_adapter_resolves_ltxvideo_and_keeps_dry_run_synthetic():
    assert isinstance(make_adapter("ltxvideo", config(dry_run=False)), LtxVideoAdapter)
    # dry_run short-circuits before the model name is read, so the protocol
    # tests keep working for the LTX runner too.
    assert isinstance(make_adapter("ltxvideo", config()), SyntheticAdapter)
    with pytest.raises(ValueError, match="unsupported model"):
        make_adapter("ltxvideo_x", config(dry_run=False))


def test_every_ltx_baseline_is_declared_supported_and_teacache_is_not():
    for method in LTX_METHODS:
        assert "ltxvideo" in REAL_SUPPORTED[method], method
        ensure_supported("ltxvideo", method)
    # TeaCache has no LTX implementation; the runner must refuse it up front
    # rather than fail inside a timed request.
    assert "ltxvideo" not in REAL_SUPPORTED["teacache"]
    with pytest.raises(SystemExit, match="N/A: teacache is not integrated for ltxvideo"):
        ensure_supported("ltxvideo", "teacache")


def test_adapter_reports_the_model_name_the_manifest_records():
    assert LtxVideoAdapter.model_name == "LtxVideo"


# ------------------------------------------------------------------- config


def test_resolved_config_absolutises_the_ltx_checkpoint(tmp_path):
    from scripts.benchmark.run_vidprom_300_experiment import resolved_config

    values = resolved_config("cuda:3", False, model_name="ltxvideo")
    assert values["device"] == "cuda:3"
    assert values["model"] == "ltxvideo"
    # Both paths are relative in the YAML and must reach the loader absolute.
    for key in ("model_path", "t5_weights"):
        assert values[key].startswith("/"), (key, values[key])
        assert "models/LTX-Video-0.9.7-dev" in values[key] or "models/t5xxl" in values[key]


def test_ltx_config_has_no_sparse_backend_knob_and_keeps_the_wan_block_geometry():
    from scripts.benchmark.run_vidprom_300_experiment import resolved_config

    values = resolved_config("cuda:0", False, model_name="ltxvideo")
    # SpargeAttn's kernel is compiled for Q=64/K=128, so LTX has no backend to
    # choose; the Wan2.1 ``outer_backend``/``ffn_backend`` keys must not leak in.
    assert "outer_backend" not in values and "ffn_backend" not in values
    for section in ("chorus2", "carnator_b5"):
        assert (values[section]["q_block_size"], values[section]["k_block_size"]) == (64, 128)


def test_block_mask_config_reads_the_method_section():
    cfg = {"q_block_size": 64, "k_block_size": 128, "top_p": 0.9, "min_top_k": 0.2}
    resolved = LtxVideoAdapter._block_mask_config(cfg)
    assert (resolved.q_block_size, resolved.k_block_size) == (64, 128)
    assert (resolved.top_p, resolved.min_top_k) == (0.9, 0.2)
    # The Wan2.1 defaults apply when the method section is silent.
    assert LtxVideoAdapter._block_mask_config({}) == LtxBlockMaskConfig()


def test_baseline_steps_and_negative_prompt_come_from_the_model_config():
    instance = adapter(baseline_steps=40, negative_prompt="blurry")
    assert instance.baseline_steps() == 40
    assert instance.negative_prompt() == "blurry"
    # Absent from the config, the negative prompt is the empty string.
    assert LtxVideoAdapter({"baseline_steps": 50}).negative_prompt() == ""


def test_cache_methods_require_an_assigned_request_cache_directory():
    instance = adapter()
    with pytest.raises(RuntimeError, match="require an assigned request cache directory"):
        instance._request_cache_root()
    instance.request_cache_dir = "/tmp/ltx-request"
    assert str(instance._request_cache_root()) == "/tmp/ltx-request"


def test_seed_only_randomises_a_negative_request_seed():
    request = SimpleNamespace(seed=7)
    assert LtxVideoAdapter._seed(request) == 7
    request.seed = -1
    assert 0 <= LtxVideoAdapter._seed(request) < 2**63


def test_teacache_is_refused_by_generate_before_any_model_work():
    with pytest.raises(ValueError, match="teacache is not integrated for LTX-Video"):
        adapter().generate(SimpleNamespace(uuid="x", prompt="p", seed=1), method="teacache")


# ---------------------------------------------------------------- latent I/O


def test_pointer_payloads_serialise_as_plain_json(tmp_path):
    instance = adapter()
    payload = {"format": LtxVideoAdapter.CHORUS2_FORMAT, "mask_dir": "/cache/masks", "steps": 50}
    path = tmp_path / "cache" / "source.json"
    instance.save_latent(payload, path)

    assert json.loads(path.read_text()) == payload
    assert instance.load_latent(path) == payload
    # The format is the LTX one, not Wan's or CogVideoX's.
    assert payload["format"] == "ltxvideo-chorus2-source/v1"
    assert LtxVideoAdapter.CARNATOR_FORMAT in LtxVideoAdapter.POINTER_FORMATS


def test_trajectory_manifest_keeps_its_tensors_in_a_sidecar(tmp_path):
    instance = adapter()
    trajectory = [torch.full((1, 2, 2), float(index)) for index in range(4)]
    path = tmp_path / "chorus" / "source.json"
    instance.save_latent(
        {"format": LtxVideoAdapter.TRAJECTORY_FORMAT, "steps": 3, "trajectory": trajectory}, path
    )

    manifest = json.loads(path.read_text())
    assert "trajectory" not in manifest  # tensors never land in the JSON
    assert manifest["trajectory_files"] == [f"latents_step_{i:03d}.pt" for i in range(4)]
    assert (tmp_path / "chorus" / "source.steps" / "latents_step_003.pt").is_file()

    loaded = instance.load_latent(path)
    assert loaded["steps"] == 3
    assert [p.name for p in loaded["trajectory_paths"]] == manifest["trajectory_files"]
    assert float(torch.load(loaded["trajectory_paths"][2]).mean()) == 2.0


def test_incomplete_trajectory_sidecar_is_rejected(tmp_path):
    instance = adapter()
    path = tmp_path / "source.json"
    instance.save_latent(
        {"format": LtxVideoAdapter.TRAJECTORY_FORMAT, "steps": 3,
         "trajectory": [torch.zeros(1, 2, 2) for _ in range(4)]},
        path,
    )
    (tmp_path / "source.steps" / "latents_step_002.pt").unlink()
    with pytest.raises(ValueError, match="trajectory sidecar is incomplete"):
        instance.load_latent(path)


def test_chorus_fusion_keeps_the_target_inside_medit_and_takes_the_source_outside():
    sample = torch.zeros(1, 4, 2)
    source = torch.ones(1, 4, 2)
    edit = torch.tensor([[[True, False], [False, True]]])  # both frames, one token each

    fused = LtxVideoAdapter._fuse_chorus_latent(sample, source, edit)
    assert fused[0, :, 0].tolist() == [0.0, 1.0, 1.0, 0.0]
    # The mask selects whole tokens, so it is the same on both channels.
    assert fused[0, :, 1].tolist() == [0.0, 1.0, 1.0, 0.0]


def test_chorus_fusion_rejects_a_mismatched_latent_or_mask():
    sample = torch.zeros(1, 4, 2)
    with pytest.raises(ValueError, match="latent shapes differ"):
        LtxVideoAdapter._fuse_chorus_latent(sample, torch.zeros(1, 4, 3), torch.zeros(1, 2, 2))
    with pytest.raises(ValueError, match="mask has 6 tokens but the latent has 4"):
        LtxVideoAdapter._fuse_chorus_latent(sample, torch.zeros_like(sample), torch.zeros(2, 3))


# ------------------------------------------------------ the probe prefix loop


def _delegate(**overrides):
    from adapters.ltx_pipeline import LtxVideoPipelineDelegate

    delegate = LtxVideoPipelineDelegate(config(**overrides))
    delegate.pipe = SimpleNamespace(scheduler=_FakeScheduler(), _execution_device="cpu")
    delegate.torch = torch
    return delegate


def _conditioning(branches: int = 3, tokens: int = 5, dim: int = 2) -> dict:
    """The triple-branch conditioning ``LtxVideoPipelineDelegate.encode`` builds."""
    return {
        "triple_embeds": torch.zeros(branches, tokens, dim),
        "triple_mask": torch.ones(branches, tokens, dtype=torch.bool),
    }


def test_probe_prefix_runs_the_first_steps_of_the_full_schedule(monkeypatch):
    """Carnator's 5-step probe is a *prefix* of the 50-step plan, not 5 steps."""
    delegate = _delegate()
    monkeypatch.setattr(
        delegate, "set_timesteps", lambda steps: torch.arange(float(steps), 0, -1.0)
    )
    monkeypatch.setattr(delegate, "_forward", lambda *args, **kwargs: torch.zeros(2, 4, 2))
    armed: list[int] = []

    latents, captured, _seconds, per_step = delegate.denoise(
        latents=torch.zeros(1, 4, 2), conditioning=_conditioning(), video_coords=torch.zeros(1, 3),
        steps=5, schedule_steps=50, begin_step_hook=armed.append,
    )

    assert [entry["completed_step"] for entry in per_step] == [1, 2, 3, 4, 5]
    # Normalised time is taken against the 50-step plan, so the guidance
    # segments the probe sees are the ones the full run would have used.
    assert [entry["normalized_time"] for entry in per_step] == pytest.approx(
        [1.0 - index / 49 for index in range(5)]
    )
    assert armed == [0, 1, 2, 3, 4]
    assert captured == {}
    assert latents.shape == (1, 4, 2)


def test_a_five_step_run_without_schedule_steps_rescales_instead(monkeypatch):
    """The contrast that makes the probe alignment matter."""
    delegate = _delegate()
    monkeypatch.setattr(delegate, "set_timesteps", lambda steps: torch.arange(float(steps), 0, -1.0))
    monkeypatch.setattr(delegate, "_forward", lambda *args, **kwargs: torch.zeros(2, 4, 2))

    _latents, _captured, _seconds, per_step = delegate.denoise(
        latents=torch.zeros(1, 4, 2), conditioning=_conditioning(), video_coords=torch.zeros(1, 3), steps=5,
    )
    assert [entry["normalized_time"] for entry in per_step] == pytest.approx(
        [1.0 - index / 4 for index in range(5)]
    )


def test_denoise_refuses_more_steps_than_the_schedule_it_was_given():
    delegate = _delegate()
    with pytest.raises(ValueError, match="steps must not exceed schedule_steps; got steps=51, schedule=50"):
        delegate.denoise(
            latents=torch.zeros(1, 4, 2), conditioning=_conditioning(), video_coords=torch.zeros(1, 3),
            steps=51, schedule_steps=50,
        )


def test_denoise_refuses_a_start_step_outside_the_executed_range(monkeypatch):
    delegate = _delegate()
    monkeypatch.setattr(delegate, "set_timesteps", lambda steps: torch.arange(float(steps), 0, -1.0))
    with pytest.raises(ValueError, match=r"start_step must be in \[0, 5\)"):
        delegate.denoise(
            latents=torch.zeros(1, 4, 2), conditioning=_conditioning(), video_coords=torch.zeros(1, 3),
            steps=5, start_step=5,
        )


def test_denoise_captures_completed_steps_and_lets_the_caller_replace_them(monkeypatch):
    delegate = _delegate()
    monkeypatch.setattr(delegate, "set_timesteps", lambda steps: torch.arange(float(steps), 0, -1.0))
    monkeypatch.setattr(delegate, "_forward", lambda *args, **kwargs: torch.zeros(2, 4, 2))

    seen: list[int] = []

    def on_step_end(completed: int, sample):
        seen.append(completed)
        return sample + completed

    _latents, captured, _seconds, per_step = delegate.denoise(
        latents=torch.zeros(1, 4, 2), conditioning=_conditioning(), video_coords=torch.zeros(1, 3), steps=3,
        on_step_end=on_step_end, capture_steps=(2,),
    )

    assert seen == [1, 2, 3]
    assert set(captured) == {2}
    # Step 1 leaves 1, replacement adds 1 -> 2; step 2 leaves 3, replacement
    # adds 2 -> 5. The capture has to be the post-replacement latent.
    assert float(captured[2].mean()) == 5.0
    assert captured[2].device.type == "cpu"
    assert [entry["branches"] for entry in per_step] == [["uncond", "cond"]] * 3


def test_denoise_reports_the_branches_a_step_actually_ran(monkeypatch):
    """CFG + STG gives three rows; the labels come from the slice, not the count."""
    delegate = _delegate(
        stg={"enabled": True, "guidance_timesteps": [1.0], "guidance_scale": [3.0],
             "stg_scale": [4.0], "rescaling_scale": [1.0], "skip_block_list": [[3]]},
    )
    delegate.pipe.transformer = SimpleNamespace(transformer_blocks=[object()] * 4)
    monkeypatch.setattr(delegate, "set_timesteps", lambda steps: torch.arange(float(steps), 0, -1.0))
    monkeypatch.setattr(delegate, "_forward", lambda *args, **kwargs: torch.zeros(3, 4, 2))

    _latents, _captured, _seconds, per_step = delegate.denoise(
        latents=torch.zeros(1, 4, 2), conditioning=_conditioning(), video_coords=torch.zeros(1, 3), steps=2,
    )
    assert [entry["num_conds"] for entry in per_step] == [3, 3]
    assert per_step[0]["branches"] == ["uncond", "cond", "cond_perturbed"]
    assert per_step[0]["stg_scale"] == 4.0
