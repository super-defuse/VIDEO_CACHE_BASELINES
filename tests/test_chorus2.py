from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from chorus2 import wan_ti2v_5b as chorus2
from carnator.outer_sparse import OuterSparseExecutor


def test_chorus2_runs_full_steps_and_starts_sparse_masks_after_step_ten():
    args = chorus2.build_parser().parse_args([])

    assert args.reuse_start_step == 0
    assert chorus2.SPARSE_START_STEP == 10
    assert args.layer_refresh_size == 8
    assert args.outer_sparse_dtype == "fp16"
    assert list(chorus2.ALL_BLOCKS) == list(range(30))
    assert not hasattr(args, "latent_injection_mode")


def test_chorus2_dry_run_records_full_target_without_latent_restart(tmp_path):
    args = chorus2.build_parser().parse_args([
        "--dry-run",
        "--artifact-root", str(tmp_path / "artifacts"),
        "--results-root", str(tmp_path / "results"),
        "--run-id", "smoke",
    ])
    args.outer_attn_backend = "sparge-sage2"

    summary = chorus2.run_experiment(args)
    written = json.loads((tmp_path / "results" / "smoke" / "summary.json").read_text())

    assert summary["status"] == "dry_run"
    assert written["config"]["latent_reuse"] is False
    assert written["config"]["reuse_start_step"] == 0
    assert written["config"]["target_denoising_steps"] == 50
    assert written["config"]["sparse_attention"]["backend"] == "sparge-sage2"
    assert written["config"]["sparse_attention"]["start_step"] == 11
    assert written["config"]["sparse_attention"]["layer_refresh_size"] == 8
    assert written["config"]["sparse_attention"]["layers"] == list(range(30))


def test_native_compact_runs_preserve_pure_dit_time_and_steps():
    manifest = {
        "output": "target.mp4",
        "completed_steps": 50,
        "timing": {
            "loop_denoise_seconds": 20.0,
            "dit_forward_seconds": 15.0,
            "end_to_end_seconds": 30.0,
        },
    }
    from carnator.adapters.wan_ti2v_5b import compact_run as compact_carnator_run

    for compact in (chorus2.compact_run(manifest), compact_carnator_run(manifest)):
        assert compact["dit_forward_seconds"] == 15.0
    assert compact["completed_steps"] == 50


def test_chorus2_rejects_an_invalid_reuse_start_step():
    args = chorus2.build_parser().parse_args(["--reuse-start-step", "10"])

    with pytest.raises(ValueError, match="reuse-start-step"):
        chorus2.validate(args)


def test_chorus2_rejects_an_incompatible_spargeattn_block_layout():
    args = chorus2.build_parser().parse_args(["--outer-mask-q-block", "128"])

    with pytest.raises(ValueError, match="SpargeAttn"):
        chorus2.validate(args)


def test_chorus2_requires_complete_external_mask_coverage():
    args = chorus2.build_parser().parse_args(["--steps", "11", "--guidance-scale", "1"])
    complete = SimpleNamespace(entries={(11, "cond", layer): {} for layer in range(30)})

    chorus2.verify_source_mask_coverage(args, complete)

    incomplete = SimpleNamespace(entries={(11, "cond", layer): {} for layer in range(29)})
    with pytest.raises(ValueError, match="does not fully cover"):
        chorus2.verify_source_mask_coverage(args, incomplete)


def test_chorus2_rejects_a_source_latent_cache():
    args = chorus2.build_parser().parse_args(["--source-latent-cache", "source"])

    with pytest.raises(ValueError, match="source-latent-cache"):
        chorus2.validate(args)


def test_chorus2_uses_only_a_full_target_c0_as_speedup_reference(tmp_path):
    c0 = tmp_path / "c0.json"
    c0.write_text(json.dumps({"steps": 50, "resume": {"resume_start_step": 0}, "timing": {"end_to_end_seconds": 300.0}}))
    assert chorus2.load_speedup_reference(c0) == {"manifest_path": str(c0), "end_to_end_seconds": 300.0}

    restart = tmp_path / "restart.json"
    restart.write_text(json.dumps({"steps": 50, "resume": {"resume_start_step": 10}, "timing": {"end_to_end_seconds": 200.0}}))
    with pytest.raises(ValueError, match="restart"):
        chorus2.load_speedup_reference(restart)


def test_layer_refresh_completes_each_group_on_its_last_layer():
    cache = SimpleNamespace(entries={(10, "cond", 0): {}, (10, "cond", 1): {}})
    executor = OuterSparseExecutor(cache, "flashinfer", layer_refresh_size=2)
    first = torch.tensor([[[True, False], [False, False]]])
    second = torch.tensor([[[False, True], [False, False]]])

    refreshed_first, first_added = executor._apply_layer_refresh(first, step=10, branch="cond", layer=0)
    refreshed_second, second_added = executor._apply_layer_refresh(second, step=10, branch="cond", layer=1)

    assert torch.equal(refreshed_first, first)
    assert first_added == 0
    assert torch.equal(refreshed_first | refreshed_second, torch.ones_like(second))
    assert second_added == 2
    assert executor.stats["refresh_events"] == 1
