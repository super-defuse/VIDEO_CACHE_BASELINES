from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters import make_adapter
from core import Request, Runner
from core.registry import make_method


def run(tmp_path: Path, method_name: str):
    config = {"dry_run": True, "seed": 3, "cache_root": "cache", "max_cache_bytes": 1024}
    adapter = make_adapter("synthetic", config)
    method = make_method(method_name, config)
    runner = Runner(adapter, method, model="synthetic", output_dir=tmp_path / method_name,
                    cache_root=tmp_path / "cache")
    request = Request(1, "one", "a red car", 4)
    return runner, runner.process(request), runner.process(Request(2, "two", "a red car", 5))


def test_chai_cold_then_hot_and_namespace(tmp_path):
    runner, cold, hot = run(tmp_path, "chai")
    assert cold.status == hot.status == "ok"
    assert not cold.cache_match.hit
    assert hot.cache_match.hit
    assert cold.executed_steps == 50
    assert hot.executed_steps == 8
    assert (tmp_path / "cache" / "synthetic" / "chai").is_dir()
    assert not (tmp_path / "cache" / "synthetic" / "baseline").exists()


def test_chorus_uses_source_prompt_cache_on_synthetic(tmp_path):
    config = {"dry_run": True, "seed": 3, "cache_root": "cache", "max_cache_bytes": 1024}
    adapter = make_adapter("synthetic", config)
    method = make_method("chorus", config)
    runner = Runner(adapter, method, model="synthetic", output_dir=tmp_path / "chorus",
                    cache_root=tmp_path / "cache")
    source = runner.process(Request(0, "source", "a dog", 4))
    target = runner.process(Request(1, "target", "a cat", 4, cache_prompt="a dog"))
    assert source.status == target.status == "ok"
    assert target.cache_match.kind == "source_prompt"
    assert target.executed_steps == 8


def test_chorus_no_tgaa_is_registered_as_a_cross_request_variant():
    method = make_method("chorus_no_tgaa", {"dry_run": True})
    assert method.name == "chorus_no_tgaa"
    assert method.capabilities.cross_request_cache
    assert method.capabilities.source_trajectory


def test_chorus_mbase_only_is_registered_as_a_cross_request_variant():
    method = make_method("chorus_mbase_only", {"dry_run": True})
    assert method.name == "chorus_mbase_only"
    assert method.capabilities.cross_request_cache
    assert method.capabilities.source_trajectory


def test_chorus_mbase_only_selects_base_for_updates_and_keeps_see_context():
    import torch

    from adapters.wan22 import Wan22Adapter

    bundle = {
        "token_base": torch.tensor([True, False, False]),
        "token_edit": torch.tensor([True, True, False]),
        "token_see": torch.tensor([True, True, True]),
        "latent_base": torch.tensor([[[True, False]]]),
        "latent_edit": torch.tensor([[[True, True]]]),
    }
    selected = Wan22Adapter._select_chorus_update_masks(bundle, "Mbase")
    assert torch.equal(selected["token_edit"], bundle["token_base"])
    assert torch.equal(selected["latent_edit"], bundle["latent_base"])
    assert torch.equal(selected["token_see"], bundle["token_see"])
    assert selected["update_mask"] == "Mbase"


def test_chorus_tgaa_probes_are_registered():
    for name in ("chorus_tgaa_joint_gamma_1_6", "chorus_tgaa_joint_gamma_2_1",
                 "chorus_tgaa_joint_gamma_2_0"):
        method = make_method(name, {"dry_run": True})
        assert method.name == name
        assert method.capabilities.cross_request_cache


def test_chorus_latent_fusion_matches_carnator_region_semantics():
    import torch

    from adapters.wan22 import Wan22Adapter

    sample = torch.full((2, 2, 3, 4), 7.0)
    source = torch.full_like(sample, -3.0)
    edit = torch.zeros((2, 3, 4), dtype=torch.bool)
    edit[:, 1:, 2:] = True
    fused = Wan22Adapter._fuse_chorus_latent(sample, source, edit)

    assert torch.equal(fused[:, :, :1, :], source[:, :, :1, :])
    assert torch.equal(fused[:, :, 1:, 2:], sample[:, :, 1:, 2:])
    assert torch.equal(fused[:, :, 1:, :2], source[:, :, 1:, :2])


def test_chorus_scheduler_history_reset_preserves_global_step():
    from adapters.wan22 import Wan22Adapter

    class Scheduler:
        model_outputs = ["a", "b"]
        timestep_list = ["t0", "t1"]
        lower_order_nums = 2
        last_sample = "sample"
        this_order = 2
        _step_index = 40

    scheduler = Scheduler()
    Wan22Adapter._reset_chorus_scheduler_history(scheduler)
    assert scheduler.model_outputs == [None, None]
    assert scheduler.timestep_list == [None, None]
    assert scheduler.lower_order_nums == 0
    assert scheduler.last_sample is None
    assert scheduler.this_order == 1
    assert scheduler._step_index == 40


def test_native_srd_keeps_native_head_dense():
    import torch

    from adapters.wan22 import Wan22Adapter

    class Block:
        def forward(self, *args):
            return args[0]

    class Head:
        def __init__(self):
            self.calls = 0

        def forward(self, x, e):
            self.calls += 1
            return x + e

    block = Block()
    head = Head()
    model = type("Model", (), {"blocks": [block], "head": head})()
    state = {"active": True}
    patches = Wan22Adapter({})._install_native_srd(
        model,
        {"token_edit": torch.tensor([True, False]), "token_see": torch.tensor([True, True])},
        state,
    )
    try:
        x = torch.ones((1, 2, 3))
        result = head.forward(x, torch.ones_like(x))
        assert head.calls == 1
        assert torch.equal(result, x + 1)
    finally:
        Wan22Adapter._restore_native_srd(patches)


def test_native_tgaa_is_all_blocks_stage_two_gated(monkeypatch):
    import types

    import pytest
    import torch

    from adapters.wan22 import Wan22Adapter

    observed: list[torch.Tensor] = []

    def flash_attention(query, key, value, *, k_lens):
        del query, value, k_lens
        observed.append(key.detach().clone())
        return key

    wan_module = types.ModuleType("wan")
    wan_modules = types.ModuleType("wan.modules")
    attention_module = types.ModuleType("wan.modules.attention")
    attention_module.flash_attention = flash_attention
    wan_module.modules = wan_modules
    wan_modules.attention = attention_module
    monkeypatch.setitem(sys.modules, "wan", wan_module)
    monkeypatch.setitem(sys.modules, "wan.modules", wan_modules)
    monkeypatch.setitem(sys.modules, "wan.modules.attention", attention_module)

    class Attention:
        num_heads = 1
        head_dim = 2

        def __init__(self):
            self.base_calls = 0

        @staticmethod
        def norm_q(value):
            return value

        @staticmethod
        def q(value):
            return value

        @staticmethod
        def norm_k(value):
            return value

        @staticmethod
        def k(value):
            return value

        @staticmethod
        def v(value):
            return value

        @staticmethod
        def o(value):
            return value

        def forward(self, x, context, context_lens):
            del context, context_lens
            self.base_calls += 1
            return torch.full_like(x, -1.0)

    attentions = [Attention() for _ in range(3)]
    model = types.SimpleNamespace(
        blocks=[types.SimpleNamespace(cross_attn=attention) for attention in attentions]
    )
    state = {"tgaa_active": False, "step": 10, "total_steps": 50, "tgaa_calls": 0}
    adapter = Wan22Adapter({})
    patches = adapter._install_native_tgaa(
        model, {"tgaa_blocks": "all", "gamma_k": 1.15, "gamma_o": 1.05}, [1], state
    )
    assert len(patches) == len(attentions)

    x = torch.ones((1, 2, 2))
    context = torch.ones((1, 2, 2))
    for attention in attentions:
        assert torch.equal(attention.forward(x, context, None), torch.full_like(x, -1.0))
    assert [attention.base_calls for attention in attentions] == [1, 1, 1]
    assert state["tgaa_calls"] == 0

    state["tgaa_active"] = True
    progress = state["step"] / (state["total_steps"] - 1)
    key_scale = 1.0 + 0.15 * (1.0 - progress)
    output_scale = 1.0 + 0.05 * (1.0 - progress)
    for attention in attentions:
        result = attention.forward(x, context, None)
        assert result[0, 1].tolist() == pytest.approx([key_scale * output_scale] * 2)
    assert [attention.base_calls for attention in attentions] == [1, 1, 1]
    assert state["tgaa_calls"] == len(attentions)
    assert all(torch.allclose(key[0, 1], torch.full_like(key[0, 1], key_scale)) for key in observed)
    assert torch.equal(context, torch.ones_like(context))

    with pytest.raises(ValueError, match="requires at least one"):
        adapter._install_native_tgaa(model, {"tgaa_blocks": "all"}, [], state)


def test_cache_namespace_isolated_by_model(tmp_path):
    config = {"dry_run": True, "cache_root": "cache", "max_cache_bytes": 1024}
    first = Runner(make_adapter("synthetic", config), make_method("chai", config),
                   model="Wan2.2", output_dir=tmp_path / "wan", cache_root=tmp_path / "cache")
    second = Runner(make_adapter("synthetic", config), make_method("chai", config),
                    model="CogVideoX1.5", output_dir=tmp_path / "cog", cache_root=tmp_path / "cache")
    first.process(Request(0, "same", "a red car", 1))
    result = second.process(Request(0, "same", "a red car", 1))
    assert not result.cache_match.hit
    assert (tmp_path / "cache" / "Wan2.2" / "chai").is_dir()
    assert (tmp_path / "cache" / "CogVideoX1.5" / "chai").is_dir()


def test_baseline_does_not_populate_cache(tmp_path):
    runner, result, _ = run(tmp_path, "baseline")
    assert result.status == "ok"
    assert not result.cache_match.hit
    assert runner.cache.bytes_used == 0


def test_result_is_jsonl_compatible(tmp_path):
    runner, result, _ = run(tmp_path, "chai")
    path = tmp_path / "results.jsonl"
    runner.run([Request(0, "zero", "a blue sky", 3)], path)
    row = json.loads(path.read_text().splitlines()[0])
    assert {"model", "method", "seed", "cache_match", "executed_steps", "latency_s", "quality_input_path"} <= row.keys()


def test_chai_model_configs_match_the_opensora_eight_step_schedule():
    for model in ("wan22", "cogvideox15"):
        config = yaml.safe_load((ROOT / "configs" / "models" / f"{model}.yaml").read_text())
        assert config["fast_steps"] == 8
        assert config["cache_attention"] == {"blocks": [0], "steps": [1, 2, 3, 4]}


def test_external_sources_are_repository_local_and_locked():
    from core.external import external_path, locked_sources

    sources = locked_sources()
    assert {"wan22", "cogvideo", "lightx2v", "spargeattn", "grounded_sam2"} <= set(sources)
    for name in ("wan22", "cogvideo", "lightx2v", "spargeattn", "grounded_sam2"):
        assert sources[name]["path"].startswith("external/")
        assert external_path(name).is_dir()


def test_runtime_code_has_no_sibling_repository_imports():
    legacy_roots = ("../CHAI", "../CARNATOR_ICLR2027", "../Grounded-SAM-2")
    for directory in (ROOT / "adapters", ROOT / "core", ROOT / "methods", ROOT / "carnator", ROOT / "scripts"):
        for source in directory.rglob("*.py"):
            content = source.read_text(encoding="utf-8")
            assert not any(legacy in content for legacy in legacy_roots), source
