from __future__ import annotations

from scripts.run.shared_cache_wan_baseline import build_target_request, resolved_config


def test_shared_target_request_keeps_source_prompt_for_chorus_diff():
    request = build_target_request("a dog", "a cat", 20260218)

    assert request.prompt == "a cat"
    assert request.cache_prompt == "a dog"
    assert request.seed == 20260218


def test_resolved_config_applies_paired_chorus_gamma_overrides():
    config = resolved_config("cuda:0", chorus_gamma_k=1.5, chorus_gamma_o=1.5)

    assert config["chorus"]["gamma_k"] == 1.5
    assert config["chorus"]["gamma_o"] == 1.5
