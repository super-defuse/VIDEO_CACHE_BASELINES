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
from scripts.run import chai_carnator_matrix


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


def test_chai_carnator_protocol_is_fixed_to_eight_steps_and_step_five_masks():
    assert chai_carnator_matrix.NAME == "CHAI-Carnator"
    assert chai_carnator_matrix.CC_STEPS == tuple(range(1, 9))
    assert chai_carnator_matrix.CC_CACHE_STEPS == (1, 2, 3, 4)
    assert chai_carnator_matrix.CC_MASK_PROBE_STEP == 5
    assert [item["baseline"] for item in chai_carnator_matrix._variant_specs()] == [
        "CC-B0", "CC-B1", "CC-B2", "CC-B3", "CC-B4", "CC-B5",
    ]
