from argparse import Namespace
import json
from types import SimpleNamespace

import pytest

from core import CacheMatch, RequestResult
from scripts.benchmark.aggregate_vidprom_300_results import aggregate_method
import scripts.benchmark.run_vidprom_300_experiment as experiment
from scripts.benchmark.run_vidprom_300_experiment import (
    SMOKE_FORCED_HIT_SOURCE_UUID,
    append_result_and_record,
    force_cache_match,
    forced_carnator_probe_match,
    forced_smoke_decision,
    pending_rows_for_resume,
    validate_forced_smoke,
)


def smoke_args(enabled: bool = True) -> Namespace:
    return Namespace(smoke_force_outcomes=enabled)


def source_rows():
    return [{"request_index": 217, "uuid": SMOKE_FORCED_HIT_SOURCE_UUID, "prompt": "a hamster with a man"}]


def test_forced_smoke_has_exactly_one_miss_then_one_hit():
    args = smoke_args()

    assert forced_smoke_decision(args, 0) == "miss"
    assert forced_smoke_decision(args, 1) == "hit"
    assert forced_smoke_decision(smoke_args(False), 1) is None
    with pytest.raises(ValueError, match="no decision"):
        forced_smoke_decision(args, 2)


def test_forced_smoke_rejects_a_different_two_request_slice():
    with pytest.raises(ValueError, match="request-indices 0,1"):
        validate_forced_smoke(
            smoke_args(),
            [{"request_index": 1}, {"request_index": 2}],
            source_rows(),
        )


def test_forced_hit_uses_the_fixed_source_and_preserves_nirvana_restart_step():
    observed = CacheMatch(True, "normal-source", 3, "prompt", 0.91)
    config = {"retrieval": {"nirvana_vid": {"reuse_step": 10}}}

    match = force_cache_match(
        observed,
        outcome="hit",
        request_index=1,
        method_name="nirvana_vid",
        source_by_uuid={row["uuid"]: row for row in source_rows()},
        config=config,
    )

    assert match.hit is True
    assert match.kind == "forced_smoke_hit"
    assert match.source_uuid == SMOKE_FORCED_HIT_SOURCE_UUID
    assert match.source_request_index == 217
    assert match.metadata["reuse_step"] == 10
    assert match.metadata["forced_smoke"]["observed_match"]["source_uuid"] == "normal-source"


def test_forced_miss_cannot_load_a_source_and_carnator_still_probes():
    observed = CacheMatch(True, "normal-source", 3, "prompt", 0.91)

    miss = force_cache_match(
        observed,
        outcome="miss",
        request_index=0,
        method_name="chorus2",
        source_by_uuid={row["uuid"]: row for row in source_rows()},
        config={},
    )
    probe = forced_carnator_probe_match(observed, request_index=0, outcome="miss")

    assert miss.hit is False
    assert miss.source_uuid is None
    assert miss.kind == "forced_smoke_miss"
    assert probe.kind == "probe_required"
    assert probe.metadata["probe_required"] is True
    assert probe.metadata["forced_smoke"]["outcome"] == "miss"


def test_resume_skips_matching_terminal_error_records(tmp_path):
    rows = [
        {"arrival_index": 0, "request_index": 4, "uuid": "first", "prompt": "first prompt"},
        {"arrival_index": 1, "request_index": 5, "uuid": "second", "prompt": "second prompt"},
    ]
    complete = {**rows[0], "status": "ok"}
    failed = {**rows[1], "status": "error"}
    (tmp_path / "requests.jsonl").write_text(
        "\n".join(json.dumps(item) for item in (complete, failed)) + "\n", encoding="utf-8"
    )
    (tmp_path / "request_records.jsonl").write_text(
        "\n".join(json.dumps(item) for item in (complete, failed)) + "\n", encoding="utf-8"
    )

    pending, resumed = pending_rows_for_resume(tmp_path, rows, resume=True)

    assert pending == []
    assert resumed == 2


def test_error_result_is_written_to_both_audit_logs(tmp_path):
    row = {"arrival_index": 8, "request_index": 12, "uuid": "target", "prompt": "target prompt"}
    payload = {
        **row,
        "status": "error",
        "error": "RuntimeError: target failure",
        "cache_match": {"hit": True, "source_uuid": "source", "kind": "prompt", "score": 0.9, "metadata": {}},
        "latency_s": 12.5,
        "denoising_s": None,
        "external_s": None,
        "hit_e2e_s": None,
        "hit_denoising_s": None,
    }

    append_result_and_record(tmp_path, payload, row, "chorus")

    result = json.loads((tmp_path / "requests.jsonl").read_text(encoding="utf-8"))
    record = json.loads((tmp_path / "request_records.jsonl").read_text(encoding="utf-8"))
    assert result["status"] == record["status"] == "error"
    assert result["error"] == record["error"] == "RuntimeError: target failure"
    assert record["hit"] is True
    assert record["hit_object"] == "source"
    pending, resumed = pending_rows_for_resume(tmp_path, [row], resume=True)
    assert pending == []
    assert resumed == 1


def test_aggregation_excludes_terminal_errors_from_hit_rate(tmp_path):
    rows = [
        {"status": "ok", "hit": True, "latency_s": 10.0, "denoising_s": 5.0, "external_s": 1.0},
        {"status": "cache_miss", "hit": False},
        {"status": "error", "hit": True, "latency_s": 2.0, "error": "target failed"},
    ]

    summary = aggregate_method("chorus", rows, tmp_path)

    assert summary["requests"] == 3
    assert summary["completed"] == 2
    assert summary["errors"] == 1
    assert summary["hits"] == 1
    assert summary["hit_rate"] == 0.5


def test_generic_run_records_an_error_and_continues(monkeypatch, tmp_path):
    rows = [
        {"arrival_index": 0, "request_index": 0, "uuid": "first", "prompt": "first prompt"},
        {"arrival_index": 1, "request_index": 1, "uuid": "second", "prompt": "second prompt"},
        {"arrival_index": 2, "request_index": 2, "uuid": "third", "prompt": "third prompt"},
    ]
    outcomes = iter([
        RequestResult(0, "first", "first prompt", "synthetic", "chorus", 1, CacheMatch(True, "source"), None, 0.1,
                      "error", error="RuntimeError: target failure"),
        RequestResult(1, "second", "second prompt", "synthetic", "chorus", 2, CacheMatch(), 0, 0.0, "cache_miss"),
        RequestResult(2, "third", "third prompt", "synthetic", "chorus", 3, CacheMatch(True, "source"), 45, 1.0, "ok",
                      output_path="/tmp/third.mp4", quality_input_path="/tmp/third.mp4", extra={"dit_forward_s": 0.5}),
    ])

    class FakeRunner:
        def __init__(self, *_args, **_kwargs):
            pass

        def process(self, _request):
            return next(outcomes)

    method = SimpleNamespace(config={})
    monkeypatch.setattr(experiment, "make_adapter", lambda *_args, **_kwargs: SimpleNamespace(model_name="synthetic"))
    monkeypatch.setattr(experiment, "make_method", lambda *_args, **_kwargs: method)
    monkeypatch.setattr(experiment, "Runner", FakeRunner)
    monkeypatch.setattr(experiment, "view_rows_for_preload", lambda _view: [])
    args = Namespace(method="chorus", gpu=5, dry_run=True, seed=1, resume=False, smoke_force_outcomes=False)

    experiment.generic_run(args, rows, tmp_path, tmp_path)

    results = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [result["status"] for result in results] == ["error", "cache_miss", "ok"]
    assert results[0]["error"] == "RuntimeError: target failure"


def test_resume_rejects_mismatched_audit_files(tmp_path):
    rows = [{"arrival_index": 0, "request_index": 4, "uuid": "first", "prompt": "first prompt"}]
    (tmp_path / "requests.jsonl").write_text(json.dumps({**rows[0], "status": "ok"}) + "\n", encoding="utf-8")
    (tmp_path / "request_records.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="disagree"):
        pending_rows_for_resume(tmp_path, rows, resume=True)
