from __future__ import annotations

import json

import pytest

from scripts.benchmark.build_cogvideo_hit_replay import METHODS, build_method_replay, build_replay


def _row(index: int, *, hit: bool, method: str) -> dict:
    source = f"source-{index}" if hit else None
    return {
        "arrival_index": index,
        "request_index": index,
        "uuid": f"target-{index}",
        "prompt": f"prompt {index}",
        "method": method,
        "status": "ok" if hit or method == "teacache" else "cache_miss",
        "error": None,
        "hit": hit,
        "hit_object": source,
    }


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_replay_requires_exactly_300_terminal_decisions(tmp_path):
    path = tmp_path / "request_records.jsonl"
    _write(path, [_row(index, hit=index % 2 == 0, method="nirvana_vid") for index in range(299)])

    with pytest.raises(ValueError, match="exactly 300"):
        build_method_replay(path, method="nirvana_vid")


def test_replay_preserves_hits_sources_and_legacy_errors(tmp_path):
    path = tmp_path / "request_records.jsonl"
    rows = [_row(index, hit=index != 1, method="chorus") for index in range(300)]
    rows[0]["status"] = "error"
    rows[0]["error"] = "legacy preprocessing failed"
    _write(path, rows)

    replay = build_method_replay(path, method="chorus")

    assert replay[0]["decision"] == "hit"
    assert replay[0]["source_uuid"] == "source-0"
    assert replay[0]["legacy_status"] == "error"
    assert replay[1]["decision"] == "miss"
    assert replay[1]["source_uuid"] is None


def test_teacache_is_explicitly_target_only(tmp_path):
    path = tmp_path / "request_records.jsonl"
    _write(path, [_row(index, hit=False, method="teacache") for index in range(300)])

    replay = build_method_replay(path, method="teacache")

    assert all(row["decision"] == "target_only" for row in replay)
    assert all(row["hit"] is None and row["source_uuid"] is None for row in replay)


def test_build_all_method_files(tmp_path):
    legacy = tmp_path / "legacy"
    for method in METHODS:
        rows = [_row(index, hit=method != "teacache" and index % 2 == 0, method=method) for index in range(300)]
        _write(legacy / method / "request_records.jsonl", rows)

    counts = build_replay(legacy, tmp_path / "replay")

    assert counts == {method: 300 for method in METHODS}
    manifest = json.loads((tmp_path / "replay" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "cogvideox15-hit-replay/v1"
