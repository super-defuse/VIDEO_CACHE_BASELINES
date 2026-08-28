from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.benchmark.run_vidprom_forced_early_signature_calibration import (
    ForcedPair,
    PromptRow,
    SourceRow,
    read_pairs,
    select_nearest_pairs,
    write_pairs,
)


def _source(tmp_path: Path, uuid: str, prompt: str) -> SourceRow:
    run_dir = tmp_path / uuid / "carnator"
    cache_dir = run_dir / "cache" / "source_dense"
    signature = cache_dir / "online_color_signature.json"
    return SourceRow(uuid, prompt, run_dir, cache_dir, signature, ("object",))


def test_select_nearest_pairs_excludes_self_and_uses_highest_cosine(tmp_path):
    target = PromptRow(0, "same", "target")
    sources = [_source(tmp_path, "same", "self"), _source(tmp_path, "other", "best"), _source(tmp_path, "third", "next")]
    pairs = select_nearest_pairs(
        [target], sources,
        [[1.0, 0.0]],
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
    )
    assert len(pairs) == 1
    assert pairs[0].source_uuid == "other"
    assert pairs[0].source_uuid != pairs[0].target_uuid
    assert pairs[0].pair_source == "clip_nearest_excluding_self"


def test_forced_pair_csv_round_trip_preserves_similarity_and_paths(tmp_path):
    source = _source(tmp_path, "source", "source prompt")
    pair = ForcedPair(
        "clip_forced_0001", 0, "target", "target prompt", ("cat",),
        source.uuid, source.prompt, source.source_nouns, 0.8765,
        "clip_nearest_excluding_self", source.source_run_dir,
        source.source_cache_dir, source.source_signature_path,
    )
    path = tmp_path / "pairs.csv"
    write_pairs(path, [pair])
    loaded = read_pairs(path)
    assert loaded == [pair]
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["target_nouns"]) == ["cat"]
    assert float(row["clip_similarity"]) == 0.8765


def test_select_nearest_pairs_rejects_embedding_count_mismatch(tmp_path):
    target = PromptRow(0, "target", "target")
    source = _source(tmp_path, "source", "source")
    try:
        select_nearest_pairs([target], [source], [], [[1.0]])
    except ValueError as exc:
        assert "embedding count" in str(exc)
    else:
        raise AssertionError("expected embedding count validation")
