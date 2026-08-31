from argparse import Namespace

import pytest

from core import CacheMatch
from scripts.benchmark.run_vidprom_300_experiment import (
    SMOKE_FORCED_HIT_SOURCE_UUID,
    force_cache_match,
    forced_carnator_probe_match,
    forced_smoke_decision,
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
