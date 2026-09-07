from __future__ import annotations

import pytest

from scripts.benchmark.summarize_vidprom_carnator_quality_prefixes import QualityRecord, parse_percentages, prefix_summary


def test_prefix_summary_ranks_each_metric_and_balanced_score():
    records = [
        QualityRecord(request_index=0, uuid="zero", clip_t=0.9, vbench_q=0.9),
        QualityRecord(request_index=1, uuid="one", clip_t=0.8, vbench_q=0.1),
        QualityRecord(request_index=2, uuid="two", clip_t=0.1, vbench_q=0.8),
        QualityRecord(request_index=3, uuid="three", clip_t=0.2, vbench_q=0.2),
    ]

    summary = prefix_summary(records, (25, 50))
    rows = {(row["ranking_method"], row["top_percent"]): row for row in summary["rows"]}

    assert summary["selected_request_indices"]["clip_t"]["25"] == [0]
    assert summary["selected_request_indices"]["clip_t"]["50"] == [0, 1]
    assert summary["selected_request_indices"]["vbench_q"]["50"] == [0, 2]
    assert summary["selected_request_indices"]["balanced_z"]["50"] == [0, 1]
    assert rows[("clip_t", 50)]["clip_t_mean"] == pytest.approx(0.85)
    assert rows[("clip_t", 50)]["vbench_q_mean"] == pytest.approx(0.5)
    assert rows[("vbench_q", 50)]["clip_t_mean"] == pytest.approx(0.5)
    assert rows[("vbench_q", 50)]["vbench_q_mean"] == pytest.approx(0.85)


def test_parse_percentages_requires_increasing_unique_integer_values():
    assert parse_percentages("5,10,90") == (5, 10, 90)

    for value in ("", "10,5", "5,5", "0,5", "five"):
        try:
            parse_percentages(value)
        except ValueError:
            continue
        raise AssertionError(f"expected parse_percentages({value!r}) to fail")
