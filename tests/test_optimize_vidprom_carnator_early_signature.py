from __future__ import annotations

import pytest

from scripts.benchmark.optimize_vidprom_carnator_early_signature import (
    DistanceAuditError,
    audit_rows,
    coverage_band,
    threshold_values,
)


def _row(request_index: int, *, same_noun_set: bool, null_substitution_distances: bool = False) -> dict:
    metrics = {
        "context_color_ciede2000": 1.0,
        "log_area_ratio": None if same_noun_set or null_substitution_distances else 0.4,
        "shape_one_minus_dice": None if same_noun_set or null_substitution_distances else 0.2,
        "changed_color_ciede2000": None if same_noun_set or null_substitution_distances else 1.2,
    }
    source_only = [] if same_noun_set else ["source"]
    target_only = [] if same_noun_set else ["target"]
    return {
        "request_index": request_index,
        "uuid": f"uuid-{request_index}",
        "prompt": "prompt",
        "clip_t": 0.3,
        "vbench_q": 0.6,
        "cache_match": {
            "metadata": {
                "selected_metrics": metrics,
                "inference_mask_selection": {
                    "source_only_nouns": source_only,
                    "target_only_nouns": target_only,
                },
            },
        },
    }


def test_audit_normalises_only_inapplicable_same_noun_distances():
    records, audit = audit_rows([_row(0, same_noun_set=True), _row(1, same_noun_set=False)])

    assert records[0].distances == (1.0, 0.0, 0.0, 0.0)
    assert records[1].distances == (1.0, 0.4, 0.2, 1.2)
    assert audit["same_noun_set_request_indices"] == [0]
    assert audit["normalised_inapplicable_to_zero"]["log_area_ratio"] == [0]


def test_audit_rejects_missing_substitution_distance():
    with pytest.raises(DistanceAuditError) as error:
        audit_rows([_row(0, same_noun_set=False, null_substitution_distances=True)])

    assert '"log_area_ratio": [0]' in str(error.value)


def test_threshold_values_keep_only_individually_feasible_observed_cutoffs():
    assert threshold_values([0.1, 0.2, 0.3, 0.4], minimum_accepted=3, maximum_values=8) == [0.3, 0.4]


def test_coverage_band_is_stable_at_a_decimal_boundary():
    assert coverage_band(0.6, 0.05) == (0.6, 0.65)
