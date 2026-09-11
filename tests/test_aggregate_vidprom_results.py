import json
import os

import pytest

from scripts.benchmark.aggregate_vidprom_300_results import (
    AggregationError,
    MODEL_METHODS,
    aggregate_method,
    main,
    rows_for,
    validate_quality_completeness,
)


def test_aggregate_method_reports_hit_only_timings(tmp_path):
    rows = [
        {"status": "ok", "hit": True, "latency_s": 4.0, "denoising_s": 3.0, "hit_e2e_s": 4.0, "hit_denoising_s": 3.0},
        {"status": "ok", "hit": False, "latency_s": 10.0, "denoising_s": 9.0, "hit_e2e_s": None, "hit_denoising_s": None},
    ]
    summary = aggregate_method("nirvana_vid", rows, tmp_path)
    assert summary["e2e_s_per_request"] == 7.0
    assert summary["hit_timed_requests"] == 1
    assert summary["hit_e2e_s_per_hit"] == 4.0
    assert summary["hit_denoising_s_per_hit"] == 3.0
    assert "retrieval_ms_per_request" not in summary


def _ok_row(**overrides):
    row = {
        "status": "ok",
        "latency_s": 500.0,
        "denoising_s": 480.0,
        "clip_t": 0.29,
        "vbench_q": 0.61,
    }
    row.update(overrides)
    return row


def _write_method(root, method, rows, *, results_rows=None, requests_mtime=1000.0, results_mtime=2000.0):
    method_root = root / method
    method_root.mkdir(parents=True, exist_ok=True)
    requests_path = method_root / "requests.jsonl"
    results_path = method_root / "results.jsonl"
    requests_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    os.utime(requests_path, (requests_mtime, requests_mtime))
    if results_rows is not None:
        results_path.write_text("".join(json.dumps(row) + "\n" for row in results_rows), encoding="utf-8")
        os.utime(results_path, (results_mtime, results_mtime))
    return requests_path, results_path


def _write_full_wan21_run(root, results_rows=None):
    for method in MODEL_METHODS["wan21"]:
        _write_method(root, method, [_ok_row()], results_rows=results_rows or [_ok_row()])


def test_rows_for_refuses_to_fall_back_to_the_raw_audit_log(tmp_path):
    requests_path, _ = _write_method(tmp_path, "chorus2", [_ok_row()], results_rows=None)

    with pytest.raises(AggregationError, match="missing scored results"):
        rows_for(requests_path)


def test_rows_for_rejects_results_older_than_the_run_log(tmp_path):
    # results.jsonl predates requests.jsonl: a stale artifact from an earlier
    # attempt that --overwrite would never have cleared.
    requests_path, results_path = _write_method(
        tmp_path, "chorus2", [_ok_row()], results_rows=[_ok_row()],
        requests_mtime=2000.0, results_mtime=1000.0,
    )

    with pytest.raises(AggregationError, match="stale scored results") as excinfo:
        rows_for(requests_path)
    message = str(excinfo.value)
    assert str(results_path) in message and str(requests_path) in message


def test_rows_for_rejects_dry_run_stub_rows(tmp_path):
    stub = [_ok_row(denoising_s=0.0, latency_s=1.37e-05)]
    requests_path, _ = _write_method(tmp_path, "baseline", stub, results_rows=stub)

    with pytest.raises(AggregationError, match="dry-run/stub results detected"):
        rows_for(requests_path)


def test_rows_for_accepts_a_healthy_run(tmp_path):
    requests_path, _ = _write_method(tmp_path, "chorus2", [_ok_row()], results_rows=[_ok_row()])

    assert len(rows_for(requests_path)) == 1


def test_quality_completeness_rejects_partially_missing_quality(tmp_path):
    rows = [_ok_row(), _ok_row(clip_t=None, vbench_q=None)]
    path = tmp_path / "results.jsonl"

    with pytest.raises(AggregationError, match="incomplete quality metrics"):
        validate_quality_completeness("carnator_boundary_b5", rows, path)


def test_quality_completeness_rejects_a_method_with_no_successful_requests(tmp_path):
    with pytest.raises(AggregationError, match="no successful requests"):
        validate_quality_completeness("chorus", [{"status": "cache_miss"}], tmp_path / "results.jsonl")


def test_quality_completeness_accepts_a_complete_run(tmp_path):
    validate_quality_completeness("chorus", [_ok_row()], tmp_path / "results.jsonl")


def test_quality_completeness_can_be_downgraded_to_a_warning(tmp_path, capsys):
    rows = [_ok_row(clip_t=None, vbench_q=None)]

    validate_quality_completeness("chorus", rows, tmp_path / "results.jsonl", allow_incomplete=True)

    assert "incomplete quality metrics" in capsys.readouterr().err


def test_main_refuses_to_write_a_summary_when_quality_is_missing(tmp_path, monkeypatch, capsys):
    quality_less = [_ok_row(clip_t=None, vbench_q=None)]
    _write_full_wan21_run(tmp_path, results_rows=quality_less)
    monkeypatch.setattr("sys.argv", ["aggregate", "--model", "wan21", "--results-root", str(tmp_path)])

    assert main() == 1
    assert not (tmp_path / "summary.csv").exists()
    assert "aggregation refused" in capsys.readouterr().err


def test_main_writes_the_summary_when_the_escape_hatch_is_used(tmp_path, monkeypatch):
    quality_less = [_ok_row(clip_t=None, vbench_q=None)]
    _write_full_wan21_run(tmp_path, results_rows=quality_less)
    monkeypatch.setattr(
        "sys.argv",
        ["aggregate", "--model", "wan21", "--results-root", str(tmp_path), "--allow-incomplete-quality"],
    )

    assert main() == 0
    assert (tmp_path / "summary.csv").exists()


def test_main_refuses_a_stale_run_and_leaves_no_summary(tmp_path, monkeypatch, capsys):
    for method in MODEL_METHODS["wan21"]:
        _write_method(
            tmp_path, method, [_ok_row()], results_rows=[_ok_row()],
            requests_mtime=2000.0, results_mtime=1000.0,
        )
    monkeypatch.setattr("sys.argv", ["aggregate", "--model", "wan21", "--results-root", str(tmp_path)])

    assert main() == 1
    assert not (tmp_path / "summary.csv").exists()
    assert "stale scored results" in capsys.readouterr().err
