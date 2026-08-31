from scripts.benchmark.aggregate_vidprom_300_results import aggregate_method


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
