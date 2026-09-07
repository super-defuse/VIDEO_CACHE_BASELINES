#!/usr/bin/env bash
# Persistent tmux worker launcher for the interrupted 300-request experiment.
set -euo pipefail

method=${1:?method is required}
root=/home/tangxuebin/project/VIDEO_CACHE_BASELINES
output_root=artifacts/vidprom_1000cache_300input_noun_match/early_signature_disabled_20260903_100305
cache_view=artifacts/vidprom_1000cache_300input_noun_match/source_view

cd "$root"

case "$method" in
  nirvana_vid)
    exec conda run --no-capture-output -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. \
      python scripts/benchmark/run_vidprom_300_experiment.py --method "$method" --gpu 3 --limit 300 --resume \
      --cache-view "$cache_view" --output-root "$output_root"
    ;;
  teacache)
    exec conda run --no-capture-output -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. \
      python scripts/benchmark/run_vidprom_300_experiment.py --method "$method" --gpu 4 --limit 300 --resume \
      --cache-view "$cache_view" --output-root "$output_root"
    ;;
  chorus)
    exec conda run --no-capture-output -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. \
      python scripts/benchmark/run_vidprom_300_experiment.py --method "$method" --gpu 5 --limit 300 --resume \
      --cache-view "$cache_view" --output-root "$output_root"
    ;;
  chorus2)
    exec conda run --no-capture-output -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. \
      python scripts/benchmark/run_vidprom_300_experiment.py --method "$method" --gpu 6 --limit 300 --resume \
      --cache-view "$cache_view" --output-root "$output_root"
    ;;
  carnator_boundary_b5)
    exec conda run --no-capture-output -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. python -c '
import sys
import scripts.benchmark.run_vidprom_300_experiment as runner
runner.EARLY_SIGNATURE_THRESHOLDS = {
    "max_context_color_distance": 1e6,
    "max_log_area_ratio": 1e6,
    "max_shape_distance": 1e6,
    "max_changed_color_distance": 1e6,
}
sys.argv = [
    "run_vidprom_300_experiment.py", "--method", "carnator_boundary_b5", "--gpu", "7",
    "--limit", "300", "--resume", "--cache-view", "artifacts/vidprom_1000cache_300input_noun_match/source_view",
    "--output-root", "artifacts/vidprom_1000cache_300input_noun_match/early_signature_disabled_20260903_100305",
]
raise SystemExit(runner.main())
'
    ;;
  *)
    echo "unsupported method: $method" >&2
    exit 2
    ;;
esac
