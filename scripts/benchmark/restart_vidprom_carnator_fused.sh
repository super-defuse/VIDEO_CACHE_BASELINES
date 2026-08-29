#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"

results_root="artifacts/vidprom_1000cache_300input/results"
carnator_root="$results_root/carnator_boundary_b5"
logs_root="artifacts/vidprom_1000cache_300input/logs"
mkdir -p "$logs_root"

failed_marker="$logs_root/carnator_fused_restart.failed"
done_marker="$logs_root/carnator_fused_restart.done"

fail() {
  printf '%s\n' "$1" > "$failed_marker"
  exit 1
}

has_merged_carnator_run() {
  /data/shared/miniconda/bin/python - "$carnator_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in ("requests.jsonl", "request_records.jsonl", "results.jsonl"):
    path = root / name
    if not path.is_file():
        raise SystemExit(1)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 300 or {row.get("arrival_index") for row in rows} != set(range(300)):
        raise SystemExit(1)
raise SystemExit(0)
PY
}

wait_for_session() {
  local session=$1
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done
}

if has_merged_carnator_run; then
  printf '%s\n' "reusing existing merged Carnator results" > "$logs_root/carnator_resume.log"
else
  for session in vidprom-fused-signatures-gpu3 vidprom-fused-signatures-gpu4 vidprom-fused-signatures-gpu5 vidprom-fused-signatures-gpu7; do
    wait_for_session "$session"
  done

  if ! /data/shared/miniconda/bin/conda run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. \
    python scripts/benchmark/run_vidprom_300_experiment.py \
    --method carnator_boundary_b5 --gpu 7 --limit 300 --check-only \
    > "$logs_root/carnator_fused_check.log" 2>&1; then
    fail "fused Carnator source-signature validation failed"
  fi

  for shard in 3 4 5 6 7; do
    start=$(( (shard - 3) * 60 ))
    stop=$(( start + 60 ))
    remap="--allow-gpu-remap"
    if [[ "$shard" -eq 7 ]]; then
      remap=""
    fi
    tmux new-session -d -s "vidprom-formal-carnator-gpu${shard}" -c "$root" \
      "env PYTHONNOUSERSITE=1 PYTHONPATH=. CARNATOR_ENV_READY=1 /data/shared/miniconda/bin/conda run -n wan python scripts/benchmark/run_vidprom_300_experiment.py --method carnator_boundary_b5 --gpu ${shard} ${remap} --limit 300 --arrival-start ${start} --arrival-stop ${stop} --run-dir $carnator_root/shards/gpu${shard} > $logs_root/carnator_gpu${shard}.log 2>&1"
  done

  for session in vidprom-formal-carnator-gpu3 vidprom-formal-carnator-gpu4 vidprom-formal-carnator-gpu5 vidprom-formal-carnator-gpu6 vidprom-formal-carnator-gpu7; do
    wait_for_session "$session"
  done

  if ! /data/shared/miniconda/bin/conda run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. \
    python scripts/benchmark/merge_vidprom_carnator_shards.py \
    --output "$carnator_root" \
    --shard-dir "$carnator_root/shards/gpu3" "$carnator_root/shards/gpu4" "$carnator_root/shards/gpu5" "$carnator_root/shards/gpu6" "$carnator_root/shards/gpu7" \
    > "$logs_root/carnator_merge.log" 2>&1; then
    fail "Carnator shard merge failed"
  fi
fi

for session in vidprom-formal-nirvana_vid vidprom-formal-chorus vidprom-formal-chorus2; do
  wait_for_session "$session"
done

quality_pids=()
for method_gpu in nirvana_vid:3 teacache:4 chorus:5 chorus2:6 carnator_boundary_b5:7; do
  method=${method_gpu%%:*}
  gpu=${method_gpu##*:}
  /data/shared/miniconda/bin/conda run -n vbench env PYTHONNOUSERSITE=1 PYTHONPATH=. \
    python scripts/benchmark/quality_runner.py \
    --results "$results_root/$method/results.jsonl" --device "cuda:$gpu" --skip-if-complete --reuse-existing-metrics \
    > "$logs_root/quality_$method.log" 2>&1 &
  quality_pids+=("$!")
done

quality_failed=0
for pid in "${quality_pids[@]}"; do
  wait "$pid" || quality_failed=1
done
if [[ "$quality_failed" -ne 0 ]]; then
  fail "one or more quality jobs failed"
fi

if ! /data/shared/miniconda/bin/conda run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. \
  python scripts/benchmark/aggregate_vidprom_300_results.py \
  > "$logs_root/aggregate.log" 2>&1; then
  fail "VidProM aggregation failed"
fi

printf '%s\n' "completed" > "$done_marker"
