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

wait_for_session() {
  local session=$1
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done
}

for session in vidprom-fused-signatures-gpu3 vidprom-fused-signatures-gpu4 vidprom-fused-signatures-gpu5 vidprom-fused-signatures-gpu7; do
  wait_for_session "$session"
done

if ! /data/shared/miniconda/bin/conda run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. \
  python scripts/benchmark/run_vidprom_300_experiment.py \
  --method carnator_boundary_b5 --gpu 7 --limit 300 --check-only \
  > "$logs_root/carnator_fused_check.log" 2>&1; then
  fail "fused Carnator source-signature validation failed"
fi

tmux new-session -d -s vidprom-formal-carnator-gpu3 -c "$root" \
  "env PYTHONNOUSERSITE=1 PYTHONPATH=. CARNATOR_ENV_READY=1 /data/shared/miniconda/bin/conda run -n wan python scripts/benchmark/run_vidprom_300_experiment.py --method carnator_boundary_b5 --gpu 3 --allow-gpu-remap --limit 300 --arrival-start 0 --arrival-stop 75 --run-dir $carnator_root/shards/gpu3 > $logs_root/carnator_gpu3.log 2>&1"
tmux new-session -d -s vidprom-formal-carnator-gpu4 -c "$root" \
  "env PYTHONNOUSERSITE=1 PYTHONPATH=. CARNATOR_ENV_READY=1 /data/shared/miniconda/bin/conda run -n wan python scripts/benchmark/run_vidprom_300_experiment.py --method carnator_boundary_b5 --gpu 4 --allow-gpu-remap --limit 300 --arrival-start 75 --arrival-stop 150 --run-dir $carnator_root/shards/gpu4 > $logs_root/carnator_gpu4.log 2>&1"
tmux new-session -d -s vidprom-formal-carnator-gpu5 -c "$root" \
  "env PYTHONNOUSERSITE=1 PYTHONPATH=. CARNATOR_ENV_READY=1 /data/shared/miniconda/bin/conda run -n wan python scripts/benchmark/run_vidprom_300_experiment.py --method carnator_boundary_b5 --gpu 5 --allow-gpu-remap --limit 300 --arrival-start 150 --arrival-stop 225 --run-dir $carnator_root/shards/gpu5 > $logs_root/carnator_gpu5.log 2>&1"
tmux new-session -d -s vidprom-formal-carnator-gpu7 -c "$root" \
  "env PYTHONNOUSERSITE=1 PYTHONPATH=. CARNATOR_ENV_READY=1 /data/shared/miniconda/bin/conda run -n wan python scripts/benchmark/run_vidprom_300_experiment.py --method carnator_boundary_b5 --gpu 7 --limit 300 --arrival-start 225 --arrival-stop 300 --run-dir $carnator_root/shards/gpu7 > $logs_root/carnator_gpu7.log 2>&1"

for session in vidprom-formal-carnator-gpu3 vidprom-formal-carnator-gpu4 vidprom-formal-carnator-gpu5 vidprom-formal-carnator-gpu7; do
  wait_for_session "$session"
done

if ! /data/shared/miniconda/bin/conda run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. \
  python scripts/benchmark/merge_vidprom_carnator_shards.py \
  --output "$carnator_root" \
  --shard-dir "$carnator_root/shards/gpu3" "$carnator_root/shards/gpu4" "$carnator_root/shards/gpu5" "$carnator_root/shards/gpu7" \
  > "$logs_root/carnator_merge.log" 2>&1; then
  fail "Carnator shard merge failed"
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
    --results "$results_root/$method/results.jsonl" --device "cuda:$gpu" \
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
