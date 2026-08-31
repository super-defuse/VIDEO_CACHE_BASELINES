#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_PID RUN_ROOT" >&2
  exit 2
fi

run_pid="$1"
run_root="$2"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
pairs="$repo_root/$run_root/pairs.csv"
quality_root="$repo_root/$run_root/quality"
wan_python="/data/shared/conda_envs/wan/bin/python"
vbench_python="/data/shared/conda_envs/vbench/bin/python"
log_root="$quality_root/logs"
mkdir -p "$log_root"

while kill -0 "$run_pid" 2>/dev/null; do
  sleep 30
done

"$vbench_python" "$repo_root/scripts/benchmark/score_carnator_early_signature_calibration.py" \
  --run-root "$repo_root/$run_root" --pairs "$pairs" --worker prepare \
  --clip-weights "$repo_root/artifacts/vbench-cache/clip_model/ViT-B-32.pt" \
  --vbench-cache "$repo_root/artifacts/vbench-cache" > "$log_root/prepare.log" 2>&1

dimensions=(subject_consistency background_consistency motion_smoothness aesthetic_quality imaging_quality dynamic_degree)
pids=()
for index in 0 1 2 3 4; do
  dimension="${dimensions[$index]}"
  "$vbench_python" "$repo_root/scripts/benchmark/score_carnator_early_signature_calibration.py" \
    --run-root "$repo_root/$run_root" --pairs "$pairs" --worker vbench \
    --dimension "$dimension" --device "cuda:$((index + 3))" \
    --clip-weights "$repo_root/artifacts/vbench-cache/clip_model/ViT-B-32.pt" \
    --vbench-cache "$repo_root/artifacts/vbench-cache" > "$log_root/vbench_${dimension}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$vbench_python" "$repo_root/scripts/benchmark/score_carnator_early_signature_calibration.py" \
  --run-root "$repo_root/$run_root" --pairs "$pairs" --worker vbench \
  --dimension dynamic_degree --device cuda:3 \
  --clip-weights "$repo_root/artifacts/vbench-cache/clip_model/ViT-B-32.pt" \
  --vbench-cache "$repo_root/artifacts/vbench-cache" > "$log_root/vbench_dynamic_degree.log" 2>&1

"$vbench_python" "$repo_root/scripts/benchmark/score_carnator_early_signature_calibration.py" \
  --run-root "$repo_root/$run_root" --pairs "$pairs" --worker clip --device cuda:4 \
  --clip-weights "$repo_root/artifacts/vbench-cache/clip_model/ViT-B-32.pt" \
  --vbench-cache "$repo_root/artifacts/vbench-cache" > "$log_root/clip.log" 2>&1
"$vbench_python" "$repo_root/scripts/benchmark/score_carnator_early_signature_calibration.py" \
  --run-root "$repo_root/$run_root" --pairs "$pairs" --worker aggregate \
  --clip-weights "$repo_root/artifacts/vbench-cache/clip_model/ViT-B-32.pt" \
  --vbench-cache "$repo_root/artifacts/vbench-cache" > "$log_root/aggregate.log" 2>&1

"$wan_python" "$repo_root/scripts/benchmark/run_vidprom_forced_early_signature_calibration.py" scan \
  --run-root "$repo_root/$run_root" --pairs "$pairs" --q-min 0 --clip-t-min 0 \
  --coverage-min 0.90 --confidence 0.95 --selection-min-lower-bound 0.80 \
  > "$log_root/scan.log" 2>&1
