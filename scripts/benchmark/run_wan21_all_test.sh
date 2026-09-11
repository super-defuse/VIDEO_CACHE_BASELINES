#!/bin/bash
# Wan2.1 10-request smoke run: all five baselines in parallel, then aggregate.
# Formal 300-request experiment: run_wan21_all_formal.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/data/shared/conda_envs/wan/bin/python"

LIMIT=10
OUTPUT_ROOT="$ROOT_DIR/artifacts/vidprom_1000cache_300input_noun_match/results_wan21_test"
REPLAY_ROOT="$ROOT_DIR/artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay"
LOG_DIR="$ROOT_DIR/artifacts/wan21_experiment_logs"
METHODS=(baseline nirvana_vid chorus chorus2 carnator_boundary_b5)
GPU_FOR=(3 4 5 6 7)  # index matches METHODS

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
echo "Wan2.1 test run: ${METHODS[*]} x $LIMIT requests -> $OUTPUT_ROOT"

PIDS=()
for i in "${!METHODS[@]}"; do
    method=${METHODS[$i]}
    log_file="$LOG_DIR/test_${method}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$method] GPU ${GPU_FOR[$i]}, log $log_file"
    "$PYTHON" "$ROOT_DIR/scripts/benchmark/run_vidprom_300_experiment.py" \
        --model wan21 --method "$method" --gpu "${GPU_FOR[$i]}" \
        --limit "$LIMIT" --replay-root "$REPLAY_ROOT" \
        --output-root "$OUTPUT_ROOT" --overwrite \
        > "$log_file" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "OK   ${METHODS[$i]}"
    else
        echo "FAIL ${METHODS[$i]} (exit $?, see $LOG_DIR)"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -eq 0 ]; then
    "$PYTHON" "$SCRIPT_DIR/aggregate_vidprom_300_results.py" \
        --model wan21 --results-root "$OUTPUT_ROOT"
fi
exit "$FAILED"
