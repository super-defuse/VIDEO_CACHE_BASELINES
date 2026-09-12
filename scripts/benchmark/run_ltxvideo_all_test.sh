#!/bin/bash
# LTX-Video 10-request smoke run: all five baselines in parallel, then aggregate.
# Formal 300-request experiment: run_ltxvideo_all_formal.sh
#
# PYTHONNOUSERSITE=1 is mandatory: the user site-packages carries diffusers
# 0.40.0, which shadows the 0.35.2 the wan environment provides and has a
# different LTXConditionPipeline signature.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/data/shared/conda_envs/wan/bin/python"
export PYTHONNOUSERSITE=1

LIMIT=10
OUTPUT_ROOT="$ROOT_DIR/artifacts/vidprom_1000cache_300input_noun_match/results_ltxvideo_test"
REPLAY_ROOT="$ROOT_DIR/artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay"
LOG_DIR="$ROOT_DIR/artifacts/ltxvideo_experiment_logs"
METHODS=(baseline nirvana_vid chorus chorus2 carnator_boundary_b5)
GPU_FOR=(3 4 5 6 7)  # index matches METHODS

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
echo "LTX-Video test run: ${METHODS[*]} x $LIMIT requests -> $OUTPUT_ROOT"

PIDS=()
for i in "${!METHODS[@]}"; do
    method=${METHODS[$i]}
    log_file="$LOG_DIR/test_${method}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$method] GPU ${GPU_FOR[$i]}, log $log_file"
    "$PYTHON" "$ROOT_DIR/scripts/benchmark/run_vidprom_300_experiment.py" \
        --model ltxvideo --method "$method" --gpu "${GPU_FOR[$i]}" \
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
        --model ltxvideo --results-root "$OUTPUT_ROOT"
fi
exit "$FAILED"
