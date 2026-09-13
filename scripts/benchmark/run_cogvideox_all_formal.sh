#!/bin/bash
# CogVideoX1.5 formal experiment: 300 requests per baseline in parallel, then
# the status check and the aggregated report.
# Smoke run first: run_cogvideox_all_test.sh
#
# PYTHONNOUSERSITE=1 is mandatory: the user site-packages carries diffusers
# 0.40.0, which shadows the 0.35.2 the cogvideo environment provides and cannot
# import a CogVideoX pipeline there at all (its own httpx is missing).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/data/shared/conda_envs/cogvideo/bin/python"
export PYTHONNOUSERSITE=1

# Chorus needs Grounded-SAM-2, whose GroundingDINO imports `supervision`.  That
# package lives only in the user site-packages, which PYTHONNOUSERSITE=1 closes
# off -- and the user site-packages cannot simply be re-enabled because its
# diffusers 0.40.0 shadows the 0.35.2 this environment provides.  The overlay
# carries just the missing packages, so nothing shadows the environment.
OVERLAY="$ROOT_DIR/.cogvideo_overlay"
if [ -d "$OVERLAY" ]; then
    export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
else
    echo "WARNING: $OVERLAY is missing; chorus will fail on 'No module named supervision'" >&2
fi

LIMIT=300
OUTPUT_ROOT="$ROOT_DIR/artifacts/vidprom_1000cache_300input_noun_match/results_cogvideox15"
REPLAY_ROOT="$ROOT_DIR/artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay"
LOG_DIR="$ROOT_DIR/artifacts/cogvideox15_experiment_logs"
METHODS=(baseline nirvana_vid chorus chorus2 carnator_boundary_b5)
GPU_FOR=(3 4 5 6 7)  # index matches METHODS

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
echo "CogVideoX1.5 formal run: ${METHODS[*]} x $LIMIT requests -> $OUTPUT_ROOT"

PIDS=()
for i in "${!METHODS[@]}"; do
    method=${METHODS[$i]}
    log_file="$LOG_DIR/formal_${method}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$method] GPU ${GPU_FOR[$i]}, log $log_file"
    # --resume keeps finished requests when a worker is restarted; drop it
    # only when you also pass --overwrite to start from scratch.
    "$PYTHON" "$ROOT_DIR/scripts/benchmark/run_vidprom_300_experiment.py" \
        --model cogvideox15 --method "$method" --gpu "${GPU_FOR[$i]}" \
        --limit "$LIMIT" --replay-root "$REPLAY_ROOT" \
        --output-root "$OUTPUT_ROOT" --resume \
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

# Exit codes alone would pass a run whose every request errored: the worker
# reports a request error through its JSON status and keeps going.  Check the
# status distribution against the replay decisions it was supposed to follow.
if [ "$FAILED" -eq 0 ]; then
    "$PYTHON" "$SCRIPT_DIR/verify_replay_status_distribution.py" \
        --output-root "$OUTPUT_ROOT" --replay-root "$REPLAY_ROOT" \
        --limit "$LIMIT" --methods "${METHODS[@]}" || FAILED=$((FAILED + 1))
fi

if [ "$FAILED" -eq 0 ]; then
    "$PYTHON" "$SCRIPT_DIR/aggregate_vidprom_300_results.py" \
        --model cogvideox15 --results-root "$OUTPUT_ROOT"
fi
exit "$FAILED"
