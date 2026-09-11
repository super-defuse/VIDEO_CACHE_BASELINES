#!/bin/bash
# Run CogVideoX1.5 experiments with first 3 requests across 5 baselines on GPU 3-7

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

REPLAY_ROOT="$ROOT/artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay"
OUTPUT_ROOT="$ROOT/artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_3requests_$(date +%Y%m%d_%H%M%S)"

if [ ! -d "$REPLAY_ROOT" ]; then
    echo "Error: Replay root not found: $REPLAY_ROOT"
    echo "Run build_cogvideo_hit_replay.py first"
    exit 1
fi

echo "Output directory: $OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

declare -A GPU_MAP=(
    ["nirvana_vid"]=3
    ["teacache"]=4
    ["chorus"]=5
    ["chorus2"]=6
    ["carnator_boundary_b5"]=7
)

METHODS=("nirvana_vid" "teacache" "chorus" "chorus2" "carnator_boundary_b5")

# Launch all methods in parallel
for method in "${METHODS[@]}"; do
    gpu="${GPU_MAP[$method]}"
    log="$OUTPUT_ROOT/${method}_run.log"

    echo "Launching $method on GPU $gpu (log: $log)"

    CUDA_VISIBLE_DEVICES=$gpu python scripts/benchmark/run_vidprom_300_experiment.py \
        --method "$method" \
        --model cogvideox15 \
        --gpu "$gpu" \
        --limit 3 \
        --replay-root "$REPLAY_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        2>&1 | tee "$log" &
done

echo "Waiting for all experiments to complete..."
wait

echo ""
echo "=== Results Summary ==="
for method in "${METHODS[@]}"; do
    log="$OUTPUT_ROOT/${method}_run.log"
    if [ -f "$log" ]; then
        echo ""
        echo "[$method]"
        tail -1 "$log"
    fi
done

echo ""
echo "All experiments completed. Output: $OUTPUT_ROOT"
