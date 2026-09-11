#!/bin/bash
# Launch 5 baselines in parallel on GPUs 3-7 (10 samples each)
# Usage: ./launch_5baselines_10samples.sh

set -e

cd "$(dirname "$0")/.."

# Activate cogvideo conda environment
eval "$(/data/shared/miniconda/condabin/conda shell.bash hook)"
conda activate cogvideo

echo "============================================"
echo "CogVideoX 5-Baseline 10-Sample Experiment"
echo "============================================"
echo ""
echo "Python: $(which python3)"
echo "Environment: $CONDA_DEFAULT_ENV"
echo ""
echo "Baselines:"
echo "  1. baseline (dense, no caching)"
echo "  2. carnator_boundary_b5"
echo "  3. chorus"
echo "  4. chorus2"
echo "  5. nirvana_vid"
echo ""
echo "GPUs: 3, 4, 5, 6, 7"
echo "Samples: 10 per baseline"
echo ""

# Check if artifacts directory exists
if [ ! -d "artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay" ]; then
    echo "Error: Replay decision directory not found!"
    exit 1
fi

# Create output directory
OUTPUT_DIR="artifacts/cogvideox15_5baselines_10samples"
mkdir -p "$OUTPUT_DIR"

# Run parallel experiments
python3 scripts/dev/run_5baselines_10samples_fixed.py \
    --gpus 3,4,5,6,7 \
    --limit 10 \
    --output-root "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/launch.log"

echo ""
echo "All experiments launched! Monitor with:"
echo "  ./scripts/dev/monitor_5baselines.sh"
