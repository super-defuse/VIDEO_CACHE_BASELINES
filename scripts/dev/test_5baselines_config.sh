#!/bin/bash
# Test the 5-baseline configuration without actually running experiments

set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo "Testing 5-Baseline Configuration"
echo "============================================"
echo ""

# Run dry-run to validate configuration
python3 scripts/dev/run_5baselines_10samples_fixed.py \
    --gpus 3,4,5,6,7 \
    --limit 10 \
    --dry-run

echo ""
echo "Configuration test completed!"
echo ""
echo "If no errors above, you can run the actual experiment with:"
echo "  ./scripts/dev/launch_5baselines_10samples.sh"
