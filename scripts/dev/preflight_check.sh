#!/bin/bash
# Pre-flight check for 5-baseline experiment

set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo "Pre-Flight Check for 5-Baseline Experiment"
echo "============================================"
echo ""

ALL_OK=true

# 1. Check replay decision files
echo "[1/5] Checking replay decision files..."
REPLAY_DIR="artifacts/vidprom_1000cache_300input_noun_match/cogvideox15_replay"

if [ ! -d "$REPLAY_DIR" ]; then
    echo "  ✗ Replay directory not found: $REPLAY_DIR"
    ALL_OK=false
else
    echo "  ✓ Replay directory exists"

    # Check individual files
    for method in carnator_boundary_b5 chorus chorus2 nirvana_vid; do
        file="$REPLAY_DIR/${method}.jsonl"
        if [ ! -f "$file" ]; then
            echo "  ✗ Missing: $file"
            ALL_OK=false
        else
            lines=$(wc -l < "$file")
            if [ $lines -lt 10 ]; then
                echo "  ✗ Insufficient data in $file: only $lines lines"
                ALL_OK=false
            else
                echo "  ✓ $method.jsonl exists ($lines lines)"
            fi
        fi
    done

    # Check manifest
    if [ ! -f "$REPLAY_DIR/manifest.json" ]; then
        echo "  ✗ Missing: manifest.json"
        ALL_OK=false
    else
        echo "  ✓ manifest.json exists"
    fi
fi
echo ""

# 2. Check GPUs
echo "[2/5] Checking GPU availability..."
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "  Total GPUs: $GPU_COUNT"

if [ $GPU_COUNT -lt 8 ]; then
    echo "  ✗ Need at least 8 GPUs (for GPU 3-7)"
    ALL_OK=false
else
    echo "  ✓ Sufficient GPUs available"
    echo ""
    echo "  Target GPUs (3-7):"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | grep -E "^[3-7]," | while read line; do
        echo "    $line"
    done
fi
echo ""

# 3. Check GPU memory
echo "[3/5] Checking GPU memory..."
for gpu in 3 4 5 6 7; do
    mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu)
    mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpu)
    mem_free=$((mem_total - mem_used))

    printf "  GPU %d: %6d MB used, %6d MB free" $gpu $mem_used $mem_free

    if [ $mem_free -lt 20000 ]; then
        echo " ✗ (need ~20GB free)"
        ALL_OK=false
    else
        echo " ✓"
    fi
done
echo ""

# 4. Check Python environment
echo "[4/5] Checking Python environment..."
if ! python3 -c "import sys; sys.path.insert(0, '.'); from core.registry import make_method" 2>/dev/null; then
    echo "  ✗ Cannot import core.registry.make_method"
    ALL_OK=false
else
    echo "  ✓ core.registry.make_method importable"
fi

if ! python3 -c "import sys; sys.path.insert(0, '.'); from adapters.cogvideox15 import CogVideoX15Adapter" 2>/dev/null; then
    echo "  ✗ Cannot import CogVideoX15Adapter"
    ALL_OK=false
else
    echo "  ✓ CogVideoX15Adapter importable"
fi

if ! python3 -c "import sys; sys.path.insert(0, '.'); from core.runner import Runner" 2>/dev/null; then
    echo "  ✗ Cannot import Runner"
    ALL_OK=false
else
    echo "  ✓ Runner importable"
fi

if ! python3 -c "import sys; sys.path.insert(0, '.'); from core.cogvideo_runner import CogVideoReplayRunner" 2>/dev/null; then
    echo "  ✗ Cannot import CogVideoReplayRunner"
    ALL_OK=false
else
    echo "  ✓ CogVideoReplayRunner importable"
fi
echo ""

# 5. Check for old/conflicting experiments
echo "[5/5] Checking for conflicting experiments..."
OLD_DIR="artifacts/cogvideox15_10samples_test"
if [ -d "$OLD_DIR" ]; then
    echo "  ⚠ Found old test directory: $OLD_DIR"
    echo "    Run ./scripts/dev/clean_failed_10samples.sh to clean up"
else
    echo "  ✓ No old test directories"
fi

# Check for running experiments
RUNNING=$(ps aux | grep "run_.*10samples.py" | grep -v grep | wc -l)
if [ $RUNNING -gt 0 ]; then
    echo "  ⚠ Found $RUNNING running experiment process(es)"
    ps aux | grep "run_.*10samples.py" | grep -v grep
else
    echo "  ✓ No conflicting processes"
fi
echo ""

# Summary
echo "============================================"
if [ "$ALL_OK" = true ]; then
    echo "✓ All checks passed!"
    echo ""
    echo "Ready to run experiment:"
    echo "  ./scripts/dev/test_5baselines_config.sh    # Test configuration first"
    echo "  ./scripts/dev/launch_5baselines_10samples.sh   # Then launch"
else
    echo "✗ Some checks failed"
    echo ""
    echo "Please fix the issues above before running the experiment."
    exit 1
fi
echo "============================================"
