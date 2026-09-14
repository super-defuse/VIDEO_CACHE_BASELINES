#!/bin/bash
# Wan2.1 Carnator step-5 Early Signature dataset: two 5-step probes per arrival
# (the replay-selected source prompt and the arrival prompt), 300 arrivals
# sharded over GPUs 3-7.  No 50-step source, no reuse, no videos -- see the
# script docstring for what this dataset can and cannot answer.
#
#   ./run_wan21_carnator_step5_signatures.sh              # full 300, then summary
#   ./run_wan21_carnator_step5_signatures.sh --limit 2    # cheap timing check
#
# The run is resumable: rerunning skips arrivals that already completed.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/data/shared/conda_envs/wan/bin/python"

GPUS="3,4,5,6,7"
RUN_ROOT="$ROOT_DIR/artifacts/vidprom_1000cache_300input_noun_match/wan21_carnator_step5"

# --limit defaults to 300 inside the Python script; pass --limit N to override.
mkdir -p "$RUN_ROOT/logs"
echo "Wan2.1 Carnator step-5 signatures: gpus=$GPUS -> $RUN_ROOT"

"$PYTHON" "$SCRIPT_DIR/run_wan21_carnator_step5_signatures.py" \
    --gpus "$GPUS" --run-root "$RUN_ROOT" --summarize "$@"
