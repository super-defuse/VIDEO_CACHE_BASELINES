#!/usr/bin/env bash
# Launch both P1 Chorus counterexample pairs in parallel: Pair A on GPU 3,
# Pair B on GPU 4.  See docs/CHORUS_COUNTEREXAMPLE_P1.md.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

CONDA="${CONDA_BIN:-/data/shared/miniconda/bin/conda}"
ENTRY="scripts/benchmark/run_chorus_counterexample_p1.py"
# The published P1 artifact (see docs/CHORUS_COUNTEREXAMPLE_P1.md).
OUT_ROOT="${OUT_ROOT:-artifacts/chorus_counterexample_p1_v2_halo2}"
# Extra flags for the entry point; the default reproduces the published P1
# artifact (Medit halo 2 instead of Chorus's own default 8).  Override with
# EXTRA_ARGS="--mask-dilation-edit 8" for the wider halo.  LOG_SUFFIX keeps
# other runs' logs intact.
EXTRA_ARGS="${EXTRA_ARGS:---mask-dilation-edit 2}"
LOG_SUFFIX="${LOG_SUFFIX:-}"
read -r -a extra_args <<< "$EXTRA_ARGS"

run_pair () {
  local pair="$1" gpu="$2"
  PYTHONNOUSERSITE=1 PYTHONPATH=. \
    "$CONDA" run -n wan env PYTHONNOUSERSITE=1 PYTHONPATH=. \
    python "$ENTRY" --stage all --with-dense \
    --pair "$pair" --gpu "$gpu" \
    --output-root "$OUT_ROOT/pair_$pair" \
    "${extra_args[@]}" \
    > "logs/chorus_p1_pair_$pair$LOG_SUFFIX.log" 2>&1
}

run_pair A 3 &
pid_a=$!
run_pair B 4 &
pid_b=$!

echo "started pair A pid=$pid_a (gpu3), pair B pid=$pid_b (gpu4)"

wait "$pid_a"; status_a=$?
wait "$pid_b"; status_b=$?
echo "pair A exit=$status_a"
echo "pair B exit=$status_b"
exit $(( status_a != 0 || status_b != 0 ))
