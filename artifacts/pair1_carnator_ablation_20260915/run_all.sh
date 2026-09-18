#!/usr/bin/env bash
# E1-E3 attribution runs for the CHAI pair1-wan-carnator "red bus -> red bus" failure.
#
# Every run replays the exact native reuse-stage command recorded in
#   project/CHAI/artifacts/runs/pair1-wan-carnator/run.log
# with exactly one flag changed, so the only difference against the published
# pair1-wan-carnator row is the isolated variable:
#
#   E1  --no-latent-injection        latent fusion off, attention/region reuse intact
#   E2  --latent-injection-mode hard band removed (source weight 49.6% -> 37.2%)
#   E3  --source-mask-dir ...        noun-gated self-contrast Mbase instead of fusion_masks
#
# Outputs land in a fresh root; the published pair1 artifacts are never written to.
set -euo pipefail

CARNATOR_ROOT=/home/tangxuebin/project/VIDEO_CACHE_BASELINES
PY=/data/shared/conda_envs/wan/bin/python
MODEL_DIR=/home/tangxuebin/models/Wan2.2-TI2V-5B
PAIR_ROOT=/home/tangxuebin/project/CHAI/artifacts/runs/pair1-wan-carnator
SOURCE_RUN="$PAIR_ROOT/carnator/artifacts/source"
SELF_CONTRAST_MASKS="$SOURCE_RUN/cache/source_dense/self_contrast_masks"
DENSE_MANIFEST="$PAIR_ROOT/carnator/target_dense_manifest.json"
OUT_ROOT="$CARNATOR_ROOT/artifacts/pair1_carnator_ablation_20260915"
SOURCE_PROMPT="a red bus driving through a city"
TARGET_PROMPT="a red car driving through a city"

run_one() {
  local name="$1" gpu="$2" mode="$3"; shift 3
  local dir="$OUT_ROOT/$name"
  mkdir -p "$dir/artifacts" "$dir/results"

  local -a cmd=(
    "$PY" -m carnator.adapters.wan_ti2v_5b
    --no-env-setup
    --model-dir "$MODEL_DIR"
    --device cuda:0
    --artifact-root "$dir/artifacts"
    --results-root "$dir/results"
    --run-id run
    --source-prompt "$SOURCE_PROMPT"
    --target-prompt "$TARGET_PROMPT"
    --seed 20260219
    --height 704 --width 1280 --frames 121
    --steps 50 --scheduler unipc --flow-shift 5.0 --guidance-scale 5.0
    --model-dtype fp16
    --mask-probe-step 5 --reuse-start-step 10 --mask-layers 15
    --latent-injection-mode "$mode"
    --boundary-alpha-function smoothstep
    --outer-attn-backend sparge-sage2
    --outer-sparse-layers 0-9,21-29
    --ffn-backend w8a8
    --skip-w8a8-dense-diagnostic
    --disable-progress
    --source-positive bus
    --source-run-dir "$SOURCE_RUN"
    --target-dense-manifest "$DENSE_MANIFEST"
  )
  [[ $# -gt 0 ]] && cmd+=("$@")

  printf 'command=%s\n\n' "${cmd[*]}" > "$dir/log.txt"

  (
    cd "$CARNATOR_ROOT"
    export CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID
    export PYTHONPATH="$CARNATOR_ROOT:$CARNATOR_ROOT/external/wan2.2"
    export CARNATOR_ENV_READY=1 CARNATOR_DEVICE=cuda:0 PYTHONNOUSERSITE=1
    "${cmd[@]}" >> "$dir/log.txt" 2>&1
  )
  echo "== $name done (gpu $gpu) -> $dir"
}

# No argument: run all three sequentially.  With an experiment name: run just that
# one, so the three can also be launched as parallel jobs on separate GPUs.
case "${1:-all}" in
  all)
    run_one e1_no_latent_injection 3 boundary-blend --no-latent-injection
    run_one e2_hard_injection      4 hard
    run_one e3_self_contrast_mask  5 boundary-blend --source-mask-dir "$SELF_CONTRAST_MASKS"
    ;;
  e1*) run_one e1_no_latent_injection 3 boundary-blend --no-latent-injection ;;
  e2*) run_one e2_hard_injection      4 hard ;;
  e3*) run_one e3_self_contrast_mask  5 boundary-blend --source-mask-dir "$SELF_CONTRAST_MASKS" ;;
  *) echo "unknown experiment: $1" >&2; exit 2 ;;
esac
