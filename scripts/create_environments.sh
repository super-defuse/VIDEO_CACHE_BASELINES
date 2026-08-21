#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
command -v conda >/dev/null 2>&1 || { echo "conda is required" >&2; exit 1; }
: "${PYTORCH_INDEX_URL:?set PYTORCH_INDEX_URL to the matching PyTorch CUDA wheel index}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.8.0+cu128}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0+cu128}"
cd "$root"

run_in_env() {
  local env_name="$1"
  shift
  # A user-site package must never satisfy or shadow a reproducible runtime
  # dependency. Keep this explicit for both pip and Python entry points.
  conda run --no-capture-output -n "$env_name" env PYTHONNOUSERSITE=1 "$@"
}

env_prefix() {
  conda env list | awk -v name="$1" '$1 == name {print $NF; exit}'
}

valid_conda_env() {
  local prefix="$1"
  [[ -d "$prefix/conda-meta" ]] || return 1
  compgen -G "$prefix/conda-meta/python-*.json" >/dev/null || return 1
  [[ ! -f "$prefix/pyvenv.cfg" ]] || return 1
  [[ -x "$prefix/bin/python" ]] || return 1
  local resolved
  resolved=$(readlink -f "$prefix/bin/python")
  [[ "$resolved" == "$prefix/"* ]] || return 1
}

ensure_conda_env() {
  local env_name="$1"
  local prefix
  prefix=$(env_prefix "$env_name" || true)
  if [[ -n "$prefix" ]] && valid_conda_env "$prefix"; then
    echo "environment is valid: $env_name ($prefix)"
    return
  fi
  if [[ -n "$prefix" && -e "$prefix" ]]; then
    local backup="${prefix}.invalid.$(date +%Y%m%d%H%M%S)"
    mv "$prefix" "$backup"
    echo "moved invalid environment to recoverable backup: $backup"
  fi
  # Use explicit channels so a user's broken defaults mirror cannot silently
  # make a fresh clone fail. The environment YAML remains the package record.
  conda create --name "$env_name" --yes --override-channels --channel conda-forge \
    python=3.10 pyyaml pip
  prefix=$(env_prefix "$env_name")
  valid_conda_env "$prefix" || {
    echo "new environment failed validation: $env_name" >&2
    exit 1
  }
}

for env_name in wan cogvideo vbench; do
  ensure_conda_env "$env_name"
done

for env_name in wan cogvideo vbench; do
  run_in_env "$env_name" python -m pip install \
    --index-url "$PYTORCH_INDEX_URL" \
    "torch==$PYTORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
  run_in_env "$env_name" python -m pip install --no-deps -e "$root"
done

conda install --name wan --yes --override-channels --channel conda-forge decord=0.6.0
run_in_env wan python -m pip install -r "$root/requirements/wan.txt"
run_in_env cogvideo python -m pip install -r "$root/requirements/cogvideo.txt"
conda install --name vbench --yes --override-channels --channel conda-forge decord=0.6.0
run_in_env vbench python -m pip install -r "$root/requirements/vbench.txt"
# The conda-forge decord build can pull NumPy 2.x before the legacy vbench
# requirements install NumPy 1.26. pip may leave NumPy 2's private _core tree
# and conda metadata behind; move stale files out of vbench so it has one
# NumPy installation. Wan and CogVideo intentionally keep NumPy 2 for FAISS
# 1.13 and therefore must not use this cleanup.
repair_numpy_residual() {
  local env_name="$1"
  local prefix
  prefix=$(env_prefix "$env_name")
  local stale_root="${TMPDIR:-/tmp}/video-cache-${env_name}-numpy2-residual-$(date +%Y%m%d%H%M%S)"
  if [[ -d "$prefix/lib/python3.10/site-packages/numpy/_core" ]]; then
    mkdir -p "$stale_root"
    mv "$prefix/lib/python3.10/site-packages/numpy/_core" "$stale_root/"
  fi
  for numpy_meta in \
    "$prefix"/conda-meta/numpy-*.json \
    "$prefix"/lib/python3.10/site-packages/numpy-2.*.dist-info; do
    if [[ -e "$numpy_meta" ]]; then
      mkdir -p "$stale_root"
      mv "$numpy_meta" "$stale_root/"
    fi
  done
}
# vbench keeps NumPy 1.26 for the legacy evaluator; remove a stale NumPy 2
# tree left by conda-forge decord before pip installs its pinned NumPy.
repair_numpy_residual vbench
run_in_env wan python -m pip install -r "$root/requirements/retrieval-wan.txt"
run_in_env cogvideo python -m pip install -r "$root/requirements/retrieval-cogvideo.txt"

echo "ready: wan, cogvideo, vbench"
echo "next: conda run -n wan env PYTHONNOUSERSITE=1 python -m spacy download en_core_web_sm"
echo "next: install the official LongCLIP module/checkpoint for CHAI"
