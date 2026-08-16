#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
command -v conda >/dev/null 2>&1 || { echo "conda is required" >&2; exit 1; }
: "${PYTORCH_INDEX_URL:?set PYTORCH_INDEX_URL to the matching PyTorch CUDA wheel index}"
cd "$root"
exists() { conda env list | awk '{print $1}' | grep -Fxq "$1"; }

for env_file in wan cogvideo vbench; do
  env_name=$(awk '/^name:/ {print $2}' "$root/environments/$env_file.yml")
  if exists "$env_name"; then
    echo "environment exists: $env_name"
  else
    conda env create -f "$root/environments/$env_file.yml"
  fi
done

for env_name in wan cogvideo vbench; do
  conda run --no-capture-output -n "$env_name" python -m pip install \
    --index-url "$PYTORCH_INDEX_URL" torch torchvision
  conda run --no-capture-output -n "$env_name" python -m pip install --no-deps -e "$root"
done

conda run --no-capture-output -n wan python -m pip install -r "$root/requirements/wan.txt"
conda run --no-capture-output -n cogvideo python -m pip install -r "$root/requirements/cogvideo.txt"
conda run --no-capture-output -n vbench python -m pip install -r "$root/requirements/vbench.txt"
echo "ready: wan, cogvideo, vbench"
