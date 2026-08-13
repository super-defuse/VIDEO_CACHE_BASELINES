#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
command -v conda >/dev/null 2>&1 || { echo "conda is required" >&2; exit 1; }
cd "$root"
exists() { conda env list | awk '{print $1}' | grep -Fxq "$1"; }

rename_legacy() {
  local old=$1 new=$2
  if exists "$new" || ! exists "$old"; then
    return
  fi
  if conda rename --help >/dev/null 2>&1; then
    conda rename -n "$old" "$new" -y
  else
    echo "conda rename is unavailable; rename $old to $new before continuing" >&2
    exit 1
  fi
}

rename_legacy chai-wan wan
rename_legacy chai-cogvideo cogvideo
rename_legacy chai-vbench vbench

for env_file in baseline chai teacache nirvana_vid carnator; do
  env_name=$(awk '/^name:/ {print $2}' "$root/environments/$env_file.yml")
  if exists "$env_name"; then
    echo "environment exists: $env_name"
  else
    conda env create -f "$root/environments/$env_file.yml"
  fi
done

for env_name in vcb-baseline vcb-chai vcb-teacache vcb-nirvana-vid vcb-carnator; do
  conda run --no-capture-output -n "$env_name" python -m pip install --no-deps -e "$root"
done

conda run --no-capture-output -n vcb-chai python -m pip install -r "$root/../CHAI/requirements-wan.txt"
conda run --no-capture-output -n vcb-teacache python -m pip install -r "$root/../CHAI/requirements-cogvideo.txt"
conda run --no-capture-output -n vcb-nirvana-vid python -m pip install -r "$root/../CHAI/requirements-wan.txt"
conda run --no-capture-output -n vcb-carnator python -m pip install -r "$root/../CARNATOR_ICLR2027/requirements.txt"

if ! exists vbench; then
  conda env create -f "$root/environments/vbench.yml"
fi
conda run --no-capture-output -n vbench python -m pip install -r "$root/../CHAI/requirements-vbench.txt"
conda run --no-capture-output -n vbench python -m pip install --no-deps -e "$root"
echo "ready: wan, cogvideo, vbench; optional isolated method environments: vcb-baseline, vcb-chai, vcb-teacache, vcb-nirvana-vid, vcb-carnator"
