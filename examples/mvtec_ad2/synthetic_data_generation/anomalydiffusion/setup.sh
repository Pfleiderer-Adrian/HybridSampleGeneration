#!/usr/bin/env bash
# Isolated Python 3.10 runtime for AnomalyDiffusion; the project itself keeps its own environment.
set -euo pipefail
module_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$module_root/../../../.." && pwd)"
home="${ANOMALYDIFFUSION_HOME:-$module_root}"
mkdir -p "$home/.tools"
uv_exe="$(command -v uv || true)"
if [[ -z "$uv_exe" ]]; then
    "$project_root/.venv/bin/python" -m pip install --target "$home/.tools/uv" uv==0.13.6
    uv_exe="$home/.tools/uv/bin/uv"
fi
if [[ ! -d "$home/upstream/.git" ]]; then
    git clone https://github.com/sjtuplayer/anomalydiffusion "$home/upstream"
    git -C "$home/upstream" checkout --detach b85297f333ebfc59b3863338445b81363efb9761
fi
if [[ ! -d "$home/.tools/taming-transformers/.git" ]]; then
    git clone https://github.com/CompVis/taming-transformers "$home/.tools/taming-transformers"
    git -C "$home/.tools/taming-transformers" checkout --detach 3ba01b241669f5ade541ce990f7650a3b8f65318
fi
if [[ ! -x "$home/.venv/bin/python" ]]; then
    "$uv_exe" venv --python 3.10 "$home/.venv"
fi
"$uv_exe" pip install --python "$home/.venv/bin/python" setuptools==69.5.1 wheel==0.45.1
"$uv_exe" pip install --no-build-isolation --python "$home/.venv/bin/python" -r "$module_root/requirements.lock"
cd -- "$project_root"
ANOMALYDIFFUSION_HOME="$home" "$home/.venv/bin/python" -c \
    'from examples.mvtec_ad2.synthetic_data_generation.anomalydiffusion.runtime import configure_upstream; configure_upstream(); print("AnomalyDiffusion runtime OK")'
