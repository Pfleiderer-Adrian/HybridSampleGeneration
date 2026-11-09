#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$task_root/../.." && pwd)"
uv_exe="$(command -v uv || true)"
if [[ -z "$uv_exe" ]]; then
    "$project_root/.venv/bin/python" -m pip install --target "$task_root/.tools/uv" uv==0.13.6
    uv_exe="$task_root/.tools/uv/bin/uv"
fi
if [[ ! -d "$task_root/upstream/.git" ]]; then
    git clone https://github.com/sjtuplayer/anomalydiffusion "$task_root/upstream"
    git -C "$task_root/upstream" checkout --detach b85297f333ebfc59b3863338445b81363efb9761
fi
if [[ ! -d "$task_root/.tools/taming-transformers/.git" ]]; then
    git clone https://github.com/CompVis/taming-transformers "$task_root/.tools/taming-transformers"
    git -C "$task_root/.tools/taming-transformers" checkout --detach 3ba01b241669f5ade541ce990f7650a3b8f65318
fi
if [[ ! -x "$task_root/.venv/bin/python" ]]; then
    "$uv_exe" venv --python 3.10 "$task_root/.venv"
fi
"$uv_exe" pip install --python "$task_root/.venv/bin/python" setuptools==69.5.1 wheel==0.45.1
"$uv_exe" pip install --no-build-isolation --python "$task_root/.venv/bin/python" -r "$task_root/requirements.lock"
cd -- "$project_root"
"$task_root/.venv/bin/python" -c 'from experiments.anomalydiffusion.runtime import configure_upstream; configure_upstream(); print("AnomalyDiffusion runtime OK")'
