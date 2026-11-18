#!/usr/bin/env bash
# Run a command of the AnomalyDiffusion CLI in its isolated environment (see __main__.py).
set -euo pipefail
module_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$module_root/../../../.." && pwd)"
export ANOMALYDIFFUSION_HOME="${ANOMALYDIFFUSION_HOME:-$module_root}"
mkdir -p "$ANOMALYDIFFUSION_HOME/.tools/cuda"
# Some GPU containers expose only the versioned driver library; cuDNN 8 asks for libcuda.so.
if [[ ! -e "$ANOMALYDIFFUSION_HOME/.tools/cuda/libcuda.so" ]]; then
    driver_library="$(ldconfig -p | awk '$1 == "libcuda.so.1" && !found {path=$NF; found=1} END {print path}')"
    if [[ -n "$driver_library" ]]; then
        ln -s "$driver_library" "$ANOMALYDIFFUSION_HOME/.tools/cuda/libcuda.so"
    fi
fi
export LD_LIBRARY_PATH="$ANOMALYDIFFUSION_HOME/.tools/cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$project_root"
exec "$ANOMALYDIFFUSION_HOME/.venv/bin/python" -m examples.mvtec_ad2.synthetic_data_generation.anomalydiffusion "$@"
