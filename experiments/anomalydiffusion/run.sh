#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$task_root/../.." && pwd)"
mkdir -p "$task_root/.tools/cuda"
# Some GPU containers expose only the versioned driver library; cuDNN 8 asks for
# libcuda.so. Keep the compatibility link local to this experiment.
if [[ ! -e "$task_root/.tools/cuda/libcuda.so" ]]; then
    driver_library="$(ldconfig -p | awk '$1 == "libcuda.so.1" && !found {path=$NF; found=1} END {print path}')"
    if [[ -n "$driver_library" ]]; then
        ln -s "$driver_library" "$task_root/.tools/cuda/libcuda.so"
    fi
fi
export LD_LIBRARY_PATH="$task_root/.tools/cuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$project_root"
exec "$task_root/.venv/bin/python" -m experiments.anomalydiffusion.run "$@"
