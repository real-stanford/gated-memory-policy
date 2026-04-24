#!/bin/bash
# Start the MIKASA remote env server.
#
# Usage:
#   bash shell_scripts/serve_mikasa_env.sh [GPU_ID] [SERVER_PORT]
#
# Defaults: GPU 1, port 18765

root_dir=$(dirname $(dirname $(realpath $0)))
cd $root_dir

gpu_id=${1:-1}
server_port=${2:-18765}

# activate mikasa conda env if not already active
if [[ -z "$CONDA_PREFIX" || "$CONDA_PREFIX" != *"mikasa"* ]]; then
    . $(conda info --base)/etc/profile.d/conda.sh
    conda activate mikasa
    echo "activated conda env: $CONDA_PREFIX"
fi

export CUDA_VISIBLE_DEVICES=$gpu_id
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=$gpu_id
export MUJOCO_EGL_DEVICE_ID=$gpu_id

echo "GPU: $gpu_id | policy server: tcp://localhost:$server_port"

python eval/mikasa_env_server.py \
    --policy-server "tcp://localhost:$server_port" \
    --output-dir eval_results