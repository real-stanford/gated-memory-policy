#! /bin/bash

root_dir=$(dirname $(dirname $(realpath $0)))
cd $root_dir

gpu_id=$1
if [ -n "$gpu_id" ]; then
    echo "Using GPU $gpu_id"
else
    echo "No GPU ID provided. Using default GPU 0."
    gpu_id=0
fi

if [[ -z "$CONDA_PREFIX" || "$CONDA_PREFIX" != *"mujoco-env" ]]; then
    . $(conda info --base)/etc/profile.d/conda.sh # This is equivalent to source the conda profile
    conda activate mujoco-env
    echo "Activated conda environment: $CONDA_PREFIX"
fi

additional_args=""

server_port=$2
if [ -n "$server_port" ]; then
    echo "Using server_port $server_port"
else
    server_port=$((18765 + $gpu_id))
    echo "Using default server_port $server_port"
fi

additional_args="$additional_args server.policy_server_address=tcp://localhost:$server_port"
echo "Using server port $server_port"

# additional_args="$additional_args task.agent.action_execution_horizon=8"
# additional_args="$additional_args rollout_episode_num=20"

python_path=$(conda info --base)/envs/mujoco-env/bin/python

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=$gpu_id
export MUJOCO_EGL_DEVICE_ID=$gpu_id


command="$python_path scripts/serve_remote_env.py $additional_args"

echo "================================================"
echo "Running command: $command"
echo "================================================"
eval $command
