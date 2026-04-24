#! /bin/sh

task_name=$1

gpu_id=$2
if [ -n "$gpu_id" ]; then
    echo "Using GPU $gpu_id"
else
    echo "No GPU ID provided. Using default GPU 0."
    gpu_id=0
fi

policy_server_port=$((18765 + $gpu_id))
echo "Using policy server port $policy_server_port"

python_path=$(conda info --base)/envs/mujoco-env/bin/python

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=$gpu_id
export MUJOCO_EGL_DEVICE_ID=$gpu_id

# task_name=robomimic_square
# task_name=robomimic_transport
# task_name=robomimic_tool_hang
# task_name=pick_and_match_color
# task_name=pick_and_match_color_rand_delay
# task_name=push_cube
# task_name=fling_cloth

command="$python_path scripts/rollout_policy_parallel.py \
    policy_server_port=$policy_server_port \
    task_name=$task_name"



echo "================================================"
echo "Running command: $command"
echo "================================================"
eval $command
