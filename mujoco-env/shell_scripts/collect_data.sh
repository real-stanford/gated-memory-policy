gpu_id=$1

dbg=$2

python_path=$(conda info --base)/envs/mujoco-env/bin/python


if [ -z "$gpu_id" ]; then
    echo "No GPU ID provided. Using default GPU 0."
    gpu_id=0
fi

if [ "$dbg" = "dbg" ]; then
    dbg=true
else
    dbg=false
fi

if $dbg; then
    parallel_workers=1
else
    parallel_workers=15
fi


export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=$gpu_id
export MUJOCO_EGL_DEVICE_ID=$gpu_id
training_server_name=localhost
# training_server_home_dir=/scratch/m000073/yihuai

additional_keywords=""
# additional_keywords="$additional_keywords total_episodes=100"

# task_name=pick_and_match_color
# run_name=pick_and_match_color

# task_name=pick_and_match_color_rand_delay
# run_name=pick_and_match_color_rand_delay_5_120

# task_name=pick_and_place_back
# run_name=pick_and_place_back

task_name=push_cube
run_name=push_cube

# task_name=fling_cloth
# run_name=fling_cloth





command="$python_path scripts/collect_heuristic_data.py \
 parallel_workers=$parallel_workers \
 task_name=$task_name run_name=$run_name $additional_keywords"

echo "Using GPU $gpu_id; Running command: \n$command"

set -e

if $dbg; then
    $command
else
    output=$(sh -c "$command")

    echo "$output"

    data_path=$(echo "$output" | grep "Episodes data saved to" | head -n 1 | sed 's/.*Episodes data saved to \(.*\)/\1/')
    data_path=$(echo "$data_path" | sed 's/\(.*\)[ \[].*/\1/')

    echo "Data path: $data_path"
fi

