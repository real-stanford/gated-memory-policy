#!/bin/bash
if [[ -n "$skip_gate_label_calculation" ]]; then
    echo "Skipping gate label calculation"
    exit 0
fi

root_dir=$(dirname $(dirname $(realpath $0)))
cd $root_dir

# The following should be exported from the other script
required_vars=("gpu_ids" "with_mem_ckpt_path" "no_mem_ckpt_path" "dataset_type" "eval_episode_num" "window_size")

if [[ -z "$logger_project_name" ]]; then
    logger_project_name=$task_name
fi

# Loop through the array and check each variable
for var_name in "${required_vars[@]}"; do
    # Use indirect parameter expansion to check the value of the variable name stored in var_name
    if [ -z "${!var_name}" ]; then
        echo "Error: $var_name is not set"
        exit 1
    fi
done


if [[ -z "$CONDA_PREFIX" || "$CONDA_PREFIX" != *"imitation" ]]; then
    . $(conda info --base)/etc/profile.d/conda.sh # This is equivalent to source the conda profile
    conda activate imitation
    echo "Activated conda environment: $CONDA_PREFIX"
fi

num_processes=$(echo $gpu_ids | tr ',' '\n' | wc -l)

accelerate_path=$(conda info --base)/envs/imitation/bin/accelerate

command="$accelerate_path launch --main_process_port 29510 --gpu_ids $gpu_ids --num_processes $num_processes \
    scripts/generate_gate_labels.py \
    --with_mem_ckpt_path=$with_mem_ckpt_path \
    --no_mem_ckpt_path=$no_mem_ckpt_path \
    --eval_episode_num=$eval_episode_num \
    --dataset_type=$dataset_type \
    --dataset_dir=$dataset_root_dir \
    --date_str=$date_str \
    --time_str=$time_str \
    --window_size=$window_size \
    $eval_additional_args"


echo $command
eval $command