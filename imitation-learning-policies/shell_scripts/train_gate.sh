#!/bin/bash

root_dir=$(dirname $(dirname $(realpath $0)))
cd $root_dir

if [[ -z "$logger_project_name" ]]; then
    logger_project_name=$task_name
fi

# The following should be exported from the other script
required_vars=("gpu_ids" "project_name" "logger_project_name" "task_name" "run_name" "dataset_type" "dataset_compressed_dir" "dataset_root_dir" "final_statistics_path")

# Loop through the array and check each variable
for var_name in "${required_vars[@]}"; do
    # Use indirect parameter expansion to check the value of the variable name stored in var_name
    if [ -z "${!var_name}" ]; then
        echo "Error: $var_name is not set"
        exit 1
    fi
done

echo "conda prefix: $CONDA_PREFIX"
echo "conda base: $(conda info --base)"
if [[ -z "$CONDA_PREFIX" || "$CONDA_PREFIX" != *"imitation" ]]; then
    . $(conda info --base)/etc/profile.d/conda.sh # This is equivalent to source the conda profile
    conda activate imitation
    echo "Activated conda environment: $CONDA_PREFIX"
fi

num_processes=$(echo $gpu_ids | tr ',' '\n' | wc -l)

accelerate_path=$(conda info --base)/envs/imitation/bin/accelerate


if [ -z "$final_statistics_path" ]; then
    final_statistics_path="${statistics_path%\.pt}_window_${window_size}.pt"
fi

command="$accelerate_path launch --main_process_port 29510 --gpu_ids $gpu_ids --num_processes $num_processes \
    scripts/train_memory_gate.py \
    +task_name=$task_name \
    +policy_name=$policy_name \
    +project_name=$project_name \
    +run_name=$run_name \
    +logger_project_name=$logger_project_name \
    +project_name=$project_name \
    +workspace.train_dataset.compressed_dir=$dataset_compressed_dir \
    +workspace.train_dataset.root_dir=$dataset_root_dir \
    +workspace.train_dataset.normalizer_dir=$normalizer_dir \
    +workspace.train_dataset.+statistics_data_path=$final_statistics_path\
    $additional_args"


echo $command
eval $command