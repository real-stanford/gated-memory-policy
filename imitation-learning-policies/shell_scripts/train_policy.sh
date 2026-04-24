#!/bin/bash

root_dir=$(dirname $(dirname $(realpath $0)))
cd $root_dir
export PYTHONPATH=$root_dir:$PYTHONPATH

# The following should be exported from the other script
required_vars=("gpu_ids" "project_name" "task_name" "logger_project_name" "run_name" "dataset_compressed_dir" "dataset_root_dir")


# Loop through the array and check each variable
for var_name in "${required_vars[@]}"; do
    # Use indirect parameter expansion to check the value of the variable name stored in var_name
    if [ -z "${!var_name}" ]; then
        echo "Error: $var_name is not set"
        exit 1
    fi
done

if [ -z "$additional_args" ]; then
    additional_args=""
fi

if [ -z "$train_server_name" ]; then
    echo "train_server_name is set to localhost"
    train_server_name="localhost"
fi

export num_processes=$(echo $gpu_ids | tr ',' '\n' | wc -l)

conda_env="imitation"

echo "conda prefix: $CONDA_PREFIX"
echo "conda base: $(conda info --base)"
if [[ -z "$CONDA_PREFIX" || "$CONDA_PREFIX" != *"$conda_env" ]]; then
    . $(conda info --base)/etc/profile.d/conda.sh # This is equivalent to source the conda profile
    conda activate $conda_env
    echo "Activated conda environment: $CONDA_PREFIX"
fi


# Check whether to use more than one GPU
if [ -n "$gpu_ids" ] && [ $num_processes -gt 1 ]; then
    # Multi-GPU training
    accelerate_path=$CONDA_PREFIX/bin/accelerate

    # export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
    # main_process_port="--main_process_port $MASTER_PORT"
    main_process_port=""
    command="$accelerate_path launch --gpu_ids $gpu_ids --num_processes $num_processes \
    $main_process_port scripts/train_policy.py +policy_name=$policy_name \
    +task_name=$task_name +logger_project_name=$logger_project_name \
    +project_name=$project_name +run_name=$run_name \
    +workspace.train_dataset.compressed_dir=$dataset_compressed_dir \
    +workspace.train_dataset.root_dir=$dataset_root_dir \
    +workspace.train_dataset.normalizer_dir=$normalizer_dir +train_server_name=$train_server_name $additional_args"
else
    # Single GPU training
    export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
    export WORLD_SIZE=1
    python_path=$CONDA_PREFIX/bin/python
    if [ -n "$gpu_ids" ]; then
        export CUDA_VISIBLE_DEVICES=$gpu_ids
        echo "CUDA_VISIBLE_DEVICES=$gpu_ids"
    fi
    export NCCL_P2P_DISABLE="1"
    export NCCL_IB_DISABLE="1"
    command="$python_path scripts/train_policy.py +policy_name=$policy_name \
    +task_name=$task_name +logger_project_name=$logger_project_name \
    +project_name=$project_name +run_name=$run_name \
    +workspace.train_dataset.compressed_dir=$dataset_compressed_dir \
    +workspace.train_dataset.root_dir=$dataset_root_dir \
    +workspace.train_dataset.normalizer_dir=$normalizer_dir +train_server_name=$train_server_name $additional_args"
fi


echo "================================================"
echo "Running command: $command"
echo "================================================"
eval $command
