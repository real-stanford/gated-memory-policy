#!/bin/bash

echo "env: "
env

set -e

. $(conda info --base)/etc/profile.d/conda.sh # This is equivalent to source the conda profile
if [[ -z "$CONDA_PREFIX" || "$CONDA_PREFIX" != *"imitation" ]]; then
    conda activate imitation
    echo "Activated conda environment: $CONDA_PREFIX"
fi

root_dir=$(dirname $(dirname $(realpath $0)))
cd $root_dir

export PYTHONPATH=$root_dir:$PYTHONPATH

project_name="gated_memory_policy"
if [ -z "$logger_project_name"]; then
    logger_project_name=$project_name
fi

policy_name="diffusion_memory_transformer" # This is good enough for all the real-world tasks
# policy_name="diffusion_transformer"
# policy_name="diffusion_transformer_large"
# policy_name="longhist_downsampled_diffusion_transformer"
# policy_name="longhist_img_only_diffusion_transformer"

## iPhUMI

export benchmark_name="iphumi"
export task_name="iphumi_place_back_with_correction" # Use all 3 datasets

# export task_name="iphumi_place_back"
# export dataset_name="pick-and-place-back-all" # Place back only data
## export dataset_name="pick-and-place-back-correction-all" # Only correction data
# export additional_args=$additional_args" +workspace.train_dataset.name=$dataset_name"

## RealWorld
# export benchmark_name="real_world"
# export task_name="real_world_iterative_casting"
# # export dataset_name="real_world_iterative_casting_xxxx" # For different versions of the dataset
# # additional_args=$additional_args" +workspace.train_dataset.name=$dataset_name"

export run_name="${task_name}_${policy_name//_transformer}_lr3e-4"

export dataset_compressed_dir="data/datasets/${benchmark_name}"
export normalizer_dir="data/datasets/${benchmark_name}"
export dataset_root_dir="data/datasets/${benchmark_name}"

if [ -n "$1" ]; then
    gpu_ids=$1
else
    if [ -z "$gpu_ids" ]; then
        gpu_ids=$(seq -s, 0 $(($(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)-1)))
    fi
fi
echo "gpu_ids is set to $gpu_ids"

# Parse gpu_ids to get the number of processes
num_processes=$(echo $gpu_ids | tr ',' '\n' | wc -l)


# Adjustable arguments


export additional_args=$additional_args" +workspace.train_dataset.dataloader_cfg.num_workers=12" # The default number of workers is 24. Reduce the number of workers server with fewer CPU cores or less memory
export additional_args=$additional_args" +workspace.train_dataset.dataloader_cfg.batch_size=1" # The default batch size in the dataset configs fits for 80GB GPUs. Reduce the batch size for GPU with smaller memory. 
export additional_args=$additional_args" +workspace.train_dataset.split_dataloader_cfg.num_workers=4" # The default number of workers is 12. Reduce the number of workers for server with fewer CPU cores or less memory. 
export additional_args=$additional_args" +workspace.train_dataset.split_dataloader_cfg.batch_size=1" # The default batch size in the dataset configs fits for 80GB GPUs. Reduce the batch size for GPU with smaller memory. 


## No Proprio
# export additional_args=$additional_args" +workspace.model.proprio_indices=[]"
# export additional_args=$additional_args" +workspace.model.proprio_length=0"
# export additional_args=$additional_args" +workspace.model.history_padding_length=0"


# For debugging (only run 10 training steps and 1 eval step)
export additional_args=$additional_args" +workspace.trainer.debug=True"

if [ -n "$base_ckpt_path" ]; then
    additional_args=$additional_args" +base_ckpt_path=$base_ckpt_path"
fi

source shell_scripts/train_policy.sh
