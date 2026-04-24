#! /bin/bash

root_dir=$(dirname $(dirname $(realpath $0)))
cd $root_dir
# Parse gpu_ids to get the number of processes


if [ -n "$1" ]; then
    gpu_ids=$1
else
    if [ -z "$gpu_ids" ]; then
        gpu_ids=0
    fi
fi
echo "gpu_ids is set to $gpu_ids"
num_processes=$(echo $gpu_ids | tr ',' '\n' | wc -l)

# Adjustable arguments

main_process_port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
main_process_port="--main_process_port $main_process_port"

# Check whether to use more than one GPU
if [ -n "$gpu_ids" ] && [ $num_processes -gt 1 ]; then
    # Multi-GPU training
    accelerate_path=$(conda info --base)/envs/imitation/bin/accelerate
    command="$accelerate_path launch --gpu_ids $gpu_ids --num_processes $num_processes $main_process_port scripts/train_diffusion_policy_resume.py --ckpt_path $resume_ckpt_path"

else
    # Single GPU training
    python_path=$(conda info --base)/envs/imitation/bin/python
    if [ -n "$gpu_ids" ]; then
        export CUDA_VISIBLE_DEVICES=$gpu_ids
        echo "CUDA_VISIBLE_DEVICES=$gpu_ids"
    fi
    export NCCL_P2P_DISABLE="1"
    export NCCL_IB_DISABLE="1"
    command="$python_path scripts/train_diffusion_policy_resume.py --ckpt_path $resume_ckpt_path"
fi

echo "================================================"
echo "Running command: $command"
echo "================================================"
eval $command
