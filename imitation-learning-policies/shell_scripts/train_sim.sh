#!/bin/bash

echo "env: "
env

set -e

. $(conda info --base)/etc/profile.d/conda.sh # This is equivalent to source the conda profile
echo "conda prefix: $CONDA_PREFIX"
echo "conda base: $(conda info --base)"
if [[ -z "$CONDA_PREFIX" || "$CONDA_PREFIX" != *"imitation" ]]; then
    conda activate imitation
    echo "Activated conda environment: $CONDA_PREFIX"
fi

root_dir=$(dirname $(dirname $(realpath $0)))
cd $root_dir # the directory of imitation-learning-policies

export additional_args=""

##### Policy name: select one of the below

export policy_name="diffusion_memory_transformer" # Memory policy without gate. Works well in most of the memory-intensive tasks. Try it first before training the gated model.
# export policy_name="diffusion_gated_transformer" # Please specify a gate checkpoint path in the additional arguments
# export policy_name="flow_memory_transformer" # Does not work well... There should be a bug somewhere (please PR if you find it); longhist_cross_attn_flow_transformer works better (gets 100% success rate on pick_and_place_back and pick_and_match_color)
# export policy_name="bc_rnn_memory"


## The below policies are not using overlapped trajectory training. Will be much slower than the above 4.
# export policy_name="longhist_cross_attn_diffusion_transformer"
# export policy_name="longhist_ptp_diffusion_transformer"
# export policy_name="midhist_cross_attn_diffusion_transformer"
# export policy_name="midhist_ptp_diffusion_transformer"

## No memory policies
# export policy_name="diffusion_transformer_large"
# export policy_name="diffusion_transformer"
# export policy_name="flow_transformer"

## Ablation Studies
# export policy_name="diffusion_continuous_gated_transformer" # Train a continuous-valued memory gate along with the policy. You may apply memory_gate_loss_weight to adjust the regularization strength.
# export policy_name="diffusion_binary_gated_transformer" # Train a binary memory gate along with the policy. You may choose different straight_through methods.
# export policy_name="longhist_self_attn_diffusion_transformer"


###### For WandB logging
export project_name="gated_memory_policy"


###### Benchmark and task names: select one each

## MemMinic
export benchmark_name="memmimic"

export task_name="pick_and_place_back"
# export task_name="pick_and_match_color"
# export task_name="push_cube"
# export task_name="fling_cloth"

## For the 10min history checkpoint training, please activate all of the below settings:
# export task_name="pick_and_match_color_rand_delay"
# export base_ckpt_path="data/checkpoints/memmimic/pick_and_match_color_pretrained_vit.ckpt" # Load pretrained ViT
# export additional_args=$additional_args" +workspace.model.max_training_traj_num=20" # Reduce this number to avoid OOM. Tested on 32GB RTX 5090.

## Robomimic
# export benchmark_name="robomimic"

# export task_name="robomimic_square_ph"
# export task_name="robomimic_square_mh"
# export task_name="robomimic_transport_ph"
# export task_name="robomimic_transport_mh"
# export task_name="robomimic_tool_hang_ph"

export dataset_compressed_dir="data/datasets/${benchmark_name}"
export normalizer_dir="data/datasets/${benchmark_name}"
export dataset_root_dir="data/datasets/${benchmark_name}"

run_name="${task_name}_${policy_name//_transformer}"

if [[ -z "$logger_project_name" ]]; then
    logger_project_name=$task_name
fi


if [[ -n "$1" ]]; then
    gpu_ids=$1
else
    if [[ -z "$gpu_ids" ]]; then
        echo "Use all GPUs"
        gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        gpu_ids=$(seq -s, 0 $((gpu_num-1))) # 0,1,2,3,...
    fi
fi
echo "gpu_ids is set to $gpu_ids"

# Parse gpu_ids to get the number of processes
export num_processes=$(echo $gpu_ids | tr ',' '\n' | wc -l)


# Adjustable arguments

export additional_args=$additional_args" +workspace.train_dataset.dataloader_cfg.num_workers=12" # The default number of workers is 24. Reduce the number of workers server with fewer CPU cores or less memory
export additional_args=$additional_args" +workspace.train_dataset.dataloader_cfg.batch_size=1" # The default batch size in the dataset configs fits for 80GB GPUs. Reduce the batch size for GPU with smaller memory. 
export additional_args=$additional_args" +workspace.train_dataset.split_dataloader_cfg.num_workers=8" # The default number of workers is 12. Reduce the number of workers for server with fewer CPU cores or less memory. 
export additional_args=$additional_args" +workspace.train_dataset.split_dataloader_cfg.batch_size=1" # The default batch size in the dataset configs fits for 80GB GPUs. Reduce the batch size for GPU with smaller memory. 

# When training "memory_transformer" and "gated_transformer", this batch size is the number of the trajectory "sets" to accelerate training. Please refer to the Supplementary Material "Overlapped Trajectory Training" for more details.

# For small datasets, we repeat dataset multiple times to avoid the overhead of dataloader prefetching process after one epoch is exhausted.
# export additional_args=$additional_args" +workspace.train_dataset.repeat_dataset_num=10"

# For debugging (only run 10 training steps and 1 eval step)
# export additional_args=$additional_args" +workspace.trainer.debug=True"

# If activated, the training process will wait for VSCode debugger to attach at port 5678. This also works for multi-GPU training.
# export additional_args=$additional_args" +attach_vscode_debugger=True" 

# export additional_args=$additional_args" +workspace.trainer.optimizer_partial.lr=3e-4"


## Ablation Configs

# Train binary/continuous gate together with the policy
# export additional_args=$additional_args" +workspace.model.denoising_network_partial.straight_through=v3"

# export additional_args=$additional_args" +workspace.trainer.memory_gate_loss_weight=0.0"
# export additional_args=$additional_args" +workspace.trainer.memory_gate_loss_weight=1e-3"
# export additional_args=$additional_args" +workspace.trainer.memory_gate_loss_weight=1e-2"

# Random noising
# export additional_args=$additional_args" +workspace.model.train_history_action_noise_level=random"
# export additional_args=$additional_args" +workspace.model.eval_history_action_noise_level=random"
# No noising
# export additional_args=$additional_args" +workspace.model.train_history_action_noise_level=none"
# export additional_args=$additional_args" +workspace.model.eval_history_action_noise_level=none"


## For fine-tuning a checkpoint or using pretrained models (e.g. ViT in the pick_and_match_color_rand_delay)
if [[ -n "$base_ckpt_path" ]]; then
    export additional_args=$additional_args" +base_ckpt_path=$base_ckpt_path"
fi

if [[ $policy_name == *"gated"* ]]; then
    export additional_args=$additional_args" +workspace.model.memory_gate.ckpt_path=data/checkpoints/${benchmark_name}/${task_name}_memory_gate.ckpt"
fi

source shell_scripts/train_policy.sh