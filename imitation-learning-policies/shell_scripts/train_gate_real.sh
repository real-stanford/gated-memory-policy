#!/bin/bash

export additional_args=""

data_prefix=data

project_name="memory_gate_training"

# task_name="iphumi_place_back"
# export with_mem_ckpt_path="${data_prefix}/iphumi_place_back/2026-01-19/22-33-43_1104_diffusion_memory_lr3e-4_no_proprio_half_41epochs/checkpoints/epoch_5_train_mean_loss_0_001.ckpt"
# export no_mem_ckpt_path="${data_prefix}/iphumi_place_back/2026-01-19/22-31-20_1104_diffusion_lr3e-4_no_proprio_half_41epochs/checkpoints/epoch_5_train_mean_loss_0_001.ckpt"

# export with_mem_ckpt_path="${data_prefix}/iphumi_place_back/2025-12-25/18-34-57_1104_diffusion_memory_lr3e-4_half/checkpoints/epoch_20_train_mean_loss_0_000.ckpt"
# export no_mem_ckpt_path="${data_prefix}/iphumi_place_back/2025-12-25/18-35-08_1104_diffusion_lr3e-4_half/checkpoints/epoch_20_train_mean_loss_0_000.ckpt"
# export with_mem_ckpt_path="data/iphumi_place_back/2025-12-25/18-34-57_1104_diffusion_memory_lr3e-4_half/checkpoints/epoch_20_train_mean_loss_0_000.ckpt"
# export no_mem_ckpt_path="data/iphumi_place_back/2025-12-25/18-35-08_1104_diffusion_lr3e-4_half/checkpoints/epoch_20_train_mean_loss_0_000.ckpt"

task_name="real_world_iterative_casting"
export dataset_name="real_world_iterative_casting_0122"
export additional_args=$additional_args" +workspace.train_dataset.name=$dataset_name"
export with_mem_ckpt_path="${data_prefix}/real_world_iterative_casting/2026-01-26/17-16-29_casting_0122_diffusion_memory_half/checkpoints/epoch_15_train_mean_loss_0_000.ckpt"
export no_mem_ckpt_path="${data_prefix}/real_world_iterative_casting/2026-01-26/17-16-29_casting_0122_diffusion_half/checkpoints/epoch_11_train_mean_loss_0_000.ckpt"
export final_statistics_path="${data_prefix}/real_world_iterative_casting/2026-01-26/23-15-42_val_comparison/val_results_statistics_window_20.pt"

shortened_task_name=${task_name#robomimic_}
shortened_task_name=${shortened_task_name#pick_and_}
export run_name="${shortened_task_name}_gate_lr1e-4"


export with_mem_idx_pool_size=500
export no_mem_idx_pool_size=20000

export dataset_type="val"
export date_str=$(date +%Y-%m-%d)
export time_str=$(date +%H-%M-%S)

export window_size=20

if [ -z "$final_statistics_path" ]; then
    export eval_episode_num=-1
    # export eval_episode_num=1
    # export eval_episode_num=100
    if [ $eval_episode_num -eq -1 ]; then
        export statistics_path="${data_prefix}/${task_name}/${date_str}/${time_str}_${dataset_type}_comparison/val_results_statistics.pt"
    else
        export statistics_path="${data_prefix}/${task_name}/${date_str}/${time_str}_${dataset_type}_comparison/${dataset_type}_results_statistics_${eval_episode_num}.pt"
    fi
else
    export skip_gate_label_calculation=1
fi
# export additional_args=$additional_args" +workspace.trainer.debug=True"

export run_name="${task_name}_gate_lr1e-4_window_${window_size}_ratio_10"
export run_dir="${task_name}/${date_str}/${time_str}_${run_name}"
export checkpoints_dir="${run_dir}/checkpoints"


source shell_scripts/generate_gate_labels.sh
source shell_scripts/train_gate.sh
