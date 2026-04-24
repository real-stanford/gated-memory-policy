#!/bin/bash

project_name="memory_gate_training"

task_name=push_cube
final_statistics_path="data/push_cube/2025-12-25/20-48-06_val_comparison/val_results_statistics_100_window_20.pt"

with_mem_ckpt_path="data/push_cube/2025-12-16/10-30-07_push_cube_diffusion_memory_lr3e-4_half_no_gate/checkpoints/epoch_20_train_mean_loss_0_000.ckpt"
no_mem_ckpt_path="data/push_cube/2025-12-16/10-30-06_push_cube_diffusion_lr3e-4_half/checkpoints/epoch_20_train_mean_loss_0_000.ckpt"

shortened_task_name=${task_name#robomimic_}
shortened_task_name=${shortened_task_name#pick_and_}
export run_name="${shortened_task_name}_gate_mse_lr1e-4"


export with_mem_idx_pool_size=500
export no_mem_idx_pool_size=20000

export dataset_type=val
export eval_episode_num=2
export date_str=$(date +%Y-%m-%d)
export time_str=$(date +%H-%M-%S)

export additional_args=""
export additional_args=$additional_args" +workspace.model.loss_fn_name=mse"
# export additional_args=$additional_args" +workspace.trainer.debug=True"

# source shell_scripts/generate_gate_labels.sh
source shell_scripts/train_gate.sh

# dataset_type=train

# source shell_scripts/train_gate.sh


