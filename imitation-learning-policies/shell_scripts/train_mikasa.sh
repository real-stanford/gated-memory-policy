#!/bin/bash

ATTACH_DEBUGGER=false

for arg in "$@"; do
  if [ "$arg" == "--attach_vscode_debugger" ]; then
    ATTACH_DEBUGGER=true
    break
  fi
done

echo "env: "
# env | grep -E 'SLURM'
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
cd $root_dir

export additional_args=""

if $ATTACH_DEBUGGER; then
    export additional_args=$additional_args" +attach_vscode_debugger=True"
fi

# export policy_name="diffusion_gated_transformer"
# export policy_name="diffusion_gated_transformer_large"
# export policy_name="diffusion_continuous_gated_transformer"
# export policy_name="diffusion_binary_gated_transformer"
# export policy_name="longhist_self_attn_diffusion_transformer"
# export policy_name="longhist_cross_attn_diffusion_transformer"
# export policy_name="ptp_diffusion_transformer"
export policy_name="diffusion_memory_transformer"
# export policy_name="flow_memory_transformer"
# export policy_name="legacy_flow_transformer"
# export policy_name="diffusion_transformer"
# export policy_name="flow_transformer"


## Robomimic
### Square
# task_name="robomimic_square_ph"
# run_name="robomimic_square_ph_diffusion"
# run_name="robomimic_square_ph_history_diffusion"
# run_name="robomimic_square_ph_history_diffusion_no_regularization"

# task_name="robomimic_square_mh"
# run_name="robomimic_square_mh_history_diffusion"
# run_name="robomimic_square_mh_history_diffusion_no_regularization"
# run_name="robomimic_square_mh_diffusion"

export project_name="mikasa"
# export task_name="mikasa_shell_game_touch"
export task_name="mikasa_remember_color_3"
# export task_name="mikasa_intercept_medium"
# export task_name="mikasa_intercept_grab_medium"

# export run_name="${mikasa_shell_game_touch}_${policy_name//_transformer}_debug"
# export run_name="${mikasa_remember_color_3}_${policy_name//_transformer}_debug"
export run_name="${task_name//mikasa_}_${policy_name//_transformer}_debug"

# export run_name="${shortened_task_name}_${policy_name//_transformer}_lr3e-4_match_nomem"
# export additional_args=$additional_args" +workspace.train_dataset.traj_num=1"
# export additional_args=$additional_args" +workspace.train_dataset.dataloader_cfg.batch_size=64"
# export additional_args=$additional_args" +workspace.train_dataset.split_dataloader_cfg.batch_size=64"
# export additional_args=$additional_args" +workspace.train_dataset.index_pool_size_per_episode=-1"
# export additional_args=$additional_args" +workspace.train_dataset.repeat_dataset_num=10"
# export additional_args=$additional_args" +workspace.model.denoising_network_partial.max_history_len=0"

# additional_args=$additional_args" +workspace.model.memory_gate.ckpt_path=data/robomimic_tool_hang_ph/2025-12-25/15-51-13_robomimic_tool_hang_ph_gate_lr1e-4/checkpoints/epoch_4_train_mean_loss_0_001.ckpt"
# run_name="robomimic_tool_hang_ph_ours_no_regularization"
# export run_name="tool_hang_ph_no_hist"
# export run_name="tool_hang_ph_ours_refactored_weight1e-3"
# export run_name="rool_hang_gated_debug"
# export run_name="tool_hang_ph_legacy_flow_interactive"

# project_name="transport"
# task_name="robomimic_transport_ph"
# run_name="robomimic_transport_ph_history_diffusion"
# run_name="robomimic_transport_ph_diffusion"

# task_name="robomimic_transport_mh"
# run_name="robomimic_transport_mh_history_diffusion"
# run_name="robomimic_transport_mh_diffusion"

# run_name="memory_gate_0_001_eye_in_hand"
# run_name="rand_interval"
# run_name="64_traj_mask_prob_0_5"
# run_name="32_traj"


## MemoryGym
### Push Cube

# export project_name="push_cube"
# export task_name="push_cube"
# additional_args=$additional_args" +workspace.model.memory_gate.ckpt_path=data/push_cube/2026-01-06/00-01-15_push_cube_gate_lr1e-4_window_10_ratio_5/checkpoints/epoch_4_train_mean_loss_0_193.ckpt"
# additional_args=$additional_args" +workspace.model.memory_gate.ckpt_path=data/push_cube/2026-01-06/00-01-15_push_cube_gate_lr1e-4_window_10_ratio_5/checkpoints/epoch_1_train_mean_loss_0_204.ckpt"
# run_name="push_cube_ours_no_regularization"
# run_name="push_cube_rand_noise_level"
# additional_args=$additional_args" +workspace.train_dataset.name=push_cube_1000episodes"


### Pick and Place Back

# project_name="pick_and_place_back"
# task_name="pick_and_place_back"
# additional_args=$additional_args" +workspace.model.memory_gate.ckpt_path=data/pick_and_place_back/2025-12-22/09-32-12_place_back_gate_lr1e-4/checkpoints/epoch_0_train_mean_loss_0_060.ckpt"
# additional_args=$additional_args" +workspace.model.memory_gate=null"
# project_name="pick_and_match_color"
# task_name="pick_and_match_color"
# run_name="init"
# additional_args=$additional_args" +workspace.train_dataset.name=2025-08-16_15-36-52_pick_and_match_color_no_drifting_1000episodes"

### Fling Cloth

# task_name="fling_cloth"
# run_name="fling_cloth_ours"
# additional_args=$additional_args" +workspace.train_dataset.name=fling_cloth_1000episodes"

## RealWorld

# project_name="iphumi_place_back"
# task_name="iphumi_place_back"
# run_name="pick-and-place-back-0906_ultrawide_history"
# additional_args=$additional_args" +workspace.train_dataset.name=pick-and-place-back-0906"

# run_name="${task_name//robomimic_}_${policy_name//_transformer}_debug_v2"


# additional_args=$additional_args" +workspace.train_dataset.name=pick-up-only"

if [[ -z "$logger_project_name" ]]; then
    logger_project_name=$task_name
fi


if [[ -n "$1" ]]; then
    gpu_ids=$1
else
    if [[ -z "$gpu_ids" ]]; then
        gpu_ids=0
    fi
fi
echo "gpu_ids is set to $gpu_ids"

# Parse gpu_ids to get the number of processes
export num_processes=$(echo $gpu_ids | tr ',' '\n' | wc -l)


# Adjustable arguments

# additional_args=$additional_args" +time_str=06-26-00"
 
export additional_args=$additional_args" +workspace.trainer.debug=True"
export additional_args=$additional_args" +workspace.trainer.rollout_every=0"

# additional_args=$additional_args" +workspace.model.train_history_action_noise_level=random"
# additional_args=$additional_args" +workspace.model.eval_history_action_noise_level=none"

additional_args=$additional_args" +workspace.train_dataset.dataloader_cfg.batch_size=1"
additional_args=$additional_args" +workspace.train_dataset.dataloader_cfg.num_workers=1"
# additional_args=$additional_args" +workspace.model.denoising_network_partial.add_memory_gate_token=True"
# additional_args=$additional_args" +workspace.model.denoising_network_partial.skip_history_attn=True"
# additional_args=$additional_args" +workspace.model.denoising_network_partial.record_data_entries=[]"

# additional_args=$additional_args" +workspace.model.denoising_network_partial.straight_through=v3"
# additional_args=$additional_args" +workspace.train_dataset.traj_num=32"
# additional_args=$additional_args" +workspace.trainer.optimizer_partial.lr=3e-4"

# additional_args=$additional_args" +workspace.trainer.memory_gate_loss_weight=1e-3"
# additional_args=$additional_args" +workspace.trainer.memory_gate_loss_weight=1e-2"

# additional_args=$additional_args" +workspace.trainer.memory_gate_loss_weight=0.0"
# additional_args=$additional_args" +workspace.model.denoising_network_partial.add_additional_self_attn=False"

## To disable history attention (for ablation study)


if [[ -n "$base_ckpt_path" ]]; then
    export additional_args=$additional_args" +base_ckpt_path=$base_ckpt_path"
fi


source shell_scripts/train_policy.sh