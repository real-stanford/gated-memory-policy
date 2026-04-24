#!/bin/bash
# Fetch a training checkpoint from a remote machine and run local eval.
#
# Usage: bash fetch_and_eval.sh
#
# Configure the variables below before running. The script will:
#   1. rsync the checkpoint directory from the remote training machine
#   2. select a checkpoint (hardcoded or latest epoch)
#   3. run eval/mikasa_eval.py with the specified settings
set -euo pipefail

# Remote training machine
REMOTE="user@hostname"
REMOTE_PREFIX="/path/to/training/data"
LOCAL_PREFIX="mikasa_models"

# Run to evaluate. Relative path under REMOTE_PREFIX.
# Task name is auto-derived from the first component
# (e.g. mikasa_remember_color_3 -> RememberColor3-v0).
RUN_PATH="mikasa_remember_color_3/2026-03-26/04-36-22_run_name"

# Eval settings
NUM_ENVS=16
NUM_EVAL_STEPS=420
SEED=1000
DIRECT_QPOS=""             # set to "--direct-qpos" for teleport mode
ABS_JOINT_POS=""           # set to "--abs-joint-pos" for absolute position control
NO_PROPRIO=""              # set to "--no-proprio" for vision-only

# Checkpoint override (leave empty to auto-pick latest epoch)
CKPT_OVERRIDE=""
# CKPT_OVERRIDE="checkpoints/epoch_20_train_mean_loss_0_000.ckpt"

# 1. Fetch checkpoint from remote
TASK_FOLDER=$(echo "${RUN_PATH}" | cut -d/ -f1)
TASK_NAME=$(echo "${TASK_FOLDER}" | sed 's/^mikasa_//' \
    | sed -E 's/_([a-z])/\U\1/g' \
    | sed -E 's/_([0-9])/\1/g' \
    | sed -E 's/^([a-z])/\U\1/')
ENV_ID="${TASK_NAME}-v0"

LOCAL_DIR="${LOCAL_PREFIX}/${RUN_PATH}"
REMOTE_DIR="${REMOTE}:${REMOTE_PREFIX}/${RUN_PATH}/"

echo "Fetching from remote: ${REMOTE_DIR} -> ${LOCAL_DIR}"
mkdir -p "${LOCAL_DIR}"
rsync -rzvP "${REMOTE_DIR}" "${LOCAL_DIR}"

# 2. Select checkpoint
if [ -n "${CKPT_OVERRIDE}" ]; then
    CKPT="${LOCAL_DIR}/${CKPT_OVERRIDE}"
    [ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
else
    CKPT_DIR="${LOCAL_DIR}/checkpoints"
    [ -d "${CKPT_DIR}" ] || { echo "ERROR: no checkpoints dir at ${CKPT_DIR}"; exit 1; }
    CKPT=$(ls "${CKPT_DIR}"/epoch_*.ckpt 2>/dev/null | sort -V | tail -1)
    [ -n "${CKPT}" ] || { echo "ERROR: no epoch_*.ckpt found in ${CKPT_DIR}"; exit 1; }
fi
echo "Using checkpoint: ${CKPT}"

# Build output dir from run suffix and epoch
RUN_SUFFIX=$(basename "${RUN_PATH}" | sed -E 's/^[0-9-]+_[a-z]+_[a-z]+_[0-9]+_diffusion_memory_lr[^_]+_//')
EPOCH=$(basename "${CKPT}" | grep -oP 'epoch_\d+')
MODE=""
[ -n "${DIRECT_QPOS}" ] && MODE="_dqpos"
[ -n "${ABS_JOINT_POS}" ] && MODE="_absjpos"
OUTPUT_DIR="eval_results/${TASK_NAME}_${RUN_SUFFIX}_${EPOCH}${MODE}"

# 3. Run eval
echo "Running eval: ${ENV_ID} | ${NUM_ENVS} envs | ${NUM_EVAL_STEPS} steps | seed ${SEED}"
echo "Output: ${OUTPUT_DIR}"

conda run --no-capture-output -n mikasa python -u eval/mikasa_eval.py \
    --env-id "${ENV_ID}" \
    --checkpoint "${CKPT}" \
    ${DIRECT_QPOS:-} \
    ${ABS_JOINT_POS:-} \
    ${NO_PROPRIO:-} \
    --num-envs "${NUM_ENVS}" \
    --num-eval-steps "${NUM_EVAL_STEPS}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}"

echo "Done: ${OUTPUT_DIR}/${ENV_ID}"
