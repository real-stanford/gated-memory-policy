#!/bin/bash
# Usage: ./collect_data.sh [env-id] [num-episodes] [ckpt-dir] [save-dir]
# Args override defaults below.
set -euo pipefail

ENV_ID="${1:-TakeItBack-v0}"
TOTAL="${2:-1000}"
CKPT_DIR="${3:-.}"
SAVE_DIR="${4:-data}"

(( TOTAL % 250 == 0 )) || { echo "Error: num-episodes must be divisible by 250"; exit 1; }

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
PER_GPU=$(( (TOTAL / NUM_GPUS / 250) * 250 ))
(( PER_GPU == 0 )) && NUM_GPUS=1 && PER_GPU=$TOTAL

echo "Collecting $TOTAL eps of $ENV_ID on $NUM_GPUS GPU(s) ($PER_GPU/gpu)"

# Each GPU saves to gpu_<i> subfolder under batched/unbatched to avoid collisions
PIDS=()
for (( i=0; i<NUM_GPUS; i++ )); do
    N=$PER_GPU
    (( i == NUM_GPUS-1 )) && N=$(( TOTAL - PER_GPU * i ))
    (( N == 0 )) && continue

    (( i > 0 )) && sleep 20

    CUDA_VISIBLE_DEVICES=$i python3 mikasa_robo_suite/dataset_collectors/get_mikasa_robo_datasets.py \
        --env-id="$ENV_ID" --path-to-save-data="$SAVE_DIR/gpu_$i" \
        --ckpt-dir="$CKPT_DIR" --num-train-data="$N" &
    PIDS+=($!)
done

FAIL=0
for P in "${PIDS[@]}"; do wait "$P" || FAIL=$((FAIL+1)); done
(( FAIL > 0 )) && echo "$FAIL process(es) failed" && exit 1

# Merge per-GPU unbatched into final path, sequential naming
OUT="$SAVE_DIR/MIKASA-Robo/unbatched/$ENV_ID"
mkdir -p "$OUT"
IDX=0
for (( i=0; i<NUM_GPUS; i++ )); do
    SRC="$SAVE_DIR/gpu_$i/MIKASA-Robo/unbatched/$ENV_ID"
    [ -d "$SRC" ] || continue
    for f in "$SRC"/train_data_*.npz; do
        [ -f "$f" ] && mv "$f" "$OUT/train_data_$((IDX++)).npz"
    done
    rm -rf "$SAVE_DIR/gpu_$i"
done

echo "Done: $IDX episodes -> $OUT"
