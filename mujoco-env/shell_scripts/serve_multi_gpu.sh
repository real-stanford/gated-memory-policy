#!/bin/sh

pkill -f "serve_remote_env.sh"
pkill -f "serve_remote_env.py"
pkill -f "ray"

gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
gpu_ids=$(seq -s, 0 $((gpu_num-1))) # 0,1,2,3,...


for gpu_id in $(seq 0 $((gpu_num-1))); do
    port=$((18765 + $gpu_id))
    echo "Serving remote env on GPU $gpu_id, port $port"
    shell_scripts/serve_remote_env.sh $gpu_id $port &
done