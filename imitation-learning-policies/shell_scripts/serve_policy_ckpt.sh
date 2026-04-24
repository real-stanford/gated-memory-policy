#!/bin/bash
set -e

ckpt_path=$1

gpu_id=$2

if [ -z "$gpu_id" ]; then
    gpu_id=0
    echo "No GPU ID provided. Using default GPU 0."
fi

if [ -z "$server_port" ]; then
    server_port=$(($gpu_id+18765))
fi
record_history_attention=$3

python_path=$(conda info --base)/envs/imitation/bin/python


root_dir=$(dirname $(dirname $(realpath $0)))
cd $root_dir

export PYTHONPATH=$root_dir:$PYTHONPATH
train_server_name="localhost" # This is to automatically fetch checkpoints from another server if other than localhost

export CUDA_VISIBLE_DEVICES=$gpu_id
echo "CUDA_VISIBLE_DEVICES=$gpu_id"

if [ -z "$record_history_attention" ]; then
    record_history_attention_arg=""
else
    record_history_attention_arg="--record_history_attention $record_history_attention"
    echo "Record history attention: $record_history_attention"
fi

server_command="$python_path scripts/run_policy_server.py --ckpt_path $ckpt_path --server_name $train_server_name $record_history_attention_arg --server_endpoint tcp://localhost:$server_port"
echo $server_command
eval $server_command
