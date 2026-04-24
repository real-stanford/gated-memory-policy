#!/bin/bash
set -e

root_dir=$(dirname $(dirname $(realpath $0)))
cd $root_dir

export PYTHONPATH=$root_dir:$PYTHONPATH

gpu_id=$1


if [[ -z "$CONDA_PREFIX" || "$CONDA_PREFIX" != *"imitation" ]]; then
    . $(conda info --base)/etc/profile.d/conda.sh # This is equivalent to source the conda profile
    conda activate imitation
    echo "Activated conda environment: $CONDA_PREFIX"
fi

python_path=$CONDA_PREFIX/bin/python

if [ -z "$gpu_id" ]; then
    gpu_id=0
fi
export CUDA_VISIBLE_DEVICES=$gpu_id
echo "CUDA_VISIBLE_DEVICES=$gpu_id"

server_port=$2
if [ -z "$server_port" ]; then
    server_port=$(($gpu_id+18765))
fi

additional_args=""
if [ -n "$server_port" ]; then
    additional_args="--server_endpoint tcp://localhost:$server_port"
fi

server_command="$python_path scripts/run_policy_server.py $additional_args"

echo $server_command
eval $server_command
