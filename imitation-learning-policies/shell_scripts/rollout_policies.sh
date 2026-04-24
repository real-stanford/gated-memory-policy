#! /bin/bash

root_dir=$(dirname $(dirname $(realpath $0)))
cd $root_dir

export PYTHONPATH=$PYTHONPATH:.
export checkpoints_dir=$1
export filter_str=$2
if [ -z "$checkpoints_dir" ]; then
    echo "checkpoints_dir is not set"
    exit 1
fi

pkill -f "serve_policy.sh"
pkill -f "run_policy_server.py"
pkill -f "serve_remote_env.sh"
pkill -f "serve_remote_env.py"
pkill -f "ray"

ulimit -c 0

gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
gpu_ids=$(seq -s, 0 $((gpu_num-1))) # 0,1,2,3,...

echo "Serving remote envs on GPUs: $gpu_ids"

port=37567
ports=()


. $(conda info --base)/etc/profile.d/conda.sh # This is equivalent to source the conda profile
# conda activate /project/cosmos/yihuaig/miniforge3/envs/imitation
conda activate imitation
echo "Activated conda environment: $CONDA_PREFIX"


for gpu_id in $(seq 0 $((gpu_num-1))); do
    port=$(($port+1))
    ports+=($port)
    echo "Serving policy and remote env on GPU $gpu_id and port $port"
    bash shell_scripts/serve_policy.sh $gpu_id $port &
    sleep 0.1
done

for gpu_id in $(seq 0 $((gpu_num-1))); do
    bash ../mujoco-env/shell_scripts/serve_remote_env.sh $gpu_id ${ports[$gpu_id]} &
    sleep 0.1
done

ports_str=$(IFS=,; echo "${ports[*]}")

command="python scripts/start_multi_gpu_mixed_policy_rollout.py $checkpoints_dir --filter_str=$filter_str --num_servers=$gpu_num --server_name=localhost --start_port=37568 --skip_existing_results"
echo $command
python scripts/start_multi_gpu_mixed_policy_rollout.py $checkpoints_dir --filter_str=$filter_str --num_servers=$gpu_num --server_name=localhost --start_port=37568 --skip_existing_results