#! /bin/sh

root_dir=$(dirname $(dirname $(realpath $0)))

python_path=$(conda info --base)/envs/mujoco-env/bin/python

$python_path $root_dir/scripts/serve_rollout_website.py \
 $root_dir/data --exclude-dirs "rollout_policy,run_heuristic_agent"