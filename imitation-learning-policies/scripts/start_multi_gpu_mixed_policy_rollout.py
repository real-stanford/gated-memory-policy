import datetime
import json
import os
import time
import glob

import click
import numpy as np

from imitation_learning.envs.remote_env import RemoteEnv


def export_results(all_results: list[dict], ckpt_dir: str, filter_str: str):
    os.makedirs(ckpt_dir, exist_ok=True)

    all_results.sort(key=lambda x: (x.get("task_name", ""), x.get("policy_name", "")))

    results_json = f"{ckpt_dir}/{filter_str}_results.json"

    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Exported results to {results_json}")
    
@click.command()
@click.argument("ckpt_dir", type=str)
@click.option("--filter_str", type=str, default="")
@click.option("--num_servers", type=int, default=8)
@click.option("--server_name", type=str, default="localhost")
@click.option("--start_port", type=int  , default=18765)
@click.option("--skip_existing_results", type=bool, flag_value=True, default=False)
def parallel_rollout_policy(
    ckpt_dir: str,
    filter_str: str,
    num_servers: int,
    server_name: str,
    start_port: int,
    skip_existing_results: bool,
):

    ckpt_dir = ckpt_dir.rstrip("/")
    ports = [start_port + i for i in range(num_servers)]

    print(f"Server ports: {ports}")
    envs = [
        RemoteEnv(
            run_name="mixed_policy_rollout",
            time_tag=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            server_endpoint=f"tcp://localhost:{ports[i]}",
            train_server_name=server_name,
            env_id=i,
        )
        for i in range(num_servers)
    ]

    all_results = []

    start_time = time.time()
    for env in envs:
        env.clear_results_buffer()
        env.reset()

    all_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if filter_str:
        all_ckpts = [ckpt for ckpt in all_ckpts if filter_str in ckpt]
    ckpt_num = len(all_ckpts)

    new_ckpts = []
    if skip_existing_results:
        for ckpt_path in all_ckpts:
            single_run_result_json = ckpt_path.replace(".ckpt", "_results.json")
            if os.path.exists(single_run_result_json):
                print(f"Single run result json {single_run_result_json} already exists. Skipping this checkpoint")
                results = json.load(open(single_run_result_json, "r"))
                all_results.append(results)
            else:
                new_ckpts.append(ckpt_path)
        all_ckpts = new_ckpts

    pending_ckpts = list(enumerate(all_ckpts))
    print(f"Found {ckpt_num} checkpoints in {ckpt_dir}. {len(pending_ckpts)} checkpoints to rollout")


    env_is_running = [False] * num_servers
    while True:
        for env in envs:
            if env.rollout_running:
                results = env.fetch_results(check_run_name=False, enforce_increasing_epoch=False)
                if len(results) > 0:
                    print(results)

                    for result in results:
                        single_run_result_json = result['ckpt_path'].replace(".ckpt", "_results.json")
                        all_results.append(result)
                        if os.path.exists(single_run_result_json):
                            print(f"Single run result json {single_run_result_json} already exists. Renaming the old file to {single_run_result_json}.old")
                            os.rename(single_run_result_json, single_run_result_json + ".old")
                        with open(single_run_result_json, "w") as f:
                            print(f"Writing result to {single_run_result_json}")
                            json.dump(result, f, indent=4)
                            
                    print(f"Time taken: {time.time() - start_time} seconds")
                    break
            if not env.rollout_running:
                if pending_ckpts:
                    idx, ckpt_path = pending_ckpts.pop(0)
                    print(f"Starting rollout for checkpoint {ckpt_path}")
                    env.start_rollout(
                        idx,
                        ckpt_path,
                        enforce_increasing_epoch=False,
                        record_history_attention=False,
                    )

            env_is_running[env.env_id] = env.rollout_running

        time.sleep(1)
        if not any(env_is_running):
            export_results(all_results, ckpt_dir, filter_str)
            break


if __name__ == "__main__":
    parallel_rollout_policy()