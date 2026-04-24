import os
import time
from typing import Any

import hydra
import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf

from env.modules.tasks.base_task import BaseTask
from env.utils.data_utils import compress_data, merge_episode_data
from env.utils.visualize_utils import (
    convert_data_to_video_parallel,
)


@ray.remote
def run_worker(hydra_cfg: DictConfig, episode_configs: list[dict[str, Any]]) -> None:
    instantiated_cfg: dict[str, Any] = hydra.utils.instantiate(hydra_cfg)
    task: BaseTask = instantiated_cfg["task"]
    task.run_episodes(episode_configs)


@hydra.main(
    config_path="../config", config_name="collect_heuristic_data", version_base=None
)
def main(cfg) -> None:
    # Ensure we're in the project root directory

    task_name = cfg.task_name
    run_name = cfg.run_name

    np.set_printoptions(precision=4, suppress=True, sign=" ")
    data_storage_dir = (
        cfg.task.data_storage_dir
    )  # This step will resolve the current timestamp
    cfg.task.data_storage_dir = data_storage_dir  # Override the data_storage_dir in cfg

    task_cfg = hydra.compose(config_name=f"task/{task_name}")
    agent_cfg = hydra.compose(config_name=f"task/agent/heuristic/{task_name}")
    OmegaConf.set_struct(
        task_cfg, False
    )  # Allows adding new fields for the first config
    OmegaConf.set_struct(
        agent_cfg, False
    )  # Allows adding new fields for the first config
    agent_cfg.task.agent = agent_cfg.task.agent.heuristic
    task_cfg = OmegaConf.merge(agent_cfg, task_cfg, cfg)
    assert isinstance(task_cfg, DictConfig)
    print(f"task_cfg:{task_cfg}")

    if not os.path.exists(data_storage_dir):
        os.makedirs(data_storage_dir, exist_ok=True)
    config_file_path = os.path.join(data_storage_dir, "sim_config.yaml")
    with open(config_file_path, "w") as f:
        OmegaConf.save(config=task_cfg, f=f)

    total_episodes: int = task_cfg.total_episodes
    worker_num: int = task_cfg.parallel_workers
    episode_configs: list[dict[str, Any]] = []
    for i in range(total_episodes):
        episode_config = {
            "episode_idx": i,
            "seed": i,
        }

        episode_configs.append(episode_config)

    if worker_num > 1:
        ray.init()
        episode_num_per_worker = total_episodes // worker_num
        if episode_num_per_worker == 0:
            episode_num_per_worker = 1
        episode_configs_per_worker = [
            episode_configs[i : i + episode_num_per_worker]
            for i in range(0, len(episode_configs), episode_num_per_worker)
        ]
        workers = [
            run_worker.remote(task_cfg, episode_configs_per_worker)
            for episode_configs_per_worker in episode_configs_per_worker
        ]
        start_time = time.time()
        ray.get(workers)
        end_time = time.time()
        ray.shutdown()
    else:
        print("Running in single-thread mode")
        instantiated_cfg: dict[str, Any] = hydra.utils.instantiate(task_cfg)
        task: BaseTask = instantiated_cfg["task"]
        start_time = time.time()
        task.run_episodes(episode_configs)
        end_time = time.time()

    print(f"==========================================================================")
    print(
        f"Time taken: {end_time - start_time:.2f} seconds for {total_episodes} episodes accross {worker_num} workers"
    )
    print(f"==========================================================================")
    if cfg.merge_data:
        merge_episode_data(data_storage_dir)
    if cfg.compress_data:
        compress_data(data_storage_dir)

    convert_data_to_video_parallel(
        data_storage_dir,
        split_videos=False,
        video_type="success",
        show_timestep=True,
    )
    convert_data_to_video_parallel(
        data_storage_dir,
        split_videos=False,
        video_type="failure",
        show_timestep=True,
    )


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"

    main()
