import os
import time
from typing import Any

import hydra
import numpy as np
import numpy.typing as npt
import ray
from omegaconf import OmegaConf

from env.modules.tasks.base_task import BaseTask


@ray.remote
def run_worker(
    hydra_cfg: dict[str, Any], episode_configs: list[dict[str, Any]]
) -> None:
    instantiated_cfg: dict[str, Any] = hydra.utils.instantiate(hydra_cfg)
    task: BaseTask = instantiated_cfg["task"]
    task.run_episodes(episode_configs)


@hydra.main(
    config_path="../config", config_name="collect_sliding_distances", version_base=None
)
def main(cfg) -> None:
    # Ensure we're in the project root directory

    np.set_printoptions(precision=4, suppress=True, sign=" ")
    data_storage_dir = (
        cfg.task.data_storage_dir
    )  # This step will resolve the current timestamp
    cfg.task.data_storage_dir = data_storage_dir  # Override the data_storage_dir in cfg
    if not os.path.exists(data_storage_dir):
        os.makedirs(data_storage_dir, exist_ok=True)
    config_file_path = os.path.join(data_storage_dir, "sim_config.yaml")
    with open(config_file_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    if "friction_num" in cfg:
        frictions = (
            np.linspace(
                np.sqrt(cfg.friction_min), np.sqrt(cfg.friction_max), cfg.friction_num
            )
            ** 2
        )
        total_episodes: int = cfg.friction_num * cfg.delta_push_vel_num * cfg.repeat_num
    else:
        frictions = np.arange(cfg.friction_min, cfg.friction_max, cfg.friction_step)
        total_episodes = len(frictions) * cfg.delta_push_vel_num * cfg.repeat_num

    worker_num: int = cfg.parallel_workers
    delta_vel_start: float = -cfg.delta_push_vel * (cfg.delta_push_vel_num - 1) / 2
    delta_vel_end: float = cfg.delta_push_vel * (cfg.delta_push_vel_num - 1) / 2
    delta_vels: npt.NDArray[np.float64] = np.linspace(
        delta_vel_start, delta_vel_end, cfg.delta_push_vel_num
    )
    episode_configs: list[dict[str, Any]] = []

    g = 9.81
    x0 = -0.475  # Actually it should be -0.5
    x = 0.125

    seed = 0
    for friction in frictions:
        default_push_vel = np.sqrt(2 * g * (x - x0) * friction)
        # Round push vel to the closest 0.001
        rounded_push_vel = (
            round(default_push_vel / cfg.delta_push_vel) * cfg.delta_push_vel
        )
        for delta_vel in delta_vels:
            for trial in range(cfg.repeat_num):
                episode_config = {
                    "seed": seed,
                    "sliding_friction": friction,
                    "push_vels_m_per_s": [rounded_push_vel + delta_vel],
                }
                episode_configs.append(episode_config)
                seed += 1
    assert (
        len(episode_configs) == total_episodes
    ), f"len(episode_configs) = {len(episode_configs)}, total_episodes = {total_episodes}"

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
            run_worker.remote(cfg, episode_configs_per_worker)
            for episode_configs_per_worker in episode_configs_per_worker
        ]
        start_time = time.time()
        ray.get(workers)
        end_time = time.time()
        ray.shutdown()
    else:
        print("Running in single-thread mode")
        instantiated_cfg: dict[str, Any] = hydra.utils.instantiate(cfg)
        task: BaseTask = instantiated_cfg["task"]
        start_time = time.time()
        task.run_episodes(episode_configs)
        end_time = time.time()

    print(f"==========================================================================")
    print(
        f"Time taken: {end_time - start_time:.2f} seconds for {total_episodes} episodes accross {worker_num} workers"
    )
    print(f"==========================================================================")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
