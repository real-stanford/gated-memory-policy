import os
import time
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
import zarr
from omegaconf import DictConfig, OmegaConf

from env.utils.visualize_utils import convert_data_to_video_parallel

if TYPE_CHECKING:
    from env.modules.tasks.base_task import BaseTask

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@hydra.main(
    config_path="../config", config_name="run_heuristic_agent", version_base=None
)
def main(cfg: DictConfig) -> None:
    # Ensure we're in the project root directory
    np.set_printoptions(precision=4, suppress=True, sign=" ")

    task_name: str = cfg.task_name
    assert task_name in [
        "push_cube",
        "pick_and_place_back",
        "pick_and_match_color",
        "pick_and_match_color_rand_delay",
        "fling_cloth",
    ]

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
    print(f"task_cfg: {task_cfg}")

    task: "BaseTask" = hydra.utils.instantiate(task_cfg)["task"]
    start_time: float = time.time()
    root = zarr.open(f"{task.data_storage_dir}/episode_data.zarr", mode="a")
    assert isinstance(root, zarr.Group)

    try:
        episode_configs = []
        for i in range(0, cfg.episode_num):
            episode_config: dict[str, Any] = {"episode_idx": i, "seed": i + cfg.start_seed}
            if task_name == "fling_cloth":
                # HACK: for testing fling_cloth
                episode_config.update(
                    # {"cloth_mass": 2.0, "speed_scales": [0.7, 0.9, 1.0]}
                    # {"cloth_mass": 0.2, "speed_scales": [0.7, 0.5, 0.4]}
                    {"cloth_mass": 0.15}
                )

                pass

            episode_configs.append(episode_config)
        task.run_episodes(episode_configs)
    except KeyboardInterrupt:
        print(f"Keyboard interrupt detected. Stopping...")

    convert_data_to_video_parallel(
        task.data_storage_dir, split_videos=False, video_type="failure"
    )
    convert_data_to_video_parallel(
        task.data_storage_dir, split_videos=False, video_type="success"
    )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["EGL_DEVICE_ID"] = "0"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    main()
