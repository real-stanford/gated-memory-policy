import os
from typing import Any

import hydra
import numpy as np
from omegaconf import OmegaConf

from env.modules.tasks.base_task import BaseTask


@hydra.main(config_path="../config", config_name="spacemouse_teleop", version_base=None)
def main(cfg) -> None:
    # Ensure we're in the project root directory
    np.set_printoptions(precision=4, suppress=True, sign=" ")

    task_name: str = cfg.task_name

    if task_name.startswith("robomimic"):
        raise ValueError(f"Robomimic task does not support spacemouse teleop yet.")

    assert task_name in ["pick_and_place_back", "pick_and_match_color_rand_delay", "pick_and_match_color", "push_cube", "fling_cloth"]

    task_cfg = hydra.compose(config_name=f"task/{task_name}")
    agent_cfg = hydra.compose(config_name=f"task/agent/spacemouse")
    OmegaConf.set_struct(
        task_cfg, False
    )  # Allows adding new fields for the first config
    OmegaConf.set_struct(
        agent_cfg, False
    )  # Allows adding new fields for the first config
    task_cfg = OmegaConf.merge(agent_cfg, task_cfg, cfg)
    print(f"task_cfg: {task_cfg}")

    instantiated_cfg: dict[str, Any] = hydra.utils.instantiate(task_cfg)

    task: BaseTask = instantiated_cfg["task"]
    for i in range(cfg.episode_num):
        # task.run_episode(episode_config={"seed": i, "object_init_bin_ids": [0]})
        # task.run_episode(episode_config={"seed": i, "object_init_bin_ids": None})

        episode_config = {
            "episode_idx": i,
            "seed": i,
        }

        if task_name == "pick_and_place_back":
            episode_config.update(
                {
                    # For pick_and_place_back
                    # "object_init_bin_ids": None,
                    # "switch_bin_colors": True,
                    # "init_bin_materials": [0, 0, 0, 0],
                }
            )
        task.run_episode(episode_config)


if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
