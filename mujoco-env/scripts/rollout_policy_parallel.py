import os
import time
from typing import Any, cast

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf, ListConfig

from env.modules.agents.manipulation_policy_parallel_agent import (
    ManipulationPolicyParallelAgent,
)
from env.modules.agents.robomimic_policy_parallel_agent import (
    RobomimicPolicyParallelAgent,
)
from env.modules.tasks.parallel_task import ParallelTask
from robot_utils.config_utils import disable_hydra_target
from env.utils.config_utils import convert_task_to_parallel
from env.utils.visualize_utils import (
    convert_data_to_video_parallel,
)
import robotmq

@hydra.main(
    config_path="../config", config_name="rollout_policy_parallel", version_base=None
)
def main(cfg) -> None:
    # Ensure we're in the project root directory
    np.set_printoptions(precision=4, suppress=True, sign=" ")
    task_name: str = cfg.task_name

    # To instantiate the time string
    time_str = (
        cfg.time_str
    )  # This step will convert {now:%Y-%m-%d-%H-%M-%S} to actual time string
    cfg.time_str = time_str

    assert task_name in [
        "pick_and_place_back",
        "pick_and_match_color",
        "pick_and_match_color_rand_delay",
        "push_cube",
        "fling_cloth",
        "robomimic_square",
        "robomimic_tool_hang",
        "robomimic_transport",
    ], f"Unknown task: {task_name}"

    task_cfg = hydra.compose(config_name=f"task/{task_name}")
    OmegaConf.set_struct(
        task_cfg, False
    )  # Allows adding new fields for the first config
    task_cfg.task = convert_task_to_parallel(task_cfg.task)
    if task_name.startswith("robomimic"):
        agent_cfg = hydra.compose(config_name=f"task/agent/robomimic_policy_parallel")
    else:
        agent_cfg = hydra.compose(config_name=f"task/agent/policy_parallel")

    OmegaConf.set_struct(
        agent_cfg, False
    )  # Allows adding new fields for the first config
    task_cfg = OmegaConf.merge(agent_cfg, task_cfg, cfg)
    print(f"task_cfg: {task_cfg}")
    assert isinstance(task_cfg, DictConfig)

    policy_server_address = task_cfg.task.agent.policy_server_address
    temp_client = robotmq.RMQClient("temp_policy_client", policy_server_address)

    # Fetch policy config from server
    policy_config: dict[str, Any] = robotmq.deserialize(
        temp_client.request_with_data(
            "policy_config",
            robotmq.serialize(True),
        )
    )

    # Configure task based on policy config
    proprio_length: int = policy_config["workspace"]["model"]["proprio_length"]
    image_length: int = policy_config["workspace"]["model"]["image_length"]
    # assert (
    #     proprio_length >= image_length
    # ), f"proprio_length ({proprio_length}) must be greater than or equal to image_length ({image_length})"
    obs_length = max(proprio_length, image_length)

    task_cfg.task.agent.obs_history_len = obs_length
    task_cfg.task.agent.image_obs_frames_ids = ListConfig(
        [
            idx - 1
            for idx in policy_config["workspace"]["model"]["image_indices"]
        ]
    )  # During training data loading, the latest image is indexed as 0, previous frames are -k
    # During rollout, the latest image is indexed as -1, thus need to subtract 1
    task_cfg.task.agent.proprio_obs_frames_ids = ListConfig(
        [
            idx - 1
            for idx in policy_config["workspace"]["model"]["proprio_indices"]
        ]
    )

    task_cfg["task"]["env"] = disable_hydra_target(task_cfg["task"]["env"])

    instantiated_cfg: dict[str, Any] = hydra.utils.instantiate(task_cfg)
    task: ParallelTask = instantiated_cfg["task"]
    task.reset()

    data_storage_dir = instantiated_cfg["task"].data_storage_dir
    if not os.path.exists(data_storage_dir):
        os.makedirs(data_storage_dir, exist_ok=True)

    assert isinstance(task.agent, ManipulationPolicyParallelAgent) or isinstance(
        task.agent, RobomimicPolicyParallelAgent
    )
    policy_config = cast(dict[str, Any], task.agent.get_policy_config())
    if hasattr(task.agent, "use_relative_pose"):
        task.agent = cast(ManipulationPolicyParallelAgent, task.agent)
        use_relative_pose: bool = policy_config["workspace"]["train_dataset"][
            "use_relative_pose"
        ]
        task.agent.use_relative_pose = use_relative_pose
    with open(os.path.join(data_storage_dir, "policy_config.yaml"), "w") as f:
        yaml.dump(policy_config, f)
    print(
        f"Policy config saved to {os.path.join(data_storage_dir, 'policy_config.yaml')}"
    )

    dataset_config = cast(
        ManipulationPolicyParallelAgent, task.agent
    ).get_dataset_config()
    with open(os.path.join(data_storage_dir, "dataset_config.yaml"), "w") as f:
        yaml.dump(dataset_config, f)
    print(
        f"Dataset config saved to {os.path.join(data_storage_dir, 'dataset_config.yaml')}"
    )
    start_time = time.time()
    episode_configs: dict[int, dict[str, Any]] = {}
    episode_num = instantiated_cfg["episode_num"]
    for episode_idx, seed in enumerate(
        range(cfg.start_seed, cfg.start_seed + episode_num)
    ):
        episode_config = {
            "episode_idx": episode_idx,
            "seed": seed,
        }
        episode_configs[episode_idx] = episode_config

    task.run_episodes(episode_configs)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    assert isinstance(task.agent, ManipulationPolicyParallelAgent)
    export_file_path = task.agent.export_recorded_data(f"rollout_{time_str}")

    # Convert rollout data to video
    convert_data_to_video_parallel(
        task.data_storage_dir, split_videos=False, video_type="success"
    )
    convert_data_to_video_parallel(
        task.data_storage_dir, split_videos=False, video_type="failure"
    )

    print(f"Exported rollout data to {export_file_path}")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
