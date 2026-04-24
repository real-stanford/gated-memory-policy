import copy
import os
import shutil
from threading import currentThread
import time
from typing import TYPE_CHECKING, Any

import hydra
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf


if TYPE_CHECKING:
    from env.modules.tasks.parallel_task import ParallelTask
    from env.modules.agents.manipulation_policy_parallel_agent import (
        ManipulationPolicyParallelAgent,
    )

import robotmq

from env.modules.tasks.parallel_task import ParallelTask
from robot_utils.config_utils import disable_hydra_target
from env.utils.config_utils import convert_task_to_parallel
from env.utils.visualize_utils import convert_data_to_video_parallel
from loguru import logger


class RemoteEnvServer:
    def __init__(
        self,
        rollout_episode_num: int,
        start_seed: int,
        policy_server_address: str,
        env_num: int,
    ):

        logger.info(
            f"RemoteEnvServer init with {rollout_episode_num=} {start_seed=} {policy_server_address=} {env_num=}"
        )

        self.server_endpoint: str = policy_server_address
        self.rollout_episode_num: int = rollout_episode_num
        self.start_seed: int = start_seed
        self.client: robotmq.RMQClient = robotmq.RMQClient(
            "mujoco_parallel_env", self.server_endpoint
        )
        self.env_num: int = env_num

    def run(self):
        while True:
            logger.info("Waiting for new checkpoint loaded")
            while (
                self.client.get_topic_status("new_checkpoint_loaded", timeout_s=1) <= 0
            ):
                # Wait until the server is connected and new checkpoint is loaded
                time.sleep(1)

            raw_data, _ = self.client.pop_data("new_checkpoint_loaded", 1)

            assert len(raw_data) == 1
            new_checkpoint_info: dict[str, Any] = robotmq.deserialize(raw_data[0])
            """
            "task_name": str,
            "run_name": str,
            "policy_name": str,
            "time_tag": str,
            "global_step": int,
            "epoch": int,
            "ckpt_path": str
            "train_server_name": str
            """

            policy_config: dict[str, Any] = robotmq.deserialize(
                self.client.request_with_data(
                    "policy_config",
                    robotmq.serialize(True),
                )
            )

            date_str = "-".join(new_checkpoint_info["time_tag"].split("-")[:3])
            time_str = "-".join(new_checkpoint_info["time_tag"].split("-")[3:])
            task_name = new_checkpoint_info["task_name"]

            # For robomimic tasks
            if "_ph" in task_name:
                task_name = task_name.replace("_ph", "")
            if "_mh" in task_name:
                task_name = task_name.replace("_mh", "")

            assert task_name in [
                "pick_and_place_back",
                "pick_and_match_color",
                "pick_and_match_color_rand_delay",
                "push_cube",
                "fling_cloth",
                "robomimic_tool_hang",
                "robomimic_square",
                "robomimic_transport",
            ]

            series_output_dir = os.path.join(
                "data/rollout_policy_online",
                f"{new_checkpoint_info['task_name']}",
                f"{date_str}",
                f"{time_str}_{new_checkpoint_info['run_name']}",
            )
            os.makedirs(series_output_dir, exist_ok=True)

            task_cfg = hydra.compose(config_name=f"task/{task_name}")
            OmegaConf.set_struct(task_cfg, False)
            task_cfg.task = convert_task_to_parallel(task_cfg.task)

            if task_name.startswith("robomimic"):
                agent_cfg = hydra.compose(
                    config_name=f"task/agent/robomimic_policy_parallel"
                )
            else:
                agent_cfg = hydra.compose(config_name=f"task/agent/policy_parallel")

            # Allows adding new fields for the first config
            OmegaConf.set_struct(agent_cfg, False)
            task_cfg = OmegaConf.merge(agent_cfg, task_cfg)
            assert isinstance(task_cfg, DictConfig)

            data_storage_dir = os.path.join(
                series_output_dir,
                f"epoch_{new_checkpoint_info['epoch']}",
            )
            logger_handler_id = logger.add(data_storage_dir + "/rollout.log")
            task_cfg["task"]["data_storage_dir"] = data_storage_dir
            task_cfg["task"]["agent"]["policy_server_address"] = self.server_endpoint
            task_cfg["task"]["env_num"] = self.env_num
            task_cfg["task"]["env"] = disable_hydra_target(task_cfg["task"]["env"])

            try:
                proprio_length: int = policy_config["workspace"]["model"]["proprio_length"]
                image_length: int = policy_config["workspace"]["model"]["image_length"]
                action_length: int = policy_config["workspace"]["model"]["action_length"]
                action_indices: list[int] = policy_config["workspace"]["model"]["action_indices"]
                image_indices: list[int] = policy_config["workspace"]["model"]["image_indices"]
                proprio_indices: list[int] = policy_config["workspace"]["model"]["proprio_indices"]
            except KeyError:
                print(f"using train_dataset instead of model for legacy checkpoint compatibility")
                proprio_length= policy_config["workspace"]["train_dataset"]["proprio_length"]
                image_length = policy_config["workspace"]["train_dataset"]["image_length"]
                action_length = policy_config["workspace"]["train_dataset"]["action_length"]
                action_indices = policy_config["workspace"]["train_dataset"]["action_indices"]
                image_indices = policy_config["workspace"]["train_dataset"]["image_indices"]
                proprio_indices = policy_config["workspace"]["train_dataset"]["proprio_indices"]

            if action_indices[0] < 0: # Using Past token prediction, that predict history actions
                action_length = sum([idx>=0 for idx in action_indices])

            obs_length = max(proprio_length, image_length)
            task_cfg["task"]["render_image_indices"] = ListConfig(
                [
                    idx - 1
                    for idx in image_indices
                ]
            )  # During training data loading, the latest image is indexed as 0, previous frames are -k
            # During rollout, the latest image is indexed as -1, thus need to subtract 1
            task_cfg.task.agent.obs_history_len = obs_length
            task_cfg.task.agent.image_obs_frames_ids = ListConfig(
                [
                    idx - 1
                    for idx in image_indices
                ]
            )  # During training data loading, the latest image is indexed as 0, previous frames are -k
            # During rollout, the latest image is indexed as -1, thus need to subtract 1
                
            task_cfg.task.agent.proprio_obs_frames_ids = ListConfig(
                [
                    idx - 1
                    for idx in proprio_indices
                ]
            )

            print(f"task_cfg.task.agent.proprio_obs_frames_ids: {task_cfg.task.agent.proprio_obs_frames_ids}")


            task_cfg["task"]["agent"]["image_obs_frames_ids"] = task_cfg["task"][
                "render_image_indices"
            ]
            task_cfg["task"]["agent"]["action_prediction_horizon"] = action_length

            logger.info(f"task_cfg: {task_cfg}")

            task: "ParallelTask" = hydra.utils.instantiate(task_cfg)["task"]

            if os.path.exists(task.data_storage_dir):
                shutil.rmtree(task.data_storage_dir)
            os.makedirs(task.data_storage_dir, exist_ok=True)

            if hasattr(task.agent, "use_relative_pose"):
                if TYPE_CHECKING:
                    assert isinstance(task.agent, ManipulationPolicyParallelAgent)
                task.agent.use_relative_pose = policy_config["workspace"][
                    "train_dataset"
                ]["use_relative_pose"]
                logger.info(f"{task.agent.use_relative_pose=}")
            with open(
                os.path.join(task.data_storage_dir, "policy_config.yaml"), "w"
            ) as f:
                yaml.dump(policy_config, f)

            episode_configs: dict[int, dict[str, Any]] = {}
            for episode_idx, seed in enumerate(
                range(
                    self.start_seed,
                    self.start_seed + self.rollout_episode_num,
                )
            ):
                episode_configs[episode_idx] = {
                    "episode_idx": episode_idx,
                    "seed": seed,
                }

            task.reset()

            success_rate, mean_reward = task.run_episodes(episode_configs)

            rollout_results = copy.deepcopy(new_checkpoint_info)
            rollout_results.pop("train_server_name")
            rollout_results["success_rate"] = success_rate

            self.client.put_data("rollout_results", robotmq.serialize(rollout_results))
            logger.info(f"done putting rollout results: {rollout_results}")
            self.client.request_with_data(
                "done_rollout",
                robotmq.serialize(rollout_results),
            )
            logger.info("done requesting done_rollout")

            convert_data_to_video_parallel(
                task.data_storage_dir, split_videos=False, video_type="success"
            )
            convert_data_to_video_parallel(
                task.data_storage_dir, split_videos=False, video_type="failure"
            )

            new_dir_name = (
                task.data_storage_dir
                + f"_success_rate_{success_rate}".replace(".", "_")
            )
            if os.path.exists(new_dir_name):
                shutil.rmtree(new_dir_name)
            os.rename(task.data_storage_dir, new_dir_name)
            logger.info(f"done putting rollout results: {rollout_results}")
            current_dir = os.getcwd()
            logger.info(f"Results saved to {os.path.join(current_dir, new_dir_name)}")
            logger.remove(logger_handler_id)
