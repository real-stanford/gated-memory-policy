import copy
import time
from typing import TYPE_CHECKING, Any, Union, cast

import zarr
from omegaconf import DictConfig, OmegaConf

from env.modules.common import robot_data_type

if TYPE_CHECKING:
    from env.modules.agents.base_parallel_agent import BaseParallelAgent
    from env.modules.envs.base_env import BaseEnv
    from env.modules.common import data_buffer_type

import os
from copy import deepcopy

import hydra
import numpy as np
import ray

from robot_utils.config_utils import enable_hydra_target
from env.utils.data_utils import convert_to_list, flatten_episode_data
from loguru import logger


@ray.remote
class Worker:
    def __init__(
        self,
        worker_id: int,
        env_cfg: Union[dict[str, Any], DictConfig],
        render_image_indices: list[int],
        obs_history_len: int,
        action_history_len: int,
        action_execution_horizon: int,
    ):
        self.worker_id: int = worker_id
        env_cfg = enable_hydra_target(env_cfg)
        self.env: "BaseEnv" = hydra.utils.instantiate(env_cfg)
        self.env.init_simulator()
        self.render_image_indices: list[int] = render_image_indices
        self.obs_history_len: int = obs_history_len
        self.action_history_len: int = action_history_len
        self.action_execution_horizon: int = action_execution_horizon
        self.episode_config: dict[str, Any] = {}

    def init_episode(self, episode_config: dict[str, Any]):
        """
        episode_config:
            "episode_idx": int
            "seed": int
            "object_init_bin_ids": list[int] (optional, length should be the same as the number of objects)
            "object_poses_xyz_wxyz": list[numpy.ndarray] (optional, length should be the same as the number of objects)
            "robot_init_tcp_poses_xyz_wxyz": list[numpy.ndarray] (optional, length should be the same as the number of robots)
            "robot_init_gripper_width": list[float] (optional, length should be the same as the number of robots)
        Return: dict with keys
            worker_id, episode_idx, robots_obs, env_objs_obs, executed_actions, done, reward, episode_config
            Same as `step()` method.
        """
        self.episode_config = copy.deepcopy(episode_config)
        obs, info = self.env.reset(self.episode_config)
        obs_without_image, _, _, _ = self.env.step(None, render_image=False)
        robots_obs: "data_buffer_type" = []
        env_objs_obs: "data_buffer_type" = []
        executed_actions: "data_buffer_type" = []

        for i in range(self.obs_history_len):
            if i - self.obs_history_len in self.render_image_indices:
                robots_obs.append(obs["robots_obs"])
                env_objs_obs.append(obs["env_objs_obs"])
            else:
                # To make sure the all data has the same shape
                robots_obs.append(obs_without_image["robots_obs"])
                env_objs_obs.append(obs_without_image["env_objs_obs"])

        for i in range(self.action_history_len):
            executed_actions.append(info["executed_action"])

        data_dict = {
            "worker_id": self.worker_id,
            "episode_idx": self.episode_config["episode_idx"],
            "robots_obs": robots_obs,
            "env_objs_obs": env_objs_obs,
            "executed_actions": executed_actions,
            "done": False,
            "reward": False,
            "episode_config": convert_to_list(copy.deepcopy(self.episode_config)),
        }
        return data_dict

    def step(self, actions: "data_buffer_type"):
        """
        action: nested list of actions for all robots. Example:
            [
                [ # timestep 0
                    { # robot0
                        "name": str,
                        "tcp_xyz_wxyz": numpy.ndarray,
                        "gripper_width": numpy.ndarray,
                    },
                    { # robot_1
                        ...
                    },
                ],
                ... # More timesteps
            ]

        Return: dict with keys (worker_id, episode_idx, robots_obs, env_objs_obs, executed_actions, done, reward).
            All the data items are nested numpy arrays except for episode_idx (int) worker_id (int), done (bool), reward (bool).
            "robots_obs": [
                [ # timestep 0
                    { # robot0
                        "name": str,
                        "arm_qpos": numpy.ndarray,
                        "arm_qvel": numpy.ndarray,
                        "arm_qacc": numpy.ndarray,
                        "tcp_xyz_wxyz": numpy.ndarray,
                        "gripper_width": numpy.ndarray,
                    },
                    { # robot_1
                        ...
                    },
                ],
                [ # timestep 1
                    { # robot0
                        ...
                    },
                    { # robot_1
                        ...
                    },
                ],
                ... # More timesteps
            ]
            "env_objs_obs": [
                [ # timestep 0
                    { # env
                        "name": str,
                        "bin_center_xyz": numpy.ndarray,
                        "timestamp": float,
                    }
                    { # obj0
                        "name": str,
                        "object_pose_xyz_wxyz": numpy.ndarray,
                        "bin_id": int,
                        "tcp_relative_poses_xyz_wxyz": numpy.ndarray,
                    }
                    { # obj_1
                        ...
                    }
                ]
                ... # More timesteps
            ]
            "done": bool, # If the episode ends, the last element will be True
            "reward": bool, # If the episode is successful, the last element will be True
            "executed_actions": [
                [ # timestep 0
                    { # robot0
                        ...
                    },
                ],
            ]
            The length is the same as the action length.
        """

        robots_obs_list: list[list[robot_data_type]] = []
        env_objs_obs_list: list[list[robot_data_type]] = []
        executed_actions_list: list[list[robot_data_type]] = []
        done = False
        reward = False

        for idx, robots_action in enumerate(actions):
            if idx >= self.action_execution_horizon:
                break
            render_image = (
                idx - self.action_execution_horizon in self.render_image_indices
            )
            obs, reward, done, info = self.env.step(robots_action, render_image)
            robots_obs_list.append(obs["robots_obs"])
            env_objs_obs_list.append(obs["env_objs_obs"])
            executed_actions_list.append(info["executed_action"])
            if done:
                break

        data_dict = {
            "worker_id": self.worker_id,
            "episode_idx": self.episode_config["episode_idx"],
            "robots_obs": robots_obs_list,
            "env_objs_obs": env_objs_obs_list,
            "executed_actions": executed_actions_list,
            "done": done,
            "reward": reward,
        }
        return data_dict


class ParallelTask:
    def __init__(
        self,
        name: str,
        env: Union[dict[str, Any], DictConfig],
        env_num: int,
        agent: "BaseParallelAgent",
        render_image_indices: list[int],
        data_storage_dir: str,
        successful_reward: float,
        **kwargs,
    ):
        logger.info(f"ParallelTask redundant kwargs: {kwargs}")
        self.name: str = name
        self.env_num: int = env_num

        self.agent: "BaseParallelAgent" = agent
        self.render_image_indices: list[int] = render_image_indices
        self.data_storage_dir: str = data_storage_dir

        if len(self.data_storage_dir) > 0 and not os.path.exists(self.data_storage_dir):
            logger.info(f"Creating data storage path: {self.data_storage_dir}")
            os.makedirs(self.data_storage_dir, exist_ok=True)

        self.episode_configs: dict[int, dict[str, Any]] = {}
        self.env_objs_obs_buffers: dict[int, "data_buffer_type"] = {}
        self.executed_actions_buffers: dict[int, "data_buffer_type"] = {}
        self.robots_obs_buffers: dict[int, "data_buffer_type"] = {}

        self.episode_data: dict[int, Any] = {}
        """
        episode_idx: 
            "robots_obs": "data_buffer_type"
            "env_objs_obs": "data_buffer_type"
            "executed_actions": "data_buffer_type"
            "predicted_trajs": list["data_buffer_type"]
            "reward": bool
            "episode_length": int
            "episode_config": dict[str, Any]
        """

        self.worker2episode: dict[int, int] = (
            {}
        )  # Maps active workers to episodes. Inactive workers will be removed.
        # This dictionary should be no more than self.env_num elements.
        self.episode2worker: dict[int, int] = (
            {}
        )  # Maps episodes to the workers it is running on.
        # This dictionary may include inactive workers (multiple episodes running on the same worker)

        if isinstance(env, DictConfig):
            env_dict = OmegaConf.to_container(env)
        else:
            env_dict = env
        self.env_cfg: dict[str, Any] = cast(dict[str, Any], env_dict)

        if "control_freq" in self.env_cfg.keys():
            assert (
                self.env_cfg["control_freq"] == agent.agent_update_freq_hz
            ), f"The control frequency of the environment {self.env_cfg['control_freq']} should be the same as the agent update frequency {agent.agent_update_freq_hz}"

        if not ray.is_initialized():
            ray.init(local_mode=False, num_cpus=env_num)
            print("Ray initialized")

        self.workers: list[Any] = [
            Worker.remote(
                i,
                deepcopy(env),
                self.render_image_indices,
                agent.obs_history_len,
                agent.action_history_len,
                agent.action_execution_horizon,
            )
            for i in range(env_num)
        ]

        self.rng = np.random.default_rng()
        self.successful_reward: float = successful_reward

    def update_workers(
        self,
        batch_actions: dict[int, "data_buffer_type"],
        init_episode_indices: list[int],
    ):
        if self.workers is None:
            raise ValueError(
                "Workers are not initialized. Please initialize ParallelTask with env or call recreate_workers()."
            )
        running_worker_ids: list[int] = []
        init_worker_ids: list[int] = []
        for episode_idx in batch_actions.keys():
            assert (
                episode_idx in self.worker2episode.values()
            ), f"Episode id {episode_idx} is not in the worker2episode mapping. Please check the episode initialization."
            assert (
                episode_idx not in init_episode_indices
            ), f"Episode id {episode_idx} should not bein both batch_actions and init_episode_indices."
            running_worker_ids.append(self.episode2worker[episode_idx])
        assert (
            len(batch_actions.keys()) + len(init_episode_indices) <= self.env_num
        ), f"The number of episode ids {len(batch_actions.keys()) + len(init_episode_indices)} should be less than or equal to the number of environments {self.env_num}."

        # Assign new workers to init_episode_indices
        unused_worker_ids = [
            i for i in range(self.env_num) if i not in running_worker_ids
        ]
        init_worker_ids = unused_worker_ids[: len(init_episode_indices)]
        for i in range(len(init_episode_indices)):
            self.worker2episode[init_worker_ids[i]] = init_episode_indices[i]
            self.episode2worker[init_episode_indices[i]] = init_worker_ids[i]

        for i in range(self.env_num):
            if i not in init_worker_ids and i not in running_worker_ids:
                if i in self.worker2episode.keys():
                    self.worker2episode.pop(i)  # Remove inactive workers

        results = ray.get(
            [
                self.workers[i].step.remote(batch_actions[self.worker2episode[i]])
                for i in running_worker_ids
            ]
            + [
                self.workers[i].init_episode.remote(
                    self.episode_configs[self.worker2episode[i]]
                )
                for i in init_worker_ids
            ]
        )

        result_dict: dict[int, dict[str, Any]] = {
            result["episode_idx"]: result for result in results
        }
        return result_dict

    def reset(self):
        self.clear_episode_buffers()
        self.episode_data = {}
        self.agent.reset()

    def clear_episode_buffers(self):
        self.env_objs_obs_buffers = {}
        self.executed_actions_buffers = {}
        self.robots_obs_buffers = {}

    def init_episode_buffers(self, data_dict: dict[int, dict[str, "data_buffer_type"]]):
        for episode_idx, obs in data_dict.items():
            self.robots_obs_buffers[episode_idx] = obs["robots_obs"]
            self.env_objs_obs_buffers[episode_idx] = obs["env_objs_obs"]
            self.executed_actions_buffers[episode_idx] = obs["executed_actions"]
            new_episode_config = obs["episode_config"]
            self.episode_configs[episode_idx] = cast(dict[str, Any], new_episode_config)

    def update_episode_buffers(
        self, data_dict: dict[int, dict[str, "data_buffer_type"]]
    ):
        for episode_idx, data in data_dict.items():
            assert (
                len(data["robots_obs"])
                == len(data["env_objs_obs"])
                == len(data["executed_actions"])
            )
            data_len = len(data["robots_obs"])

            if data_len >= self.agent.obs_history_len:
                self.robots_obs_buffers[episode_idx] = copy.deepcopy(
                    data["robots_obs"][-self.agent.obs_history_len :]
                )
                self.env_objs_obs_buffers[episode_idx] = copy.deepcopy(
                    data["env_objs_obs"][-self.agent.obs_history_len :]
                )
                self.executed_actions_buffers[episode_idx] = copy.deepcopy(
                    data["executed_actions"][-self.agent.action_history_len :]
                )
            else:
                self.robots_obs_buffers[episode_idx].extend(
                    copy.deepcopy(data["robots_obs"])
                )
                self.env_objs_obs_buffers[episode_idx].extend(
                    copy.deepcopy(data["env_objs_obs"])
                )
                self.executed_actions_buffers[episode_idx].extend(
                    copy.deepcopy(data["executed_actions"])
                )
                while (
                    len(self.robots_obs_buffers[episode_idx])
                    > self.agent.obs_history_len
                ):
                    self.robots_obs_buffers[episode_idx].pop(0)
                while (
                    len(self.executed_actions_buffers[episode_idx])
                    > self.agent.action_history_len
                ):
                    self.executed_actions_buffers[episode_idx].pop(0)
                while (
                    len(self.env_objs_obs_buffers[episode_idx])
                    > self.agent.obs_history_len
                ):
                    self.env_objs_obs_buffers[episode_idx].pop(0)

    def get_episode_buffers(
        self, episode_indices: list[int]
    ) -> dict[int, dict[str, "data_buffer_type"]]:
        raw_episode_buffers = {
            episode_idx: {
                "robots_obs": self.robots_obs_buffers[episode_idx],
                "env_objs_obs": self.env_objs_obs_buffers[episode_idx],
                "executed_actions": self.executed_actions_buffers[episode_idx],
            }
            for episode_idx in episode_indices
        }

        return raw_episode_buffers

    def update_episode_data(self, data_dict: dict[int, dict[str, Any]]):
        ended_episode_indices = []
        for episode_idx, data in data_dict.items():
            if episode_idx not in self.episode_data.keys():
                self.episode_data[episode_idx] = {}
            for key, value in data.items():
                if key in [
                    "done",
                    "reward",
                    "episode_idx",
                    "episode_length",
                    "worker_id",
                ]:
                    continue
                if key not in self.episode_data[episode_idx].keys():
                    self.episode_data[episode_idx][key] = []
                self.episode_data[episode_idx][key].extend(value)

            if data["done"]:
                self.episode_data[episode_idx]["final_reward"] = data["reward"]
                self.episode_data[episode_idx]["is_successful"] = (
                    data["reward"] >= self.successful_reward
                )
                if self.episode_data[episode_idx]["is_successful"]:
                    logger.info(
                        f"Episode {episode_idx} ended successfully with reward: {data['reward']}"
                    )
                else:
                    logger.info(
                        f"Episode {episode_idx} failed with reward: {data['reward']}"
                    )

                self.episode_data[episode_idx]["episode_length"] = len(
                    self.episode_data[episode_idx]["executed_actions"]
                )
                self.episode_data[episode_idx]["episode_config"] = self.episode_configs[
                    episode_idx
                ]
                self.worker2episode.pop(self.episode2worker[episode_idx])
                ended_episode_indices.append(episode_idx)
        return ended_episode_indices

    def _process_episode_config(self, episode_config: dict[str, Any]) -> dict[str, Any]:
        """
        Process the episode config based on different seeds. Should be implemented by each task.
        """
        raise NotImplementedError("Should be implemented by each task")

    def _process_episode_configs(self, episode_configs: dict[int, dict[str, Any]]):
        for episode_idx, cfg in episode_configs.items():
            episode_configs[episode_idx] = self._process_episode_config(cfg)
        return episode_configs

    def run_episodes(self, episode_configs: dict[int, dict[str, Any]]):

        root = zarr.open(f"{self.data_storage_dir}/episode_data.zarr", mode="a")
        assert isinstance(root, zarr.Group)

        assert (
            self.workers is not None
        ), "Workers are not initialized. Please initialize ParallelTask with env or call recreate_workers()."

        self.episode_configs = self._process_episode_configs(episode_configs)
        self.episode2worker = {}
        self.worker2episode = {}

        episode_indices = [i for i, cfg in episode_configs.items()]
        successful_indices = []
        rewards = []

        started_episode_idx = 0
        while True:
            running_num = len(self.worker2episode.keys())
            init_num = min(
                self.env_num - running_num, len(episode_indices) - started_episode_idx
            )
            if init_num == 0 and running_num == 0:
                break
            init_episode_indices = episode_indices[
                started_episode_idx : started_episode_idx + init_num
            ]
            started_episode_idx += init_num
            update_episode_indices = sorted(list(self.worker2episode.values()))

            start_time = time.time()
            policy_inference_time = 0.0

            if running_num == 0:
                new_data_dict = self.update_workers({}, init_episode_indices)
                self.init_episode_buffers(new_data_dict)
            else:
                episode_buffers = self.get_episode_buffers(update_episode_indices)

                policy_inference_start_time = time.time()

                action_dict = self.agent.predict_actions_parallel(episode_buffers)
                policy_inference_time = time.time() - policy_inference_start_time

                new_data_dict = self.update_workers(action_dict, init_episode_indices)
                init_data_dict = {
                    episode_idx: new_data_dict[episode_idx]
                    for episode_idx in init_episode_indices
                }
                update_data_dict = {
                    episode_idx: new_data_dict[episode_idx]
                    for episode_idx in update_episode_indices
                }
                self.init_episode_buffers(init_data_dict)
                self.update_episode_buffers(update_data_dict)

            logger.info(
                f"{running_num=}, {init_num=}, {update_episode_indices=}, {init_episode_indices=}, \
policy_time={policy_inference_time: .3f}, env_step_time={time.time() - start_time - policy_inference_time: .3f}"
            )
            ended_episode_indices = self.update_episode_data(
                new_data_dict
            )  # Will pop the ended episodes from self.worker2episode

            # Save the ended episode data to zarr store to reduce memory usage

            for episode_idx in ended_episode_indices:
                data = self.episode_data.pop(episode_idx)
                rewards.append(data["final_reward"])
                if data["final_reward"] >= self.successful_reward:
                    successful_indices.append(episode_idx)
                ep_group = root.create_group(f"episode_{episode_idx}")
                flattened_data = flatten_episode_data(data)
                for key, value in flattened_data.items():
                    if isinstance(
                        value, np.ndarray
                    ):  # No need to disable compression for faster readout
                        ep_group.create_dataset(
                            key, data=value, compression=None, dtype=value.dtype
                        )
                    else:
                        if isinstance(value, np.bool_):
                            ep_group.attrs[key] = bool(value)
                        else:
                            ep_group.attrs[key] = value

        success_rate = len(successful_indices) / len(episode_indices)
        logger.info(f"Success rate: {success_rate}")
        mean_reward = np.mean(rewards)
        logger.info(f"Mean reward: {mean_reward}")

        return success_rate, mean_reward
