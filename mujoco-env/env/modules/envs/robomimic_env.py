import copy
import gc
from typing import Any, cast

from mujoco import MjData, MjModel
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from omegaconf import DictConfig, OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite

from env.modules.common import robot_data_type
from env.modules.judges.base_judge import BaseJudge
from loguru import logger


class RobomimicEnv:
    def __init__(
        self,
        judge: BaseJudge,
        robot_obs_low_dim_keys: list[str],
        env_meta: dict[str, Any] | DictConfig,
    ):

        self.episode_current_timestamp: float = 0.0
        self.rng: np.random.Generator = np.random.default_rng()
        self.judge: BaseJudge = judge
        if isinstance(env_meta, DictConfig):
            env_meta_dict = OmegaConf.to_container(env_meta)
            assert isinstance(env_meta_dict, dict)
        else:
            env_meta_dict = env_meta
        env_meta_dict["env_kwargs"]["use_object_obs"] = False

        camera_names = env_meta_dict["env_kwargs"]["camera_names"]
        robot_obs_rgb_keys = []
        env_objs_obs_rbg_keys = []
        for camera_name in camera_names:
            if camera_name.startswith("robot"):
                # robot0_eye_in_hand -> eye_in_hand_image
                robot_obs_rgb_keys.append(
                    "_".join(camera_name.split("_")[1:]) + "_image"
                )
            else:
                # agentview -> agentview_image
                # sideview -> sideview_image
                # shouldercamera0 -> shouldercamera0_image
                env_objs_obs_rbg_keys.append(camera_name + "_image")

        self.control_freq: float = env_meta["env_kwargs"]["control_freq"]

        robot_num = len(env_meta["env_kwargs"]["robots"])

        self.robot_num = robot_num

        modality_mapping = {"low_dim": [], "rgb": []}
        for robot_idx in range(robot_num):
            modality_mapping["low_dim"].extend(
                f"robot{robot_idx}_" + key for key in robot_obs_low_dim_keys
            )
            modality_mapping["rgb"].extend(
                f"robot{robot_idx}_" + key for key in robot_obs_rgb_keys
            )

        modality_mapping["rgb"].extend(f"{key}" for key in env_objs_obs_rbg_keys)

        self.robot_obs_low_dim_keys = robot_obs_low_dim_keys
        self.robot_obs_rgb_keys = robot_obs_rgb_keys
        self.env_objs_obs_rbg_keys = env_objs_obs_rbg_keys

        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
        logger.info(env_meta_dict)
        # env_meta["env_kwargs"].pop("lite_physics")
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta_dict,
            env_name=env_meta_dict["env_name"],
            render=False,
            render_offscreen=True,
            use_image_obs=True,
        )
        assert isinstance(env, EnvRobosuite)
        self.env: EnvRobosuite = env

        self.last_obs: dict[str, Any] = {}
        self.last_action: np.ndarray = np.zeros(self.env.action_dimension)
        self.last_reward: float = 0.0
        self.last_done: bool = False
        self.seed: int = 0

    def init_simulator(self):
        self.episode_current_timestamp = 0.0
        self.last_obs = copy.deepcopy(self.env.reset())
        self.last_action = np.zeros(self.env.action_dimension)
        self.last_reward = 0.0
        self.last_done = False

    def _load_mjcf_models(self):
        raise NotImplementedError("RobomimicEnv does not need to load mjcf models")

    def _solve_arm_ik(self):
        raise NotImplementedError("RobomimicEnv does not need to solve arm IK")

    def step(self, actions: list[robot_data_type] | None, render_image: bool = True):
        # Robomimic env will always render images, but will only save images if render_image is True
        if actions is not None:
            composed_action = np.zeros(self.env.action_dimension)
            for robot_idx in range(self.robot_num):
                composed_action[robot_idx * 7 : robot_idx * 7 + 3] = actions[robot_idx][
                    # "delta_pos_xyz
                    "pos_xyz"
                ]
                composed_action[robot_idx * 7 + 3 : robot_idx * 7 + 6] = actions[
                    robot_idx
                ][
                    # "delta_rot_rpy"
                    "ori_xyz"
                ]
                composed_action[robot_idx * 7 + 6 : robot_idx * 7 + 7] = actions[
                    robot_idx
                ]["delta_gripper_qpos"]

            last_obs, self.last_reward, self.last_done, self.last_info = copy.deepcopy(
                self.env.step(copy.deepcopy(composed_action))
            )
            for key in last_obs:
                self.last_obs[key] = np.array(last_obs[key]).copy()

            self.episode_current_timestamp += 1 / self.control_freq

        obs = {
            "robots_obs": self._get_robots_obs(render_image),
            "env_objs_obs": self._get_env_objs_obs(render_image),
        }

        self.judge.update(**obs)
        info = {
            "judge_states": self.judge.get_states(),
            "executed_action": self._get_executed_action(),
            "timestamp": self.episode_current_timestamp,
        }

        return (
            obs,
            self.judge.get_reward(),
            self.judge.get_done(),
            info,
        )

    def _get_robots_obs(self, render_image: bool):
        # The environment will always render images
        robots_obs: list[robot_data_type] = []
        for robot_idx in range(self.robot_num):
            robot_obs: robot_data_type = {}
            robot_obs["name"] = f"robot{robot_idx}"
            for key in self.robot_obs_low_dim_keys:
                robot_obs[key] = self.last_obs[f"robot{robot_idx}_{key}"]
            if render_image:
                for key in self.robot_obs_rgb_keys:
                    robot_obs[key] = (
                        (self.last_obs[f"robot{robot_idx}_{key}"] * 255)
                        .astype(np.uint8)
                        .transpose(1, 2, 0)
                        # self.last_obs[f"robot{robot_idx}_{key}"]
                        # .transpose(1, 2, 0)
                        # .copy()
                    )  # Convert to uint8 for compatibility
                    # img = cv2.cvtColor(robot_obs[key], cv2.COLOR_RGB2BGR)
                    # cv2.imshow("robot0_rgb", img)
                    # cv2.waitKey(1)
                    # robot_obs.pop(key)
            robots_obs.append(robot_obs)

        return robots_obs

    def _get_env_objs_obs(self, render_image: bool):
        env_objs_obs: list[robot_data_type] = []
        env_obs: robot_data_type = {}

        # To make it compatible since the judge is written inside the env for robosuite
        env_obs["done"] = self.last_done
        env_obs["reward"] = self.last_reward
        env_obs["timestamp"] = self.episode_current_timestamp
        if render_image:
            for key in self.env_objs_obs_rbg_keys:
                env_obs[key] = (
                    (self.last_obs[f"{key}"] * 255)
                    .astype(np.uint8)
                    .transpose(1, 2, 0)
                    .copy()
                    # self.last_obs[f"{key}"]
                    # .transpose(1, 2, 0)
                    # .copy()
                )  # Convert to uint8 for compatibility

        env_objs_obs.append(env_obs)
        return env_objs_obs

    def _get_executed_action(self):
        executed_actions: list[robot_data_type] = []
        for robot_idx in range(self.robot_num):
            robot_action: robot_data_type = {}
            robot_action["delta_pos_xyz"] = self.last_action[
                robot_idx * 7 : robot_idx * 7 + 3
            ]
            robot_action["delta_rot_rpy"] = self.last_action[
                robot_idx * 7 + 3 : robot_idx * 7 + 6
            ]
            robot_action["delta_gripper_qpos"] = self.last_action[
                robot_idx * 7 + 6 : robot_idx * 7 + 7
            ]
            executed_actions.append(robot_action)
        return executed_actions

    @property
    def model(self):
        return cast(MjModel, self.env.env.sim.model._model)
    @property
    def data(self):
        return cast(MjData, self.env.env.sim.data._data)

    def reset(self, episode_config: dict[str, Any] | None = None):
        if episode_config is not None:
            self.seed = episode_config["seed"]
            # Robosuite uses numpy random to set seed for initialization. This is generally a bad practice.
            # Should initialize a random number generator specifically for this class.

        np.random.seed(self.seed)
        gc.collect()  # To prevent memory leak
        self.env.reset()
        self.last_obs, self.last_reward, self.last_done, self.last_info = copy.deepcopy(
            self.env.step(np.zeros(self.env.action_dimension))
        )
        self.episode_current_timestamp = 0.0

        obs = {
            "robots_obs": self._get_robots_obs(render_image=True),
            "env_objs_obs": self._get_env_objs_obs(render_image=True),
        }
        self.judge.reset(episode_config)
        info = {
            "judge_states": self.judge.get_states(),
            "executed_action": self._get_executed_action(),
            "timestamp": self.episode_current_timestamp,
        }
        return obs, info

    def reset_robot_joints(self):
        pass

    def _wait_until_stable(self, vel_threshold: float = 1e-4, max_steps: int = 1000):
        raise NotImplementedError("RobomimicEnv does not need to wait until stable")
