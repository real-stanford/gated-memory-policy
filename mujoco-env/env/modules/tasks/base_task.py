import time
from typing import TYPE_CHECKING, Any, Optional, cast

from env.modules.agents.spacemouse_agent import SpacemouseAgent

if TYPE_CHECKING:
    from env.modules.envs.base_env import BaseEnv
    from env.modules.agents.base_agent import BaseAgent
    from env.modules.agents.heuristic_agent import HeuristicAgent

import copy
import os

import cv2
import numpy as np
import zarr
from robot_utils.teleop_utils.keyboard import KeyboardClient

from env.modules.common import data_buffer_type, robot_data_type
from env.utils.data_utils import convert_to_list, flatten_episode_data
from loguru import logger

# Takes too long to compress/decompress; will reduce the size of the dataset to 1/3
# from imagecodecs.numcodecs import Jpegxl
# from numcodecs import register_codec
# register_codec(Jpegxl)


class BaseTask:
    def __init__(
        self,
        name: str,
        env: "BaseEnv",
        agent: "BaseAgent",
        keyboard_address: str,
        render_all_images: bool,
        use_viewer: bool,
        show_camera_images: bool,
        play_speed: float,
        data_storage_dir: str,
        successful_reward: float,
        viewer_cam_init_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.name: str = name
        self.env: "BaseEnv" = env

        self.episode_robots_obs: data_buffer_type = []
        self.episode_env_objs_obs: data_buffer_type = []
        self.episode_executed_actions: data_buffer_type = []
        self.episode_rewards: list[float] = []
        self.episode_predicted_trajs: list[tuple[int, data_buffer_type]] = []
        self.episode_config: dict[str, Any] = {}
        self.robots_obs_buffer: data_buffer_type = []
        self.executed_actions_buffer: data_buffer_type = []
        self.env_objs_obs_buffer: data_buffer_type = []

        self.env.init_simulator()
        self.use_viewer: bool = use_viewer
        self.show_camera_images: bool = show_camera_images
        self.viewer_cam_init_kwargs: Optional[dict[str, Any]] = viewer_cam_init_kwargs
        if self.use_viewer:
            from mujoco import viewer as mjviewer

            self.mj_viewer: mjviewer.Handle = mjviewer.launch_passive(
                self.env.model, self.env.data
            )
            if viewer_cam_init_kwargs is not None:
                for key, value in viewer_cam_init_kwargs.items():
                    if hasattr(self.mj_viewer.cam, key):
                        setattr(self.mj_viewer.cam, key, value)
            self.mj_viewer.sync()
        self.play_speed: float = play_speed
        self.render_all_images: bool = render_all_images

        self.agent: "BaseAgent" = agent
        self.keyboard: Optional[KeyboardClient] = None
        if keyboard_address:
            self.keyboard = KeyboardClient(keyboard_address)

        assert (
            self.env.control_freq == self.agent.agent_update_freq_hz
        ), f"The control frequency of the environment {self.env.control_freq} should be the same as the agent update frequency {self.agent.agent_update_freq_hz}"

        self.data_storage_dir: str = data_storage_dir
        if not os.path.exists(self.data_storage_dir):
            logger.info(f"Creating data storage path: {self.data_storage_dir}")
            os.makedirs(self.data_storage_dir, exist_ok=True)

        self.rng = np.random.default_rng()
        self.successful_reward: float = successful_reward

        self.time_step: int = 0

    def _process_episode_config(self, episode_config: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Subclasses should implement this method")

    @property
    def customized_obs_dict(self) -> dict[str, Any]:
        """Each task may define its own customized observation"""
        return {}

    def __del__(self):
        if self.use_viewer:
            self.mj_viewer.close()
        if self.show_camera_images:
            cv2.destroyAllWindows()

    def reset(self, episode_config: Optional[dict[str, Any]] = None):
        """
        Reset the task to its initial state and clear all the collected data.
        Each task should (randomly) initialize the missing parameters of the episode config.
        Including setting seed for random number generators.
        """
        self.time_step = 0

        if episode_config is None:
            if len(self.episode_config) == 0:
                raise ValueError(
                    "Episode config is empty. Please provide a valid episode config."
                )
        else:
            episode_config = self._process_episode_config(episode_config)
            self.episode_config = copy.deepcopy(episode_config)

        self.env.reset(self.episode_config)
        self.init_episode_buffers()
        if hasattr(self.agent, "seed"):
            cast("HeuristicAgent", self.agent).seed = self.episode_config["seed"]

        self.agent.reset(episode_config)

    def clear_episode_data(self):

        self.episode_robots_obs = []
        self.episode_env_objs_obs = []
        self.episode_executed_actions = []
        self.episode_predicted_trajs = []
        self.robots_obs_buffer = []
        self.executed_actions_buffer = []
        self.env_objs_obs_buffer = []

    def init_episode_buffers(self):
        self.clear_episode_data()
        # Fill the bufferes with initial values
        obs, reward, done, info = self.env.step(None, render_image=True)
        for _ in range(self.agent.obs_history_len):
            self.robots_obs_buffer.append(copy.deepcopy(obs["robots_obs"]))
        for _ in range(self.agent.action_history_len):
            self.executed_actions_buffer.append(copy.deepcopy(info["executed_action"]))
        for _ in range(self.agent.obs_history_len):
            self.env_objs_obs_buffer.append(copy.deepcopy(obs["env_objs_obs"]))

    def print_task_states(self):
        print_str = "\n"
        print_str += f"Time step: {self.time_step}\n"
        print_str += f"Episode config: \n"
        for key, value in self.episode_config.items():
            print_str += f"    {key}: {value}\n"

        print_str += f"Customized observations: \n"
        for key, value in self.customized_obs_dict.items():
            print_str += f"    {key}: {value}\n"

        print_str += f"Judge states: \n"
        for key, value in self.env.judge.get_states().items():
            print_str += f"    {key}: {value}\n"
        logger.info(print_str)

    def run_episode(
        self,
        episode_config: dict[str, Any],
    ):
        """Run an episode of the task with a fixed configuration."""

        self.reset(episode_config)  # Will use the new episode config to reset the environment. This will override the previous configurations defined in the yaml file.
        logger.info(f"Starting episode with config: {self.episode_config}")

        self.time_step = 0
        stop_simulation = False
        last_step_time = time.time()
        run_single_step = False
        episode_inculde_error_action = False
        # To synchronize simulation and real world time
        # only used if use_viewer is True
        obs, reward, done, info = self.env.step(None, render_image=True)

        while not done:
            # interaction with keyboard
            if self.keyboard:
                if run_single_step:
                    run_single_step = False
                    stop_simulation = True
                keys = self.keyboard.get_keys()
                end_episode = False
                for key in keys:
                    if key == "q":
                        logger.info("Episode ended by user")
                        end_episode = True
                        break
                    elif key == "R":
                        logger.info("Episode reset by user")
                        self.reset()
                        last_step_time = time.time()
                        continue
                    elif key == "r":
                        logger.info("Robots reset by user")

                        self.env.reset_robot_joints()
                        self.init_episode_buffers()
                        self.agent.reset()
                        self.time_step = 0
                        last_step_time = time.time()
                        continue
                    elif key == "d":
                        logger.info(
                            f"\n============== Display task states =============="
                        )
                        self.print_task_states()
                        self.print_lastest_data()
                    elif key == "s":
                        stop_simulation = not stop_simulation
                        if not stop_simulation:
                            last_step_time = time.time()
                    elif key == "x":  # Switch robot for spacemouse control
                        if isinstance(self.agent, SpacemouseAgent):
                            self.agent.switch_robot()
                        else:
                            logger.info(
                                "Agent is not a SpacemouseAgent. x key to switch robot is not allowed"
                            )
                    elif key == "S":
                        # Run single step
                        run_single_step = True
                        stop_simulation = False
                        last_step_time = time.time()
                    elif key == "C":
                        # Get camera attributes
                        print_str = ""
                        camera_attributes = [
                            "lookat",
                            "distance",
                            "azimuth",
                            "elevation",
                        ]
                        print_str += f"\nCamera attributes:\n"
                        for attribute in camera_attributes:
                            print_str += f"  {attribute}: {getattr(self.mj_viewer.cam, attribute)}\n"
                        print_str += f"To get camera position and orientation, run `copy camera` in the viewer console and paste it into the xml files\n"
                        logger.info(print_str)

                if end_episode:
                    break

            if stop_simulation:
                time.sleep(0.1)
                if self.use_viewer:
                    self.mj_viewer.sync()
                if self.show_camera_images:
                    if self.robots_obs_buffer and self.env_objs_obs_buffer:
                        robots_obs = self.robots_obs_buffer[-1]
                        env_objs_obs = self.env_objs_obs_buffer[-1]
                        self.plot_images(robots_obs, env_objs_obs)
                continue

            future_actions = self.agent.predict_actions(
                self.robots_obs_buffer,
                self.env_objs_obs_buffer,
                self.executed_actions_buffer,
            )
            predicted_traj = (self.time_step + 1, future_actions)
            self.episode_predicted_trajs.append(predicted_traj)
            assert len(future_actions) == self.agent.action_prediction_horizon

            if hasattr(self.env, "robots"):
                assert len(future_actions[0]) == len(self.env.robots)

            for k in range(self.agent.action_execution_horizon):
                if self.render_all_images:
                    render_image = True
                else:
                    render_image = (
                        k - self.agent.action_execution_horizon
                        in self.agent.image_obs_frames_ids
                    )

                action = future_actions[k]

                # First record the data before taking the action
                # action should be followed by the current observation
                self.episode_robots_obs.append(copy.deepcopy(obs["robots_obs"]))
                self.episode_env_objs_obs.append(copy.deepcopy(obs["env_objs_obs"]))
                self.episode_executed_actions.append(copy.deepcopy(action))
                self.episode_rewards.append(reward)

                obs, reward, done, info = self.env.step(action, render_image)

                # Update the buffers with the new observation for action prediction
                self.robots_obs_buffer.append(obs["robots_obs"])
                self.robots_obs_buffer.pop(0)
                self.env_objs_obs_buffer.append(obs["env_objs_obs"])
                self.env_objs_obs_buffer.pop(0)
                self.executed_actions_buffer.append(action)
                self.executed_actions_buffer.pop(0)

                if self.use_viewer:
                    self.mj_viewer.sync()
                if render_image and self.show_camera_images:
                    self.plot_images(obs["robots_obs"], obs["env_objs_obs"])

                if self.use_viewer or self.show_camera_images:
                    sleep_time = (
                        last_step_time
                        + (1 / self.env.control_freq) / self.play_speed
                        - time.time()
                    )
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                last_step_time = time.time()
                if "is_error" in action[0] and action[0]["is_error"]:
                    episode_inculde_error_action = True

                self.time_step += 1

        self.episode_robots_obs.append(copy.deepcopy(obs["robots_obs"]))
        self.episode_env_objs_obs.append(copy.deepcopy(obs["env_objs_obs"]))
        self.episode_executed_actions.append(copy.deepcopy(action))
        self.episode_rewards.append(reward)

        episode_data = {
            "robots_obs": copy.deepcopy(self.episode_robots_obs),
            "env_objs_obs": copy.deepcopy(self.episode_env_objs_obs),
            "executed_actions": copy.deepcopy(self.episode_executed_actions),
            "predicted_trajs": copy.deepcopy(self.episode_predicted_trajs),
            "rewards": copy.deepcopy(self.episode_rewards),
            "final_reward": reward,
            "is_successful": reward >= self.successful_reward,
            "include_error_action": episode_inculde_error_action,
            "episode_length": len(self.episode_robots_obs),
            "episode_config": convert_to_list(copy.deepcopy(self.episode_config)),
        }

        if episode_data["is_successful"]:
            logger.info(
                f"Episode ended successfully with reward: {self.episode_rewards[-1]}"
            )
        else:
            logger.info(f"Episode failed with reward: {self.episode_rewards[-1]}")

        return episode_data

    def run_episodes(self, episode_configs: list[dict[str, Any]]):

        root = zarr.open(f"{self.data_storage_dir}/episode_data.zarr", mode="a")
        assert isinstance(root, zarr.Group), "root must be a zarr.Group"
        rewards: list[float] = []
        successes: list[bool] = []
        # Create episode groups
        for i, episode_config in enumerate(episode_configs):
            # assert "seed" in episode_config
            episode_idx = episode_config["episode_idx"]
            episode_data = self.run_episode(episode_config)
            flattened_episode_data = flatten_episode_data(episode_data)
            if "final_reward" in flattened_episode_data:
                rewards.append(flattened_episode_data["final_reward"])
            if "is_successful" in flattened_episode_data:
                successes.append(flattened_episode_data["is_successful"])
            # Create a group for this episode
            ep_group = root.create_group(f"episode_{episode_idx}")

            # Store episode data
            for key, value in flattened_episode_data.items():
                if isinstance(value, np.ndarray):
                    if value.size > 1e5 and value.ndim == 4:  # Usually image data
                        chunk_size = tuple([10] + list(value.shape[-3:]))
                        # chunk_size = tuple([1] + value.shape[-3:])

                        ep_group.create_dataset(
                            key,
                            data=value,
                            chunks=chunk_size,
                            dtype=value.dtype,
                            # compressor=Jpegxl(level=99), # This takes too long to encode
                            # Will reduce the size of the dataset to 1/3
                        )
                    else:
                        ep_group.create_dataset(
                            key, data=value, compression=None, dtype=value.dtype
                        )
                else:
                    if isinstance(value, np.bool_):
                        ep_group.attrs[key] = bool(value)
                    else:
                        ep_group.attrs[key] = value

        current_dir = os.getcwd()
        print(
            # f"Episodes data saved to {os.path.join(current_dir, self.data_storage_dir)}"
            f"Episodes data saved to {self.data_storage_dir}"
        )  # Use print here to parse the output
        try:
            if len(successes) > 0:
                logger.info(f"Success rate: {np.mean(successes)}")
            if len(rewards) > 0:
                logger.info(f"Average reward: {np.mean(rewards)}")
        except ValueError as e:
            logger.info(f"Error: {e}")
            logger.info(f"Successes: {successes}")
            logger.info(f"Rewards: {rewards}")

    def plot_images(
        self,
        robots_obs: list[robot_data_type],
        env_objs_obs: list[robot_data_type],
    ):
        single_robot_split_images = []
        for single_robots_obs in robots_obs:
            for key, value in single_robots_obs.items():
                if (
                    isinstance(value, np.ndarray)
                    and value.dtype == np.uint8
                    and len(value.shape) == 3
                    and value.shape[2] == 3
                ):
                    single_robot_split_images.append(value)

        if len(single_robot_split_images) > 0:
            merged_image = np.concatenate(
                single_robot_split_images, axis=1
            )  # shape: (height * camera_num, width * robot_num, 3)
            merged_image_bgr = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)
            cv2.imshow(
                "robot_camera_viewer",
                merged_image_bgr,
            )
            cv2.waitKey(1)

        env_split_images = []
        for key, value in env_objs_obs[0].items():
            if (
                isinstance(value, np.ndarray)
                and value.dtype == np.uint8
                and len(value.shape) == 3
                and value.shape[2] == 3
            ):
                env_split_images.append(value)
        if len(env_split_images) > 0:
            max_height = max([image.shape[0] for image in env_split_images])
            for i, image in enumerate(env_split_images):
                if image.shape[0] < max_height:
                    env_split_images[i] = np.concatenate(
                        [
                            image,
                            np.zeros(
                                (max_height - image.shape[0], image.shape[1], 3),
                                dtype=image.dtype,
                            ),
                        ],
                        axis=0,
                    )

            merged_image = np.concatenate(
                env_split_images, axis=1
            )  # shape: (height * camera_num, width, 3)
            merged_image_bgr = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)
            cv2.imshow(
                "env_camera_viewer",
                merged_image_bgr,
            )
            cv2.waitKey(1)

    def print_lastest_data(self):
        print_str = ""
        print_str += f"Robot observations: \n"
        if self.episode_robots_obs:
            last_robots_obs = self.episode_robots_obs[-1]
            for obs in last_robots_obs:
                print_str += f"    Robot {obs['name']}: \n"
                for key, value in obs.items():
                    if (
                        key == "name"
                        or key.find("camera") != -1
                        or key.find("image") != -1
                    ):
                        continue
                    print_str += f"        {key}: {value}\n"
        if self.episode_env_objs_obs:
            last_env_objs_obs = self.episode_env_objs_obs[-1]
            global_obs = last_env_objs_obs[0]
            print_str += f"Environment observation: \n"
            for key, value in global_obs.items():
                if key == "name" or key.find("camera") != -1 or key.find("image") != -1:
                    continue
                if isinstance(value, np.ndarray) and value.dtype == np.float64:
                    print_str += f"    {key}: {value}\n"
            if hasattr(self.env, "objects"):
                for obs, obj in zip(last_env_objs_obs[1:], self.env.objects):
                    print_str += f"    Object {obj.name}: \n"
                    for key, value in obs.items():
                        if key == "name" or key.find("camera") != -1:
                            continue
                        print_str += f"        {key}: {value}\n"
        if self.episode_executed_actions:
            last_action = self.episode_executed_actions[-1]
            print_str += f"Last action: \n"
            for action in last_action:
                print_str += f"    Robot {action['name']}: \n"
                for key, value in action.items():
                    if key == "name" or key.find("camera") != -1:
                        continue
                    print_str += f"        {key}: {value}\n"
        logger.info(print_str)
