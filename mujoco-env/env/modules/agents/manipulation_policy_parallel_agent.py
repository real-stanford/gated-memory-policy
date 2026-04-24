import copy
from typing import cast

import numpy as np
import numpy.typing as npt
from robotmq import RMQClient, deserialize, serialize

from env.modules.agents.base_parallel_agent import BaseParallelAgent
from env.modules.common import data_buffer_type, robot_data_type
from robot_utils.data_utils import aggregate_dict
from env.utils.data_utils import flatten_episode_data
from robot_utils.pose_utils import (
    get_absolute_pose,
    get_relative_pose,
    quat_wxyz_to_rot_6d_batch,
    rot_6d_to_quat_wxyz_batch,
)
from loguru import logger


class ManipulationPolicyParallelAgent(BaseParallelAgent):
    def __init__(
        self,
        policy_server_address: str,
        policy_obs_keys: list[str],
        use_relative_pose: bool,
        smoothen_action_weight: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.policy_server_address: str = policy_server_address
        self.policy_client: RMQClient = RMQClient(
            "manipulation_policy_client", policy_server_address
        )
        self.policy_obs_keys: list[str] = policy_obs_keys
        self.use_relative_pose: bool = use_relative_pose
        self.smoothen_action_weight: float = smoothen_action_weight

    def get_policy_config(self):
        return deserialize(
            self.policy_client.request_with_data(
                "policy_config",
                serialize(True),
            )
        )

    def get_dataset_config(self):
        if self.policy_client.get_topic_status("dataset_config", 0.5) >= 0:
            return deserialize(
                self.policy_client.request_with_data(
                    "dataset_config",
                    serialize(True),
                )
            )
        else:
            return ""

    def predict_actions_parallel(
        self,
        episode_buffers_dict: dict[int, dict[str, data_buffer_type]],
    ) -> dict[int, data_buffer_type]:
        """
        episode_buffers_dict:
        {
            episode_idx(int): {
                "robots_obs": [
                    [ # timestep 0
                        { # robot 0
                            "tcp_xyz_wxyz": [...],
                            "gripper_width": [...],
                            "name": [...],
                        },
                        { # robot 1
                            "tcp_xyz_wxyz": [...],
                            "gripper_width": [...],
                            "name": [...],
                        },
                        ...
                    ],
                    [ # timestep 1
                        ...
                    ],
                    ...
                ],
                "env_objs_obs": [...],
                "executed_actions": [...],
            }
        }
        """

        env_num = len(episode_buffers_dict)
        flattened_buffers_dict = {
            episode_idx: flatten_episode_data(episode_buffers_dict[episode_idx])
            for episode_idx in episode_buffers_dict
        }
        """
        flattened_buffers_dict: 
        {
            episode_idx: {
                "robot0_tcp_xyz_wxyz": [...], # (obs_history_len, 7)
                "robot0_gripper_width": [...], # (obs_history_len, 1)
                "robot0_name": [...], # (obs_history_len, 1)
                ...
            }
        }
        """

        aggregated_buffer = aggregate_dict(
            flattened_buffers_dict, convert_to_numpy=True, key_name="episode_idx"
        )

        """
        aggregated_buffer:
        {
            "episode_idx": [...], # (env_num)
            "robot0_tcp_xyz_wxyz": [...], # (env_num, obs_history_len, 7)
            "robot0_gripper_width": [...], # (env_num, obs_history_len, 1)
            ...
        }
        """

        batch_obs = {key: aggregated_buffer[key] for key in self.policy_obs_keys}

        # HACK: only deal with the first robot for now

        for key in batch_obs.keys():
            if "camera" in key or "image" in key:
                batch_obs[key] = np.moveaxis(batch_obs[key], -1, -3)

                if batch_obs[key].shape[1] > len(self.image_obs_frames_ids):
                    batch_obs[key] = batch_obs[key][
                        :, self.image_obs_frames_ids
                    ]  # (env_num, image_frame_num, 3, 256, 256)
            elif key.endswith("xyz_wxyz") or key.endswith("gripper_width"):
                if batch_obs[key].shape[1] > len(self.proprio_obs_frames_ids):
                    batch_obs[key] = batch_obs[key][
                        :, self.proprio_obs_frames_ids
                    ]  # (env_num, proprio_frame_num, ...)

        # for key in self.policy_obs_keys:
        #     assert key in batch_obs, f"Key {key} not found in flattened input"
        #     logger.info(f"{key}: {batch_obs[key].shape}")

        current_poses = [
            batch_obs[f"robot{j}_tcp_xyz_wxyz"][:, -1, :] for j in range(self.robot_num)
        ]  # (robot_num, env_num, 7)

        if self.use_relative_pose:
            for k in range(env_num):
                for i in range(self.obs_history_len):
                    for j in range(self.robot_num):
                        current_pose = current_poses[j][k]
                        batch_obs[f"robot{j}_tcp_xyz_wxyz"][k, i, :] = (
                            get_relative_pose(
                                batch_obs[f"robot{j}_tcp_xyz_wxyz"][k, i, :],
                                current_pose,
                            )
                        )

            for j in range(self.robot_num):
                batch_obs[f"robot{j}_tcp_xyz_wxyz"] = batch_obs[
                    f"robot{j}_tcp_xyz_wxyz"
                ][
                    :, :-1
                ]  # HACK: simply remove the last frame, which is not used
                batch_obs[f"robot{j}_gripper_width"] = batch_obs[
                    f"robot{j}_gripper_width"
                ][
                    :, -1:
                ]  # HACK: only keep the last frame

        for j in range(self.robot_num):
            batch_obs[f"robot{j}_10d"] = np.concatenate(
                [
                    batch_obs[f"robot{j}_tcp_xyz_wxyz"][:, :, :3],
                    quat_wxyz_to_rot_6d_batch(
                        batch_obs[f"robot{j}_tcp_xyz_wxyz"][:, :, 3:]
                    ),
                    batch_obs[f"robot{j}_gripper_width"],
                ],
                axis=-1,
            )
            batch_obs.pop(f"robot{j}_tcp_xyz_wxyz")
            batch_obs.pop(f"robot{j}_gripper_width")

        batch_obs["episode_idx"] = list(episode_buffers_dict.keys())

        serialized_data = serialize(batch_obs)

        raw_results = self.policy_client.request_with_data(
            "policy_inference", serialized_data, timeout_s=5
        )
        assert raw_results
        raw_actions = cast(dict[str, npt.NDArray[np.float32]], deserialize(raw_results))
        if isinstance(raw_actions, str):
            raise RuntimeError(f"Policy inference error: {raw_actions}")

        for i in range(self.robot_num):
            raw_action_pos_xyz = raw_actions[f"action{i}_10d"][:, :, :3]
            raw_action_quat_wxyz = rot_6d_to_quat_wxyz_batch(
                raw_actions[f"action{i}_10d"][:, :, 3:9]
            )
            raw_actions[f"action{i}_tcp_xyz_wxyz"] = np.concatenate(
                [raw_action_pos_xyz, raw_action_quat_wxyz],
                axis=-1,
            )
            raw_actions[f"action{i}_gripper_width"] = raw_actions[f"action{i}_10d"][
                :, :, 9:
            ]
            raw_actions.pop(f"action{i}_10d")
        actions_dict: dict[int, data_buffer_type] = {}

        for k, episode_idx in enumerate(aggregated_buffer["episode_idx"]):
            actions_dict[episode_idx] = []
            robots_obs = episode_buffers_dict[episode_idx]["robots_obs"]
            actions: data_buffer_type = []
            for i in range(self.action_prediction_horizon):
                robots_action: list[robot_data_type] = []
                for j in range(self.robot_num):
                    action = {}
                    for key, val in raw_actions.items():
                        if key.startswith(f"action{j}_"):
                            action[key.replace(f"action{j}_", "")] = val[
                                k, i
                            ]  # Unbatch

                    action["tcp_xyz_wxyz"][self.disabled_movement_mask] = (
                        self.default_actions[j][self.disabled_movement_mask]
                    )

                    if not self.move_gripper:
                        action["gripper_width"] = np.array(
                            [self.default_gripper_widths[j]]
                        )

                    if self.use_relative_pose:
                        executed_actions = episode_buffers_dict[episode_idx][
                            "executed_actions"
                        ][-1][j]["tcp_xyz_wxyz"]
                        assert isinstance(executed_actions, np.ndarray)
                        action["tcp_xyz_wxyz"] = get_absolute_pose(
                            executed_actions, action["tcp_xyz_wxyz"]
                        )
                    action["name"] = robots_obs[0][0]["name"]
                    robots_action.append(action)
                actions.append(robots_action)

            if self.smoothen_action_weight > 0:
                last_action = episode_buffers_dict[episode_idx]["executed_actions"][-1][
                    0
                ]["tcp_xyz_wxyz"]
                for i in range(self.action_execution_horizon):
                    # Only smoothen the executed actions
                    if i == self.action_prediction_horizon - 1:
                        # Do not smoothen the last action in the prediction sequence
                        # Usually when action_execution_horizon < action_prediction_horizon, all the actions will be smoothened
                        continue
                    this_action = actions[i][0]["tcp_xyz_wxyz"]
                    next_action = actions[i + 1][0]["tcp_xyz_wxyz"]
                    weight = self.smoothen_action_weight
                    smoothened_action = (
                        (1 - 2 * weight) * this_action
                        + weight * last_action
                        + weight * next_action
                    )
                    last_action = copy.deepcopy(actions[i][0]["tcp_xyz_wxyz"])
                    actions[i][0]["tcp_xyz_wxyz"] = smoothened_action

            actions_dict[episode_idx] = actions

        return actions_dict

    def reset(self):
        raw_results = self.policy_client.request_with_data(
            "policy_reset", serialize(True)
        )
        assert raw_results, "Failed to reset policy agent"

    def export_recorded_data(self, file_name: str):
        """Will trigger the policy server to export the recorded data"""
        export_file_path = deserialize(
            self.policy_client.request_with_data(
                "export_recorded_data", serialize(file_name)
            )
        )
        logger.info(f"Exported recorded data to {export_file_path}")
