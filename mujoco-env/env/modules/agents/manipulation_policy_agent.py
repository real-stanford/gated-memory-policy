import copy
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from robotmq import RMQClient, deserialize, serialize

from env.modules.agents.base_agent import BaseAgent
from env.modules.common import data_buffer_type
from env.utils.data_utils import flatten_episode_data
from robot_utils.pose_utils import (
    get_absolute_pose,
    get_relative_pose,
    quat_wxyz_to_rot_6d_batch,
    rot_6d_to_quat_wxyz_batch,
)
from loguru import logger


class ManipulationPolicyAgent(BaseAgent):
    def __init__(
        self,
        policy_server_address: str,
        policy_obs_keys: list[str],
        use_relative_pose: bool,
        smoothen_action_weight: float,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.policy_client: RMQClient = RMQClient(
            "manipulation_policy_client", policy_server_address
        )
        self.policy_obs_keys: list[str] = policy_obs_keys
        self.use_relative_pose: bool = use_relative_pose
        assert (
            smoothen_action_weight >= 0 and smoothen_action_weight < 0.5
        ), f"{smoothen_action_weight=} should be in [0, 0.5)"
        self.smoothen_action_weight: float = smoothen_action_weight
        """
        smoothened_action = (1-2*weight) * this_action + weight * last_action + weight * next_action
        """
        self.current_episode_idx: int = 0 # Used to record the current episode index. Only reset when the episode index changes.
        self.skip_policy_reset: bool = False 
        # If True, will not request policy reset when self.reset() is called. This is used to ensure the data is correctly recorded.

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

    def predict_actions(
        self,
        robots_obs: data_buffer_type,
        env_objs_obs: data_buffer_type,
        history_actions: data_buffer_type,
    ) -> data_buffer_type:
        obs = {
            "robots_obs": robots_obs,
            "env_objs_obs": env_objs_obs,
        }

        flattened_obs = flatten_episode_data(obs)

        for key in flattened_obs.keys():
            if key.endswith("camera"):
                flattened_obs[key] = np.moveaxis(
                    flattened_obs[key], -1, -3
                )  # (obs_history_len, 3, 256, 256)

                if flattened_obs[key].shape[0] > len(self.image_obs_frames_ids):
                    flattened_obs[key] = flattened_obs[key][
                        self.image_obs_frames_ids
                    ]  # (image_frame_num, 3, 256, 256)

            elif key.endswith("xyz_wxyz") or key.endswith("gripper_width"):
                if flattened_obs[key].shape[0] > len(self.proprio_obs_frames_ids):
                    flattened_obs[key] = flattened_obs[key][
                        self.proprio_obs_frames_ids
                    ]  # (proprio_frame_num, ...)

        # for key in self.policy_obs_keys:
        #     assert key in flattened_obs, f"Key {key} not found in flattened input"
        #     logger.info(f"{key}: {flattened_obs[key].shape}")

        batch_obs = {
            key: flattened_obs[key][np.newaxis, ...] for key in self.policy_obs_keys
        }  # Add the batch axis
        current_poses = [
            robots_obs[-1][j]["tcp_xyz_wxyz"] for j in range(self.robot_num)
        ]

        if self.use_relative_pose:
            for i in range(self.obs_history_len):
                for j in range(self.robot_num):
                    current_pose = current_poses[j]
                    batch_obs[f"robot{j}_tcp_xyz_wxyz"][0, i, :] = get_relative_pose(
                        batch_obs[f"robot{j}_tcp_xyz_wxyz"][0, i, :], current_pose
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
                    :, 1:
                ]  # HACK: only keep the last frame. This only works for

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
        batch_obs["episode_idx"] = [self.current_episode_idx]

        print(f"Start requesting policy inference")
        raw_results = self.policy_client.request_with_data(
            "policy_inference",
            serialize(batch_obs),
            timeout_s=10,
        )
        assert raw_results
        raw_actions = cast(
            dict[str, npt.NDArray[np.float32]], deserialize(raw_results)
        )  #
        if isinstance(raw_actions, str):
            raise RuntimeError(f"Policy inference error: {raw_actions}")
        # parse raw actions

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

        actions: data_buffer_type = []

        for i in range(self.action_prediction_horizon):
            robots_action = []
            for j in range(self.robot_num):
                action = {}

                for key, val in raw_actions.items():
                    if key.startswith(f"action{j}_"):
                        action[key.replace(f"action{j}_", "")] = val[0, i]  # Unbatch

                action["tcp_xyz_wxyz"][self.disabled_movement_mask] = (
                    self.default_actions[j][self.disabled_movement_mask]
                )

                if not self.move_gripper:
                    action["gripper_width"] = np.array([self.default_gripper_widths[j]])

                last_action = history_actions[-1][j]["tcp_xyz_wxyz"]
                if self.use_relative_pose:
                    action["tcp_xyz_wxyz"] = get_absolute_pose(
                        last_action, action["tcp_xyz_wxyz"]
                    )
                action["name"] = robots_obs[0][0]["name"]
                robots_action.append(action)

            actions.append(robots_action)
        last_action = history_actions[-1][0]["tcp_xyz_wxyz"]
        if self.smoothen_action_weight > 0:  # HACK: assume there's only one robot
            for i in range(self.action_execution_horizon):
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

        return actions

    def reset(self, episode_config: dict[str, Any] | None = None):
        if episode_config is not None and "episode_idx" in episode_config:
            self.current_episode_idx = episode_config["episode_idx"]

        if not self.skip_policy_reset:
            raw_results = self.policy_client.request_with_data(
                "policy_reset", serialize(True)
            )
            
            assert raw_results, "Failed to reset policy agent"

    def export_recorded_data(self, file_name: str):
        """Will trigger the policy server to export the recorded data"""
        export_file_path = deserialize(
            self.policy_client.request_with_data(
                "export_recorded_data", serialize(file_name), 100
            )
        )
        logger.info(f"Exported recorded data to {export_file_path}")
