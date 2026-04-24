from typing import Any, cast

import numpy as np
import numpy.typing as npt
from robotmq import RMQClient, deserialize, serialize
from transforms3d.quaternions import quat2axangle

from env.modules.agents.base_agent import BaseAgent
from env.modules.common import data_buffer_type
from env.utils.data_utils import flatten_episode_data
from robot_utils.pose_utils import quat_wxyz_to_rot_6d_batch, rot_6d_to_quat_wxyz_batch
from robot_utils.logging_utils import print_once
from loguru import logger


class RobomimicPolicyAgent(BaseAgent):
    def __init__(
        self,
        policy_server_address: str,
        policy_obs_keys: list[str],
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.policy_client: RMQClient = RMQClient(
            "robomimic_policy_client", policy_server_address
        )
        self.policy_obs_keys: list[str] = policy_obs_keys
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
            if key.endswith("camera") or key.endswith("image"):  # rgb keys
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

        for i in range(self.robot_num):
            batch_obs[f"robot{i}_10d"] = np.concatenate(
                [
                    batch_obs[f"robot{i}_eef_pos"],
                    quat_wxyz_to_rot_6d_batch(batch_obs[f"robot{i}_eef_quat"]),
                    batch_obs[f"robot{i}_gripper_qpos"][:, :, :1],
                ],
                axis=-1,
            )
            # logger.info(f"{batch_obs[f'robot{i}_10d']=}")
            batch_obs.pop(f"robot{i}_eef_pos")
            batch_obs.pop(f"robot{i}_eef_quat")
            batch_obs.pop(f"robot{i}_gripper_qpos")
        batch_obs["episode_idx"] = [self.current_episode_idx]  # No need for episode indices

        print_once(f"Waiting for policy server to respond...")
        raw_results = self.policy_client.request_with_data(
            "policy_inference",
            serialize(batch_obs),
        )
        print_once(f"Policy server responded")

        assert raw_results
        raw_actions = cast(dict[str, npt.NDArray[np.float32]], deserialize(raw_results))
        if isinstance(raw_actions, str):
            raise RuntimeError(f"Policy inference error: {raw_actions}")

        actions: data_buffer_type = []

        for i in range(self.action_prediction_horizon):
            robots_action = []
            for j in range(self.robot_num):
                action = {}
                raw_action_pos_xyz = raw_actions[f"action{j}_10d"][0, i, :3]
                raw_action_quat_wxyz = rot_6d_to_quat_wxyz_batch(
                    raw_actions[f"action{j}_10d"][0, i, 3:9]
                )
                action["pos_xyz"] = raw_action_pos_xyz
                ax, ang = quat2axangle(raw_action_quat_wxyz)
                rot_vec = ax * ang
                action["ori_xyz"] = rot_vec
                action["delta_gripper_qpos"] = raw_actions[f"action{j}_10d"][0, i, 9:]

                action["name"] = robots_obs[0][j]["name"]
                robots_action.append(action)

            actions.append(robots_action)

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
        return export_file_path
