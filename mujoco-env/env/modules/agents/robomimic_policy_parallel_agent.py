from typing import cast

import numpy as np
import numpy.typing as npt
from robotmq import RMQClient, deserialize, serialize
from transforms3d.quaternions import quat2axangle

from env.modules.agents.base_parallel_agent import BaseParallelAgent
from env.modules.common import data_buffer_type, robot_data_type
from robot_utils.data_utils import aggregate_dict
from env.utils.data_utils import flatten_episode_data
from robot_utils.pose_utils import quat_wxyz_to_rot_6d_batch, rot_6d_to_quat_wxyz_batch


class RobomimicPolicyParallelAgent(BaseParallelAgent):
    def __init__(
        self,
        policy_server_address: str,
        policy_obs_keys: list[str],
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.policy_server_address: str = policy_server_address
        self.policy_client: RMQClient = RMQClient(
            "robomimic_policy_client", policy_server_address
        )
        self.policy_obs_keys: list[str] = policy_obs_keys

    def get_policy_config(self):
        return deserialize(
            self.policy_client.request_with_data(
                "policy_config",
                serialize(True),
            )
        )

    def get_dataset_config(self):
        return ""

    def predict_actions_parallel(
        self,
        episode_buffers_dict: dict[int, dict[str, data_buffer_type]],
    ) -> dict[int, data_buffer_type]:

        env_num = len(episode_buffers_dict)

        flattened_buffers_dict = {
            episode_idx: flatten_episode_data(episode_buffers_dict[episode_idx])
            for episode_idx in episode_buffers_dict
        }

        aggregated_buffer = aggregate_dict(
            flattened_buffers_dict, convert_to_numpy=True, key_name="episode_idx"
        )

        batch_obs = {key: aggregated_buffer[key] for key in self.policy_obs_keys}

        for key in batch_obs.keys():
            if key.endswith("camera") or key.endswith("image"):  # rgb keys
                batch_obs[key] = np.moveaxis(
                    batch_obs[key], -1, -3
                )  # (env_num, obs_history_len, 3, 256, 256)

                if batch_obs[key].shape[1] > len(self.image_obs_frames_ids):
                    batch_obs[key] = batch_obs[key][
                        :, self.image_obs_frames_ids, ...
                    ]  # (env_num, image_frame_num, 3, 256, 256)

            elif key.endswith("xyz_wxyz") or key.endswith("gripper_width"):
                if batch_obs[key].shape[1] > len(self.proprio_obs_frames_ids):
                    batch_obs[key] = batch_obs[key][
                        :, self.proprio_obs_frames_ids
                    ]  # (env_num, proprio_frame_num, ...)

        # for key in self.policy_obs_keys:
        #     assert key in batch_obs, f"Key {key} not found in flattened input"
        #     logger.info(f"{key}: {batch_obs[key].shape}")

        for j in range(self.robot_num):
            batch_obs[f"robot{j}_10d"] = np.concatenate(
                [
                    batch_obs[f"robot{j}_eef_pos"],
                    quat_wxyz_to_rot_6d_batch(batch_obs[f"robot{j}_eef_quat"]),
                    batch_obs[f"robot{j}_gripper_qpos"][:, :, :1],
                ],
                axis=-1,
            )
        for j in range(self.robot_num):
            batch_obs.pop(f"robot{j}_eef_pos")
            batch_obs.pop(f"robot{j}_eef_quat")
            batch_obs.pop(f"robot{j}_gripper_qpos")
        batch_obs["episode_idx"] = np.array(list(episode_buffers_dict.keys()))

        raw_results = self.policy_client.request_with_data(
            "policy_inference", serialize(batch_obs), timeout_s=5
        )
        assert raw_results
        raw_actions = cast(dict[str, npt.NDArray[np.float32]], deserialize(raw_results))
        if isinstance(raw_actions, str):
            raise RuntimeError(f"Policy inference error: {raw_actions}")

        actions_dict: dict[int, data_buffer_type] = {}

        for k, episode_idx in enumerate(aggregated_buffer["episode_idx"]):
            robots_obs = episode_buffers_dict[episode_idx]["robots_obs"]
            actions: data_buffer_type = []

            for i in range(self.action_prediction_horizon):
                robots_action: list[robot_data_type] = []
                for j in range(self.robot_num):
                    action = {}
                    raw_action_pos_xyz = raw_actions[f"action{j}_10d"][k, i, :3]
                    raw_action_quat_wxyz = rot_6d_to_quat_wxyz_batch(
                        raw_actions[f"action{j}_10d"][k, i, 3:9]
                    )
                    action["pos_xyz"] = raw_action_pos_xyz
                    ax, ang = quat2axangle(raw_action_quat_wxyz)
                    rot_vec = ax * ang
                    action["ori_xyz"] = rot_vec
                    action["delta_gripper_qpos"] = raw_actions[f"action{j}_10d"][
                        k, i, 9:
                    ]
                    action["name"] = robots_obs[0][j]["name"]
                    robots_action.append(action)

                actions.append(robots_action)

            actions_dict[episode_idx] = actions

        return actions_dict

    def reset(self):
        raw_results = self.policy_client.request_with_data(
            "policy_reset", serialize(True)
        )
        assert raw_results, "Failed to reset policy agent"
