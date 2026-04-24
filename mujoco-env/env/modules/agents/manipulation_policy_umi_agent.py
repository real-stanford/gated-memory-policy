from typing import cast

import numpy as np
import numpy.typing as npt
from robotmq import RMQClient, deserialize, serialize

from env.modules.agents.base_agent import BaseAgent
from env.modules.common import data_buffer_type
from env.utils.data_utils import flatten_episode_data
from robot_utils.pose_utils import get_absolute_pose, get_relative_pose
from loguru import logger


class ManipulationPolicyUmiAgent(BaseAgent):
    def __init__(
        self,
        policy_server_address: str,
        robot_num: int,
        agent_update_freq_hz: float,
        action_prediction_horizon: int,
        action_execution_horizon: int,
        obs_history_len: int,
        action_history_len: int,
        image_obs_frames_ids: list[int],
        policy_obs_keys: list[str],
    ):
        super().__init__(
            robot_num,
            agent_update_freq_hz,
            action_prediction_horizon,
            action_execution_horizon,
            obs_history_len,
            action_history_len,
            image_obs_frames_ids,
        )

        self.policy_client = RMQClient(
            "manipulation_policy_client", policy_server_address
        )
        self.policy_obs_keys = policy_obs_keys

    def get_policy_config(self):
        # return deserialize(
        #     self.policy_client.request_with_data(
        #         "policy_config",
        #         b"",
        #     )
        # )
        return {}

    def get_dataset_config(self):
        # return deserialize(
        #     self.policy_client.request_with_data(
        #         "dataset_config",
        #         b"",
        #     )
        # )
        return {}

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
        flattened_obs["robot0_wrist_camera"] = (
            np.moveaxis(flattened_obs["robot0_wrist_camera"], -1, 1).astype(np.float32)
            / 255.0
        )
        flattened_obs["robot0_wrist_camera"] = flattened_obs["robot0_wrist_camera"][
            self.image_obs_frames_ids
        ]

        for key in self.policy_obs_keys:
            assert key in flattened_obs, f"Key {key} not found in flattened input"
            logger.info(f"{key}: {flattened_obs[key].shape}")

        batch_obs = {
            key: flattened_obs[key][np.newaxis, ...] for key in self.policy_obs_keys
        }  # Add the batch axis
        current_poses = [
            robots_obs[-1][j]["tcp_xyz_wxyz"] for j in range(self.robot_num)
        ]
        for i in range(self.obs_history_len):
            for j in range(self.robot_num):
                current_pose = current_poses[j]
                batch_obs[f"robot_{j}_tcp_xyz_wxyz"][0, i, :] = get_relative_pose(
                    batch_obs[f"robot_{j}_tcp_xyz_wxyz"][0, i, :], current_pose
                )
        # batch_obs["robot0_tcp_xyz_wxyz"] = batch_obs["robot0_tcp_xyz_wxyz"][
        #     :, :-1
        # ]  # HACK: simply remove the last frame, which is not used

        raw_results = self.policy_client.request_with_data(
            "policy_inference",
            serialize(batch_obs),
        )
        assert raw_results

        # raw_actions = cast(
        #     dict[str, npt.NDArray[np.float32]], deserialize(raw_results)
        # )
        raw_actions = cast(npt.NDArray[np.float32], deserialize(raw_results))

        if isinstance(raw_actions, str):
            raise RuntimeError(f"Policy inference error: {raw_actions}")
        logger.info(type(raw_actions))
        # logger.info([(k, v.shape) for k, v in raw_actions.items()])
        raw_actions = {
            "action_tcp_xyz_wxyz": raw_actions[:, :7],
            "action_gripper_width": raw_actions[:, 7:],
        }

        # logger.info(f"============== tcp_xyz_wxyz ==============")
        # logger.info(raw_actions["action_tcp_xyz_wxyz"])
        # logger.info(f"============== gripper_width ==============")
        # logger.info(raw_actions["action_gripper_width"])
        actions = []

        for i in range(self.action_prediction_horizon):
            robots_action = []
            for j in range(self.robot_num):
                action = {}
                current_pose = current_poses[j]

                for key, val in raw_actions.items():
                    if key.startswith(f"action{j}_"):
                        # action[key.replace(f"action{j}_", "")] = val[0, i]  # Unbatch
                        action[key.replace(f"action{j}_", "")] = val[i]
                action["tcp_xyz_wxyz"] = get_absolute_pose(
                    current_pose, action["tcp_xyz_wxyz"]
                )
                action["name"] = robots_obs[0][0]["name"]
                robots_action.append(action)

            actions.append(robots_action)

        logger.info(f"============== tcp_xyz_wxyz ==============")
        for action in actions:
            logger.info(action[0]["tcp_xyz_wxyz"])

        logger.info(f"============== gripper_width ==============")
        for action in actions:
            logger.info(action[0]["gripper_width"])

        return actions

    def reset(self):
        pass
