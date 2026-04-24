import copy
import datetime
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from robotmq import RMQClient, deserialize, serialize
from loguru import logger

from real_env.agents.base_agent import BaseAgent
from robot_utils.pose_utils import get_absolute_pose, get_relative_pose


class PolicyAgent(BaseAgent):
    def __init__(
        self,
        policy_server_endpoint: str,
        action_as_proprio: bool,
        keep_one_step_dummy_proprio: bool,
        use_relative_action: bool,
        smooth_action_window_size: int,
        **kwargs,
    ):
        """
        keep_one_step_dummy_proprio: For original UMI checkpoints, the last step of proprio pose is actually identity
        smooth_action_window_size: The window size for smoothing the actions. 1 means no smoothing.
        """

        self.policy_rmq_client = RMQClient(
            client_name=f"{kwargs['name']}_policy",
            server_endpoint=policy_server_endpoint,
        )

        logger.info(f"Requesting policy config from {policy_server_endpoint}")

        reply = self.policy_rmq_client.request_with_data(
            topic="policy_config", data=serialize(True), timeout_s=5.0
        )
        self.config: dict[str, Any] = deserialize(reply)
        if "proprio_history_len" not in kwargs:
            kwargs["proprio_history_len"] = (
                self.config["workspace"]["model"]["proprio_length"] + 1
            )
            print(f"Using proprio_history_len: {kwargs['proprio_history_len']}")
        if "action_history_len" not in kwargs:
            kwargs["action_history_len"] = (
                self.config["workspace"]["model"]["proprio_length"] + 1
            )
            print(f"Using action_history_len: {kwargs['action_history_len']}")
        super().__init__(**kwargs)
        self.action_as_proprio: bool = action_as_proprio
        self.keep_one_step_dummy_proprio: bool = keep_one_step_dummy_proprio
        self.use_relative_action: bool = use_relative_action
        if self.keep_one_step_dummy_proprio:
            assert (
                self.use_relative_action
            ), "PolicyAgent with keep_one_step_dummy_proprio must use relative action"
        logger.info(
            f"time: {self.config['date_str']}-{self.config['time_str']}, policy_name: {self.config['policy_name']}, run_name: {self.config['run_name']}"
        )

        assert (
            smooth_action_window_size >= 1
        ), "Smooth action window size must be at least 1"
        assert (
            smooth_action_window_size % 2 == 1
        ), "Smooth action window size must be odd"
        self.smooth_action_window_size: int = smooth_action_window_size

    def get_config(self) -> dict[str, Any]:
        return copy.deepcopy(self.config)

    def predict_actions(
        self,
        observations: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.uint8]],
        history_actions: dict[str, npt.NDArray[np.float64]],
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        N: proprio_history_len
        M: action_prediction_horizon
        For the input, the observations and actions are in global robot frame.
        They will be converted to relative poses before sending to the policy server.
        observations: {
            "robot0_main_camera": (N, H, W, 3),
            "robot0_ultrawide_camera": (N, H, W, 3),
            "robot0_depth_camera": (N, H, W, 1),
            "robot0_eef_xyz_wxyz": (N, 7),
            "robot0_eef_xyz_wxyz_wrt_start": (N, 7), # TODO: For the UMI cup checkpoint
            "robot0_gripper_width": (N, 1),
            "timestamps": (N, ),
        }
        history_actions: {
            "action0_gripper_width": (M, 1),
            "action0_eef_xyz_wxyz": (M, 7),
            "timestamps": (M, ),
        }

        K: action_prediction_horizon
        Returned actions are already converted to absolute actions in global robot frame.
        return: {
            "action0_gripper_width": (K, 1),
            "action0_eef_xyz_wxyz": (K, 7),
            "timestamps": (K, ),
        }
        """

        # print(f"Input observations:")
        # for k, v in observations.items():
        #     print(f"{k}: {v.shape}")

        observations = copy.deepcopy(observations)
        history_actions = copy.deepcopy(history_actions)
        obs_last_timestamp: npt.NDArray[np.float64] = observations["timestamps"][-1]
        obs_last_eef_xyz_wxyz: npt.NDArray[np.float64] = observations[
            "robot0_eef_xyz_wxyz"
        ][-1]
        # obs_last_gripper_width: npt.NDArray[np.float64] = observations[
        #     "robot0_gripper_width"
        # ][-1]

        # print(f"{observations['robot0_eef_xyz_wxyz'].shape=}")
        # print(f"{history_actions['action0_eef_xyz_wxyz'].shape=}")

        if self.action_as_proprio:
            observations["robot0_eef_xyz_wxyz"] = history_actions[
                "action0_eef_xyz_wxyz"
            ]
            observations["robot0_gripper_width"] = history_actions[
                "action0_gripper_width"
            ]

        actual_obs_length = len(observations["robot0_eef_xyz_wxyz"])

        if self.use_relative_action:
            if not self.keep_one_step_dummy_proprio:
                actual_obs_length -= 1
                observations["robot0_gripper_width"] = observations[
                    "robot0_gripper_width"
                ][
                    1:
                ]  # Remove the earliest gripper width to align with the observations

            observations["robot0_eef_xyz_wxyz"] = np.array(
                [
                    get_relative_pose(
                        new_pose_xyz_wxyz=observations["robot0_eef_xyz_wxyz"][i],
                        init_pose_xyz_wxyz=observations["robot0_eef_xyz_wxyz"][-1],
                    )
                    for i in range(actual_obs_length)
                ]
            )

        # Use gripper relative width as actions
        # observations["robot0_gripper_width"] = np.array(
        #     [
        #         observations["robot0_gripper_width"][i] - observations["robot0_gripper_width"][-1]
        #         for i in range(len(observations["robot0_gripper_width"]) - 1)
        #     ]
        # )

        observations["episode_idx"] = np.array(
            0
        )  # Don't need to run multiple episodes at the same time

        # for k, v in observations.items():
        #     print(f"observations: {k}, {v.shape}, {v.dtype}")
        # print(
        #     f"observations['robot0_gripper_width']: {observations['robot0_gripper_width']}, observations['robot0_eef_xyz_wxyz']: {observations['robot0_eef_xyz_wxyz']}"
        # )

        reply = self.policy_rmq_client.request_with_data(
            topic="policy_inference", data=serialize(observations), timeout_s=5.0
        )

        raw_actions = deserialize(reply)
        if isinstance(raw_actions, str):
            raise ValueError(f"Policy agent return an error message: {raw_actions}")
        assert isinstance(raw_actions, dict)
        raw_actions = cast(dict[str, npt.NDArray[np.float64]], raw_actions)
        print(f"raw_actions: {raw_actions}")

        # print(
        #     f"action0_eef_xyz_wxyz: {raw_actions['action0_eef_xyz_wxyz']}, action0_gripper_width: {raw_actions['action0_gripper_width']}"
        # )
        raw_eef_actions = raw_actions["action0_eef_xyz_wxyz"]

        if self.use_relative_action:
            absolute_eef_poses = np.array(
                [
                    get_absolute_pose(
                        init_pose_xyz_wxyz=obs_last_eef_xyz_wxyz,  # Still need to use the actual eef pose to calculate the absolute actions
                        relative_pose_xyz_wxyz=raw_eef_actions[i, :],
                    )
                    for i in range(len(raw_eef_actions))
                ]
            )
        else:
            absolute_eef_poses = raw_eef_actions

        # absolute_gripper_widths = np.array(
        #     [
        #         obs_last_gripper_width + rel_actions["action0_gripper_width"][i]
        #         for i in range(len(rel_actions["action0_gripper_width"]))
        #     ]
        # )

        if self.smooth_action_window_size > 1:
            for axis in range(3):  # Only smooth the x, y, z axes
                side_length = (self.smooth_action_window_size - 1) // 2
                absolute_eef_poses[side_length:-side_length, axis] = np.convolve(
                    absolute_eef_poses[:, axis],
                    np.ones(self.smooth_action_window_size)
                    / self.smooth_action_window_size,
                    mode="valid",
                )
                # Smooth the edges
                for timestep in range(1, side_length):
                    absolute_eef_poses[timestep, axis] = np.mean(
                        absolute_eef_poses[: timestep * 2 + 1, axis]
                    )
                    absolute_eef_poses[-timestep - 1, axis] = np.mean(
                        absolute_eef_poses[-timestep * 2 - 1 :, axis]
                    )

        absolute_gripper_widths = raw_actions["action0_gripper_width"]

        new_timestamps = (
            obs_last_timestamp
            + np.arange(
                # -1, self.policy_agent.action_prediction_horizon - 1
                # HACK: The first action from UMI policy is actually zero
                0,
                self.action_prediction_horizon,
            )  # The first action of iPhUMI policy is non-zero
            / self.agent_update_freq_hz
        )  # The first action is the same as the last image timestamp (already in the past)

        actions = {
            "action0_eef_xyz_wxyz": absolute_eef_poses,
            "action0_gripper_width": absolute_gripper_widths,
            "timestamps": new_timestamps,
        }

        return actions

    def reset(self):
        reply = self.policy_rmq_client.request_with_data(
            topic="policy_reset", data=serialize("RESET"), timeout_s=5.0
        )
        reply_str = deserialize(reply)

        if isinstance(reply_str, str):
            assert reply_str == "OK", f"Policy agent returned an error: {reply_str}"
        elif isinstance(reply_str, bool):
            assert reply_str, f"Policy agent returned an error: {reply_str}"
        else:
            raise ValueError(
                f"Policy agent returned an unexpected type: {type(reply_str)}"
            )

    def export_data(self):
        time_str = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )  # YYYY-MM-DD_HH-MM-SS
        reply = deserialize(
            self.policy_rmq_client.request_with_data(
                topic="export_recorded_data", data=serialize(time_str), timeout_s=5.0
            )
        )

        print(f"Policy agent exported data to: {reply}")
