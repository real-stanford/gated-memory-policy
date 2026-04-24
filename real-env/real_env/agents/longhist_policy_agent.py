import copy
from typing import Any, cast

from robotmq import deserialize, serialize
from real_env.agents.policy_agent import PolicyAgent
import numpy.typing as npt
import numpy as np

from robot_utils.pose_utils import get_absolute_pose, get_relative_pose


class LongHistPolicyAgent(PolicyAgent):
    def __init__(
        self,
        record_action_len_per_traj: int,
        record_image_len_per_traj: int,
        aggregated_traj_num: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert (
            not self.keep_one_step_dummy_proprio
        ), "LongHistPolicyAgent does not support keep_one_step_dummy_proprio"
        if not self.action_as_proprio:
            assert (
                record_action_len_per_traj == 0
            ), "LongHistPolicyAgent only supports record_action_len_per_traj == 0 when action_as_proprio is false"

        assert (
            record_image_len_per_traj == self.image_history_len
        ), "LongHistPolicyAgent only supports record_image_len_per_traj == image_history_len"

        self.record_action_len_per_traj: int = record_action_len_per_traj
        self.record_image_len_per_traj: int = record_image_len_per_traj
        self.aggregated_traj_num: int = aggregated_traj_num

        self.recorded_actions: dict[str, list[npt.NDArray[np.float64]]] = {}
        """
        {
            "action0_gripper_width": [
                (record_action_len_per_traj, 1),
                (record_action_len_per_traj, 1),
                ...
            ],
            "action0_eef_xyz_wxyz": [
                (record_action_len_per_traj, 7),
                (record_action_len_per_traj, 7),
                ...
            ],
        }
        """

        self.recorded_images: dict[str, list[npt.NDArray[np.uint8]]] = {}
        """
        {
            "robot0_main_camera": [
                (record_image_len_per_traj, H, W, 3),
                (record_image_len_per_traj, H, W, 3),
                ...
            ],
        }
        """

    def get_padded_history_absolute_actions(
        self, latest_history_actions: dict[str, npt.NDArray[np.float64]]
    ):
        """
        latest_history_action: {
            "action0_gripper_width": (M, 1),
            "action0_eef_xyz_wxyz": (M, 7),
        }
        return: {
            "action0_gripper_width": (aggregated_traj_num * record_action_len_per_traj + M, 1),
            "action0_eef_xyz_wxyz": (aggregated_traj_num * record_action_len_per_traj + M, 7),
        }
        """
        history_actions: dict[str, npt.NDArray[np.float64]] = {}

        for key, val in latest_history_actions.items():
            if not key.startswith("action"):
                continue

            assert (
                val.ndim == 2
            ), f"Latest history action {key} should be a 2D array, but got {val.shape}"

            dof = val.shape[1]

            if len(self.recorded_actions) == 0:
                history_actions[key] = np.repeat(
                    val[:1, ...],
                    self.aggregated_traj_num * self.record_action_len_per_traj,
                    axis=0,
                )

            else:
                concatenated_recorded_actions = np.concatenate(
                    self.recorded_actions[key], axis=0
                )
                if len(self.recorded_actions[key]) < self.aggregated_traj_num:
                    padded_action_length = (
                        self.aggregated_traj_num - len(self.recorded_actions[key])
                    ) * self.record_action_len_per_traj
                    padded_actions = np.repeat(
                        concatenated_recorded_actions[:1], padded_action_length, axis=0
                    )
                    history_actions[key] = np.concatenate(
                        [padded_actions, concatenated_recorded_actions], axis=0
                    )
                else:
                    history_actions[key] = concatenated_recorded_actions

            assert history_actions[key].shape == (
                self.aggregated_traj_num * self.record_action_len_per_traj,
                dof,
            ), f"History actions {key} should be ({self.aggregated_traj_num * self.record_action_len_per_traj}, {dof}), but got {history_actions[key].shape}"

            history_actions[key] = np.concatenate([history_actions[key], val], axis=0)

        return history_actions

    def get_padded_history_images(
        self, latest_history_images: dict[str, npt.NDArray[Any]]
    ):
        """
        latest_history_images: {
            "robot0_main_camera": (N, H, W, 3),
            "robot0_ultrawide_camera": (N, H, W, 3),
            "robot0_depth_camera": (N, H, W, 1),
        }
        return: (aggregated_traj_num * record_image_len_per_traj + N, H, W, 3)
        """

        history_images: dict[str, npt.NDArray[np.uint8]] = {}

        for key, val in latest_history_images.items():
            if not key.endswith("_camera") and not key.endswith("_image"):
                continue

            if "depth" in key:
                assert (
                    val.ndim == 3
                ), f"Latest history image {key} should be a 3D array, but got {val.shape}"
            else:
                assert (
                    val.ndim == 4
                ), f"Latest history image {key} should be a 4D array, but got {val.shape}"

            if len(self.recorded_images) == 0:
                history_images[key] = np.repeat(
                    val[:1, ...],
                    self.aggregated_traj_num * self.record_image_len_per_traj,
                    axis=0,
                )

            else:
                concatenated_recorded_images = np.concatenate(
                    self.recorded_images[key], axis=0
                )
                if len(self.recorded_images[key]) < self.aggregated_traj_num:
                    padded_image_length = (
                        self.aggregated_traj_num - len(self.recorded_images[key])
                    ) * self.record_image_len_per_traj
                    padded_images = np.repeat(
                        concatenated_recorded_images[:1], padded_image_length, axis=0
                    )
                    history_images[key] = np.concatenate(
                        [padded_images, concatenated_recorded_images], axis=0
                    )
                else:
                    history_images[key] = concatenated_recorded_images

            assert history_images[key].shape == (
                self.aggregated_traj_num * self.record_image_len_per_traj,
                *val.shape[1:],
            ), f"History images {key} should be ({self.aggregated_traj_num * self.record_image_len_per_traj}, *{val.shape[1:]}), but got {history_images[key].shape}"

            history_images[key] = np.concatenate([history_images[key], val], axis=0)

        return history_images

    def update_recorded_actions(self, actions: dict[str, npt.NDArray[Any]]):
        """
        actions: {
            "action0_gripper_width": (M, 1),
            "action0_eef_xyz_wxyz": (M, 7),
            "timestamps": (M, ),
        }
        """
        for key, val in actions.items():
            if not key.startswith("action"):
                continue

            assert (
                val.ndim == 2
            ), f"Actions {key} should be a 2D array, but got {val.shape}"

            dof = val.shape[1]

            if key not in self.recorded_actions:
                self.recorded_actions[key] = []

            self.recorded_actions[key].append(
                val[: self.record_action_len_per_traj, ...]
            )

            while len(self.recorded_actions[key]) > self.aggregated_traj_num:
                self.recorded_actions[key].pop(0)

    def update_recorded_images(self, images: dict[str, npt.NDArray[Any]]):
        """
        images: {
            "robot0_main_camera": (N, H, W, 3),
            "robot0_ultrawide_camera": (N, H, W, 3),
            "robot0_depth_camera": (N, H, W, 1),
        }
        """
        for key, val in images.items():
            if not key.endswith("_camera") and not key.endswith("_image"):
                continue

            if key not in self.recorded_images:
                self.recorded_images[key] = []

            if "depth" in key:
                assert (
                    val.ndim == 3
                ), f"Images {key} should be a 3D array, but got {val.shape}"
            else:
                assert (
                    val.ndim == 4
                ), f"Images {key} should be a 4D array, but got {val.shape}"

            self.recorded_images[key].append(val)

            while len(self.recorded_images[key]) > self.aggregated_traj_num:
                self.recorded_images[key].pop(0)

    def predict_actions(
        self,
        observations: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.uint8]],
        history_actions: dict[str, npt.NDArray[np.float64]],
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        Same as PolicyAgent.predict_actions, but will record history actions and images for policy inference.

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

        observations = copy.deepcopy(observations)
        history_actions = copy.deepcopy(history_actions)
        obs_last_timestamp: npt.NDArray[np.float64] = observations["timestamps"][-1]
        obs_last_eef_xyz_wxyz: npt.NDArray[np.float64] = observations[
            "robot0_eef_xyz_wxyz"
        ][-1]

        padded_history_actions = self.get_padded_history_absolute_actions(
            history_actions
        )
        padded_history_images = self.get_padded_history_images(observations)

        self.update_recorded_images(observations)

        observations.update(padded_history_images)

        if self.action_as_proprio:

            if self.use_relative_action:
                actual_action_length = len(
                    padded_history_actions["action0_eef_xyz_wxyz"]
                )
                actual_action_length -= 1
                observations["robot0_gripper_width"] = padded_history_actions[
                    "action0_gripper_width"
                ][
                    1:
                ]  # Remove the earliest gripper width to align with the observations

                observations["robot0_eef_xyz_wxyz"] = np.array(
                    [
                        get_relative_pose(
                            new_pose_xyz_wxyz=padded_history_actions[
                                "action0_eef_xyz_wxyz"
                            ][i],
                            init_pose_xyz_wxyz=obs_last_eef_xyz_wxyz,
                        )
                        for i in range(actual_action_length)
                    ]
                )
            else:
                observations["robot0_eef_xyz_wxyz"] = padded_history_actions[
                    "action0_eef_xyz_wxyz"
                ]
                observations["robot0_gripper_width"] = padded_history_actions[
                    "action0_gripper_width"
                ]
        else:
            if self.use_relative_action:
                raise NotImplementedError(
                    "LongHistPolicyAgent does not support use_relative_action when action_as_proprio is false"
                )

        observations["episode_idx"] = np.array(
            0
        )  # Don't need to run multiple episodes at the same time

        print(
            f"observations: {observations['robot0_eef_xyz_wxyz'].shape=}, {observations['robot0_gripper_width'].shape=}"
        )

        reply = self.policy_rmq_client.request_with_data(
            topic="policy_inference", data=serialize(observations), timeout_s=5.0
        )

        rel_actions = deserialize(reply)
        if isinstance(rel_actions, str):
            raise ValueError(f"Policy agent return an error message: {rel_actions}")
        assert isinstance(rel_actions, dict)
        rel_actions = cast(dict[str, npt.NDArray[np.float64]], rel_actions)

        # print(
        #     f"action0_eef_xyz_wxyz: {rel_actions['action0_eef_xyz_wxyz']}, action0_gripper_width: {rel_actions['action0_gripper_width']}"
        # )
        rel_eef_actions = rel_actions["action0_eef_xyz_wxyz"]

        if self.use_relative_action:
            absolute_eef_poses = np.array(
                [
                    get_absolute_pose(
                        init_pose_xyz_wxyz=obs_last_eef_xyz_wxyz,  # Still need to use the actual eef pose to calculate the absolute actions
                        relative_pose_xyz_wxyz=rel_eef_actions[i, :],
                    )
                    for i in range(len(rel_eef_actions))
                ]
            )
        else:
            absolute_eef_poses = rel_eef_actions

        absolute_gripper_widths = rel_actions["action0_gripper_width"]

        new_timestamps = (
            obs_last_timestamp
            + np.arange(
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

        self.update_recorded_actions(actions)

        return actions

    def reset(self):
        super().reset()
        self.recorded_actions = {}
        self.recorded_images = {}
