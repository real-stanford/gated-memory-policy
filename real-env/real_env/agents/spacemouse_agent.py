import copy
import time
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from robot_utils.teleop_utils.spacemouse import SpacemouseClient
from transforms3d.euler import euler2quat, quat2euler

from real_env.agents.base_agent import BaseAgent


class SpacemouseAgent(BaseAgent):
    def __init__(
        self,
        pose_orientation_types: list[int],
        rmq_server_address: str,
        position_speed_m_per_s: float,
        rotation_speed_rad_per_s: float,
        gripper_speed_m_per_s: float,
        gripper_width_m: float,
        gripper_max_overshoot_m: float,
        **kwargs,
    ):

        super().__init__(name="Spacemouse", **kwargs)
        self.spacemouse_client: SpacemouseClient = SpacemouseClient(rmq_server_address)
        self.position_speed_m_per_s: float = position_speed_m_per_s
        self.rotation_speed_rad_per_s: float = rotation_speed_rad_per_s
        self.gripper_speed_m_per_s: float = gripper_speed_m_per_s
        self.gripper_width_m: float = gripper_width_m
        self.gripper_max_overshoot_m: float = gripper_max_overshoot_m
        self.pose_orientation_types: list[int] = pose_orientation_types
        """
        Need to manually test which type is the correct one. Will be different because of different robot base orientation.
        In general, suppose the robot base frame is z pointing up, this orientation type can be calculated given the z direction of the robot home pose:
        - gripper z pointing to +x -> type 0
        - gripper z pointing to +y -> type 1
        - gripper z pointing to -x -> type 2
        - gripper z pointing to -y -> type 3
        """
        assert (
            len(self.pose_orientation_types) == self.robot_num
        ), f"pose_orientation_types must have {self.robot_num} elements, but got {len(self.pose_orientation_types)}"

        self.control_robot_idx: int = (
            -1
        )  # -1 means controlling all robots simultaneously

    def predict_actions(
        self,
        observations: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.uint8]],
        history_actions: dict[str, npt.NDArray[np.float64]],
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Return the actions for the robots based on the observations
        Args:
            observations: e.g. {
                "robot0_eef_xyz_wxyz": (proprio_history_len, 7),
                "robot0_gripper_width": (proprio_history_len, 1),
                "robot0_wrist_camera": (proprio_history_len, 256, 256, 3),
                "timestamps": (proprio_history_len, ),
            }
            history_actions: e.g. {
                "action0_eef_xyz_wxyz": (action_history_len, 7),
                "action0_gripper_width": (action_history_len, 1),
                "timestamps": (action_history_len, ),
            }
        Returns:
            e.g. {
                "action0_eef_xyz_wxyz": (action_prediction_horizon, 7),
                "action0_gripper_width": (action_prediction_horizon, 1),
                "timestamps": (action_prediction_horizon, ),
            }
        """

        # According to history observations, decide which action to execute
        # Handle cases when policy trajectory fails to execute
        if history_actions[f"action0_eef_xyz_wxyz"].shape[0] > 1:
            distance_to_last_action = np.linalg.norm(
                history_actions[f"action0_eef_xyz_wxyz"][-1, :3]
                - observations[f"robot0_eef_xyz_wxyz"][-1, :3]
            )
            distance_to_fist_action = np.linalg.norm(
                history_actions[f"action0_eef_xyz_wxyz"][0, :3]
                - observations[f"robot0_eef_xyz_wxyz"][-1, :3]
            )
            if distance_to_last_action < distance_to_fist_action:
                history_action_idx = -1
            else:
                # The last trajectory is not properly executed
                history_action_idx = 0
        else:
            history_action_idx = -1

        (
            spacemouse_movements,
            spacemouse_buttons,
        ) = self.spacemouse_client.get_average_state(10)
        new_actions: dict[str, npt.NDArray[np.float64]] = {
            k: v[-self.action_history_len :] for k, v in history_actions.items()
        }
        new_actions["timestamps"] = (
            np.array([time.monotonic()]) + 1 / self.agent_update_freq_hz
        )
        for i in range(self.robot_num):
            if i != self.control_robot_idx and self.control_robot_idx != -1:
                continue
            actual_gripper_width: float = cast(
                npt.NDArray[np.float64],
                history_actions[f"action{i}_gripper_width"][history_action_idx],
            )[0]
            quat = np.array(
                history_actions[f"action{i}_eef_xyz_wxyz"][history_action_idx][3:]
            )
            rpy = np.array(quat2euler(quat))

            if self.pose_orientation_types[i] == 0:
                rpy_vel = spacemouse_movements[[4, 3, 5]]
                rpy_vel[0] *= -1
            elif self.pose_orientation_types[i] == 1:
                rpy_vel = spacemouse_movements[[3, 4, 5]]
            elif self.pose_orientation_types[i] == 2:
                rpy_vel = spacemouse_movements[[4, 3, 5]]
                rpy_vel[1] *= -1
            elif self.pose_orientation_types[i] == 3:
                rpy_vel = spacemouse_movements[[3, 4, 5]]
                rpy_vel[0] *= -1
                rpy_vel[1] *= -1
            else:
                raise ValueError(
                    f"Unknown pose orientation type: {self.pose_orientation_types[i]}"
                )

            updated_rpy = (
                rpy
                + rpy_vel * self.rotation_speed_rad_per_s / self.agent_update_freq_hz
            )
            updated_quat = euler2quat(*updated_rpy)

            updated_xyz = (
                cast(
                    npt.NDArray[np.float64],
                    history_actions[f"action{i}_eef_xyz_wxyz"][history_action_idx],
                )[:3]
                + spacemouse_movements[:3]
                * self.position_speed_m_per_s
                / self.agent_update_freq_hz
            )
            new_eef_xyz_wxyz = np.concatenate(
                [updated_xyz, updated_quat], dtype=np.float64
            )
            new_actions[f"action{i}_eef_xyz_wxyz"][0, :] = new_eef_xyz_wxyz

            updated_gripper_width: float = cast(
                npt.NDArray[np.float64],
                history_actions[f"action{i}_gripper_width"][history_action_idx],
            )[0]
            if spacemouse_buttons[0] == 1:
                updated_gripper_width += (
                    self.gripper_speed_m_per_s / self.agent_update_freq_hz
                )
            elif spacemouse_buttons[1] == 1:
                updated_gripper_width -= (
                    self.gripper_speed_m_per_s / self.agent_update_freq_hz
                )
            if updated_gripper_width < 0:
                updated_gripper_width = 0
            elif updated_gripper_width > self.gripper_width_m:
                updated_gripper_width = self.gripper_width_m

            # Prevent overshooting too much
            if (
                actual_gripper_width - updated_gripper_width
                > self.gripper_max_overshoot_m
            ):
                updated_gripper_width = max(
                    0, actual_gripper_width - self.gripper_max_overshoot_m
                )

            # print(f"SpacemouseAgent: {updated_gripper_width=:.3f}")

            new_actions[f"action{i}_gripper_width"][0, 0] = updated_gripper_width
        return new_actions

    def switch_robot(self):
        self.control_robot_idx = (self.control_robot_idx + 1) % (self.robot_num + 1)
        if self.control_robot_idx == self.robot_num:
            self.control_robot_idx = -1
            print(f"SpacemouseAgent: Switching to simultaneous control of all robots")
        else:
            print(f"SpacemouseAgent: Switching to robot {self.control_robot_idx}")

    def reset(self):
        pass
