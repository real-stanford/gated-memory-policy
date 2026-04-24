import copy
from typing import Any, Optional, cast

import numpy as np
import numpy.typing as npt
from robot_utils.teleop_utils.spacemouse import SpacemouseClient
from transforms3d.euler import euler2quat, quat2euler

from env.modules.agents.base_agent import BaseAgent
from env.modules.common import data_buffer_type, robot_data_type
from loguru import logger


class SpacemouseAgent(BaseAgent):
    def __init__(
        self,
        rmq_server_address: str,
        position_speed_m_per_s: float,
        rotation_speed_rad_per_s: float,
        gripper_speed_m_per_s: float,
        gripper_width_m: float,
        gripper_max_overshoot_m: float,
        robot_base_poses_xyz_wxyz: list[list[float]],
        **kwargs,
    ):
        # assert (
        #     "robot_num" in kwargs and kwargs["robot_num"] == 1
        # ), "Only one robot is supported for spacemouse teleop"
        super().__init__(**kwargs)
        # assert (
        #     len(robot_base_poses_xyz_wxyz) == 1
        # ), "Only one robot base pose is supported for spacemouse teleop"
        self.spacemouse_client: SpacemouseClient = SpacemouseClient(rmq_server_address)
        self.position_speed_m_per_s: float = position_speed_m_per_s
        self.rotation_speed_rad_per_s: float = rotation_speed_rad_per_s
        self.gripper_speed_m_per_s: float = gripper_speed_m_per_s
        self.gripper_width_m: float = gripper_width_m
        self.gripper_max_overshoot_m: float = gripper_max_overshoot_m
        self.pose_orientation_types: list[int] = [
            self._get_pose_orientation_type(robot_base_poses_xyz_wxyz[i])
            for i in range(len(robot_base_poses_xyz_wxyz))
        ]
        self.control_robot_idx: int = -1  # -1 means simultaneous control of all robots

    def _get_pose_orientation_type(self, robot_base_poses_xyz_wxyz: list[float]) -> int:
        quat_wxyz = np.array(robot_base_poses_xyz_wxyz[3:])
        if quat_wxyz[0] < 0:
            quat_wxyz = -quat_wxyz
        default_orientations_wxyz = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.707, 0.0, 0.0, 0.707],
                [0.0, 0.0, 0.0, 1.0],
                [0.707, 0.0, 0.0, -0.707],
            ]
        )
        for i, default_orientation in enumerate(default_orientations_wxyz):
            if np.linalg.norm(quat_wxyz - default_orientation) < 0.1:
                return i
        raise ValueError(
            f"robot base pose orientation (wxyz) should be close to one of the following: {default_orientations_wxyz}, but got {quat_wxyz}"
        )

    def predict_actions(
        self,
        robots_obs: data_buffer_type,
        env_objs_obs: data_buffer_type,
        history_actions: data_buffer_type,
    ) -> data_buffer_type:
        spacemouse_movements, spacemouse_buttons = (
            self.spacemouse_client.get_average_state(10)
        )
        results: data_buffer_type = []

        for history_action in history_actions:
            new_actions: list[robot_data_type] = []  # For each robot
            for i in range(self.robot_num):
                if i != self.control_robot_idx and self.control_robot_idx != -1:
                    # Preserve the last action
                    new_actions.append(copy.deepcopy(history_action[i]))
                    continue
                robot_name = cast(str, robots_obs[-1][i]["name"])
                actual_gripper_width: float = cast(
                    npt.NDArray[np.float64], robots_obs[-1][i]["gripper_width"]
                )[0]
                new_action: robot_data_type = {}
                rpy = np.array(quat2euler(history_action[i]["tcp_xyz_wxyz"][3:]))

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
                    + rpy_vel
                    * self.rotation_speed_rad_per_s
                    / self.agent_update_freq_hz
                )
                updated_quat = euler2quat(*updated_rpy)

                updated_xyz = (
                    cast(npt.NDArray[np.float64], history_action[i]["tcp_xyz_wxyz"])[:3]
                    + spacemouse_movements[:3]
                    * self.position_speed_m_per_s
                    / self.agent_update_freq_hz
                )
                new_action["tcp_xyz_wxyz"] = np.concatenate(
                    [updated_xyz, updated_quat], dtype=np.float64
                )
                new_action["tcp_xyz_wxyz"][self.disabled_movement_mask] = (
                    self.default_actions[i][self.disabled_movement_mask]
                )
                if not self.move_gripper:
                    new_action["gripper_width"] = self.default_gripper_widths[i]

                updated_gripper_width: float = cast(
                    npt.NDArray[np.float64], history_action[i]["gripper_width"]
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

                new_action["gripper_width"] = np.array([updated_gripper_width])
                new_action["name"] = robot_name
                new_actions.append(new_action)
            results.append(new_actions)
        return results

    def switch_robot(self):
        self.control_robot_idx = (self.control_robot_idx + 1) % (self.robot_num + 1)
        if self.control_robot_idx == self.robot_num:
            self.control_robot_idx = -1
            logger.info(
                f"SpacemouseAgent: Switching to simultaneous control of all robots"
            )
        else:
            logger.info(f"SpacemouseAgent: Switching to robot {self.control_robot_idx}")

    def reset(self, episode_config: Optional[dict[str, Any]] = None):
        pass
