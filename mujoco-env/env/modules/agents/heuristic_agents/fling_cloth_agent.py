from typing import Any

import numpy as np
import numpy.typing as npt

from env.modules.agents.heuristic_agent import HeuristicAgent
from env.modules.common import data_buffer_type, f64arr, robot_data_type
from env.utils.pose_utils import (
    ActionInterpolator,
    CubicInterpolator,
    LinearActionInterpolator,
)


class FlingClothAgent(HeuristicAgent):
    def __init__(
        self,
        reset_poses_xyz_wxyz: list[list[float]],
        fling_waypoints_z_bias: float,
        fling_waypoints_xz: list[list[float]],
        fling_timestamp_intervals_s: list[float],
        fling_speed_scaling_mask: list[bool],
        pause_before_fling_s: float,
        pause_before_reset_s: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.reset_poses_xyz_wxyz: f64arr = np.array(reset_poses_xyz_wxyz)
        assert self.reset_poses_xyz_wxyz.shape == (
            2,
            7,
        ), f"Invalid reset poses {self.reset_poses_xyz_wxyz}"
        assert np.allclose(
            self.reset_poses_xyz_wxyz[0, (0, 2)], self.reset_poses_xyz_wxyz[1, (0, 2)]
        ), "Reset poses must have the same x and z coordinates"

        self.fling_waypoints_z_bias: float = fling_waypoints_z_bias
        self.fling_start_xz: f64arr = self.reset_poses_xyz_wxyz[0, (0, 2)]
        self.fling_waypoints_xz: f64arr = np.array(fling_waypoints_xz)
        self.fling_waypoints_xz[:, 1] += self.fling_waypoints_z_bias
        assert self.fling_waypoints_xz.shape[1] == 2

        # self.fling_speeds_m_per_s: f64arr = np.array(fling_speeds_m_per_s)
        # assert len(self.fling_speeds_m_per_s) == len(self.fling_waypoints_xz) - 1
        assert (
            len(fling_timestamp_intervals_s) == len(self.fling_waypoints_xz) - 1
        ), f"{len(fling_timestamp_intervals_s)=}, {len(self.fling_waypoints_xz)=}"
        assert len(fling_speed_scaling_mask) == len(
            fling_timestamp_intervals_s
        ), f"{len(fling_speed_scaling_mask)=}, {len(fling_timestamp_intervals_s)=}"

        self.fling_waypoint_intervals_s: f64arr = np.array(fling_timestamp_intervals_s)
        self.fling_speed_scaling_mask: npt.NDArray[np.bool_] = np.array(
            fling_speed_scaling_mask, dtype=np.bool_
        )
        # Fling waypoint interval time will be scaled if the mask value is true

        self.fling_speed_scales: list[float] | None

        self.pause_before_fling_s: float = pause_before_fling_s
        self.pause_before_reset_s: float = pause_before_reset_s
        self.fixed_gripper_width: npt.NDArray[np.float64] = np.array([0.0])

        # State variables
        self.phase: str = "resetting"
        """
        Valid phases:
            resetting,
            pause_before_fling,
            flinging,
            pause_before_reset
        """
        self.last_cloth_grid_pos: f64arr | None = None
        self.trial_cnt: int = 0
        self.interpolator: ActionInterpolator | None = None

    def predict_actions(
        self,
        robots_obs: data_buffer_type,
        env_objs_obs: data_buffer_type,
        history_actions: data_buffer_type,
    ) -> data_buffer_type:

        if self.fling_speed_scales is None:
            raise RuntimeError(
                "fling_speed_scales should be set through reset(episode_config)"
            )

        actions: data_buffer_type = []  # [1 action, 2 robots]
        left_robot_action: robot_data_type = {}
        right_robot_action: robot_data_type = {}
        left_robot_action["name"] = robots_obs[0][0]["name"]
        right_robot_action["name"] = robots_obs[0][1]["name"]
        left_robot_action["is_error"] = np.array([False])
        right_robot_action["is_error"] = np.array([False])
        left_robot_action["is_critical"] = np.array([False])
        right_robot_action["is_critical"] = np.array([False])
        left_robot_action["gripper_width"] = np.array([0.0])
        right_robot_action["gripper_width"] = np.array([0.0])

        # self.default_actions

        current_xz = history_actions[-1][0]["tcp_xyz_wxyz"][[0, 2]]
        indices_without_y = [0, *tuple(range(2, 7))]
        indices_without_xz = [1, *tuple(range(3, 7))]
        assert np.allclose(
            current_xz, history_actions[-1][1]["tcp_xyz_wxyz"][[0, 2]]
        ), "Left and right robot must have the same x and z coordinates"
        assert np.allclose(
            self.reset_poses_xyz_wxyz[0, indices_without_xz],
            history_actions[-1][0]["tcp_xyz_wxyz"][indices_without_xz],
        ), f"Only x and z coordinates are allowed to change, {self.reset_poses_xyz_wxyz[0]=} {history_actions[-1][0]['tcp_xyz_wxyz']=}"
        assert np.allclose(
            self.reset_poses_xyz_wxyz[1, indices_without_xz],
            history_actions[-1][1]["tcp_xyz_wxyz"][indices_without_xz],
        ), f"Only x and z coordinates are allowed to change, {self.reset_poses_xyz_wxyz[1]=} {history_actions[-1][1]['tcp_xyz_wxyz']=}"

        if self.phase == "resetting":
            if self.interpolator is None:
                self.interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=history_actions[-1][0]["tcp_xyz_wxyz"],
                    end_pose_xyz_wxyz=self.reset_poses_xyz_wxyz[0],
                    start_gripper_width=self.fixed_gripper_width,
                    end_gripper_width=self.fixed_gripper_width,
                    pos_speed_m_per_s=self.position_speed_m_per_s,
                    rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                    gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                )
            left_robot_action["tcp_xyz_wxyz"], left_robot_action["gripper_width"] = (
                self.interpolator.interpolate(1 / self.agent_update_freq_hz)
            )

            if self.interpolator.is_finished:
                self.phase = "pause_before_fling"
                self.phase_start_cnt = 0
                self.interpolator = None

        elif self.phase == "pause_before_fling":

            left_robot_action["tcp_xyz_wxyz"] = history_actions[-1][0][
                "tcp_xyz_wxyz"
            ]  # Keep the same pose

            if (
                self.phase_start_cnt
                < self.pause_before_fling_s * self.agent_update_freq_hz
            ):
                self.phase_start_cnt += 1
            else:
                self.phase = "flinging"
                self.phase_start_cnt = 0
                waypoint_num = len(self.fling_waypoints_xz)
                pose_waypoints = np.array([self.reset_poses_xyz_wxyz[0]] * waypoint_num)
                pose_waypoints[:, [0, 2]] = self.fling_waypoints_xz
                gripper_width_waypoints = np.array([[0.0]] * waypoint_num)

                fling_speed_scale = self.fling_speed_scales[
                    min(self.trial_cnt, len(self.fling_speed_scales) - 1)
                ]
                fling_waypoint_intervals_s = self.fling_waypoint_intervals_s.copy()
                fling_waypoint_intervals_s[
                    self.fling_speed_scaling_mask
                ] /= fling_speed_scale

                fling_waypoint_timestamps_s = np.hstack(
                    [
                        np.zeros(1),
                        np.cumsum(fling_waypoint_intervals_s),
                    ]
                )

                self.interpolator = CubicInterpolator(
                    pose_waypoints=pose_waypoints,
                    gripper_width_waypoints=gripper_width_waypoints,
                    timestamps_s=fling_waypoint_timestamps_s,
                )
                # self.interpolator.visualize_spline()

        elif self.phase == "flinging":
            assert self.interpolator is not None
            (
                left_robot_action["tcp_xyz_wxyz"],
                left_robot_action["gripper_width"],
            ) = self.interpolator.interpolate(1 / self.agent_update_freq_hz)

            if self.interpolator.is_finished:
                self.phase = "pause_before_reset"
                self.phase_start_cnt = 0
                self.interpolator = None
                self.trial_cnt += 1

        elif self.phase == "pause_before_reset":
            left_robot_action["tcp_xyz_wxyz"] = history_actions[-1][0][
                "tcp_xyz_wxyz"
            ]  # Keep the same pose

            if (
                self.phase_start_cnt
                < self.pause_before_reset_s * self.agent_update_freq_hz
            ):
                self.phase_start_cnt += 1
            else:
                self.phase = "resetting"
                self.phase_start_cnt = 0
                self.interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=history_actions[-1][0]["tcp_xyz_wxyz"],
                    end_pose_xyz_wxyz=self.reset_poses_xyz_wxyz[0],
                    start_gripper_width=self.fixed_gripper_width,
                    end_gripper_width=self.fixed_gripper_width,
                    pos_speed_m_per_s=self.position_speed_m_per_s,
                    rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                    gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                )

        right_robot_action["tcp_xyz_wxyz"] = left_robot_action["tcp_xyz_wxyz"].copy()
        right_robot_action["tcp_xyz_wxyz"][1] = self.reset_poses_xyz_wxyz[1][
            1
        ]  # Only y coordinate is different from the left robot
        # right_robot_action["tcp_xyz_wxyz"][0] -= 0.01

        actions.append([left_robot_action, right_robot_action])
        return actions

    def reset(self, episode_config: dict[str, Any] | None = None):
        if episode_config is not None:
            if "speed_scales" in episode_config:
                assert isinstance(episode_config["speed_scales"], list)
                self.fling_speed_scales = episode_config["speed_scales"].copy()

        assert (
            self.fling_speed_scales is not None
        ), "fling_speed_scales should be provided in episode_config"

        self.phase = "resetting"
        self.phase_start_cnt = 0
        self.interpolator = None
        self.trial_cnt = 0
