from typing import Any

import numpy as np

from env.modules.agents.heuristic_agent import HeuristicAgent
from env.modules.common import castf64, data_buffer_type, f64arr, robot_data_type
from env.utils.pose_utils import (
    ActionInterpolator,
    FinalSpeedActionInterpolator,
    LinearActionInterpolator,
    QuadraticActionInterpolator,
)
from loguru import logger

class PushCubeAgent(HeuristicAgent):
    def __init__(
        self,
        push_vels_m_per_s: list[float],
        push_start_pose_xyz_wxyz: list[float],
        push_end_y_m: float,
        slow_down_end_y_m: float,
        draw_back_end_y_m: float,
        gripper_width_m: list[float],
        object_stable_threshold_m_per_s: float,
        draw_back_1_duration_s: float,
        draw_back_2_duration_s: float,
        reset_duration_s: float,
        target_y_m: float,
        stop_trying_tolerance_m: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.push_vels_m_per_s: f64arr = np.sort(np.array(push_vels_m_per_s))
        self.push_start_pose_xyz_wxyz: f64arr = np.array(push_start_pose_xyz_wxyz)
        self.push_end_y_m: float = push_end_y_m
        self.draw_back_end_y_m: float = draw_back_end_y_m
        self.slow_down_end_y_m: float = slow_down_end_y_m
        self.interpolator: ActionInterpolator | None = None
        self.gripper_width_m: f64arr = np.array(gripper_width_m)
        self.object_stable_threshold: float = object_stable_threshold_m_per_s
        self.draw_back_1_duration_s: float = draw_back_1_duration_s
        self.draw_back_2_duration_s: float = draw_back_2_duration_s
        """
        Use a fixed time to draw back the robot
        """
        self.reset_duration_s: float = reset_duration_s
        """
        Use a fixed time to reset to the pushing start position
        """
        self.target_y_m: float = target_y_m
        self.stop_trying_tolerance_m: float = stop_trying_tolerance_m

        self.phase: str = "waiting_until_stable"
        """
        Valid phases:
            waiting_until_stable, # Only at the beginning of the episode
            drawing_back_1, # Will draw the robot back to the pushing start position, will take draw_back_1_duration_s second
            drawing_back_2, # Will draw the robot back to ahead of the pushing start position, will take draw_back_2_duration_s seconds
            resetting,
            pushing,
            slowing_down,
        """
        self.phase_start_cnt: int = 0
        self.last_object_pose: f64arr | None = None
        self.trial_cnt: int = 0
        self.object_reset_cnt: int = 0
        self.reached_target_tolerance: bool = False

    def reset(self, episode_config: dict[str, Any] | None = None):
        self.phase = "waiting_until_stable"
        self.phase_start_cnt = 0
        self.last_object_pose = None
        self.trial_cnt = 0
        self.object_reset_cnt = 0
        self.reached_target_tolerance = False

    def predict_actions(
        self,
        robots_obs: data_buffer_type,
        env_objs_obs: data_buffer_type,
        history_actions: data_buffer_type,
    ) -> data_buffer_type:
        actions: data_buffer_type = []  # [1 action, 1 robot]
        robot_action: robot_data_type = {}
        robot_action["name"] = robots_obs[0][0]["name"]
        robot_action["is_error"] = np.array([False])
        robot_action["is_critical"] = np.array([False])
        robot_action["gripper_width"] = self.gripper_width_m

        self.phase_start_cnt += 1
        regulated_robot_pose = self.push_start_pose_xyz_wxyz.copy()
        regulated_robot_pose[1] = robots_obs[0][0]["tcp_xyz_wxyz"][
            1
        ]  # Only allow y movement
        new_object_pose = castf64(env_objs_obs[0][1]["object_pose_xyz_wxyz"])

        if self.last_object_pose is not None:
            object_is_stable = (
                self.object_stable_threshold / self.agent_update_freq_hz
                > np.linalg.norm(new_object_pose[:3] - self.last_object_pose[:3])
            )
        else:
            object_is_stable = False

        # Check whether the cube is stopped and close to the target
        if not self.reached_target_tolerance and object_is_stable:
            if (
                np.abs(new_object_pose[1] - self.target_y_m)
                < self.stop_trying_tolerance_m
            ):
                self.reached_target_tolerance = True
                if self.trial_cnt < len(self.push_vels_m_per_s):
                    logger.info(
                        f"Reached target tolerance, stopping trying at trial {self.trial_cnt - 1}. Original push vel num: {len(self.push_vels_m_per_s)}"
                    )
                    self.push_vels_m_per_s = self.push_vels_m_per_s[: self.trial_cnt]

        if self.phase == "waiting_until_stable":
            if self.last_object_pose is None:
                self.last_object_pose = castf64(
                    env_objs_obs[0][1]["object_pose_xyz_wxyz"]
                )
            else:
                # Wait until the object pose is stable
                if object_is_stable:
                    self.phase = "resetting"
                    self.phase_start_cnt = 0
                    pos_speed_m_per_s = (
                        abs(regulated_robot_pose[1] - self.push_start_pose_xyz_wxyz[1])
                        / self.reset_duration_s
                    )

                    self.interpolator = LinearActionInterpolator(
                        start_pose_xyz_wxyz=regulated_robot_pose,
                        end_pose_xyz_wxyz=self.push_start_pose_xyz_wxyz,
                        start_gripper_width=self.gripper_width_m,
                        end_gripper_width=self.gripper_width_m,
                        pos_speed_m_per_s=pos_speed_m_per_s,
                        rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                        gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                    )

            robot_action["tcp_xyz_wxyz"] = regulated_robot_pose.copy()

        elif self.phase == "drawing_back_1":
            assert self.interpolator is not None

            robot_action["tcp_xyz_wxyz"], robot_action["gripper_width"] = (
                self.interpolator.interpolate(1 / self.agent_update_freq_hz)
            )

            if self.interpolator.is_finished:
                self.phase = "drawing_back_2"
                self.phase_start_cnt = 0
                start_pose = self.push_start_pose_xyz_wxyz.copy()
                end_pose = self.push_start_pose_xyz_wxyz.copy()
                end_pose[1] = self.draw_back_end_y_m
                reset_pos_speed_m_per_s = (
                    abs(start_pose[1] - end_pose[1]) / self.draw_back_2_duration_s
                )
                self.interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=start_pose,
                    end_pose_xyz_wxyz=end_pose,
                    start_gripper_width=self.gripper_width_m,
                    end_gripper_width=self.gripper_width_m,
                    pos_speed_m_per_s=reset_pos_speed_m_per_s,
                    rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                    gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                )

        elif self.phase == "drawing_back_2":
            assert self.interpolator is not None

            robot_action["tcp_xyz_wxyz"], robot_action["gripper_width"] = (
                self.interpolator.interpolate(1 / self.agent_update_freq_hz)
            )

            if self.interpolator.is_finished:
                self.phase = "resetting"
                self.phase_start_cnt = 0
                draw_back_end_pose = self.push_start_pose_xyz_wxyz.copy()
                draw_back_end_pose[1] = self.draw_back_end_y_m
                reset_pos_speed_m_per_s = (
                    abs(self.push_start_pose_xyz_wxyz[1] - draw_back_end_pose[1])
                    / self.reset_duration_s
                )
                self.interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=draw_back_end_pose,
                    end_pose_xyz_wxyz=self.push_start_pose_xyz_wxyz,
                    start_gripper_width=self.gripper_width_m,
                    end_gripper_width=self.gripper_width_m,
                    pos_speed_m_per_s=reset_pos_speed_m_per_s,
                    rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                    gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                )

        elif self.phase == "resetting":
            # Slowly move the robot to the pushing start position

            assert self.interpolator is not None
            robot_action["tcp_xyz_wxyz"], robot_action["gripper_width"] = (
                self.interpolator.interpolate(1 / self.agent_update_freq_hz)
            )

            if self.interpolator.is_finished:

                self.phase = "pushing"
                push_end_pose_xyz_wxyz = self.push_start_pose_xyz_wxyz.copy()
                push_end_pose_xyz_wxyz[1] = self.push_end_y_m
                if self.trial_cnt < len(self.push_vels_m_per_s):
                    push_vel = self.push_vels_m_per_s[self.trial_cnt]
                else:
                    push_vel = self.push_vels_m_per_s[-1]
                self.interpolator = FinalSpeedActionInterpolator(
                    start_pose_xyz_wxyz=self.push_start_pose_xyz_wxyz,
                    end_pose_xyz_wxyz=push_end_pose_xyz_wxyz,
                    start_gripper_width=self.gripper_width_m,
                    end_gripper_width=self.gripper_width_m,
                    final_speed_m_per_s=push_vel,
                    dt=1 / self.agent_update_freq_hz,
                    final_speed_step_num=2,
                )

        elif self.phase == "pushing":

            assert self.interpolator is not None
            robot_action["is_error"] = np.array([False])
            robot_action["is_critical"] = np.array([True])
            robot_action["tcp_xyz_wxyz"], robot_action["gripper_width"] = (
                self.interpolator.interpolate(1 / self.agent_update_freq_hz)
            )
            if self.interpolator.is_finished:
                self.phase = "slowing_down"
                self.phase_start_cnt = 0
                self.interpolator = None
                self.last_object_pose = None
                slowing_down_start_pose = self.push_start_pose_xyz_wxyz.copy()
                slowing_down_start_pose[1] = self.push_end_y_m
                slowing_down_end_pose = self.push_start_pose_xyz_wxyz.copy()
                slowing_down_end_pose[1] = self.slow_down_end_y_m

                if self.trial_cnt < len(self.push_vels_m_per_s):
                    push_vel = self.push_vels_m_per_s[self.trial_cnt]
                else:
                    push_vel = self.push_vels_m_per_s[-1]
                self.interpolator = QuadraticActionInterpolator(
                    start_pose_xyz_wxyz=slowing_down_start_pose,
                    end_pose_xyz_wxyz=slowing_down_end_pose,
                    start_gripper_width=self.gripper_width_m,
                    end_gripper_width=self.gripper_width_m,
                    start_speed_m_per_s=push_vel,
                    final_speed_m_per_s=0.0,
                )

        elif self.phase == "slowing_down":
            assert self.interpolator is not None
            robot_action["name"] = robots_obs[0][0]["name"]
            robot_action["is_error"] = np.array([False])
            robot_action["is_critical"] = np.array([False])
            robot_action["tcp_xyz_wxyz"], robot_action["gripper_width"] = (
                self.interpolator.interpolate(1 / self.agent_update_freq_hz)
            )
            if self.interpolator.is_finished:
                self.phase = "drawing_back_1"
                self.phase_start_cnt = 0
                self.interpolator = None
                self.trial_cnt += 1

                start_pose = self.push_start_pose_xyz_wxyz.copy()
                start_pose[1] = self.slow_down_end_y_m
                end_pose = self.push_start_pose_xyz_wxyz.copy()
                draw_back_pos_speed_m_per_s = (
                    abs(start_pose[1] - end_pose[1]) / self.draw_back_1_duration_s
                )
                self.interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=start_pose,
                    end_pose_xyz_wxyz=end_pose,
                    start_gripper_width=self.gripper_width_m,
                    end_gripper_width=self.gripper_width_m,
                    pos_speed_m_per_s=draw_back_pos_speed_m_per_s,
                    rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                    gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                )
        else:
            raise ValueError(f"Invalid phase: {self.phase}")

        self.last_object_pose = new_object_pose
        actions.append([robot_action])
        return actions
