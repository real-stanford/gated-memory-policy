from typing import Any
from typing_extensions import override

import numpy as np
import numpy.typing as npt

from real_env.agents.heuristic_agent import HeuristicAgent

from real_env.agents.interpolator_utils import (
    ActionInterpolator,
    FinalSpeedActionInterpolator,
    LinearActionInterpolator,
    QuadraticActionInterpolator,
)


class CastingAgent(HeuristicAgent):
    """
    Heuristic agent starts at drag_start_pose_xyz_wxyz (home/safe position), waits for
    waiting_before_dragging_duration_s, then drags to drag_end_xyz_wxyz at specified drag_vels_m_per_s
    (maintaining constant velocity for final_speed_step_num steps), then slows down to
    slow_down_end_xyz_wxyz, then lifts arm up to lift_arm_up_xyz_wxyz with duration of
    lift_arm_up_duration_s, then moves to reset_to_back_xyz_wxyz with duration of
    reset_to_back_duration_s, finally returns to drag_start_pose_xyz_wxyz (home position) with
    duration of reset_to_start_duration_s.

    Note: drag_start_pose_xyz_wxyz serves as both the drag start position and the safe/home position.
    Waits for waiting_until_sliding_stop_duration_s after slowing down before resetting.
    """

    def __init__(
        self,
        # Dragging motion related
        drag_start_pose_xyz_wxyz: list[float],
        drag_end_xyz_wxyz: list[float],
        drag_vels_m_per_s: list[float],
        final_speed_step_num: int,
        # Slowing down related
        slow_down_end_xyz_wxyz: list[float],
        # Reset motion related
        lift_arm_up_xyz_wxyz: list[float],
        lift_arm_up_duration_s: float,
        reset_to_back_xyz_wxyz: list[float],
        reset_to_back_duration_s: float,
        reset_to_start_duration_s: float,
        initial_reset_duration_s: float,  # Duration for first-time resetting from unknown position
        waiting_after_lifting_duration_s: float,
        # Gripper related
        # gripper_width_m: float,
        # Waiting duration parameters
        waiting_before_dragging_duration_s: float,
        waiting_after_resetting_to_back_duration_s: float,
        waiting_until_sliding_stop_duration_s: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Dragging motion
        self.drag_start_pose_xyz_wxyz: npt.NDArray[np.float64] = np.array(
            drag_start_pose_xyz_wxyz
        )
        self.drag_end_xyz_wxyz: npt.NDArray[np.float64] = np.array(drag_end_xyz_wxyz)
        # self.drag_vels_m_per_s: npt.NDArray[np.float64] = np.sort(
        #     np.array(drag_vels_m_per_s)
        # )
        self.drag_vels_m_per_s = np.array(drag_vels_m_per_s)
        self.final_speed_step_num: int = final_speed_step_num

        # Slowing down
        self.slow_down_end_xyz_wxyz: npt.NDArray[np.float64] = np.array(
            slow_down_end_xyz_wxyz
        )

        # Reset motion
        self.lift_arm_up_xyz_wxyz: npt.NDArray[np.float64] = np.array(
            lift_arm_up_xyz_wxyz
        )
        self.lift_arm_up_duration_s: float = lift_arm_up_duration_s
        self.reset_to_back_xyz_wxyz: npt.NDArray[np.float64] = np.array(
            reset_to_back_xyz_wxyz
        )
        self.reset_to_back_duration_s: float = reset_to_back_duration_s
        self.reset_to_start_duration_s: float = reset_to_start_duration_s
        self.initial_reset_duration_s: float = initial_reset_duration_s
        # Gripper
        # self.gripper_width_m: npt.NDArray[np.float64] = np.array([gripper_width_m])

        # Waiting duration parameters
        self.waiting_after_resetting_to_back_duration_s: float = (
            waiting_after_resetting_to_back_duration_s
        )
        self.waiting_before_dragging_duration_s: float = (
            waiting_before_dragging_duration_s
        )
        self.waiting_until_sliding_stop_duration_s: float = (
            waiting_until_sliding_stop_duration_s
        )
        self.waiting_after_lifting_duration_s: float = waiting_after_lifting_duration_s

        # State tracking
        self.interpolator: ActionInterpolator | None = None
        self.phase: str = "reset_to_back"  # Always start with reset_to_back
        self.last_commanded_pos: npt.NDArray[np.float64] | None = None
        self.last_actual_pos: npt.NDArray[np.float64] | None = None
        self.last_timestamp: float | None = None
        """
        Phase execution flow (same for all trials):

        TRIAL LOOP (repeats for each velocity in drag_vels_m_per_s):
          1. reset_to_back - Move to reset_to_back_xyz_wxyz
             (First trial: from unknown position; subsequent: from previous reset_to_start)
          2. waiting_after_resetting_to_back - Wait at drag_start_pose for waiting_after_resetting_to_back_duration_s
          3. reset_to_start - Move from reset_to_back to drag_start_pose (home)
          4. waiting_before_dragging - Wait at drag_start_pose for waiting_before_dragging_duration_s
          5. dragging - Accelerate to drag_vel, maintain for final_speed_step_num steps
          6. slowing_down - Decelerate from drag_vel to stop at slow_down_end_xyz_wxyz
          7. waiting_until_sliding_stop - Wait for cube to settle
          8. lift_arm_up - Retract to lift_arm_up_xyz_wxyz
             Loop back to step 1

        FINAL:
          8. completed - All trials done, stay at home position
        """
        self.phase_start_cnt: int = 0
        self.trial_cnt: int = 0

    def reset(self, episode_config: dict[str, Any] | None = None) -> None:
        """Reset agent state at the beginning of each episode."""
        self.phase = "reset_to_back"
        self.phase_start_cnt = 0
        self.trial_cnt = 0
        self.interpolator = None

    @override
    def predict_actions(
        self,
        observations: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.uint8]],
        history_actions: dict[str, npt.NDArray[np.float64]],
    ) -> dict[str, npt.NDArray[np.float64]]:

        self.phase_start_cnt += 1

        current_robot_pose: npt.NDArray[np.float64] = observations[
            "robot0_eef_xyz_wxyz"
        ][-1]
        # current_gripper_width: npt.NDArray[np.float64] = observations[
        #     "robot0_gripper_width"
        # ][-1]

        tcp_xyz_wxyz: npt.NDArray[np.float64] = current_robot_pose.copy()
        # gripper_width: npt.NDArray[np.float64] = current_gripper_width.copy()

        if self.phase == "reset_to_back":
            # Move to reset_to_back position
            # First trial: from current (unknown) position
            # Subsequent trials: from previous lift_arm_up position

            if self.interpolator is None:
                # Use longer duration for first trial (from unknown position)
                if self.trial_cnt == 0:
                    dist = np.linalg.norm(
                        current_robot_pose[:3] - self.reset_to_back_xyz_wxyz[:3]
                    )
                    print(f"dist: {dist}")
                    if dist < 0.03:
                        reset_duration = (
                            0.2  # The robot is already at the reset_to_back position
                        )
                    else:
                        reset_duration = (
                            self.initial_reset_duration_s - 0.2
                        )  # Subtract 0.2s to account for the shorter duration
                else:
                    reset_duration = self.reset_to_back_duration_s

                pos_speed_m_per_s = float(
                    np.linalg.norm(
                        self.reset_to_back_xyz_wxyz[:3] - current_robot_pose[:3]
                    )
                    / reset_duration
                )

                self.interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=current_robot_pose,
                    end_pose_xyz_wxyz=self.reset_to_back_xyz_wxyz,
                    start_gripper_width=np.array([0.0]),
                    end_gripper_width=np.array([0.0]),
                    pos_speed_m_per_s=pos_speed_m_per_s,
                    rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                    gripper_speed_m_per_s=1.0,
                )

            tcp_xyz_wxyz, gripper_width = self.interpolator.interpolate(
                1 / self.agent_update_freq_hz
            )

            if self.interpolator.is_finished:
                self.phase = "waiting_after_resetting_to_back"
                self.phase_start_cnt = 0
                self.interpolator = None

        elif self.phase == "waiting_after_resetting_to_back":
            # Wait after resetting to back
            tcp_xyz_wxyz = self.reset_to_back_xyz_wxyz.copy()
            gripper_width = 0.0

            waiting_steps = int(
                self.waiting_after_resetting_to_back_duration_s
                * self.agent_update_freq_hz
            )
            if self.phase_start_cnt >= waiting_steps:
                self.phase = "reset_to_start"
                self.phase_start_cnt = 0

                pos_speed_m_per_s = float(
                    np.linalg.norm(
                        self.drag_start_pose_xyz_wxyz[:3]
                        - self.reset_to_back_xyz_wxyz[:3]
                    )
                    / self.reset_to_start_duration_s
                )

                self.interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=self.reset_to_back_xyz_wxyz,
                    end_pose_xyz_wxyz=self.drag_start_pose_xyz_wxyz,
                    start_gripper_width=np.array([0.0]),
                    end_gripper_width=np.array([0.0]),
                    pos_speed_m_per_s=pos_speed_m_per_s,
                    rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                    gripper_speed_m_per_s=1.0,
                )

        elif self.phase == "reset_to_start":
            # Move from reset_to_back to drag_start_pose (home)
            assert (
                self.interpolator is not None
            ), "Interpolator should be initialized in reset_to_back phase"

            tcp_xyz_wxyz, gripper_width = self.interpolator.interpolate(
                1 / self.agent_update_freq_hz
            )

            if self.interpolator.is_finished:
                self.phase = "waiting_before_dragging"
                self.phase_start_cnt = 0
                self.interpolator = None

        elif self.phase == "waiting_before_dragging":
            # Wait before starting to drag
            tcp_xyz_wxyz = self.drag_start_pose_xyz_wxyz.copy()
            gripper_width = 0.0

            waiting_steps = int(
                self.waiting_before_dragging_duration_s * self.agent_update_freq_hz
            )
            if self.phase_start_cnt >= waiting_steps:
                self.phase = "dragging"
                self.phase_start_cnt = 0

                if self.trial_cnt < len(self.drag_vels_m_per_s):
                    drag_vel = self.drag_vels_m_per_s[self.trial_cnt]
                else:
                    drag_vel = self.drag_vels_m_per_s[-1]

                self.interpolator = FinalSpeedActionInterpolator(
                    start_pose_xyz_wxyz=self.drag_start_pose_xyz_wxyz,
                    end_pose_xyz_wxyz=self.drag_end_xyz_wxyz,
                    start_gripper_width=np.array([0.0]),
                    end_gripper_width=np.array([0.0]),
                    final_speed_m_per_s=drag_vel,
                    dt=1 / self.agent_update_freq_hz,
                    final_speed_step_num=self.final_speed_step_num,
                )

        elif self.phase == "dragging":
            assert (
                self.interpolator is not None
            ), "Interpolator should be already initialized in waiting_before_dragging phase"

            tcp_xyz_wxyz, gripper_width = self.interpolator.interpolate(
                1 / self.agent_update_freq_hz
            )
            print(f"Dragging: {tcp_xyz_wxyz[3:]}")

            if self.interpolator.is_finished:
                self.phase = "slowing_down"
                self.phase_start_cnt = 0
                self.interpolator = None

                if self.trial_cnt < len(self.drag_vels_m_per_s):
                    drag_vel = self.drag_vels_m_per_s[self.trial_cnt]
                else:
                    drag_vel = self.drag_vels_m_per_s[-1]

                self.interpolator = QuadraticActionInterpolator(
                    start_pose_xyz_wxyz=self.drag_end_xyz_wxyz,
                    end_pose_xyz_wxyz=self.slow_down_end_xyz_wxyz,
                    start_gripper_width=np.array([0.0]),
                    end_gripper_width=np.array([0.0]),
                    start_speed_m_per_s=drag_vel,
                    final_speed_m_per_s=0.0,
                )

        elif self.phase == "slowing_down":
            assert self.interpolator is not None

            tcp_xyz_wxyz, gripper_width = self.interpolator.interpolate(
                1 / self.agent_update_freq_hz
            )

            if self.interpolator.is_finished:
                self.phase = "waiting_until_sliding_stop"
                self.phase_start_cnt = 0
                self.interpolator = None

        elif self.phase == "waiting_until_sliding_stop":
            tcp_xyz_wxyz = self.slow_down_end_xyz_wxyz.copy()

            waiting_steps = int(
                self.waiting_until_sliding_stop_duration_s * self.agent_update_freq_hz
            )
            if self.phase_start_cnt >= waiting_steps:
                self.phase = "lift_arm_up"
                self.phase_start_cnt = 0

                pos_speed_m_per_s = float(
                    np.linalg.norm(
                        self.lift_arm_up_xyz_wxyz[:3] - self.slow_down_end_xyz_wxyz[:3]
                    )
                    / self.lift_arm_up_duration_s
                )

                self.interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=self.slow_down_end_xyz_wxyz,
                    end_pose_xyz_wxyz=self.lift_arm_up_xyz_wxyz,
                    start_gripper_width=np.array([0.0]),
                    end_gripper_width=np.array([0.0]),
                    pos_speed_m_per_s=pos_speed_m_per_s,
                    rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                    gripper_speed_m_per_s=1.0,
                )

        elif self.phase == "lift_arm_up":
            assert (
                self.interpolator is not None
            ), "Interpolator should be initialized in lift_arm_up phase"

            tcp_xyz_wxyz, gripper_width = self.interpolator.interpolate(
                1 / self.agent_update_freq_hz
            )

            if self.interpolator.is_finished:
                # Check if all trials are completed
                self.phase = "waiting_after_lifting"
                self.phase_start_cnt = 0
                self.interpolator = None

        elif self.phase == "waiting_after_lifting":
            tcp_xyz_wxyz = self.lift_arm_up_xyz_wxyz.copy()
            gripper_width = 0.0

            waiting_steps = int(
                self.waiting_after_lifting_duration_s * self.agent_update_freq_hz
            )
            if self.phase_start_cnt >= waiting_steps:
                self.trial_cnt += 1
                if self.trial_cnt >= len(self.drag_vels_m_per_s):
                    self.phase = "completed"
                    self.phase_start_cnt = 0
                    self.interpolator = None
                else:
                    # Continue to next trial: reset_to_back
                    self.phase = "reset_to_back"
                    self.phase_start_cnt = 0
                    self.interpolator = None

        elif self.phase == "completed":
            # All trials completed, stay at the current position
            # Use current robot pose to avoid commanding a jump
            tcp_xyz_wxyz = current_robot_pose.copy()

        else:
            raise ValueError(f"Unknown phase: {self.phase}")

        # Debug: Print position delta, actual dt, and commanded velocity
        # current_timestamp = time.monotonic()
        # if self.last_commanded_pos is not None and self.last_timestamp is not None and self.last_actual_pos is not None:
        #     cmd_pos_delta = np.linalg.norm(tcp_xyz_wxyz[:3] - self.last_commanded_pos[:3])
        #     actual_pos_delta = np.linalg.norm(current_robot_pose[:3] - self.last_actual_pos[:3])
        #     actual_dt = current_timestamp - self.last_timestamp
        #     expected_dt = 1 / self.agent_update_freq_hz
        #     commanded_velocity = cmd_pos_delta / expected_dt
        #     actual_velocity = actual_pos_delta / actual_dt if actual_dt > 0 else 0.0
        #     # print(
        #     #     f"Phase: {self.phase:20s} | "
        #     #     f"expected_dt: {expected_dt:.6f}s | "
        #     #     f"actual_dt: {actual_dt:.6f}s | "
        #     #     f"cmd_pos_delta: {cmd_pos_delta:.6f}m | "
        #     #     f"actual_pos_delta: {actual_pos_delta:.6f}m | "
        #     #     f"cmd_vel: {commanded_velocity:.4f}m/s | "
        #     #     f"actual_vel: {actual_velocity:.4f}m/s"
        #     # )
        # self.last_commanded_pos = tcp_xyz_wxyz.copy()
        # self.last_actual_pos = current_robot_pose.copy()
        # self.last_timestamp = current_timestamp
        # print(f"phase: {self.phase} | predict_actions: {tcp_xyz_wxyz[:3]} | is_done: {self.phase == 'completed'}")
        return {
            "action0_eef_xyz_wxyz": tcp_xyz_wxyz[np.newaxis, :],
            "is_done": self.phase == "completed",
            # "action0_gripper_width": gripper_width[np.newaxis, :],
        }
