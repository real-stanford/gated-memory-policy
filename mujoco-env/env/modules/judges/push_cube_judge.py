from typing import Any, cast

import numpy as np
import numpy.typing as npt

from env.modules.common import castf64, robot_data_type
from env.modules.judges.base_judge import BaseJudge
from loguru import logger


class PushCubeJudge(BaseJudge):
    def __init__(
        self,
        target_pos_xy: list[float],
        init_pos_xy: list[float],
        center_tolerance_m: float,
        object_stable_threshold_m_per_s: float,
        total_trial_num: int,
        continual_successful_trial_num: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_pos_xy: npt.NDArray[np.float64] = np.array(target_pos_xy)
        self.init_pos_xy: npt.NDArray[np.float64] = np.array(init_pos_xy)
        self.center_tolerance_m: float = center_tolerance_m
        self.object_stable_threshold_m_per_s: float = object_stable_threshold_m_per_s
        self.total_trial_num: int = total_trial_num
        self.continual_successful_trial_num: int = continual_successful_trial_num

        # States
        self.current_trial_num: int = 0
        self.cube_status: str = "resetting"  # moving, stopped
        self.stop_in_target_zone: bool = False
        self.time_limit_reached: bool = False
        self.last_object_pose: npt.NDArray[np.float64] | None = None
        self.time_since_finished_s: float | None = None
        self.finished_timestamp: float | None = None
        self.last_trial_timestamp: float = 0.0
        self.trials_successful: list[bool] = []

    def reset(self, episode_config: dict[str, Any] | None):

        self.cube_status = "resetting"
        self.stop_in_target_zone = False
        self.current_trial_num = 0
        self.last_object_pose = None
        self.time_since_finished_s = None
        self.finished_timestamp = None
        self.time_limit_reached = False
        self.last_trial_timestamp = 0.0
        self.trials_successful = []

    def update(
        self,
        robots_obs: list[robot_data_type],
        env_objs_obs: list[robot_data_type],
    ):
        object_pose = castf64(env_objs_obs[1]["object_pose_xyz_wxyz"])

        current_timestamp = cast(float, env_objs_obs[0]["timestamp"])

        if self.finished_timestamp is not None:
            self.time_since_finished_s = current_timestamp - self.finished_timestamp
        if current_timestamp - self.last_trial_timestamp > self.time_limit_s:
            self.time_limit_reached = True

        if self.last_object_pose is not None:
            object_vel = object_pose[:2] - self.last_object_pose[:2]
            object_speed = np.linalg.norm(object_vel)
            object_stable = object_speed < self.object_stable_threshold_m_per_s
            if self.cube_status == "resetting":
                if not object_stable and object_pose[1] > self.init_pos_xy[1] + 0.1:
                    self.cube_status = "moving"

            elif self.cube_status == "moving":
                if object_stable and object_pose[1] > self.init_pos_xy[1] + 0.1:
                    self.cube_status = "stopped"
                    self.current_trial_num += 1
                    self.last_trial_timestamp = current_timestamp
                    max_dist = np.max(np.abs(object_pose[:2] - self.target_pos_xy))
                    logger.info(
                        f"current_trial_num: {self.current_trial_num} object_y: {object_pose[1]: .4f}"
                    )
                    if max_dist < self.center_tolerance_m:
                        self.stop_in_target_zone = True
                        self.trials_successful.append(True)
                    else:
                        self.trials_successful.append(False)
                    if self.current_trial_num >= self.total_trial_num:
                        # All the trials are done
                        self.finished_timestamp = current_timestamp

            elif self.cube_status == "stopped":
                if np.max(np.abs(object_pose[:2] - self.init_pos_xy)) < 0.1:
                    self.cube_status = "resetting"
                    self.stop_in_target_zone = False
        self.last_object_pose = object_pose

    def get_reward(self) -> float:
        if len(self.trials_successful) == self.total_trial_num:
            if self.continual_successful_trial_num == 0:
                # Report average success rate
                return np.sum(self.trials_successful) / self.total_trial_num

            if (
                np.sum(self.trials_successful[-self.continual_successful_trial_num :])
                == self.continual_successful_trial_num
            ):
                # Last 3 trials are all successful
                return 1.0
        return 0.0

    def get_done(self) -> bool:
        if self.time_limit_reached:
            return True
        if (
            self.time_since_finished_s is not None
            and self.time_since_finished_s > self.prolong_after_success_s
        ):
            return True
        return False

    def get_states(self) -> dict[str, Any]:
        return {
            "cube_status": self.cube_status,
            "stop_in_target_zone": self.stop_in_target_zone,
            "current_trial_num": self.current_trial_num,
            "trials_successful": self.trials_successful,
            "total_trial_num": self.total_trial_num,
        }
