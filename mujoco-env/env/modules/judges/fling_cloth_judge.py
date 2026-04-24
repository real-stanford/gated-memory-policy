from typing import Any, cast

import numpy as np
import numpy.typing as npt

from env.modules.common import castf64, robot_data_type
from env.modules.judges.base_judge import BaseJudge
from loguru import logger


class FlingClothJudge(BaseJudge):
    def __init__(
        self,
        cloth_edge_threshold_x: float,  # The edge of black line that is closer to the robot
        grid_num_threshold: int,
        fling_up_threshold_z: float,
        fling_down_threshold_z: float,
        success_check_delay_s: float,
        continual_successful_trial_num: int,
        total_trial_num: int,
        horizontal_vertex_num: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Constants
        self.cloth_edge_threshold_x: float = cloth_edge_threshold_x
        self.grid_num_threshold: int = grid_num_threshold
        assert (
            fling_up_threshold_z > fling_down_threshold_z
        ), "fling_up_threshold_z must be greater than fling_down_threshold_z"
        self.fling_up_threshold_z: float = fling_up_threshold_z
        self.fling_down_threshold_z: float = fling_down_threshold_z
        self.success_check_delay_s: float = success_check_delay_s
        self.continual_successful_trial_num: int = continual_successful_trial_num
        """
        If continual_successful_trial_num is 0, the judge will report the average success rate.
        If continual_successful_trial_num is greater than 0, the judge will report 1.0 if the last continual_successful_trial_num trials are all successful.
        """
        self.total_trial_num: int = total_trial_num
        self.horizontal_vertex_num: int = horizontal_vertex_num

        # States
        self.current_trial_num: int = 0
        self.cloth_status: str = (
            "resetting"  # flinging_up, flinging_down, stopped, resetting
        )
        self.stopping_timestamp: float | None = None
        self.time_limit_reached: bool = False
        self.finished_timestamp: float | None = None
        self.last_cloth_pos: npt.NDArray[np.float64] | None = None
        self.time_since_finished_s: float | None = None
        self.last_trial_timestamp: float = 0.0
        self.trials_successful: list[bool] = []
        self.successful_grid_num: list[int] = []

    def reset(self, episode_config: dict[str, Any] | None):

        self.cloth_status = "resetting"
        self.current_trial_num = 0
        self.last_cloth_pos = None
        self.stopping_timestamp = None
        self.time_since_finished_s = None
        self.finished_timestamp = None
        self.time_limit_reached = False
        self.last_trial_timestamp = 0.0
        self.trials_successful = []
        self.successful_grid_num = []

    def update(
        self,
        robots_obs: list[robot_data_type],
        env_objs_obs: list[robot_data_type],
    ):
        cloth_grid_xyz = castf64(env_objs_obs[1]["pos_xyz"])

        current_timestamp = cast(float, env_objs_obs[0]["timestamp"])

        if self.finished_timestamp is not None:
            self.time_since_finished_s = current_timestamp - self.finished_timestamp
        if current_timestamp - self.last_trial_timestamp > self.time_limit_s:
            self.time_limit_reached = True

        if self.cloth_status == "resetting":
            if robots_obs[0]["tcp_xyz_wxyz"][2] > self.fling_down_threshold_z:
                self.cloth_status = "flinging_up"

        elif self.cloth_status == "flinging_up":
            if robots_obs[0]["tcp_xyz_wxyz"][2] > self.fling_up_threshold_z:
                self.cloth_status = "flinging_down"

        elif self.cloth_status == "flinging_down":
            if robots_obs[0]["tcp_xyz_wxyz"][2] < self.fling_down_threshold_z:
                self.cloth_status = "stopped"
                self.stopping_timestamp = current_timestamp

        elif self.cloth_status == "stopped":
            assert self.stopping_timestamp is not None, "stopping_timestamp is None"
            if current_timestamp - self.stopping_timestamp > self.success_check_delay_s:
                successful_grid_num = int(
                    np.sum(
                        cloth_grid_xyz[: self.horizontal_vertex_num, 0]
                        <= self.cloth_edge_threshold_x
                    )
                )  # x positive direction points to the robot
                logger.info(f"{self.current_trial_num=} {successful_grid_num=}")
                if successful_grid_num >= self.grid_num_threshold:
                    self.trials_successful.append(True)
                else:
                    self.trials_successful.append(False)
                self.successful_grid_num.append(successful_grid_num)
                self.last_trial_timestamp = current_timestamp
                self.stopping_timestamp = None
                self.current_trial_num += 1
                self.cloth_status = "resetting"

                if self.current_trial_num >= self.total_trial_num:
                    self.finished_timestamp = current_timestamp

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
            "cloth_status": self.cloth_status,
            "current_trial_num": self.current_trial_num,
            "trials_successful": self.trials_successful,
            "total_trial_num": self.total_trial_num,
            "successful_grid_num": self.successful_grid_num,
        }
