from typing import Any

import numpy as np
import numpy.typing as npt

from env.modules.common import data_buffer_type
from loguru import logger


class BaseAgent:
    def __init__(
        self,
        robot_num: int,
        agent_update_freq_hz: float,
        action_prediction_horizon: int,
        action_execution_horizon: int,
        obs_history_len: int,
        action_history_len: int,
        image_obs_frames_ids: list[int], # Aligned with render_image_indices in the task config
        proprio_obs_frames_ids: list[int],
        movement_constraints: list[str] | None = None,
        default_actions: list[list[float]] | None = None,
        move_gripper: bool = True,
        default_gripper_widths: list[float] | None = None,
        **kwargs,
    ):
        self.robot_num: int = robot_num
        self.agent_update_freq_hz: float = agent_update_freq_hz
        self.action_prediction_horizon: int = action_prediction_horizon
        self.action_execution_horizon: int = action_execution_horizon
        self.obs_history_len: int = obs_history_len
        self.action_history_len: int = action_history_len
        self.image_obs_frames_ids: list[int] = image_obs_frames_ids
        self.proprio_obs_frames_ids: list[int] = proprio_obs_frames_ids

        assert all(
            k < 0 and k >= -self.obs_history_len for k in self.image_obs_frames_ids
        ), f"Image observation frame id should be in the range of [-{self.obs_history_len}, -1], but got {self.image_obs_frames_ids}"
        assert all(
            k < 0 and k >= -self.obs_history_len for k in self.proprio_obs_frames_ids
        ), f"Proprio observation frame id should be in the range of [-{self.obs_history_len}, -1], but got {self.proprio_obs_frames_ids}"

        movement_constraint_keys = ["x", "y", "z", "qw", "qx", "qy", "qz"]
        self.disabled_movement_mask: npt.NDArray[np.bool_] = np.ones(
            len(movement_constraint_keys), dtype=np.bool_
        )
        if movement_constraints is not None:
            assert all(
                constraint in movement_constraint_keys
                for constraint in movement_constraints
            )
            for idx, key in enumerate(movement_constraint_keys):
                if key in movement_constraints:
                    self.disabled_movement_mask[idx] = False
            assert default_actions is not None
            assert (
                len(default_actions) == self.robot_num
            ), f"default_actions should have {self.robot_num} actions, but got {len(default_actions)}"
            assert len(default_actions[0]) == len(movement_constraint_keys)
            self.default_actions = np.array(default_actions)
        else:
            self.disabled_movement_mask[:] = False
            self.default_actions = np.zeros(
                (self.robot_num, len(movement_constraint_keys))
            )

        self.move_gripper: bool = move_gripper
        if default_gripper_widths is not None:
            assert len(default_gripper_widths) == self.robot_num
            self.default_gripper_widths: npt.NDArray[np.float64] = np.array(
                default_gripper_widths
            )
        else:
            self.default_gripper_widths = np.zeros(self.robot_num)

        logger.info(f"BaseAgent redundant kwargs: {kwargs}")

    def predict_actions(
        self,
        robots_obs: data_buffer_type,
        env_objs_obs: data_buffer_type,
        history_actions: data_buffer_type,
    ) -> data_buffer_type:
        """Return the actions for the robots based on the observations
        Args:
            observation: The observations of the robots, shape: (obs_history_len, robot_num, ) where each item is a dictionary with specific keys
        Returns:
            actions: The actions for the robots, shape: (action_horizon, robot_num, ) where each item is a dictionary with specific keys
        """

        raise NotImplementedError

    def reset(self, episode_config: dict[str, Any] | None = None):
        raise NotImplementedError
