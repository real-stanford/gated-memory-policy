from typing import Any

import numpy as np
import numpy.typing as npt

from real_env.agents.base_agent import BaseAgent


class HeuristicAgent(BaseAgent):
    """
    Heuristic agent base class.
    """

    def __init__(
        self,
        position_speed_m_per_s: float,
        rotation_speed_rad_per_s: float,
        gripper_speed_m_per_s: float,
        seed: int,
        action_prediction_horizon: int,
        action_execution_horizon: int,
        proprio_history_len: int,
        action_history_len: int,
        **kwargs,
    ):
        super().__init__(
            action_prediction_horizon=action_prediction_horizon,
            action_execution_horizon=action_execution_horizon,
            proprio_history_len=proprio_history_len,
            action_history_len=action_history_len,
            image_obs_frames_ids=[],
            **kwargs,
        )
        self.position_speed_m_per_s: float = position_speed_m_per_s
        self.rotation_speed_rad_per_s: float = rotation_speed_rad_per_s
        self.gripper_speed_m_per_s: float = gripper_speed_m_per_s
        self.seed: int = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def predict_actions(
        self,
        observations: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.uint8]],
        history_actions: dict[str, npt.NDArray[np.float64]],
    ) -> dict[str, npt.NDArray[np.float64]]:
        raise NotImplementedError

    def reset(self, episode_config: dict[str, Any] | None = None) -> None:
        raise NotImplementedError
