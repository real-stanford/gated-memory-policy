from typing import Any

import numpy as np

from env.modules.agents.base_agent import BaseAgent
from env.modules.common import data_buffer_type


class HeuristicAgent(BaseAgent):
    def __init__(
        self,
        position_speed_m_per_s: float,
        rotation_speed_rad_per_s: float,
        gripper_speed_m_per_s: float,
        seed: int,
        **kwargs,
    ):
        super().__init__(
            action_prediction_horizon=1,
            action_execution_horizon=1,
            obs_history_len=1,
            action_history_len=1,
            image_obs_frames_ids=[],
            proprio_obs_frames_ids=[-1],
            **kwargs,
        )
        self.position_speed_m_per_s: float = position_speed_m_per_s
        self.rotation_speed_rad_per_s: float = rotation_speed_rad_per_s
        self.gripper_speed_m_per_s: float = gripper_speed_m_per_s
        self.seed: int = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def predict_actions(
        self,
        robots_obs: data_buffer_type,
        env_objs_obs: data_buffer_type,
        history_actions: data_buffer_type,
    ) -> data_buffer_type:
        raise NotImplementedError

    def reset(self, episode_config: dict[str, Any] | None = None):
        raise NotImplementedError
