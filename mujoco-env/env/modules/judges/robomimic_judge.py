from typing import Any, Optional

from env.modules.common import robot_data_type
from env.modules.judges.base_judge import BaseJudge


class RobomimicJudge(BaseJudge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.done: bool = False
        self.reward: float = 0.0
        self.max_reward: float = 0.0
        self.time_limit_reached: bool = False
        self.finished_timestamp: Optional[float] = None
        self.time_since_finished_s: Optional[float] = None

    def update(
        self,
        robots_obs: list[robot_data_type],
        env_objs_obs: list[robot_data_type],
    ):
        self.done = env_objs_obs[0]["done"]
        self.reward = env_objs_obs[0]["reward"]
        self.max_reward = max(self.max_reward, self.reward)
        if (self.done or self.reward == 1.0) and self.finished_timestamp is None:
            self.finished_timestamp = env_objs_obs[0]["timestamp"]

        if self.finished_timestamp is not None:
            self.time_since_finished_s = (
                env_objs_obs[0]["timestamp"] - self.finished_timestamp
            )

        if env_objs_obs[0]["timestamp"] > self.time_limit_s:
            self.time_limit_reached = True

    def reset(self, episode_config: dict[str, Any] | None = None):
        self.done = False
        self.reward = 0.0
        self.max_reward = 0.0
        self.time_limit_reached = False
        self.time_since_finished_s = None
        self.finished_timestamp = None

    def get_reward(self) -> float:
        return self.max_reward

    def get_done(self) -> bool:
        if self.time_limit_reached:
            return True

        if self.time_since_finished_s is not None:
            if self.time_since_finished_s > self.prolong_after_success_s:
                return True

        return False

    def get_states(self) -> dict[str, Any]:
        return {
            "done": self.done,
            "reward": self.reward,
            "max_reward": self.max_reward,
        }
