from typing import Any

from env.modules.common import robot_data_type


class BaseJudge:
    def __init__(self, prolong_after_success_s: float, time_limit_s: float):
        self.prolong_after_success_s: float = prolong_after_success_s
        self.time_limit_s: float = time_limit_s

    def update(
        self,
        robots_obs: list[robot_data_type],
        env_objs_obs: list[robot_data_type],
    ):
        raise NotImplementedError

    def reset(self, episode_config: dict[str, Any] | None = None):
        raise NotImplementedError

    def get_reward(self) -> float:
        raise NotImplementedError

    def get_done(self) -> bool:
        raise NotImplementedError

    def get_states(self) -> dict[str, Any]:
        raise NotImplementedError
