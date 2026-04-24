from typing import Any

from env.modules.common import robot_data_type
from env.modules.judges.base_judge import BaseJudge


class PressButtonJudge(BaseJudge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, episode_config: dict[str, Any]):
        pass

    def update(
        self,
        robots_obs: list[robot_data_type],
        env_objs_obs: list[robot_data_type],
    ):
        pass

    def get_reward(self) -> bool:
        return False

    def get_done(self) -> bool:
        return False

    def get_states(self) -> dict[str, Any]:
        return {}
