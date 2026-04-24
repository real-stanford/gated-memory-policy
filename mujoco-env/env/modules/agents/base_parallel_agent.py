
from env.modules.agents.base_agent import BaseAgent
from env.modules.common import data_buffer_type


class BaseParallelAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict_actions_parallel(
        self, episode_buffers_dict: dict[int, dict[str, data_buffer_type]]
    ) -> dict[int, data_buffer_type]:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
