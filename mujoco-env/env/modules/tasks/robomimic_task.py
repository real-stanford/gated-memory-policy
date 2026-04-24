from typing import Any, Optional

from mujoco import viewer as mjviewer

from env.modules.tasks.base_task import BaseTask
from env.modules.tasks.parallel_task import ParallelTask


class RobomimicTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process_episode_config(self, episode_config: dict[str, Any]) -> dict[str, Any]:
        return episode_config
    
    def reset(self, episode_config: Optional[dict[str, Any]] = None):
        super().reset(episode_config)
        if self.use_viewer:
            self.mj_viewer.close()
            self.mj_viewer = mjviewer.launch_passive(self.env.model, self.env.data)


class RobomimicParallel(ParallelTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process_episode_config(self, episode_config: dict[str, Any]) -> dict[str, Any]:
        return episode_config
