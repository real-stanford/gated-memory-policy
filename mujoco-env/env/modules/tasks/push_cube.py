from typing import Any, Optional

import numpy as np

from env.modules.agents.heuristic_agents.push_cube_agent import PushCubeAgent
from env.modules.tasks.base_task import BaseTask
from env.modules.tasks.parallel_task import ParallelTask
from robot_utils.logging_utils import print_once
from loguru import logger


def _process_episode_config(
    self: "PushCube | PushCubeParallel", episode_config: dict[str, Any]
) -> dict[str, Any]:

    if "seed" not in episode_config:
        episode_config["seed"] = 0
        print_once("seed is not provided, will be set to 0")

    self.rng = np.random.default_rng(episode_config["seed"])

    if "sliding_friction" not in episode_config:

        # Sample sliding friction based on the quadratic relationship
        # This matches the uniform distribution of the optimal velocities
        min_friction_sqrt = self.sliding_friction_min**0.5
        max_friction_sqrt = self.sliding_friction_max**0.5
        episode_config["sliding_friction"] = (
            self.rng.uniform(
                min_friction_sqrt,
                max_friction_sqrt,
            )
            ** 2
        )

    return episode_config


class PushCube(BaseTask):
    def __init__(
        self,
        sampled_vels_m_per_s: list[float],
        sampled_sliding_frictions: list[float],
        sliding_friction_min: float,
        sliding_friction_max: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampled_vels_m_per_s = np.array(sampled_vels_m_per_s)
        self.sampled_sliding_frictions = np.array(sampled_sliding_frictions)
        self.sliding_friction_min = sliding_friction_min
        self.sliding_friction_max = sliding_friction_max

        assert len(self.sampled_vels_m_per_s) == len(
            self.sampled_sliding_frictions
        ), f"{len(self.sampled_vels_m_per_s)=}, {len(self.sampled_sliding_frictions)=}"

        trial_max_step = 0
        while True:
            trial_max_step += 1
            if 2**trial_max_step - 1 > len(self.sampled_vels_m_per_s):
                raise ValueError(
                    f"Number of sampled vels should be 2**trial_num - 1, but got {len(self.sampled_vels_m_per_s)}"
                )
            if 2**trial_max_step - 1 == len(self.sampled_vels_m_per_s):
                break
        self.trial_max_step = trial_max_step

    def get_vel_sequence(self, new_sliding_friction: float):
        """
        Given a new sliding friction, return a sequence of velocities to try.
        The sequence should be according to the trial process of a binary search
        """
        closest_idx = np.argmin(
            np.abs(self.sampled_sliding_frictions - new_sliding_friction)
        )
        if self.sliding_friction_min == self.sliding_friction_max:
            # Fixed friction. Will only try the closest velocity
            trial_indices = [closest_idx]
        else:
            trial_indices = []
            current_step_size = 2 ** (self.trial_max_step - 1)
            trial_idx = current_step_size - 1
            # e.g. if there are 31 velocities in total, the first trial index should be 15
            # So that there are 15 velocities on both sides of the current trial index
            trial_indices.append(trial_idx)

            while trial_idx != closest_idx:
                current_step_size = current_step_size // 2  # e.g. 8, 4, 2, 1
                if trial_idx > closest_idx:
                    trial_idx -= current_step_size
                else:
                    trial_idx += current_step_size
                trial_indices.append(trial_idx)
        logger.info(f"trial_indices: {trial_indices}")
        return self.sampled_vels_m_per_s[trial_indices]

    def reset(self, episode_config: Optional[dict[str, Any]] = None):
        super().reset(episode_config)

        if "sliding_friction" in self.episode_config:
            if isinstance(self.agent, PushCubeAgent):
                self.agent.push_vels_m_per_s = self.get_vel_sequence(
                    self.episode_config["sliding_friction"]
                )
                logger.info(
                    f"sliding_friction: {self.episode_config['sliding_friction']}"
                )
                logger.info(f"push_vels_m_per_s: {self.agent.push_vels_m_per_s}")


PushCube._process_episode_config = _process_episode_config


class PushCubeParallel(ParallelTask):
    def __init__(
        self,
        sliding_friction_min: float,
        sliding_friction_max: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sliding_friction_min = sliding_friction_min
        self.sliding_friction_max = sliding_friction_max


PushCubeParallel._process_episode_config = _process_episode_config
