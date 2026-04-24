from typing import Any

import numpy as np
import numpy.typing as npt

from env.modules.agents.heuristic_agents.fling_cloth_agent import FlingClothAgent
from env.modules.tasks.base_task import BaseTask
from env.modules.tasks.parallel_task import ParallelTask
from loguru import logger


def _process_episode_config(
    self: "FlingCloth | FlingClothParallel", episode_config: dict[str, Any]
) -> dict[str, Any]:

    self.rng = np.random.default_rng(episode_config["seed"])

    if "cloth_mass" not in episode_config:
        if self.sampled_masses_only:
            episode_config["cloth_mass"] = self.rng.choice(self.sampled_cloth_masses)
            if self.mass_perturbation_std > 0:
                episode_config["cloth_mass"] += self.rng.normal(
                    0, self.mass_perturbation_std
                )
        else:
            min_mass = self.cloth_mass_min
            max_mass = self.cloth_mass_max
            episode_config["cloth_mass"] = self.rng.uniform(min_mass, max_mass)

    if (
        "speed_scales" not in episode_config
        and isinstance(self.agent, FlingClothAgent)
        and isinstance(self, FlingCloth)
    ):
        episode_config["speed_scales"] = self.get_speed_scale_sequence(
            episode_config["cloth_mass"]
        )

    return episode_config


class FlingCloth(BaseTask):
    def __init__(
        self,
        sampled_speed_scales: list[float],
        sampled_cloth_masses: list[float],
        cloth_mass_min: float,
        cloth_mass_max: float,
        sampled_masses_only: bool,
        mass_perturbation_std: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(sampled_speed_scales) == len(sampled_cloth_masses)
        self.sampled_speed_scales: npt.NDArray[np.float64] = np.array(
            sampled_speed_scales
        )
        self.sampled_cloth_masses: npt.NDArray[np.float64] = np.array(
            sampled_cloth_masses
        )
        self.cloth_mass_min: float = cloth_mass_min
        self.cloth_mass_max: float = cloth_mass_max
        self.sampled_masses_only: bool = sampled_masses_only
        self.mass_perturbation_std: float = mass_perturbation_std

        trial_max_step = 0
        while True:
            trial_max_step += 1
            if 2**trial_max_step - 1 > len(self.sampled_cloth_masses):
                raise ValueError(
                    f"Number of sampled vels should be 2**trial_num - 1, but got {len(self.sampled_cloth_masses)}"
                )
            if 2**trial_max_step - 1 == len(self.sampled_cloth_masses):
                break
        self.trial_max_step = trial_max_step

    def get_speed_scale_sequence(self, new_cloth_mass: float):
        closest_idx = np.argmin(np.abs(self.sampled_cloth_masses - new_cloth_mass))

        trial_indices: list[int] = []
        current_step_size = 2 ** (self.trial_max_step - 1)
        trial_idx = current_step_size - 1
        trial_indices.append(trial_idx)
        while trial_idx != closest_idx:
            current_step_size = current_step_size // 2  # e.g. 4, 2, 1
            if trial_idx > closest_idx:
                trial_idx -= current_step_size
            else:
                trial_idx += current_step_size
            trial_indices.append(trial_idx)

        logger.info(f"trial_indices: {trial_indices}")
        return self.sampled_speed_scales[trial_indices].tolist()


FlingCloth._process_episode_config = _process_episode_config


class FlingClothParallel(ParallelTask):
    def __init__(
        self,
        sampled_cloth_masses: list[float],
        cloth_mass_min: float,
        cloth_mass_max: float,
        sampled_masses_only: bool,
        mass_perturbation_std: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampled_cloth_masses: npt.NDArray[np.float64] = np.array(
            sampled_cloth_masses
        )
        self.cloth_mass_min: float = cloth_mass_min
        self.cloth_mass_max: float = cloth_mass_max
        self.sampled_masses_only: bool = sampled_masses_only
        self.mass_perturbation_std: float = mass_perturbation_std


FlingClothParallel._process_episode_config = _process_episode_config
