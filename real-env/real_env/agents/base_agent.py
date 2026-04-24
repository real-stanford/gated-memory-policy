from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        robot_num: int,
        agent_update_freq_hz: float,
        action_prediction_horizon: int,
        action_execution_horizon: int,
        image_history_len: int,
        proprio_history_len: int,
        action_history_len: int,
        **kwargs,
    ):
        print(f"[Warning] BaseAgent unused kwargs: {kwargs}")
        self.name: str = name
        self.robot_num: int = robot_num
        self.agent_update_freq_hz: float = agent_update_freq_hz
        self.action_prediction_horizon: int = action_prediction_horizon
        self.action_execution_horizon: int = action_execution_horizon
        self.image_history_len: int = image_history_len
        self.proprio_history_len: int = proprio_history_len
        self.action_history_len: int = action_history_len

    @abstractmethod
    def predict_actions(
        self,
        observations: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.uint8]],
        history_actions: dict[str, npt.NDArray[np.float64]],
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Return the actions for the robots based on the observations
        observations: {
            "robot0_main_camera": (N, H, W, 3),
            "robot0_ultrawide_camera": (N, H, W, 3),
            "robot0_depth_camera": (N, H, W, 1),
            "robot0_eef_xyz_wxyz": (N, 7),
            "robot0_gripper_width": (N, 1),
            "timestamps": (N, ),
        }
        """
        ...

    @abstractmethod
    def reset(self):
        ...
