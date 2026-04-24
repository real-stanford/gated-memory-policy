import enum
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.spatial.transform as st


@dataclass
class Trajectory:
    data: npt.NDArray[np.float64]
    timestamps: npt.NDArray[np.float64]

    def __post_init__(self):
        if isinstance(self.data, list):
            self.data = np.array(self.data)
        if isinstance(self.timestamps, list):
            self.timestamps = np.array(self.timestamps)

        assert isinstance(self.data, np.ndarray)
        assert isinstance(self.timestamps, np.ndarray)
        assert self.data.ndim == 2, f"{self.data.ndim=}"
        assert self.timestamps.ndim == 1, f"{self.timestamps.ndim=}"
        assert (
            self.data.shape[0] == self.timestamps.shape[0]
        ), f"{self.data.shape=}, {self.timestamps.shape=}"

        if len(self.timestamps) > 1:
            assert np.all(
                self.timestamps[1:] > self.timestamps[:-1]
            ), "Timestamps should be strictly increasing"


class ControllerCommand(enum.IntEnum):
    RESET = 0
    CONNECT = 1
    DISCONNECT = 2
    SCHEDULE_JOINT_TRAJ = 3
    SCHEDULE_EEF_TRAJ = 4
    START_RECORDING = 5
    STOP_RECORDING = 6


class CameraCommand(enum.IntEnum):
    RESET = 0
    START_RECORDING = 1
    STOP_RECORDING = 2
