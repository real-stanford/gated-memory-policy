import json
import time
from abc import ABC
from typing import Any

import numpy as np
import numpy.typing as npt
from robotmq import RMQClient, deserialize, serialize

from real_env.common.data_classes import ControllerCommand
from robot_utils.pose_utils import interpolate_xyz_wxyz


class BaseControllerClient(ABC):
    def __init__(
        self,
        name: str,
        server_endpoint: str,
        data_remaining_time_s: float,
    ):
        self.server_endpoint: str = server_endpoint
        self.name: str = name
        self.rmq_client = RMQClient(
            client_name=name,
            server_endpoint=server_endpoint,
        )
        self.data_remaining_time_s: float = data_remaining_time_s

    def reset(self):
        self.rmq_client.put_data(
            topic="command",
            data=serialize(
                {
                    "command": ControllerCommand.RESET,
                }
            ),
        )

    def get_config(self) -> dict[str, Any]:
        raw_data, _ = self.rmq_client.peek_data(topic="config", n=-1)
        config_str = deserialize(raw_data[0])
        print(f"Config: {config_str}")
        return json.loads(config_str)


class BaseJointClient(BaseControllerClient):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.joint_pos: list[npt.NDArray[np.float64]] = []
        self.timestamps: list[float] = []
        self.dof: int

        print(f"Waiting for the {self.name} to start, endpoint: {self.server_endpoint}")
        while True:
            topic_status = self.rmq_client.get_topic_status(
                topic="joint_state",
                timeout_s=1.0,
            )
            if topic_status > 0:
                raw_data, _ = self.rmq_client.peek_data(
                    topic="joint_state",
                    n=1,
                )
                self.dof = len(deserialize(raw_data[0])["joint_pos"])
                print(f"{self.name} started. DOF: {self.dof}")
                break
            time.sleep(0.1)

    def get_joint_pos(self, timestamp: float):
        """
        Will apply pose interpolation to get the pose at the given timestamp.
        """

        # Fetch latest results
        raw_data, _ = self.rmq_client.pop_data(
            topic="joint_state",
            n=0,
            automatic_resend=False,  # Will kill the task if the robot is disconnected
        )
        for data in raw_data:
            data = deserialize(data)
            self.joint_pos.append(data["joint_pos"])
            if len(self.timestamps) > 0:
                assert (
                    data["timestamp"] > self.timestamps[-1]
                ), f"Timestamp is not increasing: {data['timestamp']} < {self.timestamps[-1]}"
            self.timestamps.append(data["timestamp"])

        # Apply interpolation
        if timestamp >= self.timestamps[-1]:
            result = self.joint_pos[-1]
        elif timestamp >= self.timestamps[0]:
            interpolate_idx = np.searchsorted(self.timestamps, timestamp, side="right")
            ratio = (timestamp - self.timestamps[interpolate_idx - 1]) / (
                self.timestamps[interpolate_idx] - self.timestamps[interpolate_idx - 1]
            )
            result = (
                self.joint_pos[interpolate_idx - 1] * (1 - ratio)
                + self.joint_pos[interpolate_idx] * ratio
            )
        else:
            print(
                f"{self.name} Timestamp is too old: {timestamp:.3f} < {self.timestamps[0]:.3f}. Please increase data_remaining_time_s."
            )
            result = self.joint_pos[0]

        # Remove old data
        old_idx = np.searchsorted(
            self.timestamps, timestamp - self.data_remaining_time_s, side="left"
        )
        if old_idx == len(self.timestamps):
            old_idx -= 1  # Always keep the last pose
        self.joint_pos = self.joint_pos[old_idx:]
        self.timestamps = self.timestamps[old_idx:]

        return result

    def schedule_joint_traj(
        self,
        joint_traj_pos: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
        use_relative_timestamps: bool,
    ):
        """
        args:
            joint_traj_pos: (N, dof)
            timestamps: (N, )
        """
        assert (
            joint_traj_pos.shape[1] == self.dof
        ), f"trajectory input should have shape (N, {self.dof}), but got {joint_traj_pos.shape}"
        assert (
            timestamps.shape[0] == joint_traj_pos.shape[0]
        ), f"Timestamps shape mismatch: {timestamps.shape[0]=}, {joint_traj_pos.shape[0]=}"

        raw_data_list, _ = self.rmq_client.pop_data(
            topic="error",
            n=0,
            automatic_resend=False,  # Will kill the task if the robot is disconnected
        )
        for data in raw_data_list:
            data = deserialize(data)
            print(f"Error: {data['error']}")
        if len(raw_data_list) > 0:
            raise RuntimeError(f"Joint traj scheduling failed")

        self.rmq_client.put_data(
            topic="command",
            data=serialize(
                {
                    "command": ControllerCommand.SCHEDULE_JOINT_TRAJ,
                    "data": joint_traj_pos,
                    "timestamps": timestamps,
                    "use_relative_timestamps": use_relative_timestamps,
                }
            ),
            automatic_resend=False,  # Will kill the task if the robot is disconnected
        )


class BaseCartesianClient(BaseControllerClient):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.eef_poses: list[npt.NDArray[np.float64]] = []
        self.timestamps: list[float] = []
        print(f"Waiting for the {self.name} to start, endpoint: {self.server_endpoint}")
        while True:
            topic_status = self.rmq_client.get_topic_status(
                topic="eef_pose",
                timeout_s=1.0,
            )
            if topic_status > 0:
                print(f"{self.name} started.")
                break
            time.sleep(0.1)

    def get_eef_xyz_wxyz(self, timestamp: float):
        """
        Will apply pose interpolation to get the pose at the given timestamp.
        """

        # Fetch latest results
        raw_data, _ = self.rmq_client.pop_data(
            topic="eef_pose",
            n=0,
            automatic_resend=False,  # Will kill the task if the robot is disconnected
        )

        for data in raw_data:
            data = deserialize(data)
            self.eef_poses.append(data["eef_xyz_wxyz"])
            if len(self.timestamps) > 0:
                assert (
                    data["timestamp"] > self.timestamps[-1]
                ), f"Timestamp is not increasing: {data['timestamp']} < {self.timestamps[-1]}"
            self.timestamps.append(data["timestamp"])

        # Apply interpolation
        if timestamp >= self.timestamps[-1]:
            result = self.eef_poses[-1]
        elif timestamp >= self.timestamps[0]:
            interpolate_idx = np.searchsorted(self.timestamps, timestamp, side="right")
            result = interpolate_xyz_wxyz(
                pose_left=self.eef_poses[interpolate_idx - 1],
                pose_right=self.eef_poses[interpolate_idx],
                timestamp_left=self.timestamps[interpolate_idx - 1],
                timestamp_right=self.timestamps[interpolate_idx],
                timestamp=timestamp,
            )
        else:
            print(
                f"{self.name} Timestamp is too old: {timestamp} < {self.timestamps[0]}. Please increase data_remaining_time_s."
            )
            result = self.eef_poses[0]

        # Remove old data
        old_idx = np.searchsorted(
            self.timestamps, timestamp - self.data_remaining_time_s, side="left"
        )
        if old_idx == len(self.timestamps):
            old_idx -= 1  # Always keep the last pose
        self.eef_poses = self.eef_poses[old_idx:]
        self.timestamps = self.timestamps[old_idx:]

        return result

    def schedule_eef_traj(
        self,
        eef_traj_xyz_wxyz: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
        use_relative_timestamps: bool,
    ):
        """
        args:
            eef_traj_xyz_wxyz: (N, 7)
            timestamps: (N, )
        """
        raw_data_list, _ = self.rmq_client.pop_data(
            topic="error",
            n=0,
            automatic_resend=False,  # Will kill the task if the robot is disconnected
        )
        for data in raw_data_list:
            data = deserialize(data)
            print(f"Error: {data['error']}")
        if len(raw_data_list) > 0:
            raise RuntimeError(f"EEF traj scheduling failed")

        self.rmq_client.put_data(
            topic="command",
            data=serialize(
                {
                    "command": ControllerCommand.SCHEDULE_EEF_TRAJ,
                    "data": eef_traj_xyz_wxyz,
                    "timestamps": timestamps,
                    "use_relative_timestamps": use_relative_timestamps,
                }
            ),
            automatic_resend=False,  # Will kill the task if the robot is disconnected
        )


class BaseCartesianJointClient(BaseControllerClient):
    """Combined client for controllers using CARTESIAN_JOINT mode (e.g. ARX5).

    Provides both cartesian (eef pose) and joint (gripper) interfaces
    over a single RMQ connection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # EEF state buffers
        self.eef_poses: list[npt.NDArray[np.float64]] = []
        self.eef_timestamps: list[float] = []
        self.eef_target_poses: list[npt.NDArray[np.float64]] = []
        self.eef_target_timestamps: list[float] = []

        # Joint state buffers
        self.joint_pos: list[npt.NDArray[np.float64]] = []
        self.joint_timestamps: list[float] = []
        self.dof: int

        print(f"Waiting for {self.name} to start, endpoint: {self.server_endpoint}")
        eef_ready = False
        joint_ready = False
        while not (eef_ready and joint_ready):
            if not eef_ready:
                if (
                    self.rmq_client.get_topic_status(topic="eef_pose", timeout_s=0.5)
                    > 0
                ):
                    eef_ready = True
            if not joint_ready:
                if (
                    self.rmq_client.get_topic_status(topic="joint_state", timeout_s=0.5)
                    > 0
                ):
                    raw_data, _ = self.rmq_client.peek_data(topic="joint_state", n=1)
                    self.dof = len(deserialize(raw_data[0])["joint_pos"])
                    joint_ready = True
            time.sleep(0.1)
        print(f"{self.name} started. DOF: {self.dof}")

    def reset(self):
        self.rmq_client.put_data(
            topic="command",
            data=serialize(
                {
                    "command": ControllerCommand.RESET,
                }
            ),
        )
        self.rmq_client.pop_data(
            topic="eef_pose", n=0
        )  # Clear buffer to avoid old data after reset
        self.rmq_client.pop_data(
            topic="joint_state", n=0
        )  # Clear buffer to avoid old data after reset

        while True:
            topic_status = self.rmq_client.get_topic_status(
                topic="eef_pose",
                timeout_s=1.0,
            )
            if topic_status > 10:
                print(f"{self.name} EEF pose stream restarted.")
                break
            time.sleep(0.01)

    def get_eef_xyz_wxyz(self, timestamp: float):
        """Apply pose interpolation to get the EEF pose at the given timestamp."""
        raw_data, _ = self.rmq_client.pop_data(topic="eef_pose", n=0)
        for data in raw_data:
            data = deserialize(data)
            # Timestamps must be strictly increasing. ARX5 SDK timestamps start
            # from 0 on each controller instance, but the controller applies an
            # offset to convert them to time.monotonic() domain before publishing,
            # so monotonicity holds even across reconnects.
            if len(self.eef_timestamps) > 0:
                assert (
                    data["timestamp"] > self.eef_timestamps[-1]
                ), f"Timestamp is not increasing: {data['timestamp']} < {self.eef_timestamps[-1]}"
            self.eef_poses.append(data["eef_xyz_wxyz"])
            self.eef_timestamps.append(data["timestamp"])

        if timestamp >= self.eef_timestamps[-1]:
            result = self.eef_poses[-1]
        elif timestamp >= self.eef_timestamps[0]:
            interpolate_idx = np.searchsorted(
                self.eef_timestamps, timestamp, side="right"
            )
            result = interpolate_xyz_wxyz(
                pose_left=self.eef_poses[interpolate_idx - 1],
                pose_right=self.eef_poses[interpolate_idx],
                timestamp_left=self.eef_timestamps[interpolate_idx - 1],
                timestamp_right=self.eef_timestamps[interpolate_idx],
                timestamp=timestamp,
            )
        else:
            print(
                f"{self.name} EEF timestamp is too old: {timestamp} < {self.eef_timestamps[0]}. Please increase data_remaining_time_s."
            )
            result = self.eef_poses[0]

        old_idx = np.searchsorted(
            self.eef_timestamps,
            timestamp - self.data_remaining_time_s,
            side="left",
        )
        if old_idx == len(self.eef_timestamps):
            old_idx -= 1
        self.eef_poses = self.eef_poses[old_idx:]
        self.eef_timestamps = self.eef_timestamps[old_idx:]

        return result

    def get_eef_target_xyz_wxyz(self, timestamp: float):
        raw_data, _ = self.rmq_client.pop_data(
            topic="eef_target", n=0, automatic_resend=False
        )
        for data in raw_data:
            data = deserialize(data)
            # Timestamps must be strictly increasing. ARX5 SDK timestamps start
            # from 0 on each controller instance, but the controller applies an
            # offset to convert them to time.monotonic() domain before publishing,
            # so monotonicity holds even across reconnects.
            if len(self.eef_target_timestamps) > 0:
                assert (
                    data["timestamp"] > self.eef_target_timestamps[-1]
                ), f"Timestamp is not increasing: {data['timestamp']} < {self.eef_target_timestamps[-1]}"
            self.eef_target_poses.append(data["eef_xyz_wxyz"])
            self.eef_target_timestamps.append(data["timestamp"])

        if timestamp >= self.eef_target_timestamps[-1]:
            result = self.eef_target_poses[-1]
        elif timestamp >= self.eef_target_timestamps[0]:
            interpolate_idx = np.searchsorted(
                self.eef_target_timestamps, timestamp, side="right"
            )
            result = interpolate_xyz_wxyz(
                pose_left=self.eef_target_poses[interpolate_idx - 1],
                pose_right=self.eef_target_poses[interpolate_idx],
                timestamp_left=self.eef_target_timestamps[interpolate_idx - 1],
                timestamp_right=self.eef_target_timestamps[interpolate_idx],
                timestamp=timestamp,
            )
        else:
            print(
                f"{self.name} EEF timestamp is too old: {timestamp} < {self.eef_target_timestamps[0]}. Please increase data_remaining_time_s."
            )
            result = self.eef_target_poses[0]

        old_idx = np.searchsorted(
            self.eef_target_timestamps,
            timestamp - self.data_remaining_time_s,
            side="left",
        )
        if old_idx == len(self.eef_target_timestamps):
            old_idx -= 1
        self.eef_target_poses = self.eef_target_poses[old_idx:]
        self.eef_target_timestamps = self.eef_target_timestamps[old_idx:]

        return result

    def schedule_eef_traj(
        self,
        eef_traj_xyz_wxyz: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
        use_relative_timestamps: bool,
    ):
        raw_data_list, _ = self.rmq_client.pop_data(
            topic="error", n=0, automatic_resend=False
        )
        for data in raw_data_list:
            data = deserialize(data)
            print(f"Error: {data['error']}")
        if len(raw_data_list) > 0:
            raise RuntimeError("EEF traj scheduling failed")

        self.rmq_client.put_data(
            topic="command",
            data=serialize(
                {
                    "command": ControllerCommand.SCHEDULE_EEF_TRAJ,
                    "data": eef_traj_xyz_wxyz,
                    "timestamps": timestamps,
                    "use_relative_timestamps": use_relative_timestamps,
                }
            ),
            automatic_resend=False,  # Will kill the task if the robot is disconnected
        )

    def get_joint_pos(self, timestamp: float):
        """Apply interpolation to get joint position at the given timestamp."""
        raw_data, _ = self.rmq_client.pop_data(
            topic="joint_state", n=0, automatic_resend=False
        )
        for data in raw_data:
            data = deserialize(data)
            # See timestamp monotonicity comment in get_eef_xyz_wxyz.
            if len(self.joint_timestamps) > 0:
                assert (
                    data["timestamp"] > self.joint_timestamps[-1]
                ), f"Timestamp is not increasing: {data['timestamp']} < {self.joint_timestamps[-1]}"
            self.joint_pos.append(data["joint_pos"])
            self.joint_timestamps.append(data["timestamp"])

        if timestamp >= self.joint_timestamps[-1]:
            result = self.joint_pos[-1]
        elif timestamp >= self.joint_timestamps[0]:
            interpolate_idx = np.searchsorted(
                self.joint_timestamps, timestamp, side="right"
            )
            ratio = (timestamp - self.joint_timestamps[interpolate_idx - 1]) / (
                self.joint_timestamps[interpolate_idx]
                - self.joint_timestamps[interpolate_idx - 1]
            )
            result = (
                self.joint_pos[interpolate_idx - 1] * (1 - ratio)
                + self.joint_pos[interpolate_idx] * ratio
            )
        else:
            print(
                f"{self.name} Joint timestamp is too old: {timestamp:.3f} < {self.joint_timestamps[0]:.3f}. Please increase data_remaining_time_s."
            )
            result = self.joint_pos[0]

        old_idx = np.searchsorted(
            self.joint_timestamps,
            timestamp - self.data_remaining_time_s,
            side="left",
        )
        if old_idx == len(self.joint_timestamps):
            old_idx -= 1
        self.joint_pos = self.joint_pos[old_idx:]
        self.joint_timestamps = self.joint_timestamps[old_idx:]

        return result

    def schedule_joint_traj(
        self,
        joint_traj_pos: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
        use_relative_timestamps: bool,
    ):
        assert (
            joint_traj_pos.shape[1] == self.dof
        ), f"trajectory input should have shape (N, {self.dof}), but got {joint_traj_pos.shape}"
        assert (
            timestamps.shape[0] == joint_traj_pos.shape[0]
        ), f"Timestamps shape mismatch: {timestamps.shape[0]=}, {joint_traj_pos.shape[0]=}"

        raw_data_list, _ = self.rmq_client.pop_data(
            topic="error", n=0, automatic_resend=False
        )
        for data in raw_data_list:
            data = deserialize(data)
            print(f"Error: {data['error']}")
        if len(raw_data_list) > 0:
            raise RuntimeError("Joint traj scheduling failed")

        self.rmq_client.put_data(
            topic="command",
            data=serialize(
                {
                    "command": ControllerCommand.SCHEDULE_JOINT_TRAJ,
                    "data": joint_traj_pos,
                    "timestamps": timestamps,
                    "use_relative_timestamps": use_relative_timestamps,
                }
            ),
            automatic_resend=False,  # Will kill the task if the robot is disconnected
        )
