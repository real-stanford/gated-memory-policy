import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import zarr
from numpy.testing import measure
from robotmq import RMQServer, deserialize, serialize

from real_env.common.data_classes import ControllerCommand, Trajectory
from robot_utils.logging_utils import echo_exception
from robot_utils.time_utils import wait_until


class BaseController(ABC):
    def __init__(
        self,
        robot_name: str,
        server_endpoint: str,
        control_mode: str,
        ctrl_freq: float,
        config_str: str = "",
    ):

        # Properties
        self.robot_name: str = robot_name
        self.server_endpoint: str = server_endpoint
        assert control_mode in ["JOINT", "CARTESIAN", "CARTESIAN_JOINT"]
        self.control_mode: str = control_mode
        self.ctrl_freq: float = ctrl_freq
        self.config_str: str = config_str
        self.logger = logging.getLogger(f"{self.robot_name}Controller")

        # DEBUG
        # self.last_state_timestamp = 0.0

        # For communication with the policy server
        self.rmq_server = RMQServer(
            server_name=self.robot_name, server_endpoint=server_endpoint
        )

        self.rmq_server.add_topic(
            topic="command",
            message_remaining_time_s=10.0,
        )
        self.rmq_server.add_topic(
            topic="error",
            message_remaining_time_s=100.0,
        )

        if self.control_mode == "JOINT":
            self.rmq_server.add_topic(
                topic="joint_state",
                message_remaining_time_s=1.0,
            )
            self.rmq_server.add_topic(
                topic="joint_target",
                message_remaining_time_s=1.0,
            )

        elif self.control_mode == "CARTESIAN":
            self.rmq_server.add_topic(
                topic="eef_pose",
                message_remaining_time_s=1.0,
            )
            self.rmq_server.add_topic(
                topic="eef_target",
                message_remaining_time_s=1.0,
            )

        elif self.control_mode == "CARTESIAN_JOINT":
            self.rmq_server.add_topic(
                topic="joint_state",
                message_remaining_time_s=1.0,
            )
            self.rmq_server.add_topic(
                topic="joint_target",
                message_remaining_time_s=1.0,
            )
            self.rmq_server.add_topic(
                topic="eef_pose",
                message_remaining_time_s=1.0,
            )
            self.rmq_server.add_topic(
                topic="eef_target",
                message_remaining_time_s=1.0,
            )

        self.rmq_server.add_topic(
            topic="config",
            message_remaining_time_s=1.0,
        )

        print(f"Controller {self.robot_name} server created at {self.server_endpoint}")

        # States
        self.is_connected: bool = False
        self.episode_group: zarr.Group | None = None

        self.rmq_server.put_data(
            topic="config",
            data=serialize(self.config_str),
        )

    ##### Context Management #####

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

    ##### Robot interfaces #####

    @abstractmethod
    def connect(self):
        ...

    @abstractmethod
    def disconnect(self):
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def send_robot_target(self):
        """
        Interpolation should be done here.
        """
        ...

    ##### Joint control #####

    def get_joint_pos(
        self, wait_until_update: bool = False
    ) -> tuple[npt.NDArray[np.float64], float]:
        """
        (dof, ) np.array of joint positions, float timestamp
        """
        raise NotImplementedError

    def get_joint_target(self) -> tuple[npt.NDArray[np.float64], float]:
        """
        (dof, ) np.array of joint targets, float timestamp
        """
        raise NotImplementedError

    def get_eef_pose_and_joint_pos(
        self, wait_until_update: bool = False
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """
        (7, ) np.array of [x, y, z, qw, qx, qy, qz], (dof, ) np.array of joint positions, float timestamp
        """
        raise NotImplementedError

    ##### Cartesian control #####

    def schedule_joint_traj(
        self, joint_traj: npt.NDArray[np.float64], timestamps: npt.NDArray[np.float64]
    ):
        """
        args:
            joint_traj: (N, dof)
        """
        raise NotImplementedError

    ##### Cartesian control #####

    def get_eef_pose(
        self, wait_until_update: bool = False
    ) -> tuple[npt.NDArray[np.float64], float]:
        """
        (7, ) np.array of [x, y, z, qw, qx, qy, qz], float timestamp
        """
        raise NotImplementedError

    def get_eef_target(self) -> tuple[npt.NDArray[np.float64], float]:
        """
        (7, ) np.array of [x, y, z, qw, qx, qy, qz], float timestamp
        """
        raise NotImplementedError

    def schedule_eef_traj(
        self,
        eef_traj_xyz_wxyz: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
    ):
        """
        args:
            eef_traj_xyz_wxyz: (N, 7)
        """
        raise NotImplementedError

    @abstractmethod
    def record_data(self):
        ...

    ##### Main Loop #####

    def run(self):
        assert self.is_connected, "Robot is not connected"
        self.logger.info(f"Controller {self.robot_name} started")
        last_timestamp = time.monotonic()
        while True:
            all_data, rmq_timestamps = self.rmq_server.pop_data(topic="command", n=0)
            for data, timestamp in zip(all_data, rmq_timestamps):
                try:
                    data_dict = deserialize(data)
                    assert isinstance(data_dict, dict), f"data_dict: {data_dict}"
                    data_dict = cast(dict[str, Any], data_dict)
                    command = data_dict.pop("command")
                    assert command in ControllerCommand, f"Invalid command: {command}"

                    if command == ControllerCommand.RESET:
                        assert self.is_connected, "Robot is not connected"
                        self.reset()
                    elif command == ControllerCommand.CONNECT:
                        self.connect()
                    elif command == ControllerCommand.DISCONNECT:
                        self.disconnect()
                    elif command == ControllerCommand.SCHEDULE_JOINT_TRAJ:
                        if not self.is_connected:
                            self.connect()
                        assert self.is_connected, "Robot is not connected"
                        assert self.control_mode in ("JOINT", "CARTESIAN_JOINT")
                        use_relative_timestamps = data_dict.pop(
                            "use_relative_timestamps"
                        )
                        joint_traj = Trajectory(**data_dict)
                        if use_relative_timestamps:
                            joint_traj.timestamps += time.monotonic()
                        self.schedule_joint_traj(joint_traj.data, joint_traj.timestamps)

                    elif command == ControllerCommand.SCHEDULE_EEF_TRAJ:
                        if not self.is_connected:
                            self.connect()
                        assert self.is_connected, "Robot is not connected"
                        assert self.control_mode in ("CARTESIAN", "CARTESIAN_JOINT")
                        use_relative_timestamps = data_dict.pop(
                            "use_relative_timestamps"
                        )
                        eef_traj = Trajectory(**data_dict)
                        if use_relative_timestamps:
                            now = time.monotonic()
                            # print(f"{timestamp=}, {time.monotonic()=}")
                            # now = time.monotonic()
                            # if hasattr(self, '_dbg_ts'):
                            #     print(f"[DEBUG] dt: {timestamp - self._dbg_ts:.4f}s, freq: {1/(timestamp - self._dbg_ts):.1f} Hz")
                            # self._dbg_ts = timestamp
                            eef_traj.timestamps += now

                        # print(f"[UR5] schedule_eef_traj: {eef_traj.data=} {eef_traj.timestamps=}")
                        self.schedule_eef_traj(eef_traj.data, eef_traj.timestamps)
                    else:
                        raise ValueError(f"Invalid command: {command}")

                except (
                    AttributeError,
                    ValueError,
                    KeyError,
                    AssertionError,
                    RuntimeError,
                ) as e:
                    exception_str = echo_exception()
                    self.logger.error(
                        f"Error in controller {self.robot_name}: {e}, {exception_str}"
                    )
                    self.rmq_server.put_data(
                        topic="errors",
                        data=serialize(
                            {
                                "error": f"Error in controller {self.robot_name}: {e}, {exception_str}"
                            }
                        ),
                    )
            timestamp = time.monotonic()

            # Check for command timeout (subclasses like ARX5Cartesian define these)
            if self.is_connected and hasattr(self, "handle_no_command_timeout"):
                no_cmd_timeout_s = getattr(self, "no_cmd_timeout_s", None)
                last_cmd_time = getattr(self, "last_cmd_time", None)
                if (
                    no_cmd_timeout_s is not None
                    and last_cmd_time is not None
                    and last_cmd_time >= 0  # sentinel -1 means already reset
                    and (timestamp - last_cmd_time) > no_cmd_timeout_s
                ):
                    self.handle_no_command_timeout()  # type: ignore[attr-defined]

            if not self.is_connected:
                continue

            if self.control_mode == "JOINT":
                joint_pos, joint_pos_timestamp = self.get_joint_pos(
                    wait_until_update=True
                )
                self.rmq_server.put_data(
                    topic="joint_state",
                    data=serialize(
                        {
                            "joint_pos": joint_pos,
                            "timestamp": joint_pos_timestamp,
                        }
                    ),
                )
            elif self.control_mode == "CARTESIAN":
                eef_xyz_wxyz, eef_pose_timestamp = self.get_eef_pose(
                    wait_until_update=True
                )
                self.rmq_server.put_data(
                    topic="eef_pose",
                    data=serialize(
                        {
                            "eef_xyz_wxyz": eef_xyz_wxyz,
                            "timestamp": eef_pose_timestamp,
                        }
                    ),
                )
            elif self.control_mode == "CARTESIAN_JOINT":
                eef_xyz_wxyz, joint_pos, timestamp = self.get_eef_pose_and_joint_pos(
                    wait_until_update=True
                )

                # if timestamp <= self.last_state_timestamp:
                #     print(f"[WARNING] Received out-of-order state with timestamp {timestamp:.4f}, last timestamp: {self.last_state_timestamp:.4f}")
                # self.last_state_timestamp = timestamp

                self.rmq_server.put_data(
                    topic="joint_state",
                    data=serialize(
                        {
                            "joint_pos": joint_pos,
                            "timestamp": timestamp,
                        }
                    ),
                )
                self.rmq_server.put_data(
                    topic="eef_pose",
                    data=serialize(
                        {
                            "eef_xyz_wxyz": eef_xyz_wxyz,
                            "timestamp": timestamp,
                        }
                    ),
                )

            next_timestamp = last_timestamp + 1 / self.ctrl_freq
            # remaining_time = next_timestamp - timestamp
            wait_until(next_timestamp)
            last_timestamp = next_timestamp

            self.send_robot_target()
            self.record_data()

            if self.control_mode == "JOINT":
                joint_target, joint_target_timestamp = self.get_joint_target()
                self.rmq_server.put_data(
                    topic="joint_target",
                    data=serialize(
                        {
                            "joint_target": joint_target,
                            "timestamp": joint_target_timestamp,
                        }
                    ),
                )
            elif self.control_mode == "CARTESIAN":
                eef_xyz_wxyz, eef_target_timestamp = self.get_eef_target()
                self.rmq_server.put_data(
                    topic="eef_target",
                    data=serialize(
                        {
                            "eef_xyz_wxyz": eef_xyz_wxyz,
                            "timestamp": eef_target_timestamp,
                        }
                    ),
                )
            elif self.control_mode == "CARTESIAN_JOINT":
                joint_target, joint_target_timestamp = self.get_joint_target()
                eef_xyz_wxyz, eef_target_timestamp = self.get_eef_target()
                self.rmq_server.put_data(
                    topic="joint_target",
                    data=serialize(
                        {
                            "joint_target": joint_target,
                            "timestamp": joint_target_timestamp,
                        }
                    ),
                )
                self.rmq_server.put_data(
                    topic="eef_target",
                    data=serialize(
                        {
                            "eef_xyz_wxyz": eef_xyz_wxyz,
                            "timestamp": eef_target_timestamp,
                        }
                    ),
                )
