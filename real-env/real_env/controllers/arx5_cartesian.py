from numpy._typing._array_like import NDArray
import json
from omegaconf import OmegaConf


from numpy import float64


import time
import os
import sys
import hydra
import numpy as np
import numpy.typing as npt
import scipy.spatial.transform as st
from real_env.common.data_classes import Trajectory
from real_env.common.interpolator import PoseTrajInterpolator, JointTrajInterpolator
from real_env.controllers.base_controller import BaseController
from robot_utils.pose_utils import (
    to_wxyz,
    to_xyzw,
    rotvec2rotm,
    rotm2rpy,
    rpy2rotm,
    rotm2rotvec,
)
from robot_utils.logging_utils import echo_exception
from robologger import RobotCtrlLogger

import arx5_interface as arx5


def z_forward_to_x_forward_tranform(
    z_forward_pose: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Equivalent to TCP to EE transformation.
    Input should be a 6D rotvec and outputs a 6D vec of xyz + rpy
    """
    tcp_cartesian = z_forward_pose[:3]
    tcp_rotvec = z_forward_pose[3:]
    tcp_rot_mat = rotvec2rotm(tcp_rotvec)
    tcp2ee_rot_mat = np.array(
        [
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0],
        ]
    )
    ee_rot_mat = tcp_rot_mat @ tcp2ee_rot_mat
    ee_rpy = rotm2rpy(ee_rot_mat)
    return np.concatenate([tcp_cartesian, ee_rpy])


def x_forward_to_z_forward_transform(
    x_forward_pose: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Equivalent to EE to TCP transformation.
    Input should be a 6D xyz + rpy and outputs a 6D vec of xyz + rotvec
    """
    ee_cartesian = x_forward_pose[:3]
    ee_rpy = x_forward_pose[3:]
    ee_rot_mat = rpy2rotm(ee_rpy)
    ee2tcp_rot_mat = np.array(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ]
    )
    tcp_rot_mat = ee_rot_mat @ ee2tcp_rot_mat
    tcp_rotvec = rotm2rotvec(tcp_rot_mat)
    # opposite the rotation vector but keep the same pose
    angle_rad = np.linalg.norm(tcp_rotvec)
    vec = tcp_rotvec / angle_rad
    alternate_angle_rad = 2 * np.pi - angle_rad
    tcp_rotvec = -vec * alternate_angle_rad

    return np.concatenate([ee_cartesian, tcp_rotvec])


class ARX5Cartesian(BaseController):
    def __init__(
        self,
        model: str,
        interface_name: str,
        server_endpoint: str,
        cartesian_logger_endpoint: str,
        joint_logger_endpoint: str,
        robot_name: str,
        no_cmd_timeout_s: float,
        kp: npt.NDArray[np.float64],
        ctrl_freq: float,
        max_pos_speed_m_per_s: float,
        max_rot_speed_rad_per_s: float,
        max_gripper_speed_per_s: float,
        lookahead_time_s: float,
        traj_smooth_time_s: float,
        trim_new_trajectory_time_s: float,  # Slightly lower than 0.1
        dynamic_latency: bool,
        latency_matching_dt: float,  # TODO: check if this is correct
        latency_precision: float,
        latency_pos_weight: float,
        latency_rot_weight: float,
        config_str: str = "",
    ):
        super().__init__(
            robot_name=robot_name,
            control_mode="CARTESIAN_JOINT",
            server_endpoint=server_endpoint,
            ctrl_freq=ctrl_freq,
            config_str=config_str,
        )

        self.model: str = model
        self.interface_name: str = interface_name
        self.dynamic_latency: bool = dynamic_latency
        self.latency_matching_dt: float = latency_matching_dt
        self.latency_precision: float = latency_precision
        self.latency_pos_weight: float = latency_pos_weight
        self.latency_rot_weight: float = latency_rot_weight
        self.kp: NDArray[float64] = np.array(kp)

        self.no_cmd_timeout_s: float = no_cmd_timeout_s
        self.last_cmd_time: float | None = (
            None  # None = initial start, skip timeout until first command
        )

        # Properties
        self.max_pos_speed_m_per_s: float = max_pos_speed_m_per_s
        self.max_rot_speed_rad_per_s: float = max_rot_speed_rad_per_s
        self.max_gripper_speed_per_s: float = max_gripper_speed_per_s
        self.lookahead_time_s: float = lookahead_time_s
        self.traj_smooth_time_s: float = traj_smooth_time_s
        self.trim_new_trajectory_time_s: float = trim_new_trajectory_time_s

        self.pose_interpolator: PoseTrajInterpolator
        self.gripper_interpolator: JointTrajInterpolator

        self.arx5_cartesian_controller: arx5.Arx5CartesianController | None

        self.last_target_xyz_wxyz: npt.NDArray[np.float64]
        self.last_target_timestamp: float
        self.last_gripper_pos: float

        self.last_pose_timestamp: float

        self.cartesian_logger = RobotCtrlLogger(
            name="right_arm",
            endpoint=cartesian_logger_endpoint,
            attr={"robot_name": self.robot_name, "ctrl_freq": self.ctrl_freq},
            log_eef_pose=True,
            log_joint_pos=False,
            target_type="eef_pose",
            joint_units=None,
        )

        assert (
            self.cartesian_logger is not None
        ), "Failed to initialize cartesian logger"

        self.joint_logger = RobotCtrlLogger(
            name="right_end_effector",
            endpoint=joint_logger_endpoint,
            attr={
                "robot_name": self.robot_name,
                "ctrl_freq": self.ctrl_freq,
                "num_joints": 1,
            },
            log_eef_pose=False,
            log_joint_pos=True,
            target_type="joint_pos",
            joint_units="meters",
        )
        assert self.joint_logger is not None, "Failed to initialize joint logger"

        self.connect()

    def connect(self):
        """Initialize ARX5 cartesian controller and both interpolators (pose + gripper).
        Reads current arm and gripper state to seed interpolators with a single-point
        trajectory. ARX5 SDK timestamps start from 0 on each controller instance, so
        we compute a timestamp_offset to convert them to time.monotonic() domain.
        """
        if self.is_connected:
            return

        print("Connecting to ARX5 ...")

        # model is the arx5 arm we have - e.g. x5 or l5; interface is the CAN bus interface like "can0"
        self.arx5_cartesian_controller = arx5.Arx5CartesianController(
            self.model, self.interface_name
        )
        print(
            f"Connected to ARX5 Cartesian Controller with model {self.model} and interface {self.interface_name}"
        )
        # no need to asset, arx5 controller class constructor will throw err on failure

        self.is_connected = True

        # ARX5 SDK timestamps start from 0 on each new controller instance.
        # Compute offset to convert them to time.monotonic() domain.
        init_state = self.arx5_cartesian_controller.get_eef_state()
        monotonic_at_init = time.monotonic()
        self.timestamp_offset = monotonic_at_init - init_state.timestamp

        # init pose interpolator with current arm state
        eef_xyz_wxyz, gripper_pos, _ = self.get_eef_pose_and_joint_pos()
        self.pose_interpolator = PoseTrajInterpolator(
            init_trajectory=Trajectory(
                data=eef_xyz_wxyz[np.newaxis, :],
                timestamps=np.array([time.monotonic()]),
            ),
            traj_smooth_time_s=self.traj_smooth_time_s,
            max_pos_speed_m_per_s=self.max_pos_speed_m_per_s,
            max_rot_speed_rad_per_s=self.max_rot_speed_rad_per_s,
            trim_new_trajectory_time_s=self.trim_new_trajectory_time_s,
        )

        # init gripper interpolator with current gripper state
        self.gripper_interpolator = JointTrajInterpolator(
            init_trajectory=Trajectory(
                data=gripper_pos[np.newaxis, :],
                timestamps=np.array([time.monotonic()]),
            ),
            traj_smooth_time_s=self.traj_smooth_time_s,
            max_joint_speed_per_s=self.max_gripper_speed_per_s,
            trim_new_trajectory_time_s=self.trim_new_trajectory_time_s,
        )

        self.last_target_xyz_wxyz = eef_xyz_wxyz
        self.last_gripper_pos = gripper_pos[0]
        self.last_target_timestamp = (
            time.monotonic()
        )  # shared by both arm and gripper targets

        # None = initial start: set to -1 (sentinel, no timeout until first command).
        # -1 = post auto-reset: stays -1, schedule_eef/joint_traj will set it to
        #       time.monotonic() when the next command arrives.
        if self.last_cmd_time is None:
            self.last_cmd_time = time.monotonic()
        print(
            f"[DEBUG] sdk_timestamp={init_state.timestamp:.6f}, monotonic={monotonic_at_init:.6f}, offset={self.timestamp_offset:.6f}, converted={init_state.timestamp + self.timestamp_offset:.6f}"
        )
        print(
            f"ARX5 is connected with model {self.model} and interface {self.interface_name}. Resetting to home."
        )
        self.reset()

    def disconnect(self):
        """Reset arm to home, release motor torque (damping mode), and destroy the controller.
        Follows the same shutdown sequence as the ARX5 ZMQ server.
        """
        if not self.is_connected:
            return
        print(
            f"Disconnecting from ARX5 with model {self.model} and interface {self.interface_name}"
        )
        assert self.arx5_cartesian_controller is not None
        self.arx5_cartesian_controller.reset_to_home()
        self.arx5_cartesian_controller.set_to_damping()
        del self.arx5_cartesian_controller
        self.is_connected = False
        print("ARX5 is disconnected")

    def reset(self):
        """Reset the ARX5 cartesian controller."""
        assert self.arx5_cartesian_controller is not None
        start_time = time.monotonic()
        self.arx5_cartesian_controller.reset_to_home()
        end_time = time.monotonic()
        print(f"ARX5 reset_to_home time: {end_time - start_time}")

        init_gain: arx5.Gain = self.arx5_cartesian_controller.get_gain()
        init_gain.kp()[:] = np.array([300, 300, 400, 150, 120, 120])
        init_gain.gripper_kp = 5.0
        self.arx5_cartesian_controller.set_gain(init_gain)
        print(
            f"Set ARX5 Cartesian Controller gains to kp: {self.kp}, gripper_kp: {init_gain.gripper_kp}"
        )

        # init pose interpolator with current arm state
        (
            eef_xyz_wxyz,
            gripper_pos,
            self.last_pose_timestamp,
        ) = self.get_eef_pose_and_joint_pos()
        self.pose_interpolator = PoseTrajInterpolator(
            init_trajectory=Trajectory(
                data=eef_xyz_wxyz[np.newaxis, :],
                timestamps=np.array([time.monotonic()]),
            ),
            traj_smooth_time_s=self.traj_smooth_time_s,
            max_pos_speed_m_per_s=self.max_pos_speed_m_per_s,
            max_rot_speed_rad_per_s=self.max_rot_speed_rad_per_s,
            trim_new_trajectory_time_s=self.trim_new_trajectory_time_s,
        )
        self.last_target_xyz_wxyz = eef_xyz_wxyz

        # init gripper interpolator with current gripper state
        self.gripper_interpolator = JointTrajInterpolator(
            init_trajectory=Trajectory(
                data=gripper_pos[np.newaxis, :],
                timestamps=np.array([time.monotonic()]),
            ),
            traj_smooth_time_s=self.traj_smooth_time_s,
            max_joint_speed_per_s=self.max_gripper_speed_per_s,
            trim_new_trajectory_time_s=self.trim_new_trajectory_time_s,
        )
        self.last_gripper_pos = gripper_pos[0]
        self.last_target_timestamp = time.monotonic()

        self.last_cmd_time = (
            time.monotonic()
        )  # no auto-reset until next command arrives
        print(
            f"ARX5 reset complete with model {self.model} and interface {self.interface_name}"
        )

    def handle_no_command_timeout(self):
        """Reset after command inactivity."""
        if not self.is_connected:
            return
        assert self.arx5_cartesian_controller is not None
        print(
            f"Timeout: No command received for the configured interval {self.no_cmd_timeout_s}s. "
            f"Resetting ARX5 {self.model} on {self.interface_name} to home."
        )
        self.arx5_cartesian_controller.reset_to_home()
        self.arx5_cartesian_controller.set_to_damping()
        del self.arx5_cartesian_controller
        self.arx5_cartesian_controller = None
        self.is_connected = False
        self.last_cmd_time = (
            time.monotonic()
        )  # sentinel: skip future timeout checks until a new command arrives
        print(
            "ARX5 timeout handling complete — arm is in damping mode, waiting for reconnect"
        )

    def record_data(self):
        """Log current arm pose and gripper position to their respective loggers."""
        eef_pose, gripper_pos, timestamp = self.get_eef_pose_and_joint_pos()
        eef_target, eef_target_timestamp = self.get_eef_target()
        joint_target, joint_target_timestamp = self.get_joint_target()

        if self.cartesian_logger.update_recording_state():
            self.cartesian_logger.log_state(
                state_timestamp=timestamp,
                state_pos_xyz=eef_pose[:3].astype(np.float32),
                state_quat_wxyz=eef_pose[3:7].astype(np.float32),
            )
            self.cartesian_logger.log_target(
                target_timestamp=eef_target_timestamp,
                target_pos_xyz=eef_target[:3].astype(np.float32),
                target_quat_wxyz=eef_target[3:7].astype(np.float32),
            )

        if self.joint_logger.update_recording_state():
            self.joint_logger.log_state(
                state_timestamp=timestamp,
                state_joint_pos=gripper_pos.astype(np.float32),
            )
            self.joint_logger.log_target(
                target_timestamp=joint_target_timestamp,
                target_joint_pos=joint_target.astype(np.float32),
            )

    def send_robot_target(self):
        """Interpolate arm pose and gripper position at lookahead time, convert from
        z-forward wxyz quaternion to z-forward rotvec to x-forward RPY, and send as
        EEFState command to arx5.
        """
        assert self.arx5_cartesian_controller is not None
        current_timestamp: float = time.monotonic()
        target_timestamp: float = current_timestamp + self.lookahead_time_s
        target_pose_xyz_wxyz: npt.NDArray[
            np.float64
        ] = self.pose_interpolator.interpolate_xyz_wxyz(target_timestamp)
        cmd_gripper_pos = float(
            self.gripper_interpolator.interpolate(target_timestamp)[0]
        )

        target_pos = target_pose_xyz_wxyz[:3]
        target_rot = st.Rotation.from_quat(to_xyzw(target_pose_xyz_wxyz[3:7]))

        # current_pose_xyz_wxyz, _ = self.get_eef_pose()
        # current_pos = current_pose_xyz_wxyz[:3]
        # current_rot = st.Rotation.from_quat(to_xyzw(current_pose_xyz_wxyz[3:7]))

        # Convert z-forward rotvec to x-forward RPY for ARX5
        z_forward_6d = np.concatenate([target_pos, target_rot.as_rotvec()])
        x_forward_6d = z_forward_to_x_forward_tranform(z_forward_6d)

        self.arx5_cartesian_controller.set_eef_cmd(
            arx5.EEFState(x_forward_6d, cmd_gripper_pos)
        )

        self.last_target_xyz_wxyz = target_pose_xyz_wxyz
        self.last_gripper_pos = cmd_gripper_pos
        self.last_target_timestamp = target_timestamp

    def get_eef_pose_and_joint_pos(
        self, wait_until_update: bool = False
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Get current end-effector pose from ARX5, converted from x-forward 6D (xyz+rpy)
        to z-forward 7D (xyz+wxyz quaternion). ARX5 pose_6d() returns [x, y, z, roll, pitch, yaw].

        :param wait_until_update: If True, block until a new EEF state is available (timestamp changes).
        :return: (eef_xyz_wxyz, gripper_pos, timestamp) — 7D pose array, gripper position array, and EEF state timestamp
        """
        assert self.arx5_cartesian_controller is not None
        while True:
            x_forward_eef_object: arx5.EEFState = (
                self.arx5_cartesian_controller.get_eef_state()
            )
            if (
                not wait_until_update
                or x_forward_eef_object.timestamp != self.last_pose_timestamp
            ):
                break
            time.sleep(0.001)  # avoid busy waiting

        z_forward_rotvec: npt.NDArray[np.float64] = x_forward_to_z_forward_transform(
            x_forward_eef_object.pose_6d()
        )
        z_forward_quat_xyzw: npt.NDArray[np.float64] = st.Rotation.from_rotvec(
            z_forward_rotvec[3:]
        ).as_quat()
        z_forward_xyz_wxyz: npt.NDArray[np.float64] = np.concatenate(
            [z_forward_rotvec[:3], to_wxyz(z_forward_quat_xyzw)]
        )
        robot_timestamp: float = x_forward_eef_object.timestamp
        monotonic_timestamp: float = robot_timestamp + self.timestamp_offset

        self.last_pose_timestamp = robot_timestamp

        return (
            z_forward_xyz_wxyz,
            np.array([x_forward_eef_object.gripper_pos]),
            monotonic_timestamp,
        )

    def get_eef_target(self) -> tuple[npt.NDArray[np.float64], float]:
        """Return the last commanded end-effector pose (z-forward xyz+wxyz) and its timestamp."""
        return self.last_target_xyz_wxyz, self.last_target_timestamp

    def get_joint_target(self) -> tuple[npt.NDArray[np.float64], float]:
        """Return the last commanded gripper position as a 1-DOF joint array and its timestamp."""
        return np.array([self.last_gripper_pos]), self.last_target_timestamp

    def schedule_eef_traj(
        self,
        eef_traj_xyz_wxyz: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
    ):
        """Update the pose interpolator with a new end-effector trajectory. Applies dynamic
        latency compensation to align commanded timestamps with actual robot state if enabled.

        :param eef_traj_xyz_wxyz: (N, 7) array of z-forward poses [x, y, z, w, x, y, z].
        :param timestamps: (N,) array of monotonic timestamps corresponding to each pose.
        """
        current_timestamp = time.monotonic()
        self.last_cmd_time = current_timestamp

        if self.dynamic_latency and len(timestamps) > 1:
            try:
                delta_latency = self.pose_interpolator.find_delta_latency(
                    input_poses=eef_traj_xyz_wxyz,
                    input_times=timestamps,
                    current_timestamp=current_timestamp,
                    matching_dt=self.latency_matching_dt,
                    latency_precision=self.latency_precision,
                    pos_weight=self.latency_pos_weight,
                    rot_weight=self.latency_rot_weight,
                )
                adjusted_timestamps = timestamps - delta_latency
            except Exception as e:
                exception_str = echo_exception()
                print(
                    f"Dynamic latency detection failed: {e}, {exception_str}. Using original timestamps."
                )
                adjusted_timestamps = timestamps
        else:
            adjusted_timestamps = timestamps

        self.pose_interpolator.update(
            new_trajectory=Trajectory(
                data=eef_traj_xyz_wxyz,
                timestamps=adjusted_timestamps,
            ),
            current_timestamp=current_timestamp,
        )

    def schedule_joint_traj(
        self,
        joint_traj: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
    ):
        """Update the gripper interpolator with a new gripper trajectory.

        :param joint_traj: (N, 1) array of gripper positions.
        :param timestamps: (N,) array of monotonic timestamps corresponding to each position.
        """
        self.last_cmd_time = time.monotonic()
        self.gripper_interpolator.update(
            new_trajectory=Trajectory(
                data=joint_traj,
                timestamps=timestamps,
            ),
            current_timestamp=time.monotonic(),
        )


def run_arx5_cartesian():
    os.environ["HYDRA_FULL_ERROR"] = "1"
    np.set_printoptions(precision=4)
    assert len(sys.argv) == 2, "Usage: python run_arx5_cartesian.py <model>"
    model = sys.argv[1]
    with hydra.initialize(config_path="../configs/controllers"):
        cfg = hydra.compose(config_name="arx5_cartesian")
    cfg.model = model
    print(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
    controller: ARX5Cartesian = hydra.utils.instantiate(cfg)
    with controller:
        controller.run()


if __name__ == "__main__":
    run_arx5_cartesian()
