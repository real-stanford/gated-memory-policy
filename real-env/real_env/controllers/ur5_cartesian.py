import json
import os
import time

import numpy as np
import numpy.typing as npt
import rtde_control
import rtde_receive
import scipy.spatial.transform as st

from real_env.common.data_classes import Trajectory
from real_env.common.interpolator import PoseTrajInterpolator
from real_env.controllers.base_controller import BaseController
import hydra
from robot_utils.pose_utils import to_wxyz, to_xyzw
from robologger import RobotCtrlLogger
from omegaconf import OmegaConf


class UR5Cartesian(BaseController):
    def __init__(
        self,
        server_endpoint: str,
        robot_ip: str,
        logger_endpoint: str,
        robot_name: str,
        dynamic_latency_range_s: float,
        home_pose_xyz_wxyz: list[float] | npt.NDArray[np.float64] | None,
        ctrl_freq: float,
        gain: float,
        max_pos_speed_m_per_s: float,
        max_rot_speed_rad_per_s: float,
        lookahead_time_s: float,
        reset_time_s: float,
        traj_smooth_time_s: float,
        tcp_offset_pose: list[float] | npt.NDArray[np.float64] | None,
        trim_new_trajectory_time_s: float,
        config_str: str = "",
    ):
        super().__init__(
            robot_name=robot_name,
            control_mode="CARTESIAN",
            server_endpoint=server_endpoint,
            ctrl_freq=ctrl_freq,
            config_str=config_str,
        )

        self.dynamic_latency_range_s: float = dynamic_latency_range_s

        # Properties
        self.robot_ip: str = robot_ip
        self.gain: float = gain
        self.max_pos_speed_m_per_s: float = max_pos_speed_m_per_s
        self.max_rot_speed_rad_per_s: float = max_rot_speed_rad_per_s
        self.lookahead_time_s: float = lookahead_time_s
        self.reset_time_s: float = reset_time_s
        self.traj_smooth_time_s: float = traj_smooth_time_s
        self.trim_new_trajectory_time_s: float = trim_new_trajectory_time_s
        if home_pose_xyz_wxyz is None:
            self.home_pose_xyz_wxyz: npt.NDArray[np.float64] | None = None
        else:
            assert len(home_pose_xyz_wxyz) == 6
            self.home_pose_xyz_wxyz = np.array(home_pose_xyz_wxyz)

        if tcp_offset_pose is None:
            self.tcp_offset_pose: npt.NDArray[np.float64] = np.zeros(6)
        else:
            assert len(tcp_offset_pose) == 6
            self.tcp_offset_pose = np.array(tcp_offset_pose)

        self.interpolator: PoseTrajInterpolator
        self.control_interface: rtde_control.RTDEControlInterface
        self.receive_interface: rtde_receive.RTDEReceiveInterface

        self.last_target_xyz_wxyz: npt.NDArray[np.float64]
        self.last_target_timestamp: float
        self.monotonic_time_minus_robot_time_s: float

        self.last_pose_timestamp: float | None = None
        self.last_joint_pos_timestamp: float | None = None

        self.cartesian_logger = RobotCtrlLogger(
            name="right_arm",
            endpoint=logger_endpoint,
            attr={
                "robot_name": self.robot_name,
                "ctrl_freq": self.ctrl_freq,
                "num_joints": 6,
            },
            log_eef_pose=True,
            log_joint_pos=True,
            target_type="eef_pose",
            joint_units="radians",
        )
        assert self.cartesian_logger is not None, "Cartesian logger is not initialized"

        # Latency profiling
        self.log_state_latency_queue: list[float] = []
        self.log_target_latency_queue: list[float] = []
        self.log_latency_window_size: int = 10

    def _calibrate_time_difference(self):
        assert self.is_connected

        last_robot_timestamp = self.receive_interface.getTimestamp()
        robot_timestamp = self.receive_interface.getTimestamp()

        time_diffs: list[float] = []

        print(f"{self.robot_name}: Calibrating time difference")

        for _ in range(20):
            while robot_timestamp == last_robot_timestamp:
                robot_timestamp: float = self.receive_interface.getTimestamp()
            time_diffs.append(time.monotonic() - robot_timestamp)
            last_robot_timestamp = robot_timestamp

        self.monotonic_time_minus_robot_time_s = float(np.mean(time_diffs))

    def connect(self):
        if self.is_connected:
            return

        print(
            "Connecting UR5. The program might run into error. Restart the program if this happens."
        )
        self.control_interface = rtde_control.RTDEControlInterface(
            hostname=self.robot_ip,
        )
        print(f"Control interface connected to {self.robot_ip}")
        self.receive_interface = rtde_receive.RTDEReceiveInterface(
            hostname=self.robot_ip,
        )
        print(f"Receive interface connected to {self.robot_ip}")

        assert self.control_interface.isConnected()

        self.is_connected = True
        self._calibrate_time_difference()
        self.control_interface.setTcp(self.tcp_offset_pose)
        eef_xyz_wxyz, self.last_pose_timestamp = self.get_eef_pose()
        self.interpolator = PoseTrajInterpolator(
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
        self.last_target_timestamp = time.monotonic()
        print("UR5 is connected")

    def disconnect(self):
        if not self.is_connected:
            return
        print("Disconnecting UR5")
        self.control_interface.servoStop()
        self.control_interface.stopScript()
        self.control_interface.disconnect()
        self.receive_interface.disconnect()
        self.is_connected = False
        print("UR5 is disconnected")

    def reset(self):
        if self.home_pose_xyz_wxyz is None:
            print("Home pose is not set. Will not reset.")
            return
        current_timestamp = time.monotonic()
        current_pose, _ = self.get_eef_pose()
        reset_trajectory = Trajectory(
            data=np.concatenate(
                [
                    current_pose[np.newaxis, :],
                    self.home_pose_xyz_wxyz[np.newaxis, :],
                ],
                axis=0,
            ),
            timestamps=np.array(
                [current_timestamp, current_timestamp + self.reset_time_s]
            ),
        )
        self.interpolator.update(
            new_trajectory=reset_trajectory,
            current_timestamp=current_timestamp,
        )

    def record_data(self):
        assert (
            self.cartesian_logger is not None
        ), "Cartesian logger is not initialized but record_data() is called"

        eef_pose, eef_pose_timestamp = self.get_eef_pose()
        joint_pos, joint_pos_timestamp = self.get_joint_pos()
        eef_target, eef_target_timestamp = self.get_eef_target()

        if self.cartesian_logger.update_recording_state():
            log_state_start = time.monotonic()
            self.cartesian_logger.log_state(
                state_timestamp=eef_pose_timestamp,
                state_pos_xyz=eef_pose[:3].astype(np.float64),
                state_quat_wxyz=eef_pose[3:7].astype(np.float64),
                state_joint_pos=joint_pos.astype(np.float64),
            )
            log_state_latency_ms = (time.monotonic() - log_state_start) * 1000

            log_target_start = time.monotonic()
            self.cartesian_logger.log_target(
                target_timestamp=eef_target_timestamp,
                target_pos_xyz=eef_target[:3].astype(np.float64),
                target_quat_wxyz=eef_target[3:7].astype(np.float64),
            )
            log_target_latency_ms = (time.monotonic() - log_target_start) * 1000

            self.log_state_latency_queue.append(log_state_latency_ms)
            if len(self.log_state_latency_queue) > self.log_latency_window_size:
                self.log_state_latency_queue.pop(0)
            avg_state_latency_ms = np.mean(self.log_state_latency_queue)

            self.log_target_latency_queue.append(log_target_latency_ms)
            if len(self.log_target_latency_queue) > self.log_latency_window_size:
                self.log_target_latency_queue.pop(0)
            avg_target_latency_ms = np.mean(self.log_target_latency_queue)

            # print(f"[UR5] Ctrl logger latency - state: {avg_state_latency_ms:.2f}ms, target: {avg_target_latency_ms:.2f}ms (avg over {len(self.log_state_latency_queue)})")

    def send_robot_target(self):
        current_timestamp = time.monotonic()
        target_timestamp = current_timestamp + self.lookahead_time_s
        target_pose_xyz_wxyz = self.interpolator.interpolate_xyz_wxyz(target_timestamp)

        target_pos = target_pose_xyz_wxyz[:3]
        target_rot = st.Rotation.from_quat(to_xyzw(target_pose_xyz_wxyz[3:7]))

        current_pose_xyz_wxyz, _ = self.get_eef_pose()
        current_pos = current_pose_xyz_wxyz[:3]
        current_rot = st.Rotation.from_quat(to_xyzw(current_pose_xyz_wxyz[3:7]))

        time_diff = self.lookahead_time_s
        pos_speed = np.linalg.norm(target_pos - current_pos) / time_diff
        rot_speed = st.Rotation.magnitude(current_rot.inv() * target_rot) / time_diff

        # if pos_speed > self.max_pos_speed_m_per_s:
        #     print(
        #         f"Pos speed {pos_speed} is greater than max pos speed {self.max_pos_speed_m_per_s}. Will not be sent to the robot."
        #     )
        #     return

        # if rot_speed > self.max_rot_speed_rad_per_s:
        #     print(
        #         f"Rot speed {rot_speed} is greater than max rot speed {self.max_rot_speed_rad_per_s}. Will not be sent to the robot."
        #     )
        #     return

        pose_command = np.concatenate([target_pos, target_rot.as_rotvec()])
        # print(f"z: {pose_command[2]:.4f} interpolator: {self.interpolator.pos[:, 2]}, {self.interpolator.timestamps}, {target_timestamp:.4f}")

        self.control_interface.servoL(
            pose_command,
            0,  # acceleration is not used
            0,  # velocity is not used
            1.0 / self.ctrl_freq
            + 0.002,  # control time # HACK: prolong this control time for 1ms
            self.lookahead_time_s,  # lookahead time
            self.gain,
        )

        self.last_target_xyz_wxyz = target_pose_xyz_wxyz
        self.last_target_timestamp = target_timestamp

    def get_eef_pose(
        self, wait_until_update: bool = False
    ) -> tuple[npt.NDArray[np.float64], float]:

        if wait_until_update:
            while (
                self.last_pose_timestamp
                == self.monotonic_time_minus_robot_time_s
                + self.receive_interface.getTimestamp()
            ):
                time.sleep(0.001)

        eef_pose = self.receive_interface.getActualTCPPose()
        robot_timestamp = self.receive_interface.getTimestamp()
        timestamp = robot_timestamp + self.monotonic_time_minus_robot_time_s
        eef_rot_xyzw = st.Rotation.from_rotvec(eef_pose[3:6]).as_quat()
        eef_xyz_wxyz = np.concatenate([eef_pose[:3], to_wxyz(eef_rot_xyzw)])
        self.last_pose_timestamp = timestamp

        return eef_xyz_wxyz, timestamp

    def get_joint_pos(
        self, wait_until_update: bool = False
    ) -> tuple[npt.NDArray[np.float64], float]:
        # Get the joint positions of the UR5 robot
        if wait_until_update and self.last_joint_pos_timestamp is not None:
            while True:
                # safer comparison with float and raw returned rtde timestamp
                current_robot_timestamp = float(self.receive_interface.getTimestamp())
                if current_robot_timestamp != self.last_joint_pos_timestamp:
                    break
                time.sleep(0.001)

        joint_pos = self.receive_interface.getActualQ()
        robot_timestamp = float(self.receive_interface.getTimestamp())

        adjusted_timestamp = robot_timestamp + self.monotonic_time_minus_robot_time_s
        self.last_joint_pos_timestamp = adjusted_timestamp

        return np.array(joint_pos, dtype=np.float64), adjusted_timestamp

    def get_eef_target(self) -> tuple[npt.NDArray[np.float64], float]:
        return self.last_target_xyz_wxyz, self.last_target_timestamp

    def schedule_eef_traj(
        self,
        eef_traj_xyz_wxyz: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
    ):
        current_timestamp = time.monotonic()

        # DEBUG: Check if the trajectory is dragging
        is_dragging = self.interpolator.pos[0, 1] < 0 and eef_traj_xyz_wxyz[-1, 1] > 0
        if is_dragging:
            print(f"Current timestamp: {current_timestamp}")
            print(
                f"Current Y: {self.interpolator.pos[:, 1]}, {self.interpolator.timestamps - current_timestamp}"
            )
            print(
                f"Input Y: {eef_traj_xyz_wxyz[:,1]}, {timestamps - current_timestamp}"
            )

        if self.dynamic_latency_range_s > 0.0 and len(timestamps) > 1:
            delta_latency = self.interpolator.find_delta_latency(
                input_poses=eef_traj_xyz_wxyz,
                input_times=timestamps,
                current_timestamp=current_timestamp,
                latency_start=-self.dynamic_latency_range_s,
                latency_end=self.dynamic_latency_range_s,
            )
            # print(f"UR5: {delta_latency=}")
            adjusted_timestamps = timestamps - delta_latency
        else:
            adjusted_timestamps = timestamps

        self.interpolator.update(
            new_trajectory=Trajectory(
                data=eef_traj_xyz_wxyz,
                timestamps=adjusted_timestamps,
            ),
            current_timestamp=current_timestamp,
        )
        if is_dragging:
            print(
                f"Updated Y: {self.interpolator.pos[:, 1]}, {self.interpolator.timestamps - current_timestamp}"
            )
            print(f"================================================")
        # print(f"{self.robot_name} schedule_eef_traj: {self.interpolator.pos=} {self.interpolator.timestamps=}")


def run_ur5_cartesian():
    os.environ["HYDRA_FULL_ERROR"] = "1"
    np.set_printoptions(precision=4)
    with hydra.initialize(config_path="../configs/controllers"):
        cfg = hydra.compose(config_name="ur5_cartesian")
    OmegaConf.set_struct(cfg, False)
    print(cfg)
    cfg.config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
    controller: UR5Cartesian = hydra.utils.instantiate(cfg)
    with controller:
        controller.run()


if __name__ == "__main__":
    run_ur5_cartesian()
