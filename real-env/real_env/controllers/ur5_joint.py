import time

import click
import numpy as np
import numpy.typing as npt
import rtde_control
import rtde_receive

from real_env.common.constants import RMQ_PORTS
from real_env.common.data_classes import Trajectory
from real_env.common.interpolator import JointTrajInterpolator
from real_env.controllers.base_controller import BaseController
from robot_utils.pose_utils import to_wxyz
from robologger import RobotCtrlLogger
import scipy.spatial.transform as st


class UR5Joint(BaseController):
    def __init__(
        self,
        server_endpoint: str,
        robot_ip: str,
        logger_endpoint: str,
        robot_name: str,
        home_joint_q_rad: list[float] | npt.NDArray[np.float64] | None = None,
        ctrl_freq: float = 125.0,
        gain: float = 300.0,
        max_joint_speed_rad_per_s: float | list[float] = 5,  # XXX: subject to change
        max_rot_speed_rad_per_s: float = 1.0,  # unused for joint controller
        vel_calc_interval_min_s: float = 0.05,  # unique to joint controller
        lookahead_time_s: float = 0.1,
        reset_time_s: float = 5.0,
        traj_smooth_time_s: float = 0.0,
        # tcp_offset_pose: list[float] | npt.NDArray[np.float64] | None = None,
        dynamic_latency: bool = False,
        trim_new_trajectory_time_s: float = 0.09,  # Slightly lower than 0.1
    ):
        super().__init__(
            robot_name=robot_name,
            server_endpoint=server_endpoint,
            control_mode="JOINT",
            ctrl_freq=ctrl_freq,
        )

        self.dynamic_latency: bool = dynamic_latency

        # Properties
        self.robot_ip: str = robot_ip
        self.gain: float = gain
        self.max_joint_speed_rad_per_s: float | list[float] = max_joint_speed_rad_per_s
        self.max_rot_speed_rad_per_s: float = max_rot_speed_rad_per_s
        self.vel_calc_interval_min_s: float = vel_calc_interval_min_s
        self.lookahead_time_s: float = lookahead_time_s
        self.reset_time_s: float = reset_time_s
        self.traj_smooth_time_s: float = traj_smooth_time_s
        self.trim_new_trajectory_time_s: float = trim_new_trajectory_time_s

        if home_joint_q_rad is None:
            self.home_joint_q_rad: npt.NDArray[np.float64] | None = None
        else:
            assert len(home_joint_q_rad) == 6, "UR5 must have 6 joint q values"
            self.home_joint_q_rad = np.array(home_joint_q_rad)

        # if tcp_offset_pose is None:
        #     self.tcp_offset_pose: npt.NDArray[np.float64] = np.zeros(6) # x,y,z,rx,ry,rz
        # else:
        #     assert len(tcp_offset_pose) == 6, "TCP offset pose must have 6 values (x,y,z,rx,ry,rz)"
        #     self.tcp_offset_pose = np.array(tcp_offset_pose)

        self.interpolator: JointTrajInterpolator
        self.control_interface: rtde_control.RTDEControlInterface
        self.receive_interface: rtde_receive.RTDEReceiveInterface

        self.last_target_pos_rad: npt.NDArray[np.float64]
        self.last_target_timestamp: float
        self.monotonic_time_offset_s: float

        self.last_pos_timestamp: float

        self.joint_logger = RobotCtrlLogger(
            name="right_arm",
            endpoint=logger_endpoint,
            attr={
                "robot_name": self.robot_name,
                "ctrl_freq": self.ctrl_freq,
                "num_joints": 6,
            },
            log_eef_pose=True,
            log_joint_pos=True,
            target_type="joint_pos",
            joint_units="radians",
        )

        assert self.joint_logger is not None, "Joint logger is not initialized"

        self.last_robot_timestamp: float | None = None

    def _calibrate_time_difference(self):
        # Calibrate time difference between controller and robot
        assert self.is_connected, "Robot is not connected"

        last_robot_timestamp = float(self.receive_interface.getTimestamp())
        time_diffs: list[float] = []

        for _ in range(20):
            robot_timestamp = last_robot_timestamp
            while robot_timestamp == last_robot_timestamp:
                time.sleep(0.001)  # prevent busy wait
                robot_timestamp = float(self.receive_interface.getTimestamp())
            time_diffs.append(time.monotonic() - robot_timestamp)
            last_robot_timestamp = robot_timestamp

        self.monotonic_time_offset_s = float(np.mean(time_diffs))

    def connect(self):
        # Connect to the UR5 robot
        if self.is_connected:
            return

        print(
            f"Connecting to UR5 robot at {self.robot_ip}... If this run into error, restart the program"
        )

        self.control_interface = rtde_control.RTDEControlInterface(
            hostname=self.robot_ip
        )
        print("Control interface connected to robot @ ", self.robot_ip)

        self.receive_interface = rtde_receive.RTDEReceiveInterface(
            hostname=self.robot_ip
        )
        print("Receive interface connected to robot @ ", self.robot_ip)

        assert (
            self.control_interface.isConnected()
        ), "Failed to connect control interface"
        assert (
            self.receive_interface.isConnected()
        ), "Failed to connect receive interface"

        self.is_connected = True
        self._calibrate_time_difference()
        print("Time difference calibrated.")
        # self.control_interface.setTcp(self.tcp_offset_pose.tolist())

        joint_pos, self.last_pos_timestamp = self.get_joint_pos()

        self.interpolator = JointTrajInterpolator(
            init_trajectory=Trajectory(
                data=joint_pos[np.newaxis, :],  # shape (1, 6); additional axis for time
                timestamps=np.array([self.last_pos_timestamp]),
                ## shape (1,); initial timestamp, used returned timestamp from get_joint_pos()
                # for consistency instead of time.monotonic()
            ),
            traj_smooth_time_s=self.traj_smooth_time_s,
            max_joint_speed_per_s=self.max_joint_speed_rad_per_s,
            trim_new_trajectory_time_s=self.trim_new_trajectory_time_s,
            vel_calc_interval_min_s=self.vel_calc_interval_min_s,
        )
        self.last_target_pos_rad = joint_pos
        self.last_target_timestamp = time.monotonic()

        print("UR5 is connected.")

    def disconnect(self):
        # Disconnect from the UR5 robot
        if not self.is_connected:
            return

        print("Disconnecting from UR5...")
        self.control_interface.servoStop()
        self.control_interface.stopScript()
        self.control_interface.disconnect()
        self.receive_interface.disconnect()
        self.is_connected = False
        print("UR5 is disconnected")

    def reset(self):
        # Reset the UR5 robot to initial state
        if self.home_joint_q_rad is None:
            print("Home joint positions not set. Cannot reset.")
            return

        current_timestamp = time.monotonic()
        current_pos, _ = self.get_joint_pos()

        reset_trajectory = Trajectory(
            data=np.concatenate(
                [current_pos[np.newaxis, :], self.home_joint_q_rad[np.newaxis, :]],
                axis=0,
            ),
            timestamps=np.array(
                [current_timestamp, current_timestamp + self.reset_time_s]
            ),
        )

        self.interpolator.update(
            new_trajectory=reset_trajectory, current_timestamp=current_timestamp
        )

    def record_data(self):
        # Record data from the UR5 robot
        assert (
            self.joint_logger is not None
        ), "Joint logger is not initialized but record_data() is invoked"

        joint_pos, joint_timestamp = self.get_joint_pos()
        joint_target, joint_target_timestamp = self.get_joint_target()
        pose_xyz_wxyz, pose_timestamp = self.get_eef_pose()

        if self.joint_logger.update_recording_state():
            self.joint_logger.log_state(
                state_timestamp=joint_timestamp,
                state_joint_pos=joint_pos.astype(np.float64),
                state_pos_xyz=pose_xyz_wxyz[:3],
                state_quat_wxyz=pose_xyz_wxyz[3:],
            )

            self.joint_logger.log_target(
                target_timestamp=joint_target_timestamp,
                target_joint_pos=joint_target.astype(np.float64),
            )
            # no printing here to avoid overhead and jitter
            # print(f"Logged joint pos with timestamp {joint_timestamp}, target with timestamp {joint_target_timestamp}")

    def _apply_joint_speed_limit(
        self,
        current_joint_pos: npt.NDArray[np.float64],
        target_joint_pos: npt.NDArray[np.float64],
        dt_s: float,
    ) -> npt.NDArray[np.float64]:
        """
        Clamp target joint command to per-joint velocity limits.
        Emits a warning if clamping occurs.
        """
        # Conservative dt to avoid jitter-induced spikes
        dt = max(float(dt_s), 1.0 / float(self.ctrl_freq))

        current_joint_pos = np.asarray(current_joint_pos, dtype=np.float64)
        target_joint_pos = np.asarray(target_joint_pos, dtype=np.float64)
        delta_pos = target_joint_pos - current_joint_pos

        # Normalize limits to per-joint vector
        speed_limit: float | list[float] = self.max_joint_speed_rad_per_s
        if np.isscalar(speed_limit):
            velocity_limit = np.full_like(delta_pos, speed_limit, dtype=np.float64)
        else:
            # Allows for per joint rad limits if list-like provided
            velocity_limit = np.asarray(speed_limit, dtype=np.float64)
            assert (
                velocity_limit.shape == delta_pos.shape
            ), f"Joint speed limit shape mismatch, expected shape {delta_pos.shape}, got {velocity_limit.shape}"

        max_delta_pos = velocity_limit * dt
        delta_pos_clamped = np.clip(delta_pos, -max_delta_pos, max_delta_pos)

        if not np.allclose(delta_pos, delta_pos_clamped):
            # Report worst offending joint for easy debugging
            speeds = np.abs(delta_pos) / dt
            joint_idx = int(np.argmax(speeds - velocity_limit))
            print(
                f"[UR5] Joint speed clamped: "
                f"joint {joint_idx}, cmd={speeds[joint_idx]:.3f} rad/s, "
                f"limit={velocity_limit[joint_idx]:.3f} rad/s"
            )

        return current_joint_pos + delta_pos_clamped

    def send_robot_target(self):
        # Send target commands to the UR5 robot
        current_timestamp = time.monotonic()
        target_timestamp = current_timestamp + self.lookahead_time_s
        target_joint_pos = self.interpolator.interpolate(timestamp=target_timestamp)

        current_joint_pos, _ = self.get_joint_pos()

        commanded_joint_pos = self._apply_joint_speed_limit(
            current_joint_pos=current_joint_pos,
            target_joint_pos=target_joint_pos,
            dt_s=(target_timestamp - current_timestamp),
            # dt_s is effectively lookahead_time_s; however does not reflect jitter
        )

        self.control_interface.servoJ(
            commanded_joint_pos.tolist(),
            0.0,
            0.0,
            1.0 / self.ctrl_freq,  # blocking during for 1 control cycle
            self.lookahead_time_s,
            self.gain,
        )

        self.last_target_pos_rad = commanded_joint_pos
        self.last_target_timestamp = target_timestamp

    def get_joint_pos(
        self,
        wait_until_update: bool = False,
    ) -> tuple[npt.NDArray[np.float64], float]:
        # Get the joint positions of the UR5 robot
        if wait_until_update and self.last_robot_timestamp is not None:
            while True:
                # safer comparison with float and raw returned rtde timestamp
                current_robot_timestamp = float(self.receive_interface.getTimestamp())
                if current_robot_timestamp != self.last_robot_timestamp:
                    break
                time.sleep(0.001)

        joint_pos = self.receive_interface.getActualQ()
        robot_timestamp = float(self.receive_interface.getTimestamp())

        self.last_robot_timestamp = robot_timestamp
        adjusted_timestamp = robot_timestamp + self.monotonic_time_offset_s
        self.last_pos_timestamp = adjusted_timestamp

        return np.array(joint_pos, dtype=np.float64), adjusted_timestamp

    def get_joint_target(self):
        # Get the target joint positions
        return self.last_target_pos_rad, self.last_target_timestamp

    def get_eef_pose(self, wait_until_update=False):
        if wait_until_update:
            while (
                self.last_pose_timestamp
                == self.monotonic_time_offset_s + self.receive_interface.getTimestamp()
            ):
                time.sleep(0.001)

        eef_pose = self.receive_interface.getActualTCPPose()
        robot_timestamp = self.receive_interface.getTimestamp()
        timestamp = robot_timestamp + self.monotonic_time_offset_s
        eef_rot_xyzw = st.Rotation.from_rotvec(eef_pose[3:6]).as_quat()
        eef_xyz_wxyz = np.concatenate([eef_pose[:3], to_wxyz(eef_rot_xyzw)])
        self.last_pose_timestamp = timestamp

        return eef_xyz_wxyz, timestamp

    def schedule_joint_traj(
        self,
        joint_traj: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
    ):
        # Schedule joint trajectory
        current_timestamp = time.monotonic()

        if self.dynamic_latency and len(timestamps) > 1:
            delta_latency = self.interpolator.find_delta_latency(
                input_poses=joint_traj,
                input_times=timestamps,
                current_timestamp=current_timestamp,
            )
            print(f"[UR5] Dynamic latency adjustment: {delta_latency:.4f} s")
            adjusted_timestamps = timestamps - delta_latency
        else:
            adjusted_timestamps = timestamps

        self.interpolator.update(
            new_trajectory=Trajectory(
                data=joint_traj,
                timestamps=adjusted_timestamps,
            ),
            current_timestamp=current_timestamp,
        )


@click.command()
@click.option("--robot-ip", type=str, default="192.168.2.124")
@click.option(
    "--server-endpoint", type=str, default=f"tcp://localhost:{RMQ_PORTS.UR5_CONTROLLER}"
)
@click.option(
    "--logger-endpoint", type=str, default=f"tcp://localhost:{RMQ_PORTS.UR5_LOGGER}"
)
def run_ur5_joint(
    robot_ip: str,
    server_endpoint: str,
    logger_endpoint: str,
):
    # Run the UR5 joint control loop
    np.set_printoptions(precision=4)
    with UR5Joint(
        robot_ip=robot_ip,
        server_endpoint=server_endpoint,
        logger_endpoint=logger_endpoint,
        robot_name="right_arm",
        # tcp_offset_pose=[0.0, 0.0, 0.13, 0.0, 0.0, 0.0],  # For WSG50 + fin-ray gripper
        dynamic_latency=False,
    ) as ur5_joint:
        ur5_joint.run()


if __name__ == "__main__":
    run_ur5_joint()
