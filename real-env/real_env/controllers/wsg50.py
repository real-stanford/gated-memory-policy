"""
Script adapted from https://github.com/real-stanford/universal_manipulation_interface/blob/main/umi/real_world/wsg_binary_driver.py
and https://github.com/real-stanford/universal_manipulation_interface/blob/main/umi/real_world/wsg_controller.py

Command set please refer to https://d16vz4puxlsxm1.cloudfront.net/asset/076200133045-Prod/document_qmmvgeja4514b9lnbpjk4kbp4e/WSG%20Command%20Set%20Reference,%20Firmware%202-3.pdf
"""

import enum
import json
import socket
import struct
import time
import traceback
from typing import Any

import hydra
import numpy as np
import numpy.typing as npt
import os
from omegaconf import OmegaConf
from real_env.common.data_classes import Trajectory
from real_env.common.interpolator import JointTrajInterpolator
from real_env.controllers.base_controller import BaseController
from robologger import RobotCtrlLogger

CRC_TABLE_CCITT16 = [
    0x0000,
    0x1021,
    0x2042,
    0x3063,
    0x4084,
    0x50A5,
    0x60C6,
    0x70E7,
    0x8108,
    0x9129,
    0xA14A,
    0xB16B,
    0xC18C,
    0xD1AD,
    0xE1CE,
    0xF1EF,
    0x1231,
    0x0210,
    0x3273,
    0x2252,
    0x52B5,
    0x4294,
    0x72F7,
    0x62D6,
    0x9339,
    0x8318,
    0xB37B,
    0xA35A,
    0xD3BD,
    0xC39C,
    0xF3FF,
    0xE3DE,
    0x2462,
    0x3443,
    0x0420,
    0x1401,
    0x64E6,
    0x74C7,
    0x44A4,
    0x5485,
    0xA56A,
    0xB54B,
    0x8528,
    0x9509,
    0xE5EE,
    0xF5CF,
    0xC5AC,
    0xD58D,
    0x3653,
    0x2672,
    0x1611,
    0x0630,
    0x76D7,
    0x66F6,
    0x5695,
    0x46B4,
    0xB75B,
    0xA77A,
    0x9719,
    0x8738,
    0xF7DF,
    0xE7FE,
    0xD79D,
    0xC7BC,
    0x48C4,
    0x58E5,
    0x6886,
    0x78A7,
    0x0840,
    0x1861,
    0x2802,
    0x3823,
    0xC9CC,
    0xD9ED,
    0xE98E,
    0xF9AF,
    0x8948,
    0x9969,
    0xA90A,
    0xB92B,
    0x5AF5,
    0x4AD4,
    0x7AB7,
    0x6A96,
    0x1A71,
    0x0A50,
    0x3A33,
    0x2A12,
    0xDBFD,
    0xCBDC,
    0xFBBF,
    0xEB9E,
    0x9B79,
    0x8B58,
    0xBB3B,
    0xAB1A,
    0x6CA6,
    0x7C87,
    0x4CE4,
    0x5CC5,
    0x2C22,
    0x3C03,
    0x0C60,
    0x1C41,
    0xEDAE,
    0xFD8F,
    0xCDEC,
    0xDDCD,
    0xAD2A,
    0xBD0B,
    0x8D68,
    0x9D49,
    0x7E97,
    0x6EB6,
    0x5ED5,
    0x4EF4,
    0x3E13,
    0x2E32,
    0x1E51,
    0x0E70,
    0xFF9F,
    0xEFBE,
    0xDFDD,
    0xCFFC,
    0xBF1B,
    0xAF3A,
    0x9F59,
    0x8F78,
    0x9188,
    0x81A9,
    0xB1CA,
    0xA1EB,
    0xD10C,
    0xC12D,
    0xF14E,
    0xE16F,
    0x1080,
    0x00A1,
    0x30C2,
    0x20E3,
    0x5004,
    0x4025,
    0x7046,
    0x6067,
    0x83B9,
    0x9398,
    0xA3FB,
    0xB3DA,
    0xC33D,
    0xD31C,
    0xE37F,
    0xF35E,
    0x02B1,
    0x1290,
    0x22F3,
    0x32D2,
    0x4235,
    0x5214,
    0x6277,
    0x7256,
    0xB5EA,
    0xA5CB,
    0x95A8,
    0x8589,
    0xF56E,
    0xE54F,
    0xD52C,
    0xC50D,
    0x34E2,
    0x24C3,
    0x14A0,
    0x0481,
    0x7466,
    0x6447,
    0x5424,
    0x4405,
    0xA7DB,
    0xB7FA,
    0x8799,
    0x97B8,
    0xE75F,
    0xF77E,
    0xC71D,
    0xD73C,
    0x26D3,
    0x36F2,
    0x0691,
    0x16B0,
    0x6657,
    0x7676,
    0x4615,
    0x5634,
    0xD94C,
    0xC96D,
    0xF90E,
    0xE92F,
    0x99C8,
    0x89E9,
    0xB98A,
    0xA9AB,
    0x5844,
    0x4865,
    0x7806,
    0x6827,
    0x18C0,
    0x08E1,
    0x3882,
    0x28A3,
    0xCB7D,
    0xDB5C,
    0xEB3F,
    0xFB1E,
    0x8BF9,
    0x9BD8,
    0xABBB,
    0xBB9A,
    0x4A75,
    0x5A54,
    0x6A37,
    0x7A16,
    0x0AF1,
    0x1AD0,
    0x2AB3,
    0x3A92,
    0xFD2E,
    0xED0F,
    0xDD6C,
    0xCD4D,
    0xBDAA,
    0xAD8B,
    0x9DE8,
    0x8DC9,
    0x7C26,
    0x6C07,
    0x5C64,
    0x4C45,
    0x3CA2,
    0x2C83,
    0x1CE0,
    0x0CC1,
    0xEF1F,
    0xFF3E,
    0xCF5D,
    0xDF7C,
    0xAF9B,
    0xBFBA,
    0x8FD9,
    0x9FF8,
    0x6E17,
    0x7E36,
    0x4E55,
    0x5E74,
    0x2E93,
    0x3EB2,
    0x0ED1,
    0x1EF0,
]


def checksum_update_crc16(data: bytes, crc: int = 0xFFFF):
    for b in data:
        crc = CRC_TABLE_CCITT16[(crc ^ b) & 0x00FF] ^ (crc >> 8)
    return crc


class StatusCode(enum.IntEnum):
    E_SUCCESS = 0
    E_NOT_AVAILABLE = 1
    E_NO_SENSOR = 2
    E_NOT_INITIALIZED = 3
    E_ALREADY_RUNNING = 4
    E_FEATURE_NOT_SUPPORTED = 5
    E_INCONSISTENT_DATA = 6
    E_TIMEOUT = 7
    E_READ_ERROR = 8
    E_WRITE_ERROR = 9
    E_INSUFFICIENT_RESOURCES = 10
    E_CHECKSUM_ERROR = 11
    E_NO_PARAM_EXPECTED = 12
    E_NOT_ENOUGH_PARAMS = 13
    E_CMD_UNKNOWN = 14
    E_CMD_FORMAT_ERROR = 15
    E_ACCESS_DENIED = 16
    E_ALREADY_OPEN = 17
    E_CMD_FAILED = 18
    E_CMD_ABORTED = 19
    E_INVALID_HANDLE = 20
    E_NOT_FOUND = 21
    E_NOT_OPEN = 22
    E_IO_ERROR = 23
    E_INVALID_PARAMETER = 24
    E_INDEX_OUT_OF_BOUNDS = 25
    E_CMD_PENDING = 26
    E_OVERRUN = 27
    RANGE_ERROR = 28
    E_AXIS_BLOCKED = 29
    E_FILE_EXIST = 30


class CommandId(enum.IntEnum):
    Disconnect = 0x07
    Homing = 0x20
    PrePosition = 0x21
    Stop = 0x22
    FastStop = 0x23
    AckFastStop = 0x24

    GetGripperWidth = 0x43
    GetState = 0xBA  # Customized command
    PDControl = 0xBB  # Customized command


def args_to_bytes(*args: list[float | int | str], int_bytes: int = 1) -> bytes:
    buf = list()
    for arg in args:
        if isinstance(arg, float):
            # little endian 32bit float
            buf.append(struct.pack("<f", arg))
        elif isinstance(arg, int):
            buf.append(arg.to_bytes(length=int_bytes, byteorder="little"))
        elif isinstance(arg, str):
            buf.append(arg.encode("ascii"))
        elif isinstance(arg, bytes):
            buf.append(arg)
        else:
            raise RuntimeError(f"Unsupported type {type(arg)}")
    result = b"".join(buf)
    return result


class WSG50(BaseController):
    def __init__(
        self,
        server_endpoint: str,
        robot_ip: str,
        logger_endpoint: str,
        robot_name: str,
        robot_port: int,
        lookahead_time_s: float,
        traj_smooth_time_s: float,
        home_gripper_width_m: float,
        max_pos_speed_m_per_s: float,
        trim_new_trajectory_time_s: float,
        ctrl_freq: float,
        dynamic_latency: bool,
        config_str: str = "",
    ):
        super().__init__(
            robot_name=robot_name,
            control_mode="JOINT",
            server_endpoint=server_endpoint,
            ctrl_freq=ctrl_freq,
            config_str=config_str,
        )

        self.robot_ip: str = robot_ip
        self.robot_port: int = robot_port
        self.traj_smooth_time_s: float = traj_smooth_time_s
        self.max_pos_speed_m_per_s: float = max_pos_speed_m_per_s
        self.lookahead_time_s: float = lookahead_time_s
        self.trim_new_trajectory_time_s: float = trim_new_trajectory_time_s
        self.dynamic_latency: bool = dynamic_latency

        # States

        self.robot_socket: socket.socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )
        self.interpolator: JointTrajInterpolator  # Will be initialized in connect()
        self.last_gripper_width: npt.NDArray[np.float64]
        self.last_gripper_width_timestamp: float = 0.0
        self.last_target_gripper_width: npt.NDArray[np.float64]
        self.last_target_gripper_width_timestamp: float
        self.monotonic_time_minus_robot_time_s: float

        self.joint_logger = RobotCtrlLogger(
            name="right_end_effector",
            endpoint=logger_endpoint,
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
        assert self.joint_logger is not None, "Joint logger is not initialized"

        # Latency profiling
        self.log_state_latency_queue: list[float] = []
        self.log_target_latency_queue: list[float] = []
        self.log_latency_window_size: int = 10

    ##### Low-level APIs #####

    def _send(self, cmd_id: int, payload: bytes):

        preamble_b = 0xAA.to_bytes(1, "little") * 3
        cmd_b = int(cmd_id).to_bytes(1, "little")
        size_b = len(payload).to_bytes(2, "little")
        msg_b = preamble_b + cmd_b + size_b + payload
        checksum_b = checksum_update_crc16(msg_b).to_bytes(2, "little")
        msg_b += checksum_b
        return self.robot_socket.send(msg_b)

    def _receive(self) -> dict[str, Any]:
        # syncing
        sync = 0
        while sync != 3:
            res = self.robot_socket.recv(1)
            if res == 0xAA.to_bytes(1, "little"):
                sync += 1

        # read header
        cmd_id_b = self.robot_socket.recv(1)
        cmd_id = int.from_bytes(cmd_id_b, "little")

        # read size
        size_b = self.robot_socket.recv(2)
        size = int.from_bytes(size_b, "little")

        # read payload
        payload_b = self.robot_socket.recv(size)
        status_code = int.from_bytes(payload_b[:2], "little")

        parameters_b = payload_b[2:]

        # read checksum
        checksum_b = self.robot_socket.recv(2)

        # correct checksum ends in zero
        header_checksum = 0x50F5
        msg_checksum = checksum_update_crc16(
            cmd_id_b + size_b + payload_b + checksum_b, crc=header_checksum
        )
        if msg_checksum != 0:
            raise RuntimeError("Corrupted packet received from WSG")

        result = {
            "command_id": cmd_id,
            "status_code": status_code,
            "payload_bytes": parameters_b,
        }
        return result

    def _cmd(
        self, cmd: CommandId, *args, wait: bool = True, ignore_other: bool = False
    ) -> dict:
        payload = args_to_bytes(*args)

        res = self._send(cmd.value, payload)
        if res < 0:
            raise RuntimeError("Message send failed.")

        while True:
            msg = self._receive()
            if ignore_other and msg["command_id"] != cmd.value:
                continue

            if msg["command_id"] != cmd.value:
                raise RuntimeError(
                    "Response ID ({:02X}) does not match submitted command ID ({:02X})\n".format(
                        msg["command_id"], cmd.value
                    )
                )
            if wait:
                if msg["status_code"] == StatusCode.E_CMD_PENDING.value:
                    continue

            msg["command_id"] = CommandId(msg["command_id"])
            msg["status_code"] = StatusCode(msg["status_code"])

            if msg["status_code"] != StatusCode.E_SUCCESS.value:
                if msg["status_code"] == StatusCode.E_CMD_UNKNOWN.value:
                    print(
                        f"Command {cmd} not found. Please make sure you are running the custom script on WSG Control Panel -> Scripting -> Interacitve Scripting"
                    )
                raise RuntimeError(
                    f'Command {cmd.name} not successful: {msg["status_code"].name}'
                )

            return msg

    def _ack_fast_stop(self):
        self._cmd(CommandId.AckFastStop, "ack", wait=False, ignore_other=True)

    def _stop_robot(self):
        self._cmd(CommandId.Stop, wait=False, ignore_other=True)

    def _update_states(self, reply_bytes: bytes):
        state = reply_bytes[0]
        values = list()
        for i in range(4):
            start = i * 4 + 1
            end = start + 4
            values.append(struct.unpack("<f", reply_bytes[start:end])[0])
        info = {
            "state": state,
            "position": values[0],
            "velocity": values[1],
            "force_motor": values[2],
            "measure_timestamp": values[3] / 10,
            "is_moving": (state & 0x02) != 0,
        }

        assert isinstance(info["position"], float)
        assert isinstance(info["measure_timestamp"], float)

        self.last_gripper_width = np.array([info["position"] / 1e3])
        self.last_gripper_width_timestamp = (
            info["measure_timestamp"] + self.monotonic_time_minus_robot_time_s
        )

    def _pd_control(
        self,
        position_mm: float,
        velocity_mm_per_s: float,
        kp: float = 15.0,
        kd: float = 1e-2,  # Original value: 1e-3
        travel_force_limit: float = 80.0,
        blocked_force_limit: float | None = None,
    ):
        if blocked_force_limit is None:
            blocked_force_limit = travel_force_limit
        assert kp > 0
        assert kd >= 0
        return self._cmd(
            CommandId.PDControl,
            position_mm,
            0.0,  # HACK: not using gripper velocity
            kp,
            kd,
            travel_force_limit,
            blocked_force_limit,
        )

    def _get_raw_states(self):
        res = self._cmd(CommandId.GetState, "")
        state = res["payload_bytes"][0]
        values = list()
        for i in range(4):
            start = i * 4 + 1
            end = start + 4
            values.append(struct.unpack("<f", res["payload_bytes"][start:end])[0])
        info = {
            "state": state,
            "position": values[0],
            "velocity": values[1],
            "force_motor": values[2],
            "measure_timestamp": values[3] / 10,
            "is_moving": (state & 0x02) != 0,
        }
        return info

    def _calibrate_time_difference(self):
        assert self.is_connected

        robot_timestamp = self._get_raw_states()["measure_timestamp"]
        last_robot_timestamp = robot_timestamp

        time_diffs: list[float] = []

        print(f"{self.robot_name}: Calibrating time difference")

        for _ in range(10):
            while robot_timestamp == last_robot_timestamp:
                robot_timestamp: float = self._get_raw_states()["measure_timestamp"]
            time_diffs.append(time.monotonic() - robot_timestamp)
            last_robot_timestamp = robot_timestamp
            time.sleep(0.005)

        self.monotonic_time_minus_robot_time_s = float(np.mean(time_diffs))

    ##### Control Interface #####

    def get_joint_pos(self, wait_until_update: bool = False):

        prev_gripper_width_timestamp = self.last_gripper_width_timestamp

        if wait_until_update:
            while self.last_gripper_width_timestamp == prev_gripper_width_timestamp:
                res = self._cmd(CommandId.GetState, "")
                self._update_states(res["payload_bytes"])
        else:
            res = self._cmd(CommandId.GetState, "")
            self._update_states(res["payload_bytes"])

        return self.last_gripper_width, self.last_gripper_width_timestamp

    def get_joint_target(self):
        return self.last_target_gripper_width, self.last_target_gripper_width_timestamp

    def connect(self):
        if self.is_connected:
            return

        print(f"Connecting to {self.robot_name} at {self.robot_ip}:{self.robot_port}")
        self.robot_socket.connect((self.robot_ip, self.robot_port))
        self._ack_fast_stop()
        self.is_connected = True
        self._calibrate_time_difference()

        gripper_width, joint_pos_timestamp = self.get_joint_pos()
        self.interpolator = JointTrajInterpolator(
            init_trajectory=Trajectory(
                data=np.array([gripper_width]),
                timestamps=np.array([time.monotonic()]),
            ),
            traj_smooth_time_s=self.traj_smooth_time_s,
            max_joint_speed_per_s=self.max_pos_speed_m_per_s,
            trim_new_trajectory_time_s=self.trim_new_trajectory_time_s,
        )
        self.last_target_gripper_width = gripper_width
        self.last_target_gripper_width_timestamp = time.monotonic()
        print(f"{self.robot_name} is connected")

    def disconnect(self):
        self._stop_robot()
        del self.interpolator
        self.robot_socket.close()

    def reset(self):
        self._cmd(CommandId.Homing, 1)  # 1 means positive direction
        gripper_width, _ = self.get_joint_pos()
        self.interpolator = JointTrajInterpolator(
            init_trajectory=Trajectory(
                data=np.array([gripper_width]),
                timestamps=np.array([time.monotonic()]),
            ),
            traj_smooth_time_s=self.traj_smooth_time_s,
            max_joint_speed_per_s=self.max_pos_speed_m_per_s,
            trim_new_trajectory_time_s=self.trim_new_trajectory_time_s,
        )
        self.last_target_gripper_width = gripper_width
        self.last_target_gripper_width_timestamp = time.monotonic()

    def record_data(self):
        assert (
            self.joint_logger is not None
        ), "Joint logger is not initialized but record_data() is called"

        joint_pos, joint_pos_timestamp = self.get_joint_pos()
        joint_target, joint_target_timestamp = self.get_joint_target()

        if self.joint_logger.update_recording_state():
            log_state_start = time.monotonic()
            self.joint_logger.log_state(
                state_timestamp=joint_pos_timestamp,
                state_joint_pos=joint_pos.astype(np.float64),
            )
            log_state_latency_ms = (time.monotonic() - log_state_start) * 1000

            log_target_start = time.monotonic()
            self.joint_logger.log_target(
                target_timestamp=joint_target_timestamp,
                target_joint_pos=joint_target.astype(np.float64),
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

            # print(f"[WSG50] End effector logger latency - state: {avg_state_latency_ms:.2f}ms, target: {avg_target_latency_ms:.2f}ms (avg over {len(self.log_state_latency_queue)})")

    def send_robot_target(self):

        current_timestamp = time.monotonic()
        target_pos_mm = (
            self.interpolator.interpolate(current_timestamp + self.lookahead_time_s)[0]
            * 1e3
        )

        target_vel_mm_per_s = (
            target_pos_mm / 1e3 - self.last_target_gripper_width[0]
        ) * self.ctrl_freq
        self.last_target_gripper_width = np.array([target_pos_mm / 1e3])
        self.last_target_gripper_width_timestamp = current_timestamp

        # print(f"WSG50: {target_pos_mm=:.3f}, {target_vel_mm_per_s=:.3f}")

        res = self._pd_control(target_pos_mm, target_vel_mm_per_s)

        self._update_states(res["payload_bytes"])

    def schedule_joint_traj(
        self, joint_traj: npt.NDArray[np.float64], timestamps: npt.NDArray[np.float64]
    ):
        if self.dynamic_latency and len(timestamps) > 1:
            delta_latency = self.interpolator.find_delta_latency(
                input_poses=joint_traj,
                input_times=timestamps,
                latency_end=0.6,
                current_timestamp=time.monotonic(),
            )
            # print(f"WSG50: {delta_latency=:.3f}")
            adjusted_timestamps = timestamps - delta_latency
        else:
            adjusted_timestamps = timestamps

        self.interpolator.update(
            new_trajectory=Trajectory(
                data=joint_traj,
                timestamps=adjusted_timestamps,
            ),
            current_timestamp=time.monotonic(),
        )


def run_wsg50():
    os.environ["HYDRA_FULL_ERROR"] = "1"
    with hydra.initialize(config_path="../configs/controllers"):
        cfg = hydra.compose(config_name="wsg50")
    print(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
    wsg50 = hydra.utils.instantiate(cfg)
    with wsg50:
        wsg50.run()


if __name__ == "__main__":
    run_wsg50()
