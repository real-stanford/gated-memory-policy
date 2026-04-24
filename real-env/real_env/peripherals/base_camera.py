# real_env/peripherals/multi_camera_system.py

import json
import os
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any, cast

import cv2
from cv2.typing import MatLike
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, OmegaConf
import zarr
from robotmq import RMQServer, RMQClient, deserialize, serialize

from real_env.common.data_classes import CameraCommand
from robot_utils.logging_utils import echo_exception
from robot_utils.image_utils import resize_frame_without_distortion
from robologger.loggers.video_logger import VideoLogger
from robologger.utils.huecodec import depth2logrgb


class BaseCamera(ABC):
    """
    An abstract base class for managing a system of multiple cameras as a single entity.

    This class handles all the generic backend logic:
    - Running an RMQ server for communication with clients.
    - Processing commands like START/STOP_RECORDING and RESET.
    - Managing multiple FFmpeg processes for synchronized video recording.
    - Managing multiple Zarr datasets for synchronized timestamp recording.
    - Publishing a unified packet of images and a single timestamp.

    A concrete implementation must inherit from this class and implement the
    `get_images_bgr_hwc_and_timestamp` method, which is device-specific.
    """

    def __init__(
        self,
        name: str,
        server_endpoint: str,
        logger_endpoint: str,
        camera_latency_s: float,
        camera_configs: dict[str, dict[str, Any]],
        display_resolution_WH: tuple[int, int] | None,
        config_str: str = "",
    ):
        """
        Initializes the multi-camera system.

        Args:
            name (str): A unique name for the camera system (e.g., "iPhone15Pro").
            server_endpoint (str): The ZeroMQ endpoint for the server (e.g., "tcp://*:5555").
            logger_endpoint (str): The ZeroMQ endpoint for the logger (e.g., "tcp://*:55555").
            camera_configs (dict): A dictionary where keys are camera names (e.g., "main", "ultrawide")
                                  and values are dictionaries of their configurations
                                  (e.g., {'width': 1920, 'height': 1080, 'fps': 30}).
            camera_latency_s (float): An estimated latency to subtract from timestamps for accuracy.
            display_resolution_WH (tuple | None): If provided, displays all camera feeds in separate
                                                  windows, resized to this resolution.
        """
        self.name: str = name
        self.server_endpoint: str = server_endpoint
        self.camera_latency_s: float = camera_latency_s
        self.display_resolution_WH: tuple[int, int] | None = display_resolution_WH

        self.rmq_server = RMQServer(
            server_name=self.name, server_endpoint=self.server_endpoint
        )

        print(f"Initialized RMQServer for {self.name}")

        if isinstance(camera_configs, DictConfig):
            camera_configs = cast(
                dict[str, dict[str, Any]],
                OmegaConf.to_container(camera_configs, resolve=False),
            )

        self.camera_configs: dict[str, dict[str, Any]] = camera_configs

        for cam_name, _ in self.camera_configs.items():
            self.rmq_server.add_shared_memory_topic(
                cam_name, message_remaining_time_s=1.0, shared_memory_size_gb=0.5
            )

        self.rmq_server.add_topic("timestamp", message_remaining_time_s=1.0)
        self.rmq_server.add_topic("command", message_remaining_time_s=1.0)
        self.rmq_server.add_topic("error", message_remaining_time_s=10.0)
        self.rmq_server.add_topic("config", message_remaining_time_s=1.0)
        print(f"Added topics to RMQServer for {self.name}")

        self.ffmpeg_processes: dict[str, subprocess.Popen[bytes]] = {}
        self.episode_group: zarr.Group | None = None

        self.video_logger = VideoLogger(
            name=self.name,
            endpoint=logger_endpoint,
            attr={"camera_configs": self.camera_configs},
        )
        assert self.video_logger is not None, "Video logger is not initialized"

        # Latency profiling
        self.log_latency_queue: list[float] = []
        self.log_latency_window_size: int = 10

        self.config_str: str = config_str

    def reset(self):
        print(f"Reset command received for {self.name}; no action taken")
        pass

    @abstractmethod
    def get_images_bgr_hwc_and_timestamp(
        self,
    ) -> tuple[dict[str, MatLike], float]:
        """
        **This is the core method to be implemented by a concrete subclass.**

        It must capture images from physical camera sensor(s) - either a single camera
        (for BaseCamera compatibility) or multiple cameras synchronously.

        For single camera setups (BaseCamera compatibility):
        - Return a dictionary with a single "main" key mapping to the captured frame

        For multi-camera setups:
        - Capture images from all physical camera sensors as synchronously as possible
        - Return a dictionary mapping each camera name to its corresponding frame

        **Example implementation (see iPhone class):**
        The iPhone subclass connects to multiple network streams (main, ultrawide, depth)
        via PeerTalk, processes each stream (including depth encoding), and returns a
        synchronized packet: {"main": rgb_frame, "ultrawide": rgb_frame, "depth": encoded_depth_frame}.
        It uses fallback strategies (like last known depth image) to ensure all cameras always have data.

        Returns:
            A tuple containing:
            - A dictionary mapping camera names to their corresponding image frames.
              The frames are NumPy arrays with dtype np.uint8 and shape (H, W, 3) in BGR format.
              For single camera: {"main": frame_bgr_hwc}
              For multiple cameras: {"main": frame1_bgr_hwc, "ultrawide": frame2_bgr_hwc, "depth": depth_bgr_hwc, ...}
            - A real `time.monotonic()` timestamp for when the capture was initiated.

        Implementations should use fallback strategies rather than returning empty results.
        """
        ...

    def run(self):
        """
        The main execution loop for the camera system server.
        """
        self.rmq_server.put_data(
            topic="config",
            data=serialize(self.config_str),
        )
        print(
            f"'{self.name}' server running. Waiting for commands and streaming data..."
        )

        time_queue = []
        last_time = time.monotonic()

        while True:
            raw_data_list, _ = self.rmq_server.pop_data(topic="command", n=0)
            for raw_data in raw_data_list:
                command = deserialize(raw_data)
                try:
                    if command["command"] == CameraCommand.RESET:
                        self.reset()
                    else:
                        raise ValueError(f"Invalid command: {command['command']}")
                except (KeyError, ValueError, AssertionError) as e:
                    exception_str = echo_exception()
                    error_msg = (
                        f"Error processing command in {self.name}: {e}, {exception_str}"
                    )
                    print(error_msg)
                    self.rmq_server.put_data(
                        topic="error", data=serialize({"error": error_msg})
                    )

            # time.sleep(0.01)
            frames_dict, timestamp = self.get_images_bgr_hwc_and_timestamp()

            time_queue.append(time.monotonic() - last_time)
            while len(time_queue) > 10:
                time_queue.pop(0)
            avg_fps = 1 / np.mean(time_queue) if time_queue else 0
            print(
                f"FPS: {avg_fps:.1f}, Time: {np.mean(time_queue):.4f} seconds.",
                end="\r",
            )
            last_time = time.monotonic()

            if frames_dict:
                for cam_name, frame_bgr_HWC in frames_dict.items():
                    expected_config = self.camera_configs[cam_name]
                    if expected_config["type"] == "rgb":
                        expected_shape = (
                            expected_config["height"],
                            expected_config["width"],
                            3,
                        )
                    elif expected_config["type"] == "depth":
                        expected_shape = (
                            expected_config["height"],
                            expected_config["width"],
                        )
                    else:
                        raise ValueError(
                            f"Invalid camera type: {expected_config['type']}"
                        )
                    assert frame_bgr_HWC.shape == expected_shape, (
                        f"[{self.name}] Shape mismatch for camera '{cam_name}'. "
                        f"Expected {expected_shape}, but received {frame_bgr_HWC.shape} from get_images_bgr_hwc_and_timestamp()."
                    )
                assert (
                    timestamp > 0
                ), f"Timestamp is {timestamp}, which is not greater than 0"
                corrected_timestamp = timestamp - self.camera_latency_s

                for cam_name, frame_bgr_HWC_or_depth in frames_dict.items():
                    cam_config = self.camera_configs[cam_name]
                    if cam_config["type"] == "rgb":
                        frame_rgb_HWC_or_depth = cv2.cvtColor(
                            frame_bgr_HWC_or_depth, cv2.COLOR_BGR2RGB
                        )
                    elif cam_config["type"] == "depth":
                        assert frame_bgr_HWC_or_depth.shape == (
                            cam_config["height"],
                            cam_config["width"],
                        ), f"Depth frame shape mismatch for {cam_name}. Expected {cam_config['height']}x{cam_config['width']}, but got {frame_bgr_HWC_or_depth.shape}"
                        frame_rgb_HWC_or_depth = frame_bgr_HWC_or_depth
                    self.rmq_server.put_data(
                        topic=cam_name, data=frame_rgb_HWC_or_depth.tobytes()
                    )
                self.rmq_server.put_data(
                    topic="timestamp", data=serialize(corrected_timestamp)
                )

                assert (
                    self.video_logger is not None
                ), "Video logger is not initialized but is_recording is True in BaseCamera run()"
                if self.video_logger.update_recording_state():
                    frame_dict_for_logging: dict[str, dict[str, MatLike | float]] = {}
                    for cam_name, frame_rgb_hwc_or_depth in frames_dict.items():
                        cam_config = self.camera_configs[cam_name]
                        # if rbg cam, convert bgr to rgb
                        # if depth, we pass it in directly
                        if cam_config["type"] == "rgb":
                            frame_rgb_hwc_or_depth = cv2.cvtColor(
                                frame_rgb_hwc_or_depth, cv2.COLOR_BGR2RGB
                            )
                        frame_dict_for_logging[cam_name] = {
                            "frame": frame_rgb_hwc_or_depth,
                            "timestamp": corrected_timestamp,
                        }

                    log_start_time = time.monotonic()
                    self.video_logger.log_frames(frame_dict_for_logging)
                    log_latency_ms = (time.monotonic() - log_start_time) * 1000

                    self.log_latency_queue.append(log_latency_ms)
                    if len(self.log_latency_queue) > self.log_latency_window_size:
                        self.log_latency_queue.pop(0)
                    avg_log_latency_ms = np.mean(self.log_latency_queue)
                    # print(
                    #     f"[{self.name}] Video logger latency: {avg_log_latency_ms:.2f}ms (avg over {len(self.log_latency_queue)} frames)"
                    # )

                if self.display_resolution_WH:
                    display_block_start_time = time.monotonic()

                    for cam_name, frame_HWC in frames_dict.items():
                        cam_config = self.camera_configs[cam_name]
                        if cam_config["type"] == "depth":
                            # TODO: make depth range configurable
                            # TODO: determine zrange
                            frame_HWC = depth2logrgb(frame_HWC, zrange=(0.00, 4.0))
                            frame_HWC = cv2.cvtColor(frame_HWC, cv2.COLOR_RGB2BGR)

                        source_wh = (cam_config["width"], cam_config["height"])

                        display_frame = resize_frame_without_distortion(
                            source_frame=frame_HWC.astype(np.uint8),
                            source_wh=source_wh,
                            display_wh=self.display_resolution_WH,
                        )

                        cv2.imshow(f"{self.name} - {cam_name}", display_frame)

                    cv2.waitKey(1)

                    display_block_end_time = time.monotonic()

                    if avg_fps > 0:
                        time_budget_per_frame = 1.0 / avg_fps
                        display_duration = (
                            display_block_end_time - display_block_start_time
                        )

                        if display_duration > time_budget_per_frame / 1.5:
                            print(
                                f"\n[WARNING] Display operations are slow ({display_duration*1000:.1f}ms for {self.name}) "
                                f"and may be limiting your FPS. Consider using a lower resolution or disabling the display."
                            )


class BaseCameraClient:
    """
    A client to connect to, receive data from, and send commands to a BaseCamera server.
    """

    def __init__(self, name: str, server_endpoint: str):
        self.name: str = name
        self.server_endpoint: str = server_endpoint
        self.rmq_client: RMQClient = RMQClient(
            client_name=name, server_endpoint=server_endpoint
        )
        self.config: dict[str, Any] = self.get_config()
        self.camera_names: list[str] = list(self.config["camera_configs"].keys())

    def get_config(self) -> dict[str, Any]:
        raw_data_list, _ = self.rmq_client.peek_data(topic="config", n=-1)
        if len(raw_data_list) == 0:
            raise RuntimeError(
                "No info found. Please make sure the camera is initialized."
            )

        config_str = deserialize(raw_data_list[0])
        return json.loads(config_str)

    def get_latest_images_dict_THWC(
        self, frame_num: int
    ) -> tuple[dict[str, npt.NDArray[np.uint8]], npt.NDArray[np.float64]]:
        """
        Retrieve the latest frames from all cameras in THWC format.

        Args:
            frame_num: Number of most recent frames to retrieve from each camera.

        Returns:
            A tuple containing:
            - Dictionary mapping camera names to image arrays of shape {camera_name: (N, H, W, C)}
            - Array of timestamps corresponding to each frame (frame_num,)

        Raises:
            RuntimeError: If insufficient frames are available on the server.
        """
        raw_timestamp_list, _ = self.rmq_client.peek_data(
            topic="timestamp", n=-frame_num
        )
        if len(raw_timestamp_list) < frame_num:
            raise RuntimeError(
                f"Not enough timestamp frames on server. Expected {frame_num}, found {len(raw_timestamp_list)}."
            )

        timestamps = np.array(
            [deserialize(raw_ts) for raw_ts in raw_timestamp_list], dtype=np.float64
        )

        images_dict: dict[str, npt.NDArray[np.uint8]] = {}

        for cam_name in self.camera_names:
            cam_raw_list, _ = self.rmq_client.peek_data(topic=cam_name, n=-frame_num)
            if len(cam_raw_list) < frame_num:
                raise RuntimeError(
                    f"Not enough frames for camera '{cam_name}'. Expected {frame_num}, found {len(cam_raw_list)}."
                )

            cam_config = self.config["camera_configs"][cam_name]
            height, width = cam_config["height"], cam_config["width"]

            frames_list = []
            for raw_data in cam_raw_list:
                # print(f"raw_data size: {len(raw_data)} bytes")
                if cam_config["type"] == "rgb":
                    frame_HWC = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                        height, width, 3
                    )
                elif cam_config["type"] == "depth":
                    # print(f"len(raw_data): {len(raw_data)}")
                    frame_HWC = np.frombuffer(raw_data, dtype=np.float32).reshape(
                        height, width
                    )
                else:
                    raise ValueError(f"Unsupported camera type: {cam_config['type']}")
                frames_list.append(frame_HWC)

            images_dict[cam_name] = np.array(frames_list)

        assert np.all(
            timestamps[1:] - timestamps[:-1] >= 0
        ), "Timestamps are not in increasing order"

        return images_dict, timestamps

    def reset(self):
        print("Client sending RESET command.")
        self.rmq_client.put_data(
            topic="command", data=serialize({"command": CameraCommand.RESET})
        )
