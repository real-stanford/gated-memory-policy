import json
import os
import sys
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
import cv2
from cv2.typing import MatLike
import time

from real_env.common.constants import RMQ_PORTS
from real_env.peripherals.base_camera import BaseCamera
from typing import Any


class WebCam(BaseCamera):
    def __init__(
        self,
        camera_id: int,
        cam_img_format: str,
        camera_configs: dict[str, dict[str, Any]],
        *args,
        **kwargs,
    ):
        super().__init__(camera_configs=camera_configs, *args, **kwargs)

        print(f"Initializing WebCam with camera_id={camera_id}")
        try:
            self.cap: cv2.VideoCapture = cv2.VideoCapture(camera_id)
        except Exception as e:
            print(
                f"Error initializing WebCam with camera_id={camera_configs['main']['camera_id']}: {e}"
            )
            raise e

        if cam_img_format:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam_img_format))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_configs["main"]["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_configs["main"]["height"])
        self.cap.set(cv2.CAP_PROP_FPS, camera_configs["main"]["fps"])
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        # self.cap.set(cv2.CAP_PROP_EXPOSURE, -10)
        # self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        # self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5000)

        assert (
            actual_width == camera_configs["main"]["width"]
            and actual_height == camera_configs["main"]["height"]
        ), f"Width and height do not match. Expected ({camera_configs['main']['width']}, {camera_configs['main']['height']}), got ({actual_width}, {actual_height})"
        assert (
            actual_fps == camera_configs["main"]["fps"]
        ), f"FPS do not match. Expected {camera_configs['main']['fps']}, got {actual_fps}"

        print(
            f"WebCam initialized with width={actual_width}, height={actual_height}, fps={actual_fps}"
        )

        self.last_frame: MatLike | None = None
        self.last_timestamp: float | None = None

    def get_images_bgr_hwc_and_timestamp(self) -> tuple[dict[str, MatLike], float]:
        """
        Captures a frame from the webcam. If capture fails, it returns the last
        known successful frame as a fallback.
        """
        ret, frame = self.cap.read()
        # timestamp = time.monotonic()
        timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # print(f"latency: {time.monotonic() - timestamp:.4f}")
        # start_time = time.monotonic()

        if ret:
            diff = (
                timestamp - self.last_timestamp
                if self.last_timestamp is not None
                else 0.0
            )
            # print(f"diff: {diff:.4f}, latency: {start_time - timestamp:.4f}")
            self.last_frame = frame
            self.last_timestamp = timestamp
            return {"main": frame}, timestamp
        else:
            print(f"[{self.name}] Warning: Failed to read new frame.", end=" ")
            if self.last_frame is None:
                raise RuntimeError(
                    "No frame available. Please check if the camera is connected."
                )
            return {"main": self.last_frame}, self.last_timestamp


def run_webcam():
    if len(sys.argv) > 1:
        id = sys.argv[1]
    else:
        id = 0

    with hydra.initialize(config_path="../configs/peripherals"):
        cfg = hydra.compose(config_name=f"webcam{id}")
        print(cfg)

    config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(cfg, False)
    cfg.config_str = config_str
    camera = hydra.utils.instantiate(cfg)

    camera.run()


def run_gopro():
    if len(sys.argv) > 1:
        id = sys.argv[1]
    else:
        id = 0

    with hydra.initialize(config_path="../configs/peripherals"):
        cfg = hydra.compose(config_name=f"gopro{id}")
        print(cfg)

    config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(cfg, False)
    cfg.config_str = config_str
    camera = hydra.utils.instantiate(cfg)

    camera.run()


if __name__ == "__main__":
    run_webcam()
