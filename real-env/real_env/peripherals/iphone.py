import json
import os
import sys
import traceback
from pickle import bytes_types
import cv2
import time
from cv2.typing import MatLike
import numpy as np
import numpy.typing as npt
from typing import Any
import pypeertalk
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from real_env.peripherals.base_camera import BaseCamera
from robologger.utils import huecodec as hc


class iPhone(BaseCamera):
    """
    A concrete implementation of BaseCamera that acts as an AGGREGATOR.
    It connects to separate network streams from an iPhone app, synchronizes them,
    and re-broadcasts them as a single, unified packet.
    """

    def __init__(
        self,
        iphone_udid: str | None,
        max_continuous_error_count: int = 10,
        depth_enc_mode: str = "hue_codec",
        depth_range: tuple[float, float] = (0.02, 4.0),
        **kwargs,
    ):
        """
        Initializes the iPhone stream aggregator.
        """
        super().__init__(**kwargs)

        self.max_continuous_error_count: int = max_continuous_error_count
        self.continuous_error_count: int = 0
        self.peertalk_clients: dict[str, pypeertalk.PeerTalkClient] = {}

        devices = pypeertalk.get_connected_devices()
        if iphone_udid is None or iphone_udid == "":
            if len(devices) == 0:
                raise ValueError("No iPhone devices found")
            elif len(devices) == 1:
                iphone_device_info = devices[0]
                print(
                    f"  > Only one device found. Using device with UDID: {iphone_device_info.udid}"
                )
            else:
                for device in devices:
                    print(f"  > Device with UDID: {device.udid}")
                raise ValueError(
                    "Multiple iPhone devices found, UDID should be specified when initializing iPhone"
                )

        else:
            for device in devices:
                print(f"  > Found Device with UDID: {device.udid}")
                if device.udid == iphone_udid:
                    iphone_device_info = device
                    break
            else:
                raise ValueError(f"iPhone device with UDID {iphone_udid} not found")

        print("Initializing iPhone network stream connections...")
        for cam_name, config in self.camera_configs.items():
            print(f"  > Connecting to '{cam_name}' stream at {config['peertalk_port']}")
            self.peertalk_clients[cam_name] = pypeertalk.PeerTalkClient(
                iphone_device_info, config["peertalk_port"]
            )

        self.depth_range: tuple[float, float] = depth_range
        assert depth_enc_mode in ["gray_scale", "hue_codec", "hue_codec_inv"]
        self.depth_enc_mode: str = depth_enc_mode
        for cam_name, config in self.camera_configs.items():
            if config["type"] == "depth":
                config["depth_enc_mode"] = self.depth_enc_mode
                config["depth_range"] = self.depth_range

        self.opts: hc.EncoderOpts = hc.EncoderOpts(use_lut=False)

        self.last_depth_image: MatLike | None = None

    def get_images_bgr_hwc_and_timestamp(self) -> tuple[dict[str, MatLike], float]:
        """Captures frames from all iPhone camera streams."""
        frames_dict: dict[str, MatLike] = {}
        timestamp = time.monotonic()

        for cam_name, client in self.peertalk_clients.items():
            camera_config = self.camera_configs[cam_name]

            if camera_config["type"] == "depth":
                if self.last_depth_image is None:
                    message = client.get_latest_message(100)
                else:
                    message = client.get_latest_message(5)
                    if len(message) == 0:
                        print(
                            f"No message received from {cam_name}, using last recorded image"
                        )
                        frames_dict[cam_name] = self.last_depth_image
                        continue
                depth_image: npt.NDArray[np.float32] = np.frombuffer(
                    message, np.float32
                ).reshape(
                    self.camera_configs[cam_name]["height"],
                    self.camera_configs[cam_name]["width"],
                )

                self.last_depth_image = depth_image
                frames_dict[cam_name] = depth_image

            elif camera_config["type"] == "rgb":
                message = client.get_latest_message(100)
                start_time = time.monotonic()
                try:
                    image_bgr = cv2.imdecode(
                        np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR
                    )  # BGR
                    self.continuous_error_count = 0

                except Exception as e:
                    self.continuous_error_count += 1
                    print(f"Error decoding image: {e}")
                    print(f"Message length: {len(message)}")
                    if self.continuous_error_count > self.max_continuous_error_count:
                        raise RuntimeError(
                            f"Max continuous error count reached: {self.continuous_error_count}"
                        )
                    continue
                    # try:
                    #     message = client.get_latest_message(100)
                    #     image_bgr = cv2.imdecode(
                    #         np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR
                    #     )  # BGR
                    # except Exception as e:
                    #     print(f"Error decoding image: {e}")
                    #     print(f"Message length: {len(message)}")
                    #     continue
                end_time = time.monotonic()
                # print(f"Time taken to decode image: {end_time - start_time} seconds")
                assert image_bgr.shape == (
                    self.camera_configs[cam_name]["height"],
                    self.camera_configs[cam_name]["width"],
                    3,
                )
                frames_dict[cam_name] = image_bgr
            else:
                raise ValueError(f"Invalid camera type: {camera_config['type']}")

        return frames_dict, timestamp


def run_iphone():
    """Runs the iPhone camera server."""

    np.set_printoptions(precision=4)
    os.environ["HYDRA_FULL_ERROR"] = "1"
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "wrist"
    with hydra.initialize(config_path="../configs/peripherals", version_base=None):
        cfg = hydra.compose(config_name=f"iphone_{name}")
    print(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
    iphone: iPhone = hydra.utils.instantiate(cfg)
    iphone.run()


if __name__ == "__main__":
    run_iphone()
