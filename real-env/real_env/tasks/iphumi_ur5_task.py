import sys
import time
from real_env.common.constants import RMQ_PORTS
from real_env.tasks.umi_ur5_task import UmiUR5Task
import numpy as np
import numpy.typing as npt
import datetime
import os
import hydra
import json
from omegaconf import OmegaConf


class iPhumiUR5Task(UmiUR5Task):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def get_observations(self):
        # HACK: only use the latest image
        start_time = time.monotonic()
        (
            images_dict_THWC_RGB,
            image_timestamps,
        ) = self.wrist_camera_client.get_latest_images_dict_THWC(1)
        end_time = time.monotonic()
        print(f"get_latest_images_dict_THWC time: {end_time - start_time}")
        eef_poses_list: list[npt.NDArray[np.float64]] = []
        gripper_widths_list: list[npt.NDArray[np.float64]] = []
        print(f"image_timestamps: {image_timestamps}")

        # robot_state_timestamps = [
        #     image_timestamps[0] - 1.0 / self.policy_agent.agent_update_freq_hz,
        #     image_timestamps[0],
        # ]

        # HACK: if doesn't include camera latency, policy works much better. Not sure why...
        # robot_state_timestamps = [
        #     time.monotonic() - 2.0 / self.policy_agent.agent_update_freq_hz,
        #     time.monotonic() - 1.0 / self.policy_agent.agent_update_freq_hz,
        # ]

        robot_state_timestamps = [
            time.monotonic() - 1.0 / self.policy_agent.agent_update_freq_hz,
            time.monotonic() - 0.0 / self.policy_agent.agent_update_freq_hz,
        ]

        current_time = time.monotonic()
        print(f"time difference: {current_time - image_timestamps[0]}")
        for robot_state_timestamp in robot_state_timestamps:
            eef_poses_list.append(
                self.ur5_client.get_eef_xyz_wxyz(timestamp=robot_state_timestamp)
            )
            gripper_widths_list.append(
                self.wsg50_client.get_joint_pos(timestamp=robot_state_timestamp)
            )

        eef_poses = np.array(eef_poses_list)  # (N, 7)
        gripper_widths = np.array(gripper_widths_list)  # (N, 1)

        observations = {
            "robot0_eef_xyz_wxyz": eef_poses,
            "robot0_gripper_width": gripper_widths,
            "robot0_main_camera": images_dict_THWC_RGB["main"],  # (N, H, W, 3)
            "robot0_ultrawide_camera": images_dict_THWC_RGB[
                "ultrawide"
            ],  # (N, H, W, 3)
            "robot0_depth_camera": images_dict_THWC_RGB["depth"],  # (N, H, W, 3)
            "timestamps": image_timestamps,  # (N, )
        }

        print(f"iPhumiUR5Task get_observations:")
        for k, v in observations.items():
            print(f"{k}: {v.shape}")

        return observations


def run_iphumi_ur5_task():
    os.environ["HYDRA_FULL_ERROR"] = "1"
    assert len(sys.argv) == 2, "Usage: python run_iphumi_ur5_task.py <task_name>"
    task_name = sys.argv[1]
    with hydra.initialize(config_path="../configs/tasks"):
        cfg = hydra.compose(config_name=f"iphumi_ur5_{task_name}")
    OmegaConf.set_struct(cfg, False)
    cfg.config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
    print(cfg)
    task: iPhumiUR5Task = hydra.utils.instantiate(cfg)

    task.run()


if __name__ == "__main__":
    run_iphumi_ur5_task()
