import copy
import os
import sys
import time
import json
from typing import Any
from omegaconf import OmegaConf

import hydra
import numpy as np
import numpy.typing as npt

from real_env.controllers.base_controller_client import BaseCartesianJointClient
from real_env.tasks.base_task import BaseTask, TaskControlMode
from robot_utils.time_utils import wait_until
from real_env.agents.policy_agent import PolicyAgent
from real_env.peripherals.base_camera import BaseCameraClient
from robot_utils.pose_utils import get_relative_poses
from real_env.utils.umi_utils import draw_predefined_mask
from robot_utils.image_utils import resize_with_cropping

# TODO: on reconnect, self.timestamp.clear() to reset timestamps
class UmiARX5Task(BaseTask):
    """UMI task for ARX5 robot arm.

    Inherits directly from BaseTask for spacemouse + ARX5 controller.
    Uses a single BaseCartesianJointClient for both arm and gripper.
    Policy support can be added by subclasses.
    """

    def __init__(
        self,
        run_name: str,
        arx5_controller_endpoint: str,
        arx5_lookahead_time_s: float,
        pause_when_policy_inference: bool,
        wrist_camera_endpoint: str,
        third_person_camera_endpoint: str,
        arx5_home_gripper_pos_m: float | None,
        draw_gripper_mask: bool,
        draw_mirrors_mask: bool,
        camera_latency_s: float,
        policy_latency_s: float,
        policy_agent: PolicyAgent,
        arx5_home_pose_xyz_wxyz: npt.NDArray[np.float64] | None = None,
        crop_image: tuple[int, int] | None = None,
        match_camera_latency: bool = False,
        **kwargs,
    ) -> None:

        print("Initializing UMI ARX5 task")

        self.policy_agent: PolicyAgent = policy_agent
        self.match_camera_latency: bool = match_camera_latency
        self.crop_image: tuple[int, int] | None = crop_image
        if "<POLICY_NAME>" in run_name:
            run_name = run_name.replace(
                "<POLICY_NAME>", self.policy_agent.config["run_name"]
            )
        if "<POLICY_EPOCH>" in run_name:
            # print(self.policy_agent.config)
            run_name = run_name.replace(
                "<POLICY_EPOCH>", str(self.policy_agent.config["epoch"])
            )

        super().__init__(
            run_name=run_name,
            **kwargs,
        )

        self.clients.append(self.policy_agent)

        self.camera_latency_s: float = camera_latency_s
        self.policy_latency_s: float = policy_latency_s
        self.draw_gripper_mask = draw_gripper_mask
        self.draw_mirrors_mask = draw_mirrors_mask
        self.pause_when_policy_inference: bool = pause_when_policy_inference
        self.arx5_lookahead_time_s: float = arx5_lookahead_time_s
        self.arx5_home_pose_xyz_wxyz: npt.NDArray[np.float64] | None = (
            np.array(arx5_home_pose_xyz_wxyz)
            if arx5_home_pose_xyz_wxyz is not None
            else None
        )
        self.arx5_home_gripper_pos_m: npt.NDArray[np.float64] = np.array(
            [arx5_home_gripper_pos_m] if arx5_home_gripper_pos_m is not None else [0.08]
        )

        # Single combined client for ARX5
        self.arx5_client = BaseCartesianJointClient(
            name="ARX5",
            server_endpoint=arx5_controller_endpoint,
            data_remaining_time_s=1.0,
        )
        self.clients.append(self.arx5_client)
        self.wrist_camera_client = BaseCameraClient(
            name="wrist_camera",
            server_endpoint=wrist_camera_endpoint,
        )
        self.clients.append(self.wrist_camera_client)
        fps = self.wrist_camera_client.config["camera_configs"]["main"]["fps"]

        if (
            third_person_camera_endpoint is not None
            and third_person_camera_endpoint != ""
        ):
            self.third_person_camera_client = BaseCameraClient(
                name="third_person_camera",
                server_endpoint=third_person_camera_endpoint,
            )
            self.clients.append(self.third_person_camera_client)
        else:
            self.third_person_camera_endpoint = None
            print(
                "No third person camera endpoint provided. Skipping initialization of third person camera client."
            )

        if self.policy_agent.image_history_len > 1:
            assert (
                fps % int(self.policy_agent.agent_update_freq_hz) == 0
            ), f"FPS must be divisible by agent update frequency, got {fps=} and agent_update_freq_hz={self.policy_agent.agent_update_freq_hz}"

        # States
        eef_xyz_wxyz = self.arx5_client.get_eef_target_xyz_wxyz(
            timestamp=time.monotonic()
        )
        gripper_width = self.arx5_client.get_joint_pos(timestamp=time.monotonic())
        # eef_xyz_wxyz = self.arx5_client.
        self.spacemouse_history_actions: dict[str, npt.NDArray[np.float64]] = {
            "action0_eef_xyz_wxyz": eef_xyz_wxyz[np.newaxis, :],
            "action0_gripper_width": gripper_width[np.newaxis, :],
            "timestamps": np.array([time.monotonic()]),
        }
        self.policy_history_actions: dict[str, npt.NDArray[np.float64]] = {}
        self.last_control_timestamp: float = time.monotonic()
        self.last_control_mode: TaskControlMode = TaskControlMode.SPACEMOUSE
        self.policy_start_eef_xyz_wxyz: npt.NDArray[np.float64] | None = None

    def reset(self):

        # Alternatively: Schedule home pose to reset the ARX5 arm
        # self.arx5_client.schedule_joint_traj(
        #     joint_traj_pos=self.arx5_home_gripper_pos_m[np.newaxis, :],
        #     timestamps=np.array([reset_time_s]),
        #     use_relative_timestamps=True,
        # )
        # if self.arx5_home_pose_xyz_wxyz is not None:
        #     self.arx5_client.schedule_eef_traj(
        #         eef_traj_xyz_wxyz=self.arx5_home_pose_xyz_wxyz[np.newaxis, :],
        #         timestamps=np.array([reset_time_s]),
        #         use_relative_timestamps=True,
        #     )
        #     self.last_control_timestamp = time.monotonic() + reset_time_s
        #     self.spacemouse_history_actions = {
        #         "action0_eef_xyz_wxyz": self.arx5_home_pose_xyz_wxyz[np.newaxis, :],
        #         "action0_gripper_width": self.arx5_home_gripper_pos_m[np.newaxis, :],
        #     }
        # else:
        #     self.last_control_timestamp = time.monotonic()
        #     eef_xyz_wxyz = self.arx5_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        #     self.spacemouse_history_actions = {
        #         "action0_eef_xyz_wxyz": eef_xyz_wxyz[np.newaxis, :],
        #         "action0_gripper_width": self.arx5_home_gripper_pos_m[np.newaxis, :],
        #     }

        # A safer option: use the built-in reset method in the ARX5 controller
        self.arx5_client.reset()

        # Re-read state AFTER the arm has settled at home.
        # arx5_client.reset() is fire-and-forget, so the arm moves during the sleep.
        self.last_control_timestamp = time.monotonic()
        eef_xyz_wxyz = self.arx5_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        gripper_width = self.arx5_client.get_joint_pos(timestamp=time.monotonic())
        self.spacemouse_history_actions = {
            "action0_eef_xyz_wxyz": eef_xyz_wxyz[np.newaxis, :],
            "action0_gripper_width": gripper_width[np.newaxis, :],
            "timestamps": np.array([time.monotonic()]),
        }

    def get_observations(self):
        if self.wrist_camera_client is None:
            raise RuntimeError(
                "Camera client not initialized. Override get_observations() in subclass "
                "or provide camera_endpoint in __init__."
            )

        # Camera runs faster than the policy (e.g. 30 FPS vs 10 Hz). We need
        # image_history_len frames spaced at the policy rate, so we fetch extra
        # raw frames from the buffer and then downsample with [::downsample_factor].
        #
        # downsample_factor = camera_fps / agent_freq  (e.g. 30/10 = 3, take every 3rd frame)
        # frame_num uses the fence-post formula: image_history_len frames have
        # (image_history_len - 1) gaps between them, each gap is downsample_factor
        # raw frames wide, +1 to include the starting frame.
        # e.g. history=3, downsample=3 → (3-1)*3+1 = 7 raw frames → [::3] → indices [0,3,6] → 3 images
        if self.policy_agent.image_history_len > 1:
            downsample_factor = (
                self.wrist_camera_client.info["camera_configs"]["main"]["fps"]
                / self.policy_agent.agent_update_freq_hz
            )
            assert (
                downsample_factor.is_integer()
            ), f"Downsample factor must be an integer, got {downsample_factor}. {self.wrist_camera_client.info['camera_configs']['main']['fps']=} and {self.policy_agent.agent_update_freq_hz=}"
            downsample_factor = int(downsample_factor)

            frame_num = (
                self.policy_agent.image_history_len - 1
            ) * downsample_factor + 1
        else:
            frame_num = 1
            downsample_factor = 1

        (
            images_dict_THWC_RGB,
            image_timestamps,
        ) = self.wrist_camera_client.get_latest_images_dict_THWC(frame_num=frame_num)

        image_timestamps = image_timestamps[::downsample_factor]
        images_dict_THWC_RGB = {
            k: v[::downsample_factor] for k, v in images_dict_THWC_RGB.items()
        }
        print(f"image_timestamps: {image_timestamps}")

        eef_poses_list: list[npt.NDArray[np.float64]] = []
        gripper_widths_list: list[npt.NDArray[np.float64]] = []

        robot_state_timestamps = []

        if self.match_camera_latency:
            last_timestamp = image_timestamps[
                0
            ]  # This should be correct theoretically, but leads to inaccuracy
        else:
            last_timestamp = time.monotonic()  # This HACK works for UMI and iPhUMI
        # print(f"{time.monotonic() - last_timestamp=}") # Difference is usually 0.15s

        for i in range(self.policy_agent.proprio_history_len - 1, -1, -1):
            robot_state_timestamps.append(
                last_timestamp - i * (1.0 / self.policy_agent.agent_update_freq_hz)
            )

        for robot_state_timestamp in robot_state_timestamps:
            eef_poses_list.append(
                self.arx5_client.get_eef_xyz_wxyz(timestamp=robot_state_timestamp)
            )
            gripper_widths_list.append(
                self.arx5_client.get_joint_pos(timestamp=robot_state_timestamp)
            )

        # should have shape (N, 7) and (N, 1) where N = policy_agent.image_history_len
        eef_poses = np.array(eef_poses_list)
        gripper_widths = np.array(gripper_widths_list)

        observations = {
            "robot0_eef_xyz_wxyz": eef_poses,
            "robot0_gripper_width": gripper_widths,
            "robot0_main_camera": images_dict_THWC_RGB["main"],  # shape (N, H, W, C)
            "timestamps": image_timestamps,  # (N,)
        }

        if self.policy_start_eef_xyz_wxyz is None and (
            self.control_mode == TaskControlMode.POLICY
            or self.control_mode == TaskControlMode.POLICY_SINGLE_STEP
        ):
            self.policy_start_eef_xyz_wxyz = eef_poses[0]

        if self.policy_start_eef_xyz_wxyz is not None:
            observations["robot0_eef_wrt_start_xyz_wxyz"] = get_relative_poses(
                eef_poses, self.policy_start_eef_xyz_wxyz
            )

        # UMI legacy mask
        if self.draw_gripper_mask:
            for i in range(len(observations["robot0_main_camera"])):
                observations["robot0_main_camera"][i] = draw_predefined_mask(
                    img=observations["robot0_main_camera"][i],
                    mirror=self.draw_mirrors_mask,
                    gripper=True,
                    finger=False,
                    use_aa=False,
                )

        # resize for legacy umi tasks
        if self.crop_image is not None:
            observations["robot0_main_camera"] = resize_with_cropping(
                observations["robot0_main_camera"],
                display_wh=self.crop_image,
            )

        return observations

    def reset_policy_agent(self):
        self.policy_agent.reset()
        self.policy_history_actions = {}
        self.policy_start_eef_xyz_wxyz = None

    def run_policy_control(self):
        abs_actions = self.policy_agent.predict_actions(
            self.get_observations(),
            self.get_history_actions(
                length=self.policy_agent.action_history_len, action_type="policy"
            ),
        )

        if (
            self.pause_when_policy_inference
            or self.last_control_mode == TaskControlMode.POLICY_SINGLE_STEP
            or self.last_control_mode == TaskControlMode.SPACEMOUSE
        ):
            abs_actions["timestamps"] = (
                time.monotonic()
                + np.arange(self.policy_agent.action_prediction_horizon)
                / self.policy_agent.agent_update_freq_hz
                # + self.camera_latency_s
                # + self.policy_latency_s
            )

        # only execute N actions if single step or pausing
        if (
            self.control_mode == TaskControlMode.POLICY_SINGLE_STEP
            or self.pause_when_policy_inference
        ):
            for k, v in abs_actions.items():
                abs_actions[k] = v[: self.policy_agent.action_execution_horizon]

        self.arx5_client.schedule_eef_traj(
            eef_traj_xyz_wxyz=abs_actions["action0_eef_xyz_wxyz"][:, :],
            timestamps=abs_actions["timestamps"] + self.arx5_lookahead_time_s,
            use_relative_timestamps=False,
        )

        self.arx5_client.schedule_joint_traj(
            joint_traj_pos=abs_actions["action0_gripper_width"][:, :],
            timestamps=abs_actions["timestamps"],
            use_relative_timestamps=False,
        )

        # At the time where policy predicted the next action trajectory, the first action it predicted is probably outdated already due to
        # a combination of camera latency and inference latency. Therefore, in the interpolator the first action is probably being trimmed
        # out already. Therefore, most likely we are executing from action 1 to action 8 (with zero indexing; action 0 is outdated already).
        self.policy_history_actions = {
            k: v[: min(self.policy_agent.action_execution_horizon + 1, v.shape[0])]
            for k, v in abs_actions.items()
        }

        if (
            self.pause_when_policy_inference
            or self.last_control_mode == TaskControlMode.POLICY_SINGLE_STEP
        ):
            wait_until(
                monotonic_time=abs_actions["timestamps"][
                    self.policy_agent.action_execution_horizon - 1
                ]
                + self.camera_latency_s
            )
        else:
            wait_until(
                monotonic_time=abs_actions["timestamps"][
                    self.policy_agent.action_execution_horizon - 1
                ]
                - 0.02  # Subtracting 0.02s to make sure the next trajectory ticks are always scheduled before the existing trajectory, so there won't be a delayed tick
            )

        self.last_control_mode = self.control_mode

    def get_history_actions(self, length: int, action_type: str = "latest"):
        """
        action_type: "latest" (most recent of either source), "spacemouse", or "policy"
        """
        # Pick the right history source based on action_type
        if len(self.policy_history_actions) == 0:
            # No policy history yet — spacemouse is the only option
            history_actions = self.spacemouse_history_actions
        elif action_type == "latest":
            # Use whichever was more recent
            if (
                self.policy_history_actions["timestamps"][-1]
                > self.spacemouse_history_actions["timestamps"][-1]
            ):
                history_actions = self.policy_history_actions
            else:
                history_actions = self.spacemouse_history_actions
        elif action_type == "spacemouse":
            history_actions = self.spacemouse_history_actions
        elif action_type == "policy":
            history_actions = self.policy_history_actions
        else:
            raise ValueError(f"Invalid action type: {action_type}")

        # Pad or truncate to exactly `length` entries
        processed_history_actions: dict[str, npt.NDArray[np.float64]] = {}
        for key, value in history_actions.items():
            if value.shape[0] > length:
                processed_history_actions[key] = value[-length:]
            elif value.shape[0] < length:
                processed_history_actions[key] = np.concatenate(
                    [
                        np.repeat(value[:1, ...], length - value.shape[0], axis=0),
                        value,
                    ]
                )
            else:
                processed_history_actions[key] = value
        return processed_history_actions

    def run_spacemouse_control(self):
        eef_xyz_wxyz = self.arx5_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        gripper_width = self.arx5_client.get_joint_pos(timestamp=time.monotonic())

        if self.last_control_mode == TaskControlMode.POLICY:
            # Re-seed from actual robot state to avoid jumping to last policy target
            self.spacemouse_history_actions = {
                "action0_eef_xyz_wxyz": eef_xyz_wxyz[np.newaxis, :],
                "action0_gripper_width": gripper_width[np.newaxis, :],
                "timestamps": np.array([time.monotonic()]),
            }
            self.reset_policy_agent()

        actions = self.spacemouse_agent.predict_actions(
            observations={
                "robot0_eef_xyz_wxyz": eef_xyz_wxyz[np.newaxis, :],
                "robot0_gripper_width": gripper_width[np.newaxis, :],
            },
            history_actions=self.get_history_actions(
                length=self.spacemouse_agent.action_history_len
            ),
        )

        next_control_timestamp = (
            self.last_control_timestamp
            + 1.0 / self.spacemouse_agent.agent_update_freq_hz
        )
        wait_until(monotonic_time=next_control_timestamp)
        # print("Last control timestamp: ", self.last_control_timestamp)
        self.last_control_timestamp = next_control_timestamp
        # print(f"Spacemouse control moving to {self.last_control_timestamp:.2f}s, EEF actions: {actions['action0_eef_xyz_wxyz']}")
        self.arx5_client.schedule_eef_traj(
            eef_traj_xyz_wxyz=actions["action0_eef_xyz_wxyz"][:, :],
            timestamps=np.array(
                [
                    1.0 / self.spacemouse_agent.agent_update_freq_hz
                    + self.arx5_lookahead_time_s
                ]
            ),
            use_relative_timestamps=True,
        )
        self.arx5_client.schedule_joint_traj(
            joint_traj_pos=actions["action0_gripper_width"][:, :],
            timestamps=np.array([1.0 / self.spacemouse_agent.agent_update_freq_hz]),
            use_relative_timestamps=True,
        )
        self.spacemouse_history_actions = copy.deepcopy(actions)

        self.last_control_mode = TaskControlMode.SPACEMOUSE

    def display_robot_state(self):
        eef_xyz_wxyz = self.arx5_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        gripper_width = self.arx5_client.get_joint_pos(timestamp=time.monotonic())
        print(f"EEF pose: {eef_xyz_wxyz}")
        print(f"Gripper width: {gripper_width}")

    def disconnect(self):
        pass


def run_umi_arx5_task():
    os.environ["HYDRA_FULL_ERROR"] = "1"
    assert len(sys.argv) == 2, "Usage: python run_umi_arx5_task.py <task_name>"
    task_name = sys.argv[1]
    with hydra.initialize(config_path="../configs/tasks"):
        cfg = hydra.compose(config_name=f"umi_arx5_{task_name}")
    print(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
    task: UmiARX5Task = hydra.utils.instantiate(cfg)
    try:
        task.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run_umi_arx5_task()
