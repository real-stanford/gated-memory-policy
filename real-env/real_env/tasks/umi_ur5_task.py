import copy
import datetime
import os
import sys
import time
import curses
from typing import Any
import cv2
import numpy as np
import numpy.typing as npt
import hydra
from omegaconf import DictConfig
from robot_utils.image_utils import resize_with_cropping

np.set_printoptions(precision=4, suppress=True)

from real_env.agents.policy_agent import PolicyAgent
from real_env.common.constants import RMQ_PORTS
from real_env.controllers.base_controller_client import (
    BaseCartesianClient,
    BaseJointClient,
)
from real_env.peripherals.base_camera import BaseCameraClient
from real_env.tasks.base_task import BaseTask, TaskControlMode
from robot_utils.pose_utils import get_relative_poses
from robot_utils.time_utils import wait_until
from real_env.utils.umi_utils import draw_predefined_mask

from omegaconf import OmegaConf
import json


class UmiUR5Task(BaseTask):
    def __init__(
        self,
        run_name: str,
        draw_gripper_mask: bool,
        draw_mirrors_mask: bool,
        pause_when_policy_inference: bool,
        ur5_controller_endpoint: str,
        wsg50_controller_endpoint: str,
        ur5_lookahead_time_s: float,
        wsg50_action_execution_latency_s: float,
        wrist_camera_endpoint: str,
        third_person_camera_endpoint: str,
        ur5_home_pose_xyz_wxyz: npt.NDArray[np.float64] | None,
        wsg50_home_pos_m: float,
        camera_latency_s: float,  # TODO: This should be requested from the camera client
        policy_latency_s: float,  # TODO: This should be requested from the policy server
        policy_agent: PolicyAgent,
        match_camera_latency: bool,
        crop_image: tuple[int, int] | None,
        **kwargs,
    ) -> None:

        print(f"Initializing UMI UR5 task")

        self.policy_agent: PolicyAgent = policy_agent

        if "<POLICY_NAME>" in run_name:
            run_name = run_name.replace(
                "<POLICY_NAME>", self.policy_agent.config["run_name"]
            )
        if "<POLICY_EPOCH>" in run_name:
            # print(self.policy_agent.config)
            run_name = run_name.replace(
                "<POLICY_EPOCH>", str(self.policy_agent.config["epoch"])
            )

        self.crop_image: tuple[int, int] | None = crop_image

        super().__init__(
            run_name=run_name,
            **kwargs,
        )

        self.match_camera_latency: bool = match_camera_latency
        self.clients.append(self.policy_agent)

        self.pause_when_policy_inference: bool = pause_when_policy_inference
        self.draw_gripper_mask: bool = draw_gripper_mask
        self.draw_mirrors_mask: bool = draw_mirrors_mask
        self.ur5_client: BaseCartesianClient = BaseCartesianClient(
            name="UR5",
            server_endpoint=ur5_controller_endpoint,
            data_remaining_time_s=1.0,
        )
        self.clients.append(self.ur5_client)
        self.wsg50_client: BaseJointClient = BaseJointClient(
            name="WSG50",
            server_endpoint=wsg50_controller_endpoint,
            data_remaining_time_s=1.0,
        )
        self.clients.append(self.wsg50_client)
        self.wrist_camera_client = BaseCameraClient(
            name="wrist_camera",
            server_endpoint=wrist_camera_endpoint,
        )
        self.clients.append(self.wrist_camera_client)

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
            self.third_person_camera_client = None
            print("Third person camera client not initialized")

        fps: int = self.wrist_camera_client.config["camera_configs"]["main"]["fps"]
        if self.policy_agent.image_history_len > 1:
            assert (
                fps % int(self.policy_agent.agent_update_freq_hz) == 0
            ), f"FPS must be divisible by agent update frequency, got {fps=} and agent_update_freq_hz={self.policy_agent.agent_update_freq_hz}"

        self.ur5_lookahead_time_s: float = ur5_lookahead_time_s

        self.ur5_home_pose_xyz_wxyz: npt.NDArray[np.float64] | None = (
            np.array(ur5_home_pose_xyz_wxyz) if ur5_home_pose_xyz_wxyz else None
        )
        self.wsg50_home_pos_m: npt.NDArray[np.float64] = np.array([wsg50_home_pos_m])
        self.wsg50_action_execution_latency_s: float = wsg50_action_execution_latency_s

        # States
        eef_xyz_wxyz = self.ur5_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        gripper_width = self.wsg50_client.get_joint_pos(timestamp=time.monotonic())
        self.spacemouse_history_actions: dict[str, npt.NDArray[np.float64]] = {
            "action0_eef_xyz_wxyz": eef_xyz_wxyz[np.newaxis, :],
            "action0_gripper_width": gripper_width[np.newaxis, :],
            "timestamps": np.array([time.monotonic()]),
        }
        self.policy_history_actions: dict[str, npt.NDArray[np.float64]] = {}
        self.last_control_timestamp: float = time.monotonic()

        self.last_control_mode: TaskControlMode = TaskControlMode.SPACEMOUSE
        self.policy_start_eef_xyz_wxyz: npt.NDArray[np.float64] | None = None

        print("Warming up policy inference ...")
        start_time = time.monotonic()
        self.control_mode = TaskControlMode.POLICY
        actions = self.policy_agent.predict_actions(
            self.get_observations(),
            self.get_history_actions(length=self.policy_agent.action_history_len),
        )
        # Reset starting pose
        self.control_mode = TaskControlMode.SPACEMOUSE
        self.reset_policy_agent()

        print(
            f"Policy inference ready, time taken: {time.monotonic() - start_time} seconds."
        )

        self.camera_latency_s: float = camera_latency_s
        self.policy_latency_s: float = policy_latency_s

    def reset(self):
        reset_time_s = 5.0

        self.wsg50_client.schedule_joint_traj(
            joint_traj_pos=self.wsg50_home_pos_m[np.newaxis, :],
            timestamps=np.array([reset_time_s]),
            use_relative_timestamps=True,
        )
        if self.ur5_home_pose_xyz_wxyz is not None:
            self.ur5_client.schedule_eef_traj(
                eef_traj_xyz_wxyz=self.ur5_home_pose_xyz_wxyz[np.newaxis, :],
                timestamps=np.array([reset_time_s]),
                use_relative_timestamps=True,
            )
            self.last_control_timestamp = time.monotonic() + reset_time_s
            self.spacemouse_history_actions = {
                "action0_eef_xyz_wxyz": self.ur5_home_pose_xyz_wxyz[np.newaxis, :],
                "action0_gripper_width": self.wsg50_home_pos_m[np.newaxis, :],
            }
        else:
            self.last_control_timestamp = time.monotonic()
            eef_xyz_wxyz = self.ur5_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
            self.spacemouse_history_actions = {
                "action0_eef_xyz_wxyz": eef_xyz_wxyz[np.newaxis, :],
                "action0_gripper_width": self.wsg50_home_pos_m[np.newaxis, :],
            }

        self.policy_start_eef_xyz_wxyz = None

        time.sleep(reset_time_s)

    def reset_policy_agent(self):
        self.policy_agent.reset()
        self.policy_history_actions = {}

    def get_observations(self):
        # HACK: only use the latest image
        # HACK: This assumes camera_client exists. If camera_endpoint=None was passed,
        # subclasses must override this method (e.g., for heuristic agents without cameras)
        if self.wrist_camera_client is None:
            raise RuntimeError(
                "Camera client not initialized. Override get_observations() in subclass "
                "or provide camera_endpoint in __init__."
            )

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
        start_time = time.monotonic()

        (
            images_dict_THWC_RGB,
            image_timestamps,
        ) = self.wrist_camera_client.get_latest_images_dict_THWC(frame_num)

        end_time = time.monotonic()
        print(f"get_latest_images_dict_THWC time: {end_time - start_time}")

        image_timestamps = image_timestamps[::downsample_factor]
        images_dict_THWC_RGB = {
            k: v[::downsample_factor] for k, v in images_dict_THWC_RGB.items()
        }

        eef_poses_list: list[npt.NDArray[np.float64]] = []
        gripper_widths_list: list[npt.NDArray[np.float64]] = []
        print(f"image_timestamps: {image_timestamps}")

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
                last_timestamp - i * 1.0 / self.policy_agent.agent_update_freq_hz
            )

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
            "timestamps": image_timestamps,  # (N, )
        }

        if self.policy_start_eef_xyz_wxyz is None and (
            self.control_mode == TaskControlMode.POLICY
            or self.control_mode == TaskControlMode.POLICY_SINGLE_STEP
        ):
            self.policy_start_eef_xyz_wxyz = eef_poses[0]

        if self.policy_start_eef_xyz_wxyz is not None:
            observations["robot0_eef_wrt_start_xyz_wxyz"] = get_relative_poses(
                eef_poses,
                self.policy_start_eef_xyz_wxyz,
            )

        if self.draw_gripper_mask:
            for i in range(len(observations["robot0_main_camera"])):
                observations["robot0_main_camera"][i] = draw_predefined_mask(
                    img=observations["robot0_main_camera"][i],
                    mirror=self.draw_mirrors_mask,
                    gripper=True,
                    finger=False,
                    use_aa=False,
                )
                # cv2.imwrite(f"robot0_main_camera_{i}.png", observations["robot0_main_camera"][i])

        if self.crop_image is not None:
            observations["robot0_main_camera"] = resize_with_cropping(
                observations["robot0_main_camera"],
                display_wh=self.crop_image,
            )

        # # cv2.imwrite("robot0_main_camera_1.png", observations["robot0_main_camera"][0])
        # print(f"get_observations:")
        # for k, v in observations.items():
        #     print(f"{k}: {v.shape}")

        return observations

    def get_history_actions(self, length: int, action_type: str = "latest"):
        """
        action_type: "latest", "spacemouse", "policy"
        """
        if len(self.policy_history_actions) == 0:  # No policy history actions yet
            history_actions = self.spacemouse_history_actions
        elif action_type == "latest":
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

        processed_history_actions: dict[str, npt.NDArray[np.float64]] = {}
        for key, value in history_actions.items():
            if value.shape[0] > length:
                # Truncate the history actions
                processed_history_actions[key] = value[-length:]
            elif value.shape[0] < length:
                # Pad with the earliest action
                processed_history_actions[key] = np.concatenate(
                    [
                        np.repeat(value[:1, ...], length - value.shape[0], axis=0),
                        value,
                    ]
                )
            else:
                processed_history_actions[key] = value
        return processed_history_actions

    def run_policy_control(self):

        abs_actions = self.policy_agent.predict_actions(
            self.get_observations(),
            self.get_history_actions(
                length=self.policy_agent.action_history_len, action_type="policy"
            ),
        )

        if (
            self.last_control_mode == TaskControlMode.SPACEMOUSE
            or self.pause_when_policy_inference
            or self.control_mode == TaskControlMode.POLICY_SINGLE_STEP
        ):
            # Delay the first trajectory to make the movement smoother
            abs_actions["timestamps"] = (
                time.monotonic()
                + np.arange(self.policy_agent.action_prediction_horizon)
                / self.policy_agent.agent_update_freq_hz
                # + self.camera_latency_s
                # + self.policy_latency_s
            )
            # One tick for transitioning from the spacemouse commanded current action to the policy commanded action,
            # which given the checkpoint here we are using the first action is non-zero we should schedule it for the next tick
            # to ensure smoothness.
            # An additional tick is needed as we need a account for lookahead time for scheduling the trajectory as robot cannot
            # instantaneously move to the commanded position (action 0 actually executed at t_1 not t_0 as we schedule for future).

        # Alternatively, a better convention is to shift all the timestamps by one step. Depends on how the policy model is trained.
        # new_timestamps = image_timestamps[-1] + np.arange(
        #     self.policy_agent.action_prediction_horizon
        # ) * 1.0 / self.policy_agent.agent_update_freq_hz + 1.0 / self.policy_agent.agent_update_freq_hz

        if (
            self.pause_when_policy_inference
            or self.control_mode == TaskControlMode.POLICY_SINGLE_STEP
        ):
            # Only execute the first N actions
            for key, value in abs_actions.items():
                abs_actions[key] = value[: self.policy_agent.action_execution_horizon]

        self.ur5_client.schedule_eef_traj(
            eef_traj_xyz_wxyz=abs_actions["action0_eef_xyz_wxyz"],
            timestamps=abs_actions["timestamps"] + self.ur5_lookahead_time_s,
            use_relative_timestamps=False,
        )
        self.wsg50_client.schedule_joint_traj(
            joint_traj_pos=abs_actions["action0_gripper_width"],
            timestamps=abs_actions["timestamps"],
            use_relative_timestamps=False,
        )

        self.policy_history_actions = {
            key: value[
                : min(self.policy_agent.action_execution_horizon + 1, value.shape[0])
            ]
            for key, value in abs_actions.items()
        }
        # At the time where policy predicted the next action trajectory, the first action it predicted is probably outdated already due to
        # a copmbination of camera latency and inference latency. Therefore, in the interpolator the first action is probably being trimmed
        # out already. Therefore, most likely we are executing from action 1 to action 8 (with zero indexing; action 0 is outdated already).

        if (
            self.pause_when_policy_inference
            or self.control_mode == TaskControlMode.POLICY_SINGLE_STEP
        ):
            wait_until(
                monotonic_time=abs_actions["timestamps"][
                    self.policy_agent.action_execution_horizon - 1
                ]
                + self.camera_latency_s  # So that camera latency can be mitigated
            )
        else:
            wait_until(
                monotonic_time=abs_actions["timestamps"][
                    self.policy_agent.action_execution_horizon - 1
                ]
                - 0.02
            )  # Subtracting 0.02s to make sure the next trajectory ticks are always scheduled before the existing trajectory, so there won't be a delayed tick
        # print(
        #     f"target start: {abs_actions['action0_eef_xyz_wxyz'][0][:3]}, actual start: {abs_poses[-1][:3]}"
        # )
        # print(
        #     f"target final: {abs_actions['action0_eef_xyz_wxyz'][-1][:3]}, actual final: {self.ur5_client.get_eef_xyz_wxyz(timestamp=time.monotonic())[:3]}"
        # )

        self.last_control_mode = self.control_mode

    def run_spacemouse_control(self):

        eef_xyz_wxyz = self.ur5_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        gripper_width = self.wsg50_client.get_joint_pos(timestamp=time.monotonic())

        actions = self.spacemouse_agent.predict_actions(
            observations={
                "robot0_eef_xyz_wxyz": eef_xyz_wxyz[np.newaxis, :],
                "robot0_gripper_width": gripper_width[np.newaxis, :],
            },
            history_actions=self.get_history_actions(
                length=self.spacemouse_agent.action_history_len
            ),
        )  # (action_prediction_horizon, dof)

        if self.last_control_mode == TaskControlMode.POLICY:
            # Should not reset if using POLICY_SINGLE_STEP
            # Immediately reset the policy agent to flush the cached data / predicted videos
            self.reset_policy_agent()

        next_control_timestamp = (
            self.last_control_timestamp
            + 1.0 / self.spacemouse_agent.agent_update_freq_hz
        )
        wait_until(monotonic_time=next_control_timestamp)
        self.last_control_timestamp = next_control_timestamp

        self.ur5_client.schedule_eef_traj(
            eef_traj_xyz_wxyz=actions["action0_eef_xyz_wxyz"][:, :],
            timestamps=np.array(
                [
                    1.0 / self.spacemouse_agent.agent_update_freq_hz
                    + self.ur5_lookahead_time_s
                ]
            ),
            use_relative_timestamps=True,
        )
        self.wsg50_client.schedule_joint_traj(
            joint_traj_pos=actions["action0_gripper_width"][:, :],
            timestamps=np.array(
                [
                    1.0
                    / self.spacemouse_agent.agent_update_freq_hz
                    # + self.wsg50_action_execution_latency_s
                ]
            ),
            use_relative_timestamps=True,
        )
        self.spacemouse_history_actions = copy.deepcopy(actions)

        self.last_control_mode = TaskControlMode.SPACEMOUSE

    def display_robot_state(self):
        eef_xyz_wxyz = self.ur5_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        print(f"TCP pose: {eef_xyz_wxyz}")

    def disconnect(self):
        pass


def run_umi_ur5_task():
    os.environ["HYDRA_FULL_ERROR"] = "1"
    assert len(sys.argv) == 2, "Usage: python run_umi_ur5_task.py <task_name>"
    task_name = sys.argv[1]
    with hydra.initialize(config_path="../configs/tasks"):
        cfg = hydra.compose(config_name=f"umi_ur5_{task_name}")
    OmegaConf.set_struct(cfg, False)
    cfg.config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
    print(cfg)
    task: UmiUR5Task = hydra.utils.instantiate(cfg)
    task.run()


def run_umi_ur5_original_task():

    os.environ["HYDRA_FULL_ERROR"] = "1"
    with hydra.initialize(config_path="../configs/tasks"):
        cfg = hydra.compose(config_name="umi_ur5_original")
    print(cfg)
    task: UmiUR5Task = hydra.utils.instantiate(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
    print(cfg)
    task: UmiUR5Task = hydra.utils.instantiate(cfg)
    task.run()


if __name__ == "__main__":
    run_umi_ur5_task()
