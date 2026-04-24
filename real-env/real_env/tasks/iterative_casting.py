import copy
import json
import os
import time
import sys

import hydra
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, ListConfig, OmegaConf

from real_env.agents.heuristic_agents.casting_agent import CastingAgent
from real_env.agents.longhist_policy_agent import LongHistPolicyAgent
from real_env.agents.policy_agent import PolicyAgent
from real_env.common.constants import RMQ_PORTS
from real_env.controllers.base_controller_client import (
    BaseCartesianClient,
    BaseJointClient,
)
from real_env.peripherals.base_camera import BaseCameraClient
from real_env.tasks.base_task import BaseTask, EpisodeStatus
from robot_utils.image_utils import resize_with_cropping
from robot_utils.time_utils import wait_until
from real_env.utils.attempt_counting_util import (
    create_attempt_detector,
    show_realtime_attempt_detection,
    DEFAULT_ROI_CENTER,
    DEFAULT_ROI_HALF,
)


class IterativeCastingUR5Task(BaseTask):
    """
    Task for running drag cube heuristic agent on real UR5 robot.
    Inherits directly from BaseTask. No cameras needed, heuristic agent outputs absolute poses.
    """

    def __init__(
        self,
        agent: CastingAgent | PolicyAgent,
        ur5_controller_endpoint: str,
        ur5_lookahead_time_s: float,
        third_person_camera_endpoint: str,
        ur5_home_pose_xyz_wxyz: ListConfig,
        reset_time_s: float,
        confine_actions: bool,
        run_attempt_detection: bool,
        **kwargs,
    ):
        # Initialize base task (logger, spacemouse, keyboard)

        if "<POLICY_NAME>" in kwargs["run_name"]:
            assert isinstance(agent, PolicyAgent), "Agent must be a PolicyAgent"
            kwargs["run_name"] = kwargs["run_name"].replace(
                "<POLICY_NAME>", agent.config["run_name"]
            )
        if "<POLICY_EPOCH>" in kwargs["run_name"]:
            assert isinstance(agent, PolicyAgent), "Agent must be a PolicyAgent"
            kwargs["run_name"] = kwargs["run_name"].replace(
                "<POLICY_EPOCH>", str(agent.config["epoch"])
            )

        super().__init__(
            **kwargs,
        )

        # Initialize robot controllers
        self.ur5_client: BaseCartesianClient = BaseCartesianClient(
            name="UR5",
            server_endpoint=ur5_controller_endpoint,
            data_remaining_time_s=1.0,
        )
        self.clients.append(self.ur5_client)
        # self.wsg50_client: BaseJointClient = BaseJointClient(
        #     name="WSG50",
        #     server_endpoint=wsg50_controller_endpoint,
        #     data_remaining_time_s=1.0,
        # )

        self.third_person_camera_client = BaseCameraClient(
            name="third_person_camera",
            server_endpoint=third_person_camera_endpoint,
        )
        self.clients.append(self.third_person_camera_client)

        self.ur5_lookahead_time_s: float = ur5_lookahead_time_s
        self.reset_time_s: float = reset_time_s
        # Use the provided drag agent
        self.agent: CastingAgent | PolicyAgent = agent
        self.confine_actions: bool = confine_actions
        """
        If true, will confine the actions to: (x=-0.6; quat_w=-quat_z, quat_x=quat_y)
        So that the gripper will always stay in the same plane
        """
        self.run_attempt_detection: bool = run_attempt_detection

        # State tracking
        eef_xyz_wxyz = self.ur5_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        # gripper_width = self.wsg50_client.get_joint_pos(timestamp=time.monotonic())
        self.history_actions: dict[str, npt.NDArray[np.float64]] = {
            "action0_eef_xyz_wxyz": eef_xyz_wxyz[np.newaxis, :],
            "action0_gripper_width": np.array([[0.0]]),
        }
        self.last_control_timestamp: float = time.monotonic()
        self.last_control_mode: str = "spacemouse"
        self.last_phase: str = ""
        self.last_trial: int = -1

        self.ur5_home_pose_xyz_wxyz: npt.NDArray[np.float64] = np.array(
            ur5_home_pose_xyz_wxyz
        )

        if isinstance(self.agent, PolicyAgent):
            self.clients.append(self.agent)
            self.attempt_detector = create_attempt_detector(
                block_duration_s=7.0
            )  # NOTE: change if needed
            self.attempt_counter = 0
            self.roi_center = DEFAULT_ROI_CENTER
            self.roi_half = DEFAULT_ROI_HALF
            print("Warming up policy agent")
            self.agent.predict_actions(self.get_observations(), self.history_actions)

        print("IterativeCastingUR5Task initialized successfully")
        print(f"Agent update frequency: {self.agent.agent_update_freq_hz} Hz")

    def reset(self):
        """Reset the robot to home position and reset the agent."""

        ur5_home_pose_xyz_wxyz = self.ur5_home_pose_xyz_wxyz

        # Schedule robot movement to home
        # self.wsg50_client.schedule_joint_traj(
        #     joint_traj_pos=wsg50_home_pos_m[np.newaxis, :],
        #     timestamps=np.array([reset_time_s]),
        #     use_relative_timestamps=True,
        # )
        self.ur5_client.schedule_eef_traj(
            eef_traj_xyz_wxyz=ur5_home_pose_xyz_wxyz[np.newaxis, :],
            timestamps=np.array([self.reset_time_s]),
            use_relative_timestamps=True,
        )

        # Update control timestamp and history
        self.last_control_timestamp = time.monotonic() + self.reset_time_s
        self.history_actions = {
            "action0_eef_xyz_wxyz": ur5_home_pose_xyz_wxyz[np.newaxis, :],
            # "action0_gripper_width": wsg50_home_pos_m[np.newaxis, :],
        }

        # Reset the heuristic drag agent
        self.agent.reset()

        print(f"Resetting robot to home position: {ur5_home_pose_xyz_wxyz}")
        time.sleep(self.reset_time_s)
        self.attempt_counter = 0

        print("Reset complete")

    def get_observations(self):
        # Get current robot state
        current_time = time.monotonic()
        eef_xyz_wxyz = self.ur5_client.get_eef_xyz_wxyz(timestamp=current_time)
        # gripper_width = self.wsg50_client.get_joint_pos(timestamp=current_time)

        observations: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.uint8]] = {
            "robot0_eef_xyz_wxyz": eef_xyz_wxyz[np.newaxis, :],  # (1, 7)
            "robot0_gripper_width": np.array([[0.0]]),
            # "robot0_gripper_width": gripper_width[np.newaxis, :],  # (1, 1)
        }

        # Only fetch camera images for PolicyAgent
        if isinstance(self.agent, PolicyAgent):
            (
                img_dict,
                img_timestamps,
            ) = self.third_person_camera_client.get_latest_images_dict_THWC(1)
            observations["third_person_camera"] = img_dict["main"]
            observations["timestamps"] = np.array([current_time])

        return observations

    def get_history_actions(self, length: int):
        history_actions: dict[str, npt.NDArray[np.float64]] = {}
        for k, v in self.history_actions.items():
            if v.shape[0] > length:
                history_actions[k] = v[-length:]
            elif v.shape[0] < length:
                history_actions[k] = np.concatenate(
                    [np.repeat(v[:1, ...], length - v.shape[0], axis=0), v], axis=0
                )
            else:
                history_actions[k] = v
        return history_actions

    def run_policy_control(self) -> EpisodeStatus:
        """
        Run heuristic agent control. Agent outputs absolute poses (like spacemouse).
        No camera observations, no relative poses, just direct control.

        Returns:
            EpisodeStatus.COMPLETED if episode finished, EpisodeStatus.RUNNING otherwise
        """
        if self.last_control_mode == "spacemouse":
            self.agent.reset()
            self.last_control_timestamp = time.monotonic()
            self.attempt_counter = 0

        observations = self.get_observations()

        # Detect attempts during policy inference (PolicyAgent only)
        if isinstance(self.agent, PolicyAgent) and self.run_attempt_detection:
            start_time = time.monotonic()
            frame = observations["third_person_camera"][0]
            detected = self.attempt_detector(frame)

            # Show real-time visualization with cached detection info
            show_realtime_attempt_detection(
                frame_rgb=frame,
                attempt_count=self.attempt_counter,
                detection_occurred=detected,
                roi_center=self.roi_center,
                roi_half=self.roi_half,
                detection_info=self.attempt_detector.last_detection_info,  # type: ignore
            )

            print(
                f"Attempt detection time: {time.monotonic() - start_time:.3f} seconds"
            )

            if detected:
                self.attempt_counter += 1
                print(f"[POLICY ATTEMP] Detected attempt #{self.attempt_counter}")

                # If 3 attempts reached, send robot to home pose
                if self.attempt_counter >= 3:
                    print(
                        "[POLICY AGENT] 3 attempts reached, sending robot to home pose"
                    )
                    self.ur5_client.schedule_eef_traj(
                        eef_traj_xyz_wxyz=self.ur5_home_pose_xyz_wxyz[np.newaxis, :],
                        timestamps=np.array([self.reset_time_s]),
                        use_relative_timestamps=True,
                    )
                    self.last_control_timestamp = time.monotonic() + self.reset_time_s
                    self.history_actions = {
                        "action0_eef_xyz_wxyz": self.ur5_home_pose_xyz_wxyz[
                            np.newaxis, :
                        ],
                        "action0_gripper_width": np.array([[0.0]]),
                    }
                    time.sleep(self.reset_time_s)
                    print("[POLICY AGENT] Robot returned to home pose")
                    return EpisodeStatus.COMPLETED

        # Get absolute actions from agent
        start_time = time.monotonic()
        if isinstance(self.agent, PolicyAgent) or isinstance(
            self.agent, LongHistPolicyAgent
        ):
            observations["third_person_camera"] = resize_with_cropping(
                observations["third_person_camera"], display_wh=(256, 256), align="left"
            )
        actions = self.agent.predict_actions(
            observations=observations,
            history_actions=self.get_history_actions(self.agent.action_history_len),
        )
        print(f"Policy agent prediction time: {time.monotonic() - start_time} seconds")

        # Confine actions to be in the same frame:
        if self.confine_actions:
            actions["action0_eef_xyz_wxyz"][:, 0] = -0.6

            original_quat_wxyz = actions["action0_eef_xyz_wxyz"][:, 3:7]
            new_quat_xy = np.mean(original_quat_wxyz[:, 1:3], axis=1)
            new_quat_abs_wz = (
                np.abs(original_quat_wxyz[:, 0]) + np.abs(original_quat_wxyz[:, 3])
            ) / 2.0
            new_quat_z = new_quat_abs_wz * np.sign(original_quat_wxyz[:, 3])
            new_quat_w = new_quat_abs_wz * np.sign(original_quat_wxyz[:, 0])
            new_quat_wxyz = np.array(
                [new_quat_w, new_quat_xy, new_quat_xy, new_quat_z]
            ).transpose()
            actions["action0_eef_xyz_wxyz"] = np.concatenate(
                [actions["action0_eef_xyz_wxyz"][:, :3], new_quat_wxyz], axis=1
            )

        if "is_done" in actions and actions["is_done"]:
            print("Episode completed successfully")
            return EpisodeStatus.COMPLETED

        if isinstance(self.agent, CastingAgent):
            # Wait for next control tick
            # prev_control_timestamp = self.last_control_timestamp
            next_control_timestamp = (
                self.last_control_timestamp + 1.0 / self.agent.agent_update_freq_hz
            )
            wait_until(monotonic_time=next_control_timestamp)
            self.last_control_timestamp = next_control_timestamp

            # time_diff = next_control_timestamp - prev_control_timestamp
            # print(f"Control dt: {time_diff:.4f}s -> {1.0/time_diff:.2f} Hz")

            # CastingAgent returns single pose, use absolute timestamp
            actions["timestamps"] = np.array(
                [
                    next_control_timestamp + self.ur5_lookahead_time_s + 0.1
                ]  # HACK: additional look ahead
            )
        else:
            # PolicyAgent returns multiple poses, use multiple timestamps
            # Use absolute timestamps
            actions["timestamps"] = (
                np.arange(self.agent.action_prediction_horizon)
                / self.agent.agent_update_freq_hz
                + self.ur5_lookahead_time_s
                + time.monotonic()
            )

            y_actions = actions["action0_eef_xyz_wxyz"][:, 1]
            if y_actions[0] < 0 and y_actions[-1] > 0:
                # casting motions
                vel = (y_actions[1:] - y_actions[:-1]) / (
                    actions["timestamps"][1:] - actions["timestamps"][:-1]
                )

                max_vels = np.max(np.abs(vel))
                top_5_vels = np.sort(np.abs(vel))[-5:]
                print(f"Casting velocities: {vel} m/s")
                print(f"Casting top 5 velocities: {top_5_vels} m/s")

        # print(f"run policy control: {actions=}")

        # Schedule trajectories (actions are absolute poses)
        self.ur5_client.schedule_eef_traj(
            eef_traj_xyz_wxyz=actions["action0_eef_xyz_wxyz"],
            timestamps=actions["timestamps"],
            use_relative_timestamps=False,
        )
        # self.wsg50_client.schedule_joint_traj(
        #     joint_traj_pos=abs_heuristic_actions["action0_gripper_width"],
        #     timestamps=np.array([1.0 / self.agent.agent_update_freq_hz + 0.1]),
        #     use_relative_timestamps=True,
        # )

        # Update history and mode (filter out is_done)
        self.history_actions = {k: v for k, v in actions.items() if k != "is_done"}
        self.last_control_mode = "policy"

        # Print status only when phase or trial changes
        if isinstance(self.agent, CastingAgent):
            if (
                self.agent.phase != self.last_phase
                or self.agent.trial_cnt != self.last_trial
            ):
                print(
                    f"Phase: {self.agent.phase:<25s} | Trial: {self.agent.trial_cnt + 1}/{len(self.agent.drag_vels_m_per_s)}"
                )
                self.last_phase = self.agent.phase
                self.last_trial = self.agent.trial_cnt
        else:
            wait_until(
                monotonic_time=actions["timestamps"][
                    self.agent.action_execution_horizon - 1
                ]
            )

        return EpisodeStatus.RUNNING

    def reset_policy_agent(self):
        self.agent.reset()
        # Reset attempt counter for PolicyAgent
        if isinstance(self.agent, PolicyAgent):
            self.attempt_counter = 0
            print("[POLICY AGENT] Attempt counter reset to 0")

    def run_spacemouse_control(self):
        """Run spacemouse control (copied from UmiUR5Task)."""
        eef_xyz_wxyz = self.ur5_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        # gripper_width = self.wsg50_client.get_joint_pos(timestamp=time.monotonic())
        # print(f"{eef_xyz_wxyz.shape=}")
        gripper_width = np.array([0.0])
        if self.last_control_mode == "policy":
            if isinstance(self.agent, PolicyAgent):
                self.agent.export_data()
            self.agent.reset()

        history_actions = copy.deepcopy(self.history_actions)
        history_actions["action0_gripper_width"] = np.array([[0.0]])

        actions = self.spacemouse_agent.predict_actions(
            observations={
                "robot0_eef_xyz_wxyz": eef_xyz_wxyz[np.newaxis, :],
                "robot0_gripper_width": gripper_width[np.newaxis, :],
            },
            history_actions=history_actions,
        )

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
        # self.wsg50_client.schedule_joint_traj(
        #     joint_traj_pos=actions["action0_gripper_width"][:, :],
        #     timestamps=np.array([1.0 / self.spacemouse_agent.agent_update_freq_hz]),
        #     use_relative_timestamps=True,
        # )
        self.history_actions = copy.deepcopy(actions)
        self.last_control_mode = "spacemouse"

    def display_robot_state(self):
        """Display current robot state."""
        eef_xyz_wxyz = self.ur5_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
        print(f"TCP pose: {eef_xyz_wxyz}")

    def disconnect(self):
        """Clean up resources."""
        pass

    def export_data(self):
        if isinstance(self.agent, PolicyAgent):
            self.agent.export_data()

    def batch_run(self, num_episodes: int):
        """
        Run multiple episodes sequentially.
        Recording starts when robot reaches drag start position (waiting_before_dragging phase).

        All agent parameters (drag_vels_m_per_s, reset_to_back_duration_s, etc.)
        should be configured via YAML (e.g., batch_iterative_casting.yaml).

        Args:
            num_episodes: Number of episodes to run
        """
        print(f"[BATCH RUN] Starting batch run: {num_episodes} episodes")
        if isinstance(self.agent, CastingAgent):
            print(f"Speed sequence: {list(self.agent.drag_vels_m_per_s)}")

        for trial_idx in range(num_episodes):
            print(f"\n[BATCH RUN] Run {trial_idx + 1}/{num_episodes}")

            # move to reset position if needed
            current_eef = self.ur5_client.get_eef_xyz_wxyz(timestamp=time.monotonic())
            dist_to_reset = np.linalg.norm(
                current_eef[:3] - self.ur5_home_pose_xyz_wxyz[:3]
            )

            if dist_to_reset > 0.01:
                print(
                    f"[BATCH RUN] Moving to reset position (distance: {dist_to_reset:.3f}m)"
                )
                self.ur5_client.schedule_eef_traj(
                    eef_traj_xyz_wxyz=self.ur5_home_pose_xyz_wxyz[np.newaxis, :],
                    timestamps=np.array([self.reset_time_s]),
                    use_relative_timestamps=True,
                )
                self.last_control_timestamp = time.monotonic() + self.reset_time_s + 1.0
                time.sleep(self.reset_time_s + 1.0)
            else:
                print(f"[BATCH RUN] Already at reset position")

            # run agent until it reaches drag start position
            self.reset_policy_agent()
            if isinstance(self.agent, CastingAgent):
                while self.agent.phase != "reset_to_start":
                    self.run_policy_control()
            else:
                # For PolicyAgent, just start recording immediately
                pass

            # start recording at drag start position (includes short waiting buffer)
            print("[BATCH RUN] Starting recording at drag start position")
            self.start_episode()

            try:
                while True:
                    status = self.run_policy_control()
                    if status == EpisodeStatus.COMPLETED:
                        break

                self.stop_episode(is_successful=True)
                print(f"[BATCH RUN] Run {trial_idx + 1} completed\n")
            except KeyboardInterrupt:
                print("\n[BATCH RUN] Interrupted, stopping episode...")
                self.stop_episode(is_successful=False)
                raise

        print(f"[BATCH RUN] Batch run completed: {num_episodes} episodes finished")


def run_iterative_casting():

    np.set_printoptions(precision=4)
    os.environ["HYDRA_FULL_ERROR"] = "1"
    mode_name = sys.argv[1]
    assert mode_name in [
        "policy",
        "longhist_policy",
        "heuristic",
        "heuristic_batch",
    ], "Invalid mode name"
    with hydra.initialize(config_path="../configs/tasks", version_base=None):
        cfg = hydra.compose(config_name=f"iterative_casting_{mode_name}")
    print(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
    if mode_name == "heuristic_batch":
        num_episodes = cfg.pop("num_episodes", 1)
    else:
        num_episodes = 1
    task: IterativeCastingUR5Task = hydra.utils.instantiate(cfg)
    try:
        if mode_name == "heuristic_batch":
            task.batch_run(num_episodes=num_episodes)
        else:
            task.run()
    except KeyboardInterrupt:
        print("\nShutting down...")


# def run_iterative_casting_policy():
#     """Run single episode (interactive mode) using iterative_casting_policy.yaml config."""
#     os.environ["HYDRA_FULL_ERROR"] = "1"
#     with hydra.initialize(config_path="../configs/tasks", version_base=None):
#         cfg = hydra.compose(config_name="iterative_casting_policy")
#     print(cfg)
#     OmegaConf.set_struct(cfg, False)
#     cfg.config = OmegaConf.to_container(cfg, resolve=True)
#     task: IterativeCastingUR5Task = hydra.utils.instantiate(cfg)
#     try:
#         task.run()
#     except KeyboardInterrupt:
#         print("\nShutting down...")


# def run_batch_iterative_casting():
#     """Run batch mode using batch_iterative_casting.yaml config."""
#     os.environ["HYDRA_FULL_ERROR"] = "1"
#     with hydra.initialize(config_path="../configs/tasks", version_base=None):
#         cfg = hydra.compose(config_name="batch_iterative_casting")
#     # print(cfg)

#     # Extract batch-specific params
#     num_episodes = cfg.get("num_episodes", 1)

#     # # Create task config without num_episodes
#     # task_cfg = OmegaConf.to_container(cfg, resolve=True)
#     # task_cfg.pop("num_episodes", None)
#     # task_cfg = OmegaConf.create(task_cfg)

#     OmegaConf.set_struct(cfg, False)
#     cfg.pop("num_episodes", NoP] Detene)
#     cfg.config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
#     # Instantiate task (agent already configured with speeds from YAML)
#     task: IterativeCastingUR5Task = hydra.utils.instantiate(cfg)

#     try:
#         task.batch_run(num_episodes=num_episodes)
#     except KeyboardInterrupt:
#         print("\nBatch run interrupted...")


# if __name__ == "__main__":
# Run batch mode by default
# Edit iterative_casting.yaml or batch_iterative_casting.yaml to change parameters
# run_batch_iterative_casting()
# run_iterative_casting_policy()
