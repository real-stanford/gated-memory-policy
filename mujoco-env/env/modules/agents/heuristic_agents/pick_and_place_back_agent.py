import copy
from typing import Any, cast

import numpy as np

from env.modules.agents.heuristic_agent import HeuristicAgent
from env.modules.common import castf64, data_buffer_type, f64arr, robot_data_type
from robot_utils.pose_utils import get_absolute_pose
from env.utils.pose_utils import (
    LinearActionInterpolator,
    get_random_convex_combination,
)

class PickAndPlaceBackAgent(HeuristicAgent):
    def __init__(
        self,
        drifting_speed_m_per_s: float,
        drifting_speed_rad_per_s: float,
        critical_action_num: int,  # The first few actions of the "place" phase is considered as critical
        match_bin_material: bool,
        delta_bin_id: int,
        pregrasp_relative_to_object_poses: list[list[float]],
        keyframe_absolute_poses: list[list[float]],
        keyframe_regular_wait_time_ranges_s: list[list[float]],
        keyframe_drifting_time_ranges_s: list[list[float]],
        place_relative_to_bin_center_poses: list[float],
        lift_height_min: float,
        lift_height_max: float,
        grasp_gripper_width: list[float],
        release_gripper_width: list[float],
        keyframe_drifting_prob: float,
        wait_for_color_reveal: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wait_for_color_reveal: bool = wait_for_color_reveal
        self.delta_bin_id: int = delta_bin_id
        self.match_bin_material: bool = match_bin_material
        if self.match_bin_material:
            assert (
                delta_bin_id == 0
            ), f"delta_bin_id must be 0 if match_bin_material is True, but got {delta_bin_id}"
        self.drifting_speed_m_per_s: float = drifting_speed_m_per_s
        self.drifting_speed_rad_per_s: float = drifting_speed_rad_per_s

        # Load primitives
        self.pregrasp_relative_to_object_poses: f64arr = np.array(
            pregrasp_relative_to_object_poses
        )  # [N, 7]
        assert (
            self.pregrasp_relative_to_object_poses.shape[1] == 7
        ), "Pregrasp pose must be 7D"
        self.keyframe_absolute_poses: list[f64arr] = [
            np.array(keyframe_poses) for keyframe_poses in keyframe_absolute_poses
        ]  # K sets of keyframe poses, each set is [N, 7]
        assert all(
            keyframe_poses.shape[1] == 7
            for keyframe_poses in self.keyframe_absolute_poses
        ), "Keyframe pose must be 7D"

        self.keyframe_regular_wait_time_ranges_s: list[f64arr] = [
            np.array(keyframe_wait_time_range)
            for keyframe_wait_time_range in keyframe_regular_wait_time_ranges_s
        ]  # [K, 2]
        self.keyframe_drifting_time_ranges_s: list[f64arr] = [
            np.array(keyframe_wait_time_range)
            for keyframe_wait_time_range in keyframe_drifting_time_ranges_s
        ]  # [K, 2]
        assert all(
            keyframe_wait_time_range.shape == (2,)
            for keyframe_wait_time_range in self.keyframe_regular_wait_time_ranges_s
        ), "Keyframe wait time range must be 2D"
        assert len(self.keyframe_absolute_poses) == len(
            self.keyframe_regular_wait_time_ranges_s
        ), "Keyframe wait time range must be the same length as keyframe poses"

        self.place_relative_to_bin_center_poses: f64arr = np.array(
            place_relative_to_bin_center_poses
        )  # [N, 7]
        assert (
            self.place_relative_to_bin_center_poses.shape[1] == 7
        ), "Place pose must be 7D"
        self.lift_height_min: float = lift_height_min
        self.lift_height_max: float = lift_height_max
        self.grasp_gripper_width: f64arr = np.array(grasp_gripper_width)
        self.release_gripper_width: f64arr = np.array(release_gripper_width)
        self.keyframe_drifting_prob: float = keyframe_drifting_prob

        # Define states
        self.phase: str = (
            "waiting"  # waiting, pregrasp, descend, grasp, lift, keyframe, place, release, return, done
        )
        self.phase_start_cnt: int = 0  # Count the number of actions in each phase
        self.critical_action_num: int = critical_action_num
        self.keyframe_idx: int = 0
        self.keyframe_waited_time_s: float = 0.0
        self.keyframe_regular_target_wait_time: float = 0.0
        self.keyframe_drifting_time_s: float = 0.0
        self.interpolator: LinearActionInterpolator | None = None
        self.init_bin_id: int | None = None
        self.init_bin_material_id: int | None = None
        self.robot_initial_pose: f64arr | None = None
        self.robot_initial_gripper_width: f64arr | None = None
        self.object_initial_pose: f64arr | None = None
        self.lift_height: float | None = None
        self.object_stable_threshold: float = 0.001
        self.color_reveal_waited_time_s: float = 0.0
        self.color_reveal_pause_s: float = 1.0
        self.reset()

    def reset(self, episode_config: dict[str, Any] | None = None):
        if episode_config is not None:
            if "seed" in episode_config:
                self.rng = np.random.default_rng(episode_config["seed"])

        self.phase = "waiting"
        self.keyframe_idx = 0
        self.interpolator = None
        self.init_bin_id = None
        self.init_bin_material_id = None
        self.robot_initial_pose = None
        self.object_initial_pose = None
        self.lift_height = None
        self.color_reveal_waited_time_s = 0.0

    def predict_actions(
        self,
        robots_obs: data_buffer_type,
        env_objs_obs: data_buffer_type,
        history_actions: data_buffer_type,
    ) -> data_buffer_type:
        actions: data_buffer_type = []  # [1 action, 1 robot]
        robot_action: robot_data_type = {}
        robot_action["name"] = robots_obs[0][0]["name"]
        robot_action["is_error"] = np.array([False])
        robot_action["is_critical"] = np.array([False])

        self.phase_start_cnt += 1

        if self.phase == "waiting":  # Wait until the object pose is stable
            if self.robot_initial_pose is None:
                self.robot_initial_pose = castf64(robots_obs[0][0]["tcp_xyz_wxyz"])
            if self.robot_initial_gripper_width is None:
                self.robot_initial_gripper_width = castf64(
                    robots_obs[0][0]["gripper_width"]
                )
            if self.object_initial_pose is None:
                self.object_initial_pose = castf64(
                    env_objs_obs[0][1]["object_pose_xyz_wxyz"],
                )
            else:
                new_object_pose = castf64(env_objs_obs[0][1]["object_pose_xyz_wxyz"])
                # Wait until the object pose is stable
                if (
                    self.object_stable_threshold / self.agent_update_freq_hz
                    > np.linalg.norm(new_object_pose[:3] - self.object_initial_pose[:3])
                ):
                    # logger.info(f"============== pregrasp ===============")
                    self.phase = "pregrasp"
                    self.phase_start_cnt = 0
                    if self.init_bin_id is None:
                        self.init_bin_id = cast(int, env_objs_obs[0][1]["bin_id"])

                    pregrasp_relative_to_object_pose = get_random_convex_combination(
                        self.pregrasp_relative_to_object_poses,
                        self.rng,
                    )
                    pregrasp_absolute_pose = get_absolute_pose(
                        new_object_pose, pregrasp_relative_to_object_pose
                    )
                    self.lift_height = self.rng.uniform(
                        self.lift_height_min, self.lift_height_max
                    )
                    pregrasp_absolute_pose[2] += self.lift_height
                    self.interpolator = LinearActionInterpolator(
                        start_pose_xyz_wxyz=self.robot_initial_pose,
                        end_pose_xyz_wxyz=pregrasp_absolute_pose,
                        start_gripper_width=self.robot_initial_gripper_width,
                        end_gripper_width=self.release_gripper_width,
                        pos_speed_m_per_s=self.position_speed_m_per_s,
                        rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                        gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                    )

                self.object_initial_pose = new_object_pose
            robot_action = copy.deepcopy(history_actions[0][0])
            robot_action["is_error"] = np.array([False])
            robot_action["is_critical"] = np.array([False])
            actions.append([robot_action])

        elif self.phase == "pregrasp":
            assert self.interpolator is not None
            robot_action["tcp_xyz_wxyz"], robot_action["gripper_width"] = (
                self.interpolator.interpolate(1 / self.agent_update_freq_hz)
            )
            actions.append([robot_action])

            if self.interpolator.is_finished:
                self.phase = "descend"
                self.phase_start_cnt = 0
                # logger.info(f"============== grasp ===============")
                lifted_grasp_pose = robot_action["tcp_xyz_wxyz"]
                grasp_pose = lifted_grasp_pose.copy()
                assert self.lift_height is not None
                grasp_pose[2] -= self.lift_height
                self.interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=lifted_grasp_pose,
                    end_pose_xyz_wxyz=grasp_pose,
                    start_gripper_width=self.release_gripper_width,
                    end_gripper_width=self.release_gripper_width,
                    pos_speed_m_per_s=self.position_speed_m_per_s,
                    rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                    gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                )

        elif self.phase == "descend":
            assert self.interpolator is not None
            robot_action["tcp_xyz_wxyz"], robot_action["gripper_width"] = (
                self.interpolator.interpolate(1 / self.agent_update_freq_hz)
            )
            actions.append([robot_action])

            if self.interpolator.is_finished:
                self.phase = "grasp"
                self.phase_start_cnt = 0
                # logger.info(f"============== grasp ===============")
                grasp_pose = robot_action["tcp_xyz_wxyz"]
                self.interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=grasp_pose,
                    end_pose_xyz_wxyz=grasp_pose,
                    start_gripper_width=self.release_gripper_width,
                    end_gripper_width=self.grasp_gripper_width,
                    pos_speed_m_per_s=self.position_speed_m_per_s,
                    rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                    gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                )

        elif self.phase == "grasp":
            assert self.interpolator is not None
            robot_action["tcp_xyz_wxyz"], robot_action["gripper_width"] = (
                self.interpolator.interpolate(1 / self.agent_update_freq_hz)
            )
            actions.append([robot_action])

            if self.interpolator.is_finished:

                if self.match_bin_material:
                    self.init_bin_material_id = cast(
                        int,
                        env_objs_obs[0][0]["bin_materials"][self.init_bin_id],
                    )
                    assert (
                        self.init_bin_material_id > 0
                    ), "Bin color id must be greater than 0 (0: transparent, 1: red, 2: green, 3: blue, 4: yellow)"

                self.phase = "lift"
                self.keyframe_idx = 0
                self.phase_start_cnt = 0
                # logger.info(f"============== lift ===============")
                grasp_pose = robot_action["tcp_xyz_wxyz"]
                lift_pose = grasp_pose.copy()
                self.lift_height = self.rng.uniform(
                    self.lift_height_min, self.lift_height_max
                )
                lift_pose[2] += self.lift_height
                self.interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=grasp_pose,
                    end_pose_xyz_wxyz=lift_pose,
                    start_gripper_width=self.grasp_gripper_width,
                    end_gripper_width=self.grasp_gripper_width,
                    pos_speed_m_per_s=self.position_speed_m_per_s,
                    rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                    gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                )

        elif self.phase == "lift":
            assert self.interpolator is not None
            robot_action["tcp_xyz_wxyz"], robot_action["gripper_width"] = (
                self.interpolator.interpolate(1 / self.agent_update_freq_hz)
            )
            actions.append([robot_action])
            if self.interpolator.is_finished:
                self.phase = "keyframe"
                self.keyframe_idx = 0
                self.phase_start_cnt = 0
                # logger.info(f"============== keyframe {self.keyframe_idx} ===============")
                lift_pose = robot_action["tcp_xyz_wxyz"]
                keyframe_poses = self.keyframe_absolute_poses[self.keyframe_idx]
                keyframe_pose = get_random_convex_combination(keyframe_poses, self.rng)
                # logger.info(f"keyframe_pose{self.keyframe_idx}: {keyframe_pose}")
                include_error = self.rng.uniform(0.0, 1.0) < self.keyframe_drifting_prob
                if include_error:
                    self.keyframe_drifting_time_s = self.rng.uniform(
                        self.keyframe_drifting_time_ranges_s[self.keyframe_idx][0],
                        self.keyframe_drifting_time_ranges_s[self.keyframe_idx][1],
                    )
                    self.keyframe_regular_target_wait_time = (
                        self.keyframe_regular_wait_time_ranges_s[self.keyframe_idx][1]
                    )  # Take the upper bound
                else:
                    self.keyframe_drifting_time_s = 0.0
                    self.keyframe_regular_target_wait_time = self.rng.uniform(
                        self.keyframe_regular_wait_time_ranges_s[self.keyframe_idx][0],
                        self.keyframe_regular_wait_time_ranges_s[self.keyframe_idx][1],
                    )
                self.keyframe_waited_time_s = 0.0
                self.interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=lift_pose,
                    end_pose_xyz_wxyz=keyframe_pose,
                    start_gripper_width=self.grasp_gripper_width,
                    end_gripper_width=self.grasp_gripper_width,
                    pos_speed_m_per_s=self.position_speed_m_per_s,
                    rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                    gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                )
        elif self.phase == "keyframe":
            assert self.interpolator is not None
            robot_action["tcp_xyz_wxyz"], robot_action["gripper_width"] = (
                self.interpolator.interpolate(1 / self.agent_update_freq_hz)
            )
            actions.append([robot_action])
            if self.interpolator.is_finished:
                self.keyframe_waited_time_s += 1 / self.agent_update_freq_hz
                if (
                    self.keyframe_waited_time_s
                    >= self.keyframe_regular_target_wait_time
                ):
                    # if (
                    #     self.keyframe_waited_time_s
                    #     <= self.keyframe_regular_target_wait_time
                    #     + self.keyframe_drifting_time_s
                    # ):
                    #     actions[0][0]["is_error"] = np.array([True])
                    #     return actions

                    if self.keyframe_drifting_time_s > 0.0:
                        self.phase = "drift"
                        self.phase_start_cnt = 0
                        current_pose = robot_action["tcp_xyz_wxyz"]
                        drifting_pose = current_pose.copy()
                        drifting_pos_direction = self.rng.uniform(0.0, 1.0, (3,))
                        drifting_pos_direction = (
                            drifting_pos_direction
                            / np.linalg.norm(drifting_pos_direction)
                        )
                        drifting_pos_direction[2] = -np.abs(
                            drifting_pos_direction[2]
                        )  # Drifting position direction should be downwards
                        drifting_rot_direction = self.rng.uniform(0.0, 1.0, (4,))
                        drifting_rot_direction = (
                            drifting_rot_direction
                            / np.linalg.norm(drifting_rot_direction)
                        )
                        drifting_pos_speed = self.rng.uniform(
                            0, self.drifting_speed_m_per_s
                        )
                        drifting_rot_speed = self.rng.uniform(
                            0, self.drifting_speed_rad_per_s
                        )
                        drifting_pose[:3] += (
                            drifting_pos_speed
                            * drifting_pos_direction
                            * self.keyframe_drifting_time_s
                        )
                        drifting_pose[3:] += (
                            drifting_rot_speed
                            * drifting_rot_direction
                            * self.keyframe_drifting_time_s
                        )
                        self.interpolator = LinearActionInterpolator(
                            start_pose_xyz_wxyz=current_pose,
                            end_pose_xyz_wxyz=drifting_pose,
                            start_gripper_width=self.grasp_gripper_width,
                            end_gripper_width=self.grasp_gripper_width,
                            pos_speed_m_per_s=drifting_pos_speed,
                            rot_speed_rad_per_s=drifting_rot_speed,
                            gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                        )
                    else:
                        if self.wait_for_color_reveal:
                            bin_materials = env_objs_obs[0][0]["bin_materials"]
                            if all(m == 0 for m in bin_materials):
                                # Colors still hidden, keep waiting
                                return actions
                            # Colors revealed, pause before placing
                            self.color_reveal_waited_time_s += 1 / self.agent_update_freq_hz
                            if self.color_reveal_waited_time_s < self.color_reveal_pause_s:
                                return actions
                        self.phase = "place"
                        self.phase_start_cnt = 0
                        assert self.init_bin_id is not None
                        # logger.info(f"============== place ===============")
                        current_pose = robot_action["tcp_xyz_wxyz"]
                        place_relative_to_bin_center_pose = (
                            get_random_convex_combination(
                                self.place_relative_to_bin_center_poses, self.rng
                            )
                        )
                        place_absolute_pose = place_relative_to_bin_center_pose

                        if self.match_bin_material:
                            bin_materials = env_objs_obs[0][0]["bin_materials"]
                            target_bin_id = bin_materials.index(
                                self.init_bin_material_id
                            )
                        else:
                            target_bin_id = (self.init_bin_id + self.delta_bin_id) % 4
                        place_absolute_pose[:3] += cast(
                            f64arr,
                            env_objs_obs[0][0]["bin_center_xyz"],
                        )[target_bin_id]
                        # logger.info(f"place_absolute_pose: {place_absolute_pose}")
                        self.interpolator = LinearActionInterpolator(
                            start_pose_xyz_wxyz=current_pose,
                            end_pose_xyz_wxyz=place_absolute_pose,
                            start_gripper_width=self.grasp_gripper_width,
                            end_gripper_width=self.grasp_gripper_width,
                            pos_speed_m_per_s=self.position_speed_m_per_s,
                            rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                            gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                        )

        elif self.phase == "drift":
            assert self.interpolator is not None
            robot_action["tcp_xyz_wxyz"], robot_action["gripper_width"] = (
                self.interpolator.interpolate(1 / self.agent_update_freq_hz)
            )
            robot_action["is_error"] = np.array([True])
            actions.append([robot_action])
            if self.interpolator.is_finished:
                self.phase = "place"
                self.phase_start_cnt = 0
                assert self.init_bin_id is not None
                # logger.info(f"============== place ===============")
                current_pose = robot_action["tcp_xyz_wxyz"]
                place_relative_to_bin_center_pose = get_random_convex_combination(
                    self.place_relative_to_bin_center_poses, self.rng
                )
                place_absolute_pose = place_relative_to_bin_center_pose
                if self.match_bin_material:
                    bin_materials = env_objs_obs[0][0]["bin_materials"]
                    target_bin_id = bin_materials.index(self.init_bin_material_id)
                else:
                    target_bin_id = (self.init_bin_id + self.delta_bin_id) % 4
                place_absolute_pose[:3] += cast(
                    f64arr,
                    env_objs_obs[0][0]["bin_center_xyz"],
                )[target_bin_id]

                # logger.info(f"place_absolute_pose: {place_absolute_pose}")
                self.interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=current_pose,
                    end_pose_xyz_wxyz=place_absolute_pose,
                    start_gripper_width=self.grasp_gripper_width,
                    end_gripper_width=self.grasp_gripper_width,
                    pos_speed_m_per_s=self.position_speed_m_per_s,
                    rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                    gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                )

        elif self.phase == "place":
            if self.phase_start_cnt <= self.critical_action_num:
                robot_action["is_critical"] = np.array([True])

            assert self.interpolator is not None
            robot_action["tcp_xyz_wxyz"], robot_action["gripper_width"] = (
                self.interpolator.interpolate(1 / self.agent_update_freq_hz)
            )
            actions.append([robot_action])
            if self.interpolator.is_finished:
                self.phase = "release"
                self.phase_start_cnt = 0
                # logger.info(f"============== release ===============")
                self.interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=robot_action["tcp_xyz_wxyz"],
                    end_pose_xyz_wxyz=robot_action["tcp_xyz_wxyz"],
                    start_gripper_width=self.grasp_gripper_width,
                    end_gripper_width=self.release_gripper_width,
                    pos_speed_m_per_s=self.position_speed_m_per_s,
                    rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                    gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                )
        elif self.phase == "release":
            assert self.interpolator is not None
            robot_action["tcp_xyz_wxyz"], robot_action["gripper_width"] = (
                self.interpolator.interpolate(1 / self.agent_update_freq_hz)
            )
            actions.append([robot_action])
            if self.interpolator.is_finished:
                assert self.robot_initial_pose is not None
                self.phase = "return"
                self.phase_start_cnt = 0
                # logger.info(f"============== return ===============")
                self.interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=robot_action["tcp_xyz_wxyz"],
                    end_pose_xyz_wxyz=self.robot_initial_pose,
                    start_gripper_width=self.release_gripper_width,
                    end_gripper_width=self.release_gripper_width,
                    pos_speed_m_per_s=self.position_speed_m_per_s,
                    rot_speed_rad_per_s=self.rotation_speed_rad_per_s,
                    gripper_speed_m_per_s=self.gripper_speed_m_per_s,
                )

        elif self.phase == "return":
            assert self.interpolator is not None
            robot_action["tcp_xyz_wxyz"], robot_action["gripper_width"] = (
                self.interpolator.interpolate(1 / self.agent_update_freq_hz)
            )
            actions.append([robot_action])
            if self.interpolator.is_finished:
                self.phase = "done"
                self.phase_start_cnt = 0
                self.interpolator = None
        elif self.phase == "done":
            assert self.robot_initial_pose is not None
            robot_action["tcp_xyz_wxyz"] = self.robot_initial_pose
            robot_action["gripper_width"] = self.release_gripper_width
            actions.append([robot_action])

        return actions
