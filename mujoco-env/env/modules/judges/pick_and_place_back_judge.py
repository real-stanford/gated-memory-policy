from typing import Any, cast

import numpy as np

from env.modules.common import castf64, robot_data_type
from env.modules.judges.base_judge import BaseJudge


class PickAndPlaceBackJudge(BaseJudge):
    def __init__(
        self,
        lift_height_threshold: float,
        place_height_threshold: float,
        tcp_object_distance_threshold: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.init_bin_id: int | None = None
        self.target_bin_id: int | None = None
        self.object_picked_up: bool = False
        self.object_placed_in_bin: bool = False
        self.object_placed_bin_id: int | None = None
        self.lift_height_threshold: float = lift_height_threshold
        self.place_height_threshold: float = place_height_threshold
        self.tcp_object_distance_threshold: float = tcp_object_distance_threshold
        self.finished_timestamp: float | None = None
        self.time_since_finished_s: float | None = None
        self.time_limit_reached: bool = False
        self.match_bin_material: bool = False

    def reset(self, episode_config: dict[str, Any] | None = None):

        self.object_picked_up = False
        self.object_placed_in_bin = False
        self.object_placed_bin_id = None
        self.finished_timestamp = None
        self.time_since_finished_s = None
        self.time_limit_reached = False

        if episode_config is not None:

            assert (
                "object_init_bin_ids" in episode_config
            ), "object_init_bin_ids is required"

            if "shuffle_bin_material" in episode_config:
                self.match_bin_material = episode_config["shuffle_bin_material"]
            self.init_bin_id = episode_config["object_init_bin_ids"][0]
            assert self.init_bin_id is not None, "init_bin_id is not set"
            if self.match_bin_material:
                init_bin_material_id = episode_config["init_bin_materials"][
                    self.init_bin_id
                ]
                assert (
                    init_bin_material_id > 0
                ), "init_bin_material_id must be greater than 0 (transparent)"
                self.target_bin_id = episode_config["final_bin_materials"].index(
                    init_bin_material_id
                )
            else:
                if "delta_bin_id" not in episode_config:
                    episode_config["delta_bin_id"] = 0
                self.target_bin_id = (
                    self.init_bin_id + episode_config["delta_bin_id"]
                ) % 4

    def update(
        self,
        robots_obs: list[
            robot_data_type
        ],  # Only observe the latest state. List is for multiple robots.
        env_objs_obs: list[
            robot_data_type
        ],  # Only observe the latest state. List is for multiple objects. (env global observation is stored in env_objs_obs[0])
    ):
        assert (
            self.init_bin_id is not None
        ), "init_bin_id is not set. Please call reset() first."
        object_pose = castf64(env_objs_obs[1]["object_pose_xyz_wxyz"])
        tcp_object_distance = np.linalg.norm(
            object_pose[:3] - castf64(robots_obs[0]["tcp_xyz_wxyz"])[:3]
        )
        bin_center_xyz = castf64(env_objs_obs[0]["bin_center_xyz"])[self.init_bin_id]

        if self.finished_timestamp is not None:
            self.time_since_finished_s = (
                cast(float, env_objs_obs[0]["timestamp"]) - self.finished_timestamp
            )

        if cast(float, env_objs_obs[0]["timestamp"]) > self.time_limit_s:
            self.time_limit_reached = True

        if not self.object_picked_up:
            if (
                object_pose[2] > self.lift_height_threshold + bin_center_xyz[2]
                and tcp_object_distance < self.tcp_object_distance_threshold
            ):
                self.object_picked_up = True
            return

        if not self.object_placed_in_bin:
            object_bin_id = cast(int, env_objs_obs[1]["bin_id"])
            if (
                object_bin_id >= 0
                and object_pose[2] < self.place_height_threshold + bin_center_xyz[2]
            ):
                self.object_placed_bin_id = object_bin_id
                self.object_placed_in_bin = True
                self.finished_timestamp = cast(float, env_objs_obs[0]["timestamp"])
            return

    def get_reward(self) -> float:
        if not self.object_picked_up:
            return 0.0
        if not self.object_placed_in_bin:
            return 0.0
        if self.time_limit_reached:
            return 0.0
        return float(
            self.object_placed_in_bin
            and self.object_placed_bin_id == self.target_bin_id
        )

    def get_done(self) -> bool:
        if self.time_limit_reached:
            return True
        if not self.object_placed_in_bin:
            return False
        if self.finished_timestamp is None or self.time_since_finished_s is None:
            return False

        return self.time_since_finished_s > self.prolong_after_success_s

    def get_states(self) -> dict[str, Any]:
        return {
            "init_bin_id": self.init_bin_id,
            "target_bin_id": self.target_bin_id,
            "object_picked_up": self.object_picked_up,
            "object_placed_in_bin": self.object_placed_in_bin,
            "object_placed_bin_id": self.object_placed_bin_id,
        }
