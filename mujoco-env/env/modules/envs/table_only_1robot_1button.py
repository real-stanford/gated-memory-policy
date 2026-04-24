from typing import Any, cast

import numpy as np
import numpy.typing as npt

from env.modules.common import robot_data_type
from env.modules.envs.base_env import BaseEnv
from env.modules.objects.button import Button
from env.modules.scenes.table_only import TableOnly
from robot_utils.pose_utils import get_relative_pose
from env.utils.pose_utils import (get_random_4poses_convex_combination)


class TableOnly1Robot1Button(BaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.scene, TableOnly):
            raise ValueError("Scene must be TableOnly")
        assert "robot_init_tcp_poses_candidates_xyz_wxyz" in kwargs
        self.robot_init_tcp_poses_candidates_xyz_wxyz: list[npt.NDArray[np.float64]] = [
            np.array(pose)
            for pose in kwargs["robot_init_tcp_poses_candidates_xyz_wxyz"]
        ]

    def _get_env_objs_obs(self, render_image: bool) -> list[robot_data_type]:
        scene = cast(TableOnly, self.scene)
        table_center_xyz = scene.get_table_center_xyz()
        env_objs_obs: list[robot_data_type] = [
            {
                "name": self.scene.name,
                "table_center_xyz": table_center_xyz,
                "timestamp": self.episode_current_timestamp,
            },
        ]
        if render_image:
            env_objs_obs[0].update(self.scene.camera_images)
        for obj in self.objects:
            tcp_relative_poses_xyz_wxyz = np.array(
                [
                    get_relative_pose(robot.tcp_xyz_wxyz, obj.pose_xyz_wxyz)
                    for robot in self.robots
                ]
            )
            assert isinstance(obj, Button)
            env_objs_obs.append(
                {
                    "name": obj.name,
                    "object_pose_xyz_wxyz": obj.pose_xyz_wxyz,
                    "is_pressed": obj.is_pressed,
                }
            )
        return env_objs_obs

    def reset(self, episode_config: dict[str, Any] | None = None):
        if episode_config is not None:
            if "seed" not in episode_config:
                episode_config["seed"] = 0
            self.rng = np.random.default_rng(episode_config["seed"])

            if (
                "robot_init_tcp_poses_xyz_wxyz" not in episode_config
                or episode_config["robot_init_tcp_poses_xyz_wxyz"] is None
            ):
                episode_config["robot_init_tcp_poses_xyz_wxyz"] = [
                    get_random_4poses_convex_combination(
                        self.robot_init_tcp_poses_candidates_xyz_wxyz[0], self.rng
                    )
                ]

            # self.object_poses_xyz_wxyz = episode_config["object_poses_xyz_wxyz"]
            self.robot_init_tcp_poses_xyz_wxyz = episode_config[
                "robot_init_tcp_poses_xyz_wxyz"
            ]
            self.robot_init_gripper_width = [
                np.array([0.0])
            ]  # Gripper will stay closed

        super().reset()
