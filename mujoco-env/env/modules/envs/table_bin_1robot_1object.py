from typing import Any, cast

import numpy as np

from env.modules.common import robot_data_type
from env.modules.envs.base_env import BaseEnv
from env.modules.objects.base_rigid_object import BaseRigidObject
from env.modules.scenes.table_bin import TableBin
from robot_utils.pose_utils import get_relative_pose


class TableBin1Robot1Object(BaseEnv):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(self.scene, TableBin):
            raise ValueError("Scene must be TableBin")

        self.shuffle_bin_material: bool = False
        self.delay_init_bin_materials: bool = (
            False  # Will not initialize different bin materials until the robot gripper is reaching down
        )
        self.done_init_bin_materials: bool = False
        self.done_shuffling_bin_materials: bool = False
        self.switch_bin_materials_waited_steps: int = 0
        self.switch_bin_materials_waited_steps_threshold: int = (
            5  # Wait for 0.5s before shuffling color
        )

        self.z_threshold_once_picked_up: float = 0.25
        self.z_threshold_near_hanging: float = 0.35

        self.gripper_width_threshold: float = 0.05
        self.init_bin_materials: list[int] = [0, 0, 0, 0]  # Transparent
        self.final_bin_materials: list[int] = [0, 0, 0, 0]  # Transparent

        # Random delay: hide colors after pickup, wait, then reveal final colors
        self.rand_delay_steps: int = 0  # 0 means no random delay
        self.colors_hidden_for_delay: bool = False
        self.rand_delay_waited_steps: int = 0

    def _get_env_objs_obs(self, render_image: bool) -> list[robot_data_type]:
        scene = cast(TableBin, self.scene)

        bin_center_xyz = np.array(
            [scene.get_bin_center_xyz(bin_id) for bin_id in range(4)]
        )  # (4, 3)
        env_objs_obs: list[robot_data_type] = [
            {
                "name": self.scene.name,
                "bin_center_xyz": bin_center_xyz,
                "bin_materials": scene.bin_materials,
                "timestamp": self.episode_current_timestamp,
            },
        ]
        if render_image:
            env_objs_obs[0].update(self.scene.camera_images)
        for obj in self.objects:
            assert isinstance(obj, BaseRigidObject)
            tcp_relative_poses_xyz_wxyz = np.array(
                [
                    get_relative_pose(robot.tcp_xyz_wxyz, obj.pose_xyz_wxyz)
                    for robot in self.robots
                ]
            )
            env_objs_obs.append(
                {
                    "name": obj.name,
                    "object_pose_xyz_wxyz": obj.pose_xyz_wxyz,
                    "bin_id": scene.get_bin_id(obj.pose_xyz_wxyz[:3]),
                    "tcp_relative_poses_xyz_wxyz": tcp_relative_poses_xyz_wxyz,
                }
            )
        return env_objs_obs

    def step(self, actions: list[robot_data_type] | None, render_image: bool = True):
        obs, reward, done, info = super().step(actions, render_image)
        assert isinstance(self.scene, TableBin)
        if self.shuffle_bin_material:
            if not self.done_init_bin_materials:
                robot_z = self.robots[0].tcp_xyz_wxyz[2]
                gripper_width = self.robots[0].gripper_width_m
                if (
                    robot_z
                    < self.z_threshold_once_picked_up
                    # and gripper_width < self.gripper_width_threshold
                ):
                    self.done_init_bin_materials = True
                    self.scene.set_bin_materials(self.init_bin_materials)

            if not self.done_shuffling_bin_materials:
                robot_z = self.robots[0].tcp_xyz_wxyz[2]
                gripper_width = self.robots[0].gripper_width_m
                object_z = self.objects[0].pose_xyz_wxyz[2]

                z_threshold = self.z_threshold_once_picked_up

                if (
                    robot_z > z_threshold
                    and gripper_width < self.gripper_width_threshold
                    and object_z > z_threshold
                ):  # Holding the object and is above the threshold
                    self.switch_bin_materials_waited_steps += 1
                    if (
                        self.switch_bin_materials_waited_steps
                        > self.switch_bin_materials_waited_steps_threshold
                    ):
                        if self.rand_delay_steps > 0:
                            # Random delay mode: hide colors first, then wait, then reveal
                            if not self.colors_hidden_for_delay:
                                self.scene.set_bin_materials([0] * 4)
                                self.colors_hidden_for_delay = True
                                self.rand_delay_waited_steps = 0
                            else:
                                self.rand_delay_waited_steps += 1
                                if self.rand_delay_waited_steps >= self.rand_delay_steps:
                                    self.scene.set_bin_materials(self.final_bin_materials)
                                    self.done_shuffling_bin_materials = True
                        else:
                            self.scene.set_bin_materials(self.final_bin_materials)
                            self.done_shuffling_bin_materials = True

        return obs, reward, done, info

    def reset(self, episode_config: dict[str, Any] | None = None):
        if episode_config is not None:
            self.done_shuffling_bin_materials = False

            # Random delay setup
            rand_delay_s = episode_config.get("rand_delay_s", 0.0)
            self.rand_delay_steps = int(rand_delay_s * self.control_freq) if rand_delay_s > 0 else 0
            self.colors_hidden_for_delay = False
            self.rand_delay_waited_steps = 0

            self.init_bin_materials = episode_config["init_bin_materials"]
            self.final_bin_materials = episode_config["final_bin_materials"]

            self.object_poses_xyz_wxyz = episode_config["object_poses_xyz_wxyz"]
            self.robot_init_tcp_poses_xyz_wxyz = episode_config[
                "robot_init_tcp_poses_xyz_wxyz"
            ]
            self.robot_init_gripper_width = episode_config["robot_init_gripper_width"]

            assert isinstance(self.scene, TableBin)
            if (
                "shuffle_bin_material" in episode_config
                and episode_config["shuffle_bin_material"] is not None
            ):
                assert isinstance(episode_config["shuffle_bin_material"], bool)
                self.shuffle_bin_material = episode_config["shuffle_bin_material"]
                if (
                    "delay_init_bin_materials" in episode_config
                    and episode_config["delay_init_bin_materials"] is not None
                ):
                    assert isinstance(episode_config["delay_init_bin_materials"], bool)
                    self.delay_init_bin_materials = episode_config[
                        "delay_init_bin_materials"
                    ]
                else:
                    self.delay_init_bin_materials = False

                if self.delay_init_bin_materials:
                    self.scene.set_bin_materials([0] * 4)
                    self.done_init_bin_materials = False
                else:
                    self.scene.set_bin_materials(self.init_bin_materials)
                    self.done_init_bin_materials = True

            else:
                self.shuffle_bin_material = False
                self.scene.set_bin_materials([0] * 4)
                self.done_init_bin_materials = False
                self.done_shuffling_bin_materials = False
                assert (
                    "final_bin_materials" not in episode_config
                    or episode_config["final_bin_materials"] is None
                    or episode_config["final_bin_materials"]
                    == episode_config["init_bin_materials"]
                ), "final_bin_materials should not be provided if shuffle_bin_material is False"

                self.init_bin_materials = episode_config["init_bin_materials"]
                self.final_bin_materials = episode_config[
                    "init_bin_materials"
                ]  # Should not change
                assert isinstance(self.scene, TableBin)
                self.scene.set_bin_materials(self.init_bin_materials)

        return super().reset(episode_config)
