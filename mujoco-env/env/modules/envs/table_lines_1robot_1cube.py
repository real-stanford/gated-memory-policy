from typing import Any, cast

from env.modules.common import robot_data_type
from env.modules.envs.base_env import BaseEnv
from env.modules.objects.base_rigid_object import BaseRigidObject
from env.modules.objects.cube import Cube
from env.modules.scenes.table_lines import TableLines


class TableLines1Robot1Cube(BaseEnv):
    def __init__(self, cube_reset_waiting_time: float, **kwargs):
        super().__init__(**kwargs)
        self.cube_pass_line_time: float = 0.0
        self.cube_last_movement_time: float = 0.0
        self.cube_last_pos_y: float = 0.0

        self.cube_reset_waiting_time: float = cube_reset_waiting_time
        # Counting from the cube passing the line
        # Guarantees that the cube is stopped when friction >= 0.005

    def reset(self, episode_config: dict[str, Any] | None = None):
        cube = self.objects[0]
        assert isinstance(cube, Cube)
        if episode_config is not None:
            # self.objects[0].set_pose_xyz_wxyz()

            sliding_friction: float = episode_config["sliding_friction"]
            cube.set_sliding_friction(sliding_friction)

        obs, info = super().reset(episode_config)
        self.cube_pass_line_time = 0.0
        self.cube_last_movement_time = 0.0
        self.cube_last_pos_y = cube.pose_xyz_wxyz[1]

        return obs, info

    def _get_env_objs_obs(self, render_image: bool) -> list[robot_data_type]:
        scene = cast(TableLines, self.scene)
        env_objs_obs: list[robot_data_type] = [
            {
                "name": self.scene.name,
                "line_center_xyz": scene.line_center_xyz,
                "box_center_xyz": scene.box_center_xyz,
                "timestamp": self.episode_current_timestamp,
            },
        ]
        if render_image and len(self.scene.camera_ids) > 0:
            env_objs_obs[0].update(self.scene.camera_images)
        for obj in self.objects:
            assert isinstance(obj, BaseRigidObject)
            env_objs_obs.append(
                {
                    "name": obj.name,
                    "object_pose_xyz_wxyz": obj.pose_xyz_wxyz,
                    "object_vel_xyz_xyz": obj.vel_xyz_xyz,
                }
            )
        return env_objs_obs

    def step(self, actions: list[robot_data_type] | None, render_image: bool = True):
        obs, reward, done, info = super().step(actions, render_image)

        assert isinstance(self.scene, TableLines)
        line_center_y = self.scene.line_center_xyz[1]
        cube = self.objects[0]
        assert isinstance(cube, BaseRigidObject)
        cube_center_y = cube.pose_xyz_wxyz[1]

        if cube_center_y > line_center_y and self.cube_pass_line_time == 0.0:
            self.cube_pass_line_time = self.episode_current_timestamp

        # Teleport the cube to the initial position after it passes the line for 7 seconds
        if (
            self.cube_pass_line_time != 0.0
            and self.episode_current_timestamp - self.cube_pass_line_time
            > self.cube_reset_waiting_time
        ):
            cube.set_pose_xyz_wxyz(self.object_poses_xyz_wxyz[0])
            self.cube_pass_line_time = 0.0

        return obs, reward, done, info

        ## Teleport the cube to the initial position after stopped for 1 second
        # if np.abs(self.objects[0].pose_xyz_wxyz[1] - self.cube_last_pos_y) > 1e-5:
        #     self.cube_last_movement_time = self.episode_current_timestamp
        #     self.cube_last_pos_y = self.objects[0].pose_xyz_wxyz[1]
        # else:
        #     if (
        #         self.episode_current_timestamp - self.cube_last_movement_time
        #         > self.cube_reset_waiting_time
        #     ):
        #         self.objects[0].set_pose_xyz_wxyz(self.object_poses_xyz_wxyz[0])

        # # HACK: force the x velocity to be 0 to prevent from side slipping
        # for obj in self.objects:
        #     vel_xyz_xyz = obj.vel_xyz_xyz
        #     vel_xyz_xyz[0] = 0
        #     obj.set_vel_xyz_xyz(vel_xyz_xyz)
