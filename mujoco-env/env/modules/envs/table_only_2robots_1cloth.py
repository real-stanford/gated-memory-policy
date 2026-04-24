from typing import Any

from env.modules.common import robot_data_type
from env.modules.envs.base_env import BaseEnv
from env.modules.objects.cloth import Cloth
from env.modules.scenes.table_only import TableOnly


class TableOnly2Robots1Cloth(BaseEnv):
    """
    ---> y
    |
    v
    x

    -----------------------------------
    |                                 |
    |                                 |
    |                                 |
    |       0 -------------12         |
    |         |   cloth   |           |
    |         |           |           |
    |         |           |           |
    |         |           |           |
    |      156-------------168        |
    -----------------------------------

    robot_left              robot_right

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_mjcf_models(self):

        cloth = self.objects[0]
        assert isinstance(cloth, Cloth)
        bottom_left_index = cloth.horizontal_vertex_num * (
            cloth.vertical_vertex_num - 1
        )
        bottom_right_index = bottom_left_index + cloth.horizontal_vertex_num - 1
        world_model = super()._load_mjcf_models()
        world_model.equality.add(  # type: ignore
            "weld",
            name="cloth_attachment_left",
            body1="ur5e_left/wsg50/end_effector_mocap",
            body2=f"cloth/cloth_{bottom_left_index}",
            # relpose="0 0.01 -0.03 1 0 0 0",
            relpose="0 0 0 1 0 0 0",
            torquescale="0.1",
            solimp="0.97 0.98 0.001 0.5 5",
            # anchor="0 0 0",
        )
        world_model.equality.add(  # type: ignore
            "weld",
            name="cloth_attachment_right",
            body1="ur5e_right/wsg50/end_effector_mocap",
            body2=f"cloth/cloth_{bottom_right_index}",  # the other corner of the cloth
            # relpose="0 0.01 -0.03 1 0 0 0",
            relpose="0 0 0 1 0 0 0",
            torquescale="0.1",
            solimp="0.97 0.98 0.001 0.5 5",
            # anchor="0 0 0",
        )

        # Disable collision to avoid instability
        disable_contact_objects = [
            "ur5e_left/wrist_3_link",
            # "ur5e_left/wsg50/left_finger",
            # "ur5e_left/wsg50/right_finger",
            "ur5e_right/wrist_3_link",
            # "ur5e_right/wsg50/left_finger",
            # "ur5e_right/wsg50/right_finger",
        ]

        for i in range(cloth.horizontal_vertex_num * cloth.vertical_vertex_num):
            for obj in disable_contact_objects:
                world_model.contact.add(  # type: ignore
                    "exclude",
                    name=f"cloth_{i}_exclude_{obj}".replace("/", "_"),
                    body1=f"cloth/cloth_{i}",
                    body2=obj,
                )
        return world_model

    def reset(self, episode_config: dict[str, Any] | None = None):
        cloth = self.objects[0]
        assert isinstance(cloth, Cloth)
        if episode_config is not None:
            cloth_mass = episode_config["cloth_mass"]
            cloth.set_mass(cloth_mass)

        obs, info = super().reset(episode_config)

        return obs, info

    def step(self, actions: list[robot_data_type] | None, render_image: bool = True):
        obs, reward, done, info = super().step(actions, render_image)
        assert isinstance(self.scene, TableOnly)
        return obs, reward, done, info

    def _get_env_objs_obs(self, render_image: bool) -> list[robot_data_type]:
        cloth = self.objects[0]
        assert isinstance(cloth, Cloth)
        env_objs_obs: list[robot_data_type] = [
            {
                "name": self.scene.name,
                "timestamp": self.episode_current_timestamp,
            },
            {
                "name": "cloth",
                "pos_xyz": cloth.get_vertex_positions(),
                "2d_convex_hull_area": cloth.get_2d_convex_hull_area(),
            },
        ]
        if render_image:
            env_objs_obs[0].update(self.scene.camera_images)

        return env_objs_obs
