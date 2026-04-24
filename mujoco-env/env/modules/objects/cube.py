from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from env.modules.objects.base_rigid_object import BaseRigidObject

if TYPE_CHECKING:
    from dm_control.mjcf.physics import Physics


class Cube(BaseRigidObject):
    def __init__(
        self,
        center_height: float,
        radius: float,
        rotation_angle_range: tuple[float, float],
        color: npt.NDArray[np.float64] | None = None,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.geom_id: int | None = None
        self.friction: npt.NDArray[np.float64] | None = None
        self.color: npt.NDArray[np.float64] | None = color

        if isinstance(rotation_angle_range, list):
            rotation_angle_range = cast(
                tuple[float, float], tuple(rotation_angle_range)
            )
        self.rotation_angle_range: tuple[float, float] = rotation_angle_range

        self.center_height: float = center_height
        self.radius: float = radius

    def set_mujoco_ref(self, physics: "Physics"):
        super().set_mujoco_ref(physics)
        self.geom_id = self.model.geom("cube/cube_obj").id
        assert isinstance(self.geom_id, int)
        self.friction = self.model.geom_friction[self.geom_id]
        if self.color is not None:
            self.set_color(self.color)

    def set_sliding_friction(self, sliding_friction: float):
        assert self.physics is not None
        assert self.friction is not None
        self.friction[0] = sliding_friction
        self.model.geom_friction[self.geom_id] = self.friction

    def set_color(self, color: npt.NDArray[np.float64]):
        assert self.physics is not None
        assert self.geom_id is not None
        self.model.geom_rgba[self.geom_id] = color.copy()
        self.color = color
