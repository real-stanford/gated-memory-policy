import mujoco
import numpy as np
import numpy.typing as npt
from dm_control.mjcf import Physics

from env.modules.scenes.base_scene import BaseScene


class TableBin(BaseScene):
    """
    ------------
    |  1  |  0  |
    ------x-----
    |  2  |  3  |
    ------------
    robot arm here

    x: bin_center_xyz (z is the height of the bin surface and the object should contact the bin surface)
    bin size is for each bin (in the upper plot, the total size is 2 * bin_size_xy)
    when sampling random poses, the pose will not touch the bin borders

    coordinate system:
    ---> y
    |
    v
    x

    """

    def __init__(
        self,
        bin_center_xyz: list[float],
        bin_size_xy: list[float],
        bin_border_width: float,
        bin_border_clearance: float,
        bin_height_tolerance: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bin_center_xyz: npt.NDArray[np.float64] = np.array(bin_center_xyz)
        self.bin_size_xy: npt.NDArray[np.float64] = np.array(bin_size_xy)
        self.bin_border_width: float = bin_border_width
        self.bin_border_clearance: float = bin_border_clearance
        self.bin_height_tolerance: float = bin_height_tolerance
        self.bin_geom_ids: list[int] = []
        self.bin_material_ids: list[int] = []
        self.COLOR_MASKS: dict[str, list[float]] = {
            "transparent": [0, 0, 0, 0],  # 0
            # "can": [1.0, 0.6, 0.6, 1.0],  # 1
            # # "cereal": [0.5, 1.0, 0.5, 1.0],  # 2
            # "floor": [0.5, 1.0, 0.5, 1.0],  # 2
            # "metal": [0.6, 0.6, 1.0, 1.0],  # 3
            # "wood": [1.0, 1.0, 1.0, 1.0],  # 4
            "plaster": [1.0, 1.0, 1.0, 1.0],
            "ambra": [1.0, 1.0, 1.0, 1.0],
            "metal": [1.0, 1.0, 1.0, 1.0],
            "wood": [1.0, 1.0, 1.0, 1.0],
        }

        self.bin_materials: list[int] = [0, 0, 0, 0]  # Transparent by default

    def set_mujoco_ref(self, physics: Physics):
        super().set_mujoco_ref(physics)
        self.bin_geom_ids = [self.model.geom(f"bin_{i}_material").id for i in range(4)]
        self.bin_material_ids = [
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "dark-wood"
            )  # Default material
        ] + [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, f"material_{i}")
            for i in range(1, 5)
        ]

    def set_bin_materials(self, material_ids: list[int]):
        assert len(material_ids) == 4
        for bin_id, material_id in enumerate(material_ids):
            self.model.geom_rgba[self.bin_geom_ids[bin_id]] = list(
                self.COLOR_MASKS.values()
            )[material_id]
            if material_id > 0:
                self.model.geom_matid[self.bin_geom_ids[bin_id]] = (
                    self.bin_material_ids[material_id]
                )
            self.bin_materials[bin_id] = material_id

    def sample_random_pose_xyz_wxyz(
        self,
        bin_id: int,
        object_radius: float,
        object_center_height: float,
        rotation_angle_range: tuple[float, float],
        rng: np.random.Generator | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        The pose will have a fixed z aligned with the bin center and the rotation is restricted to the horizontal plane
        """
        x_idx = -1 if bin_id < 2 else 1
        y_idx = -1 if bin_id in [1, 2] else 1
        x_borders = [
            self.bin_center_xyz[0]
            + x_idx
            * (
                self.bin_size_xy[0]
                - self.bin_border_width / 2
                - object_radius
                - self.bin_border_clearance
            ),
            self.bin_center_xyz[0]
            + x_idx
            * (self.bin_border_width / 2 + object_radius + self.bin_border_clearance),
        ]
        y_borders = [
            self.bin_center_xyz[1]
            + y_idx
            * (
                self.bin_size_xy[1]
                - self.bin_border_width / 2
                - object_radius
                - self.bin_border_clearance
            ),
            self.bin_center_xyz[1]
            + y_idx
            * (self.bin_border_width / 2 + object_radius + self.bin_border_clearance),
        ]

        if rng is None:
            position_x = np.random.uniform(min(x_borders), max(x_borders))
            position_y = np.random.uniform(min(y_borders), max(y_borders))
            angle = np.random.uniform(rotation_angle_range[0], rotation_angle_range[1])
        else:
            position_x = rng.uniform(min(x_borders), max(x_borders))
            position_y = rng.uniform(min(y_borders), max(y_borders))
            angle = rng.uniform(rotation_angle_range[0], rotation_angle_range[1])
        position_z = (
            self.bin_center_xyz[2] + object_center_height + self.bin_height_tolerance
        )

        # Generate a 2d random rotation
        assert (
            len(rotation_angle_range) == 2
            and rotation_angle_range[0] <= rotation_angle_range[1]
        )
        rotation_w = np.cos(angle / 2)
        rotation_x = 0
        rotation_y = 0
        rotation_z = np.sin(angle / 2)

        return np.array(
            [
                position_x,
                position_y,
                position_z,
                rotation_w,
                rotation_x,
                rotation_y,
                rotation_z,
            ]
        )

    def get_bin_id(
        self, object_bottom_xyz: npt.NDArray[np.float64], ignore_height: bool = False
    ) -> int:
        """
        Get the bin id based on the position (0, 1, 2, 3),
        -2 if the position is too low or too high
        -1 if the position is not in any bin or inside the boarder
        """
        if not ignore_height and (
            object_bottom_xyz[2] < self.bin_center_xyz[2] - self.bin_height_tolerance
            or object_bottom_xyz[2] > self.bin_center_xyz[2] + self.bin_height_tolerance
        ):
            return -2
        for bin_id in range(4):
            x_idx = -1 if bin_id < 2 else 1
            y_idx = -1 if bin_id in [1, 2] else 1
            x_borders = [
                self.bin_center_xyz[0]
                + x_idx * (self.bin_size_xy[0] - self.bin_border_width / 2),
                self.bin_center_xyz[0] + x_idx * self.bin_border_width / 2,
            ]
            y_borders = [
                self.bin_center_xyz[1]
                + y_idx * (self.bin_size_xy[1] - self.bin_border_width / 2),
                self.bin_center_xyz[1] + y_idx * self.bin_border_width / 2,
            ]
            if (
                object_bottom_xyz[0] > min(x_borders)
                and object_bottom_xyz[0] < max(x_borders)
                and object_bottom_xyz[1] > min(y_borders)
                and object_bottom_xyz[1] < max(y_borders)
            ):
                return bin_id
        return -1

    def get_bin_center_xyz(self, bin_id: int) -> npt.NDArray[np.float64]:
        assert bin_id in range(4)
        x_idx = -1 if bin_id < 2 else 1
        y_idx = -1 if bin_id in [1, 2] else 1

        return np.array(
            [
                self.bin_center_xyz[0] + x_idx * self.bin_size_xy[0] / 2,
                self.bin_center_xyz[1] + y_idx * self.bin_size_xy[1] / 2,
                self.bin_center_xyz[2],
            ]
        )
