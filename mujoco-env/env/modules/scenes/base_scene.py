from typing import cast

from dm_control.mujoco.wrapper.mjbindings import enums
import numpy as np
import numpy.typing as npt
from dm_control import mjcf
from dm_control.mjcf.physics import Physics
from dm_control.mujoco import wrapper
from mujoco import MjData, MjModel, mjtFrame



class BaseScene:
    def __init__(
        self,
        name: str,
        mjcf_path: str,
        cameras: dict[str, tuple[int, int]] | None,
    ):
        self.name: str = name
        self.mjcf_path: str = mjcf_path
        if cameras is not None:
            self.cameras: dict[str, tuple[int, int]] = cameras
        else:
            self.cameras = {}

        self.physics: Physics | None = None
        self.mjcf_model: mjcf.RootElement = mjcf.from_path(self.mjcf_path)
        self.camera_ids: list[int] = []

    def set_mujoco_ref(self, physics: Physics):
        self.physics = physics
        self.camera_ids = [
            self.model.camera(camera_name).id for camera_name in self.cameras.keys()
        ]
        if "visualization_camera" in self.cameras.keys():
            self.model.vis.global_.offwidth = 1920
            self.model.vis.global_.offheight = 1440

    @property
    def model(self) -> MjModel:
        assert (
            self.physics is not None
        ), "Simulator is not initialized. Please call init_simulator() first."
        return cast(MjModel, self.physics.model.ptr)

    @property
    def data(self) -> MjData:
        assert (
            self.physics is not None
        ), "Simulator is not initialized. Please call init_simulator() first."
        return cast(MjData, self.physics.data.ptr)

    @property
    def camera_images(self) -> dict[str, npt.NDArray[np.uint8]]:
        assert self.physics is not None
        # assert len(self.camera_ids) > 0, "No camera ids set for robot"
        scene_option = wrapper.MjvOption()
        scene_option.frame = mjtFrame.mjFRAME_SITE.value
        scene_option.frame = enums.mjtFrame.mjFRAME_NONE
        images: dict[str, npt.NDArray[np.uint8]] = {}
        for camera_id, camera_name in zip(self.camera_ids, self.cameras.keys()):
            raw_image = self.physics.render(
                width=self.cameras[camera_name][0],
                height=self.cameras[camera_name][1],
                camera_id=camera_id,
                scene_option=scene_option,
            )
            assert raw_image.dtype == np.uint8
            image = cast(npt.NDArray[np.uint8], raw_image)
            images[camera_name] = image
        return images
