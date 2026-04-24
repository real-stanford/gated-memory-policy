
from dm_control import mjcf
from dm_control.mjcf.physics import Physics
from mujoco import MjData, MjModel



class BaseObject:
    def __init__(
        self,
        name: str,
        mjcf_path: str,
    ):
        self.name: str = name
        self.mjcf_path: str = mjcf_path
        self.mjcf_model: mjcf.RootElement = mjcf.from_path(self.mjcf_path)
        self.mjcf_model.root.model = name
        self.physics: "Physics | None" = None

    def set_mujoco_ref(self, physics: "Physics"):
        raise NotImplementedError("Should be implemented by subclass")

    @property
    def data(self) -> MjData:
        assert self.physics is not None
        return self.physics.data.ptr

    @property
    def model(self) -> MjModel:
        assert self.physics is not None
        return self.physics.model.ptr
