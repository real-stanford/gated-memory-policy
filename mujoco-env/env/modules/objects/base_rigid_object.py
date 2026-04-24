from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from env.modules.objects.base_object import BaseObject

if TYPE_CHECKING:
    from dm_control.mjcf.physics import Physics


class BaseRigidObject(BaseObject):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.joint_id: int | None = None
        self.qpos_adr: int | None = None
        self.dof_adr: int | None = None

    def set_mujoco_ref(self, physics: "Physics"):

        self.physics = physics
        self.joint_id = self.model.joint(f"{self.name}/").id
        qpos_adr: int = self.model.joint(f"{self.name}/").qposadr[0]
        dof_adr: int = self.model.joint(f"{self.name}/").dofadr[0]

        self.qpos_adr = qpos_adr
        self.dof_adr = dof_adr

    def set_pose_xyz_wxyz(self, pose_xyz_wxyz: npt.NDArray[np.float64]):
        assert self.physics is not None
        assert self.qpos_adr is not None
        assert self.dof_adr is not None
        self.data.qpos[self.qpos_adr : self.qpos_adr + 7] = pose_xyz_wxyz
        self.data.qvel[self.dof_adr : self.dof_adr + 6] = 0
        self.data.qacc[self.dof_adr : self.dof_adr + 6] = 0

    def set_vel_xyz_xyz(self, vel_xyz_xyz: npt.NDArray[np.float64]):
        assert self.physics is not None
        assert self.dof_adr is not None
        self.data.qvel[self.dof_adr : self.dof_adr + 6] = vel_xyz_xyz

    @property
    def pose_xyz_wxyz(self) -> npt.NDArray[np.float64]:
        assert self.physics is not None and self.qpos_adr is not None
        qpos = self.data.qpos[self.qpos_adr : self.qpos_adr + 7].copy()
        return qpos

    @property
    def vel_xyz_xyz(self) -> npt.NDArray[np.float64]:
        """Angular velocity"""
        assert self.physics is not None and self.dof_adr is not None
        qvel = self.data.qvel[self.dof_adr : self.dof_adr + 6].copy()
        return qvel
