from typing import TYPE_CHECKING

from mujoco import mj_name2id, mjtObj

from env.modules.objects.base_object import BaseObject

if TYPE_CHECKING:
    from dm_control.mjcf.physics import Physics

from loguru import logger


class BaseDeformableObject(BaseObject):
    def __init__(self, body_name: str, qadr_start: int, **kwargs):
        super().__init__(**kwargs)
        self.body_name: str = body_name
        self.vertex_ids = []
        self.qadr_start: int = qadr_start
        self.vertex_adr: int
        self.vertex_num: int

    def set_mujoco_ref(self, physics: "Physics"):
        self.physics = physics

        flex_obj_id = mj_name2id(
            self.model,
            int(mjtObj.mjOBJ_FLEX),
            f"{self.name}/{self.body_name}",
        )

        self.vertex_num = self.model.flex_vertnum[flex_obj_id]
        self.vertex_adr = self.model.flex_vertadr[flex_obj_id]
        self.vertex_ids = self.model.flex_vertbodyid[
            self.vertex_adr : self.vertex_adr + self.vertex_num
        ].copy()

    def get_vertex_positions(self):
        return self.data.flexvert_xpos[
            self.vertex_adr : self.vertex_adr + self.vertex_num
        ].copy()

    def reset_grid_pos(self):
        logger.info(f"Resetting grid pos")
        self.data.qpos[self.qadr_start : self.qadr_start + self.vertex_num * 3] = 0
        self.data.qvel[self.qadr_start : self.qadr_start + self.vertex_num * 3] = 0
