from mujoco import mj_name2id, mjtObj
from scipy.spatial import ConvexHull

from env.modules.objects.base_deformable_object import BaseDeformableObject


class Cloth(BaseDeformableObject):
    def __init__(
        self,
        horizontal_vertex_num: int,
        vertical_vertex_num: int,
        total_friction_N: float | None,
        **kwargs,
    ):
        """
        args:
            total_friction_N: Will adjust the friction coefficient according to the mass: `friction_coefficient = total_friction_N / 9.81 / total_mass`
        """
        super().__init__(**kwargs)
        self.horizontal_vertex_num: int = horizontal_vertex_num
        self.vertical_vertex_num: int = vertical_vertex_num
        self.total_friction_N: float | None = total_friction_N

    def set_mass(self, mass: float):

        h = self.horizontal_vertex_num
        v = self.vertical_vertex_num
        edge_num = (h - 1) * v + (v - 1) * h + (v - 1) * (h - 1)
        # Using triangluar mesh
        # e.g 3*4+4*3+3*3=33
        # x-x-x-x
        # |/|/|/|
        # x-x-x-x
        # |/|/|/|
        # x-x-x-x
        # |/|/|/|
        # x-x-x-x

        vertex_mass = mass / self.vertex_num
        inv_weight = 1 / vertex_mass

        self.model.flexedge_invweight0[self.vertex_adr : self.vertex_adr + edge_num] = (
            inv_weight
        )
        self.model.body_invweight0[self.vertex_ids, 0] = inv_weight

        self.model.dof_invweight0[
            self.qadr_start : self.qadr_start + self.vertex_num * 3
        ] = inv_weight

        # self.model.body_treeid
        body_id_start = mj_name2id(
            self.model, int(mjtObj.mjOBJ_BODY), f"{self.name}/{self.body_name}_0"
        )

        self.model.body_mass[body_id_start : body_id_start + self.vertex_num] = (
            vertex_mass
        )
        self.model.body_subtreemass[body_id_start : body_id_start + self.vertex_num] = (
            vertex_mass
        )
        mass_diff = mass - self.model.body_subtreemass[body_id_start - 1]
        self.model.body_subtreemass[body_id_start - 1] = mass
        self.model.body_subtreemass[0] += mass_diff

        self.model.dof_M0[self.qadr_start : self.qadr_start + self.vertex_num * 3] = (
            vertex_mass
        )
        # Adjust the friction coefficient according to the mass
        if self.total_friction_N is not None:
            flex_obj_id = mj_name2id(
                self.model,
                int(mjtObj.mjOBJ_FLEX),
                f"{self.name}/{self.body_name}",
            )
            friction_coefficient = self.total_friction_N / mass / 9.81
            self.model.flex_friction[flex_obj_id, 0] = friction_coefficient

        assert self.physics is not None
        self.physics.reset()
        # body_invweight0: (trn, rot) where the rot part for a deformable object is 0

    def get_2d_convex_hull_area(self):

        vertex_positions = self.get_vertex_positions()
        # 2d convex hull
        hull = ConvexHull(vertex_positions[:, :2])
        return hull.volume
