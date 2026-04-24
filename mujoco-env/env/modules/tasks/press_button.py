from typing import Any, cast

import numpy as np
from transforms3d import quaternions

from env.modules.scenes.table_only import TableOnly
from env.modules.tasks.base_task import BaseTask


class PressButton(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def customized_obs_dict(self) -> dict[str, Any]:
        customized_obs_dict: dict[str, Any] = {}

        robot_pose = self.env.robots[0].tcp_xyz_wxyz
        object_pose = self.env.objects[0].pose_xyz_wxyz
        tcp_relative_pose_to_item = np.zeros(7)
        tcp_relative_pose_to_item[:3] = robot_pose[:3] - object_pose[:3]
        tcp_relative_pose_to_item[3:] = quaternions.qmult(
            quaternions.qinverse(object_pose[3:]), robot_pose[3:]
        )
        customized_obs_dict["tcp_relative_pose_to_item"] = tcp_relative_pose_to_item

        tcp_relative_pose_to_table_center = np.zeros(7)
        scene = cast(TableOnly, self.env.scene)
        tcp_relative_pose_to_table_center[:3] = (
            robot_pose[:3] - scene.get_table_center_xyz()
        )
        tcp_relative_pose_to_table_center[3:] = robot_pose[3:]
        customized_obs_dict["tcp_relative_pose_to_table_center"] = (
            tcp_relative_pose_to_table_center
        )

        return customized_obs_dict
