from typing import Any

import numpy as np
import numpy.typing as npt
from transforms3d.quaternions import axangle2quat

from imitation_learning.datasets.base_dataset import BaseDataset
from imitation_learning.datasets.episodic_dataset import EpisodicDataset
from imitation_learning.datasets.multi_traj_dataset import MultiTrajDataset
from robot_utils.pose_utils import quat_wxyz_to_rot_6d


def _process_source_data(
    self: BaseDataset, data_dict: dict[str, npt.NDArray[Any]]
) -> dict[str, npt.NDArray[Any]]:

    # Merge tcp and gripper:
    for name, entry_meta in self.output_data_meta.items():
        if entry_meta.name.endswith("10d") and not entry_meta.name.startswith("action"):
            length = entry_meta.length
            prefix = name.split("10d")[0]
            pos = data_dict[prefix + "eef_pos"][-length:]
            quat = data_dict[prefix + "eef_quat"][-length:]
            gripper = data_dict[prefix + "gripper_qpos"][-length:, :1]
            # Use the first number of the gripper qpos
            rot_6d = np.zeros((length, 6))
            for i in range(length):
                rot_6d[i] = quat_wxyz_to_rot_6d(quat[i])
            data_dict[name] = np.concatenate(
                [pos, rot_6d, gripper], axis=-1
            )  # [traj_len, 3], [traj_len, 6], [traj_len, 1] -> [traj_len, 10]

    if "abs_actions" in data_dict:
        actions = data_dict.pop("abs_actions")
        robot_num = actions.shape[-1] // 7
        for robot_idx in range(robot_num):
            pos = actions[..., robot_idx * 7 : robot_idx * 7 + 3]
            rot_vec = actions[..., robot_idx * 7 + 3 : robot_idx * 7 + 6]
            gripper = actions[..., robot_idx * 7 + 6 : robot_idx * 7 + 7]
            rot_6d = np.zeros((actions.shape[0], 6))
            for i in range(actions.shape[0]):
                theta = np.linalg.norm(rot_vec[i])
                quat = axangle2quat(rot_vec[i], theta)
                rot_6d[i] = quat_wxyz_to_rot_6d(quat)
            data_dict[f"action{robot_idx}_10d"] = np.concatenate(
                [pos, rot_6d, gripper], axis=-1
            )  # [traj_len, 3], [traj_len, 6], [traj_len, 1] -> [traj_len, 10]

    return data_dict


class RobomimicMultiTrajDataset(MultiTrajDataset, EpisodicDataset):
    pass


RobomimicMultiTrajDataset._process_source_data = _process_source_data


class RobomimicSingleTrajDataset(EpisodicDataset):
    pass


RobomimicSingleTrajDataset._process_source_data = _process_source_data
