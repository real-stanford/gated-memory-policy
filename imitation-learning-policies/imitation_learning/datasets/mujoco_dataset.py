from typing import Any

import numpy as np
import numpy.typing as npt

from imitation_learning.datasets.base_dataset import BaseDataset
from imitation_learning.datasets.episodic_dataset import EpisodicDataset
from imitation_learning.datasets.multi_traj_dataset import MultiTrajDataset
from robot_utils.pose_utils import (get_relative_pose, quat_wxyz_to_rot_6d)


def _process_source_data(
    self: BaseDataset, data_dict: dict[str, npt.NDArray[Any]]
) -> dict[str, npt.NDArray[Any]]:
    for entry_meta in self.source_data_meta.values():
        if entry_meta.name in data_dict.keys() and entry_meta.name.endswith(
            "xyz_wxyz"
        ):  # You can also write relative pose transforms in other formats
            raw_poses = data_dict[entry_meta.name]
            if self.use_relative_pose:
                relative_poses = raw_poses.copy()
                if entry_meta.name.startswith("action"):
                    reference_pose_idx_name = 0  # Use the last action (corresponding to relative index -1) as the reference frame
                else:
                    reference_pose_idx_name = (
                        -1
                    )  # Use the current observation (corresponding to relative index 0) as the reference frame
                reference_pose_idx = entry_meta.include_indices.index(
                    reference_pose_idx_name
                )  # Get the index of the reference pose
                for i in range(len(relative_poses)):
                    relative_poses[i] = get_relative_pose(
                        raw_poses[i], raw_poses[reference_pose_idx]
                    )
                if entry_meta.name.startswith("action"):
                    data_dict[entry_meta.name] = relative_poses[1:]
                else:
                    data_dict[entry_meta.name] = relative_poses[:-1]

        if entry_meta.name in data_dict.keys() and entry_meta.name.endswith(
            "wrist_camera"
        ):
            future_entry_name = entry_meta.name.replace("robot", "future")
            if future_entry_name in self.output_data_meta.keys():
                future_entry_length = self.output_data_meta[future_entry_name].length
                obs_entry_length = self.output_data_meta[entry_meta.name].length
                assert (
                    data_dict[entry_meta.name].shape[0]
                    == future_entry_length + obs_entry_length
                ), f"Future entry length {future_entry_length} + obs entry length {obs_entry_length} != {data_dict[future_entry_name].shape[0]}"
                data_dict[future_entry_name] = data_dict[entry_meta.name][
                    obs_entry_length:, :, :, :
                ]  # Predict the last few frames
                data_dict[entry_meta.name] = data_dict[entry_meta.name][
                    :obs_entry_length, :, :, :
                ]  # Use the first few frames as the input

    # Merge tcp and gripper:
    for name, entry_meta in self.output_data_meta.items():
        if entry_meta.name.endswith("10d"):
            length = entry_meta.length
            prefix = name.split("10d")[0]
            pose = data_dict[prefix + "tcp_xyz_wxyz"][-length:]
            rot_6d = np.zeros((length, 6))
            for i in range(len(pose)):
                rot_6d[i] = quat_wxyz_to_rot_6d(pose[i][3:])
            gripper = data_dict[prefix + "gripper_width"][-length:]
            # print(f"pose: {pose.shape}, rot_6d: {rot_6d.shape}, gripper: {gripper.shape}")
            data_dict[name] = np.concatenate(
                [pose[:, :3], rot_6d, gripper], axis=-1
            )  # [traj_len, 3], [traj_len, 6], [traj_len, 1] -> [traj_len, 10]

    if "action0_is_error" in data_dict.keys():
        data_dict["action_is_error"] = data_dict.pop("action0_is_error")
    if "action0_is_critical" in data_dict.keys():
        data_dict["action_is_critical"] = data_dict.pop("action0_is_critical") 
        # If there are multiple robots, this flag should be a union of all robots' flags

    return data_dict


class MujocoMultiTrajDataset(MultiTrajDataset, EpisodicDataset):
    pass


MujocoMultiTrajDataset._process_source_data = _process_source_data


class MujocoSingleTrajDataset(EpisodicDataset):
    pass


MujocoSingleTrajDataset._process_source_data = _process_source_data

