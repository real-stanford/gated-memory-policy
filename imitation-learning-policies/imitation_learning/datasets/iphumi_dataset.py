import numpy as np
import numpy.typing as npt
from typing import Any, cast

import zarr

from imitation_learning.datasets.multi_traj_dataset import MultiTrajDataset

from imitation_learning.utils.imagecodecs_numcodecs import register_codecs
from imitation_learning.datasets.aggregated_dataset import AggregatedDataset
register_codecs()
from imitation_learning.utils.pose_repr_util import convert_pose_mat_rep
from imitation_learning.utils.pose_util import mat_to_pose10d, pose_to_mat

def _process_source_data(
    self: AggregatedDataset, data_dict: dict[str, npt.NDArray[Any]]
) -> dict[str, npt.NDArray[Any]]:
    """
    Will calculate the following data:
        relative poses
        poses wrt episode start
    This step does not include normalization and data augmentation
    Input data_dict:
        camera0_main_rgb: (..., H, W, 3) uint8
        camera0_ultrawide_rgb: (..., H, W, 3) uint8
        robot0_demo_start_pose: (1, 6) float64 (optional)
        robot0_eef_pos: (..., 3) float32
        robot0_eef_rot_axis_angle: (..., 3) float32
        robot0_gripper_width: (..., 1) float32

    Output data_dict:
        robot0_10d: (..., 10),
        robot0_main_camera: (..., 3, image_size, image_size),
        robot0_ultrawide_camera: (..., 3, image_size, image_size),
        robot0_depth_camera: (..., 1, image_size, image_size),
        robot0_eef_rot_axis_angle_wrt_start: (..., 3),
        action0_10d: (..., 10),
        traj_idx: (1),
        episode_idx: (1),

    """

    processed_data_dict: dict[str, npt.NDArray[Any]] = {}


    for i in range(self.robot_num):


        if f"camera{i}_main_rgb" in data_dict and f"robot{i}_main_camera" in self.output_data_meta:
            processed_data_dict[f"robot{i}_main_camera"] = data_dict[
                f"camera{i}_main_rgb"
            ]

        if f"camera{i}_ultrawide_rgb" in data_dict and f"robot{i}_ultrawide_camera" in self.output_data_meta:
            processed_data_dict[f"robot{i}_ultrawide_camera"] = data_dict[
                f"camera{i}_ultrawide_rgb"
            ]

        if f"camera{i}_depth" in data_dict and f"robot{i}_depth_camera" in self.output_data_meta:
            processed_data_dict[f"robot{i}_depth_camera"] = data_dict[
                f"camera{i}_depth"
            ]

        if f"camera{i}_rgb" in data_dict and f"robot{i}_main_camera" in self.output_data_meta:
            processed_data_dict[f"robot{i}_main_camera"] = data_dict[
                f"camera{i}_rgb"
            ]

        if f"robot{i}_eef_pos" in data_dict and (f"robot{i}_10d" in self.output_data_meta or f"action{i}_10d" in self.output_data_meta):

            pose_mat = pose_to_mat(
                np.concatenate(
                    [
                        data_dict[f"robot{i}_eef_pos"],
                        data_dict[f"robot{i}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )
            )

            pose_indices = self.source_data_meta["robot0_eef_pos"].include_indices
            assert pose_indices == self.source_data_meta[
                "robot0_eef_rot_axis_angle"
            ].include_indices, f"robot0_eef_pos and robot0_eef_rot_axis_angle must be aligned"

            gripper_width_indices = self.source_data_meta["robot0_gripper_width"].include_indices

            if self.use_relative_gripper_width:
                assert len(pose_indices) == len(gripper_width_indices), f"{len(pose_indices)=}, {len(gripper_width_indices)=}"
            else:
                assert len(pose_indices) == len(gripper_width_indices) + 1, f"{len(pose_indices)=}, {len(gripper_width_indices)=}"

            zero_idx = pose_indices.index(0)
            assert len(pose_indices) - zero_idx == self.output_data_meta[f"action{i}_10d"].length + 1, f"{len(pose_indices)=}, {zero_idx=}, {self.output_data_meta[f'robot{i}_10d'].length=}"

            rel_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=pose_mat[zero_idx],
                pose_rep="relative",
                backward=False,
            )
            pose = mat_to_pose10d(rel_pose_mat)

            if self.use_relative_gripper_width:
                gripper_zero_idx = gripper_width_indices.index(0)
                assert gripper_zero_idx == zero_idx, f"{gripper_zero_idx=}, {zero_idx=}"
                processed_gripper_width = cast(npt.NDArray[np.float32], data_dict[f"robot{i}_gripper_width"] - data_dict[f"robot{i}_gripper_width"][
                    gripper_zero_idx
                ])
            else:
                processed_gripper_width = data_dict[f"robot{i}_gripper_width"]

            if f"robot{i}_10d" in self.output_data_meta:
                assert zero_idx == self.output_data_meta[f"robot{i}_10d"].length, f"{pose_indices=}, {zero_idx=}, {self.output_data_meta[f'robot{i}_10d'].length=}"
                robot_state_meta = self.output_data_meta[f"robot{i}_10d"]
                robot_state_10d = np.zeros((robot_state_meta.length, *robot_state_meta.shape), dtype=np.float32)

                robot_state_10d[:, :9] = pose[:robot_state_meta.length]
                robot_state_10d[:, 9:] = processed_gripper_width[:robot_state_meta.length]

                processed_data_dict[f"robot{i}_10d"] = robot_state_10d


            if f"action{i}_10d" in self.output_data_meta:
                action_meta = self.output_data_meta[f"action{i}_10d"]
                action_10d = np.zeros((action_meta.length, *action_meta.shape), dtype=np.float32)
                action_10d[:, :9] = pose[-action_meta.length:]
                action_10d[:, 9:] = processed_gripper_width[-action_meta.length:]
                processed_data_dict[f"action{i}_10d"] = action_10d

            if f"robot{i}_demo_start_pose" in data_dict and f"robot{i}_eef_rot_axis_angle_wrt_start" in self.output_data_meta:
                # Calculate relative poses wrt episode start
                # Only used in tasks when initial pose matters, for example the cup arrangement task which requires rotate the handle to the right

                try:
                    wrt_start_entry_meta = self.output_data_meta[
                        f"robot{i}_eef_rot_axis_angle_wrt_start"
                    ]
                    assert (
                        data_dict[f"robot{i}_demo_start_pose"].shape[0] == 1
                    ), "robot0_demo_start_pose must be (1, 6)"
                    # add noise to episode start pose. Copied from the original UMI codebase.
                    start_pose: npt.NDArray[np.float64] = data_dict[
                        f"robot{i}_demo_start_pose"
                    ][0]
                    start_pose += np.random.default_rng(self.seed).normal(
                        scale=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                        size=start_pose.shape,
                    )
                    start_pose_mat = pose_to_mat(start_pose)
                    rel_pose_mat = convert_pose_mat_rep(
                        pose_mat,
                        base_pose_mat=start_pose_mat,
                        pose_rep="relative",
                        backward=False,
                    )
                    rel_obs_pose = mat_to_pose10d(rel_pose_mat)

                    # Only keep the first wrt_start_entry_meta.length frames
                    processed_data_dict[f"robot{i}_eef_rot_axis_angle_wrt_start"] = (
                        rel_obs_pose[: wrt_start_entry_meta.length, 3:]
                    )

                except ValueError:
                    # No wrt_start_entry_meta, so no relative poses wrt episode start
                    # print(f"No wrt_start_entry_meta for robot{i}")
                    pass

    # for key, val in processed_data_dict.items():
    #     if val.size < 1000:
    #         print(f"rank: {os.environ['RANK']} {key}: {val}")
    return processed_data_dict


class iPhUMISingleTrajDataset(AggregatedDataset):
    def __init__(self, align_to_ultrawide: bool, **kwargs):
        """
        If align_to_ultrawide is True, the dataset will be aligned to the ultrawide camera. (will only use 1/6 of the data, but will guarantee the main camera is aligned to the ultrawide camera)
        """
        self.align_to_ultrawide: bool = align_to_ultrawide
        
        super().__init__(**kwargs)
        assert self.use_relative_pose, "iPhUMI dataset must use relative pose"
        if align_to_ultrawide:
            assert "camera0_ultrawide_rgb" in kwargs["source_data_meta"], f"camera0_ultrawide_rgb must be in source_data_meta, but got {kwargs['source_data_meta'].keys()}"
            for index in kwargs["source_data_meta"]["camera0_ultrawide_rgb"].include_indices:
                assert index % 6 == 0, f"When align_to_ultrawide is True, camera0_ultrawide_rgb indices should be a multiple of 6. {index=}"

    def _create_index_pool(self):
        super()._create_index_pool()
        if self.align_to_ultrawide:
            new_index_pool: list[tuple[int, int]] = []

            ultrawide_indices: npt.NDArray[np.int32] = np.array(self.zarr_store["meta"]["upsample_index_camera0_ultrawide_rgb"], dtype=np.int32)
            regular_indices: npt.NDArray[np.int32] = np.array(self.zarr_store["meta"]["downsample_index_camera0_ultrawide_rgb"], dtype=np.int32)
            # Only use the data that is aligned to the ultrawide camera
            for episode_idx, traj_idx in self.index_pool:
                start_idx = self.episode_starts[episode_idx]
                global_idx = start_idx + traj_idx
                if regular_indices[ultrawide_indices[global_idx]] == global_idx:
                    new_index_pool.append((episode_idx, traj_idx))

            print(f"Aligning to ultrawide camera, index pool size changed from {len(self.index_pool)} to {len(new_index_pool)}")
            self.index_pool = new_index_pool

iPhUMISingleTrajDataset._process_source_data = _process_source_data

class iPhUMIMultiTrajDataset(MultiTrajDataset, AggregatedDataset):
    def __init__(self, align_to_ultrawide: bool, **kwargs):
        """
        If align_to_ultrawide is True, the dataset will be aligned to the ultrawide camera. (will only use 1/6 of the data, but will guarantee the main camera is aligned to the ultrawide camera)
        """

        self.align_to_ultrawide: bool = align_to_ultrawide
        self.ultrawide_interval_min: int = kwargs["traj_interval_min"] * kwargs["down_sample_steps"] // 6
        self.ultrawide_interval_max: int = kwargs["traj_interval_max"] * kwargs["down_sample_steps"] // 6
        
        super().__init__(**kwargs)
        if align_to_ultrawide:
            assert "camera0_ultrawide_rgb" in kwargs["source_data_meta"], f"camera0_ultrawide_rgb must be in source_data_meta, but got {kwargs['source_data_meta'].keys()}"
            for index in kwargs["source_data_meta"]["camera0_ultrawide_rgb"].include_indices:
                assert index % 6 == 0, f"When align_to_ultrawide is True, camera0_ultrawide_rgb indices should be a multiple of 6. {index=}"

    def init_overall_index_pool(self):
        """
        Override it if self.align_to_ultrawide is True.
        """
        if self.align_to_ultrawide:
            self.overall_index_pool = {} 
            
            ultrawide_indices = self.zarr_store["meta"]["upsample_index_camera0_ultrawide_rgb"]
            # This maps main camera indices (0->60N) to ultrawide camera indices (0->10N)
            regular_indices = self.zarr_store["meta"]["downsample_index_camera0_ultrawide_rgb"]
            # This maps ultrawide camera indices (0->10N) to main camera indices (0->60N)

            for episode_idx in self.used_episode_indices:
                ultrawide_length = self.episode_frame_nums[episode_idx] // 6 # ultrawide only has 10Hz frequency while others have 60Hz
                middle_start_idx = ultrawide_length - (self.traj_num - 1) * self.ultrawide_interval_min 
                # print(f"{middle_start_idx=}, {self.traj_num=}, {self.ultrawide_interval_min=}, {self.ultrawide_interval_max=}, {ultrawide_length=}")
                middle_start_idx = max(middle_start_idx, self.traj_interval_max)
                episode_starting_idx_max = int(ultrawide_length * self.starting_percentile_max)
                ultrawide_local_indices = list(range(max(min(episode_starting_idx_max, middle_start_idx), self.ultrawide_interval_max)))

                # local: index if only use ultrawide camera
                # global: index for all cameras
                
                local_indices = []
                for ultrawide_local_index in ultrawide_local_indices:
                    ultrawide_episode_start_idx = ultrawide_indices[self.episode_starts[episode_idx]]
                    ultrawide_global_idx = ultrawide_episode_start_idx + ultrawide_local_index
                    regular_global_idx = regular_indices[ultrawide_global_idx]
                    regular_local_idx = regular_global_idx - self.episode_starts[episode_idx]
                    local_indices.append(regular_local_idx)

                self.overall_index_pool[episode_idx] = local_indices

        else:
            super().init_overall_index_pool()

    

    def _check_ultrawide_alignment(self, episode_idx: int, regular_local_idx: int):
        ultrawide_indices = self.zarr_store["meta"]["upsample_index_camera0_ultrawide_rgb"]
        regular_indices = self.zarr_store["meta"]["downsample_index_camera0_ultrawide_rgb"]

        regular_global_idx = self.episode_starts[episode_idx] + regular_local_idx
        ultrawide_global_idx = int(ultrawide_indices[regular_global_idx])

        # print(f"{regular_global_idx=} {regular_indices[ultrawide_global_idx]=}")
        
        assert regular_global_idx == regular_indices[ultrawide_global_idx], \
            f"regular_global_idx {regular_global_idx} != regular_indices {regular_indices[ultrawide_global_idx]}"

    def _sample_multi_traj_indices(self, episode_idx: int, start_idx: int):
        if self.align_to_ultrawide:
            ultrawide_indices = self.zarr_store["meta"]["upsample_index_camera0_ultrawide_rgb"]
            assert isinstance(ultrawide_indices, zarr.Array)
            regular_indices = self.zarr_store["meta"]["downsample_index_camera0_ultrawide_rgb"]

            ultrawide_episode_start_idx = int(ultrawide_indices[self.episode_starts[episode_idx]])

            indices: list[int] = []
            next_regular_local_idx = start_idx
            indices.append(next_regular_local_idx)

            next_regular_global_idx = self.episode_starts[episode_idx] + next_regular_local_idx
            next_ultrawide_global_idx = int(ultrawide_indices[next_regular_global_idx])
            next_ultrawide_local_idx = int(next_ultrawide_global_idx - ultrawide_episode_start_idx)
            

            self._check_ultrawide_alignment(episode_idx, next_regular_local_idx)

            for i in range(self.traj_num - 1):
                next_ultrawide_local_idx = next_ultrawide_local_idx + int(self.rng.integers(
                    low=self.ultrawide_interval_min, high=self.ultrawide_interval_max + 1
                ))
                next_ultrawide_global_idx = next_ultrawide_local_idx + ultrawide_episode_start_idx
                if next_ultrawide_global_idx < len(regular_indices):
                    next_regular_global_idx = int(regular_indices[next_ultrawide_global_idx])
                    next_regular_local_idx = next_regular_global_idx - self.episode_starts[episode_idx]

                    self._check_ultrawide_alignment(episode_idx, next_regular_local_idx)
                else:
                    # Traj is padding, so we don't need to check the alignment
                    next_regular_local_idx = self.episode_frame_nums[episode_idx] + 1 # This is out of range, so "traj_is_padding" will be True

                indices.append(next_regular_local_idx)
            return indices

        else:
            return super()._sample_multi_traj_indices(episode_idx, start_idx)

    def resample_index_pool(self):
        """
        Multi-traj dataset should align to ultrawide in this stage, instead of _create_index_pool (will be called before _sample_multi_traj_indices)
        """
        super().resample_index_pool()
        if self.align_to_ultrawide:
            new_index_pool: list[tuple[int, int]] = []
            ultrawide_indices: npt.NDArray[np.int32] = np.array(self.zarr_store["meta"]["upsample_index_camera0_ultrawide_rgb"], dtype=np.int32)
            regular_indices: npt.NDArray[np.int32] = np.array(self.zarr_store["meta"]["downsample_index_camera0_ultrawide_rgb"], dtype=np.int32)
            # Only use the data that is aligned to the ultrawide camera
            for episode_idx, traj_idx in self.index_pool:
                start_idx = self.episode_starts[episode_idx]
                global_idx = start_idx + traj_idx
                if regular_indices[ultrawide_indices[global_idx]] == global_idx:
                    new_index_pool.append((episode_idx, traj_idx))

            print(f"Aligning to ultrawide camera, index pool size changed from {len(self.index_pool)} to {len(new_index_pool)}")
            self.index_pool = new_index_pool

iPhUMIMultiTrajDataset._process_source_data = _process_source_data

# class UMIDataset(AggregatedDataset):
#     pass

# UMIDataset._process_source_data = _process_source_data