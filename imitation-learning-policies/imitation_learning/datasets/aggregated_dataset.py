from typing import Any
import numpy as np
import numpy.typing as npt

import torch
import tqdm
import zarr
from imitation_learning.common.datatypes import batch_type
from imitation_learning.datasets.base_dataset import BaseDataset

from robot_utils.torch_utils import aggregate_batch


class AggregatedDataset(BaseDataset):
    """
    Dataset loader for the iPhUMI dataset.
    Example structure:
    /
    ├── data
    │   ├── camera0_main_rgb (N, image_size, image_size, 3) uint8
    │   ├── camera0_ultrawide_rgb (N/6, image_size, image_size, 3) uint8
    │   ├── robot0_demo_end_pose (N, 6) float64
    │   ├── robot0_demo_start_pose (N, 6) float64
    │   ├── robot0_eef_pos (N, 3) float32
    │   ├── robot0_eef_rot_axis_angle (N, 3) float32
    │   └── robot0_gripper_width (N, 1) float32
    └── meta
        └── episode_ends (K,) int64

    """

    def __init__(
        self,
        **kwargs,
    ):

        super().__init__(**kwargs)

        data_store = self.zarr_store["data"]
        assert isinstance(data_store, zarr.Group)

        self.data_store: zarr.Group = data_store
        self.data_store_keys: list[str] = list(data_store.keys())

        self.episode_ends: npt.NDArray[np.int64] = np.array(
            self.zarr_store["meta"]["episode_ends"]
        )
        self.store_episode_num: int = len(self.episode_ends)

        self._update_episode_indices()

        self.episode_starts: npt.NDArray[np.int64] = np.zeros_like(self.episode_ends)

        for i, end in enumerate(self.episode_ends):
            if i == 0:
                self.episode_starts[i] = 0
            else:
                self.episode_starts[i] = self.episode_ends[i - 1]

            self.episode_frame_nums[i] = end - self.episode_starts[i]
            self.episode_valid_indices_min[i] = (
                self.max_history_length - self.history_padding_length
            )
            self.episode_valid_indices_max[i] = (
                self.episode_frame_nums[i]
                + self.future_padding_length
                - self.max_future_length
            )

        
        self.avg_frame_num: float = float(np.mean(list(self.episode_frame_nums.values())))
        self.std_frame_num: float = float(np.std(list(self.episode_frame_nums.values())))
        self.max_frame_num: int = int(np.max(list(self.episode_frame_nums.values())))
        self.min_frame_num: int = int(np.min(list(self.episode_frame_nums.values())))
        print(f"Frame num stats: avg: {self.avg_frame_num: .1f}, std: {self.std_frame_num: .1f}, max: {self.max_frame_num}, min: {self.min_frame_num}")
        
        self._create_index_pool()

        print(
            f"Dataset: {self.name}, store_episode_num: {self.store_episode_num}, include_episode_num: {self.include_episode_num}, used_episode_num: {self.used_episode_num}"
        )

    def _check_data_validity(self):
        # Not implemented yet. Will skip checking for now.
        pass

    def _process_source_data(
        self, data_dict: dict[str, npt.NDArray[Any]]
    ) -> dict[str, npt.NDArray[Any]]:
        # Override this function to process the source data
        raise NotImplementedError("This function should be overridden by the subclass.")

    def _get_single_traj_data(
        self,
        episode_idx: int,
        traj_idx: int,
        output_entry_names: list[str] | None = None,
    ):

        episode_length = self.episode_frame_nums[episode_idx]
        start_idx = self.episode_starts[episode_idx]

        source_data_dict: dict[str, Any] = {}

        if output_entry_names is not None and len(output_entry_names) > 0:
            source_entry_names = [
                self.output_data_meta[target_entry_name].source_entry_names
                for target_entry_name in output_entry_names
            ]
            source_entry_names = [
                item for sublist in source_entry_names for item in sublist
            ]
        else:
            source_entry_names = None

        

        for entry_meta in self.source_data_meta.values():
            if (
                source_entry_names is not None
                and len(source_entry_names) > 0
                and entry_meta.name not in source_entry_names
            ):
                # Skip the entries that are not needed to make data loading faster
                continue

            if entry_meta.name not in self.data_store_keys:
                continue
            indices = [traj_idx + i for i in entry_meta.include_indices]
            if entry_meta.rand_idx_offset_max > 0:
                indices = [i + self.rng.integers(-entry_meta.rand_idx_offset_max, entry_meta.rand_idx_offset_max + 1) for i in indices]
            # Crop the indices to the valid range. Will introduce padding if the indices are out of range.
            indices = [
                (0 if i < 0 else episode_length - 1 if i >= episode_length else i)
                for i in indices
            ]
            global_indices = [start_idx + i for i in indices]
            if entry_meta.name.endswith("ultrawide_rgb"):
                global_indices = self.zarr_store["meta"][f"upsample_index_{entry_meta.name}"][global_indices]

            source_data_dict[entry_meta.name] = np.array(
                self.data_store[entry_meta.name][global_indices]
            )

        processed_data_dict = self._process_source_data(source_data_dict)

        torch_data_dict: dict[str, Any] = {}
        torch_data_dict["episode_idx"] = torch.tensor([episode_idx])
        torch_data_dict["traj_idx"] = torch.tensor([traj_idx])

        for entry_meta in self.output_data_meta.values():

            if (
                output_entry_names is not None
                and len(output_entry_names) > 0
                and entry_meta.name not in output_entry_names
            ):
                # Skip the entries that are not needed to make data loading faster
                continue

            assert entry_meta.name in processed_data_dict, f"entry_meta.name: {entry_meta.name} not in processed_data_dict: {processed_data_dict.keys()}"
            processed_data = processed_data_dict[entry_meta.name]
            if isinstance(processed_data, np.ndarray):
                if entry_meta.data_type == "image":
                    processed_data = self.process_image_data(
                        processed_data
                    )  # -> (..., C, H, W), float32 (0~1)
                processed_data = torch.from_numpy(processed_data.astype(np.float32))
                

            torch_data_dict[entry_meta.name] = processed_data

        return torch_data_dict
        

    def __getitem__(self, idx: int):
        """
        output_data_dict:
            obs:
                camera0_main_rgb: (..., H, W, 3) float32 (0~1)
                camera0_ultrawide_rgb: (..., H, W, 3) float32 (0~1)
                camera0_depth: (..., H, W, 1) float32 (0~1)
                robot0_gripper_width: (..., 1) float32
                robot0_eef_pos: (..., 3) float32
                robot0_eef_rot_axis_angle: (..., 3) float32
        """
        episode_idx, traj_idx = self.index_pool[idx]

        torch_data_dict = self._get_single_traj_data(episode_idx, traj_idx)

        if self.normalizer is not None:
            torch_data_dict = self.normalizer.normalize(torch_data_dict)

        torch_data_dict = self.transforms.apply(torch_data_dict)

        # for key, val in torch_data_dict.items():
        #     if val.numel() < 1000:
        #         print(f"rank: {os.environ['RANK']} {key}: {val}")

        for entry_meta in self.output_data_meta.values():
            assert torch_data_dict[entry_meta.name].shape == (
                entry_meta.length,
                *entry_meta.shape,
            ), f"entry_meta: {entry_meta.name}, torch_data_dict[entry_meta.name].shape: {torch_data_dict[entry_meta.name].shape}, entry_meta.length: {entry_meta.length}, entry_meta.shape: {entry_meta.shape}"
            
        if self.statistics_data is not None:
            torch_data_dict["with_mem_variances"] = self.statistics_data[episode_idx]["with_mem_variances"][traj_idx]
            torch_data_dict["no_mem_variances"] = self.statistics_data[episode_idx]["no_mem_variances"][traj_idx]
            torch_data_dict["with_mem_errors"] = self.statistics_data[episode_idx]["with_mem_errors"][traj_idx]
            torch_data_dict["no_mem_errors"] = self.statistics_data[episode_idx]["no_mem_errors"][traj_idx]


        return torch_data_dict

    def sample_data(
        self,
        output_entry_names: list[str],
        sample_num: int,
        augment_data: bool,
        normalize_data: bool,
        sampled_indices: npt.NDArray[np.int64] | None = None,
    ) -> batch_type:

        if sample_num == -1:
            sample_num = len(self.index_pool)

        if sampled_indices is None:
            sampled_indices = np.random.default_rng(self.seed).choice(
                len(self.index_pool), min(sample_num, len(self.index_pool)), replace=False
            )
        else:
            assert len(sampled_indices) == sample_num, f"sampled_indices should be of length {sample_num}, but got {len(sampled_indices)}"


        samples = []
        print(f"Sampling {sample_num} data from {len(self.index_pool)} trajectories.")
        for idx in tqdm.tqdm(sampled_indices):
            episode_idx, traj_idx = self.index_pool[idx]
            samples.append(self._get_single_traj_data(episode_idx, traj_idx, output_entry_names))

        all_samples_data_dict: batch_type = aggregate_batch(
            samples, aggregate_fn=torch.stack
        )

        return all_samples_data_dict