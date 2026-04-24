from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
import tqdm
import zarr

from imitation_learning.common.datatypes import batch_type
from imitation_learning.datasets.base_dataset import BaseDataset
from robot_utils.torch_utils import aggregate_batch


class EpisodicDataset(BaseDataset):
    """
    Dataset that loads data from a zarr store in episode-wise format.
    The zarr dataset should be in the following format:
    - data.zarr
        - .zgroup
        - .zattrs # For normalizers
        - episode_0
            - .zattrs
            - .zgroup
            - robot0_tcp_xyz_wxyz
                - .zarray
                - 0.0
                - ...
            - robot0_gripper_width
            - robot0_camera_images
            - ...
        - episode_1
        - ...
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.store_episode_num: int = len(self.zarr_store)

        self._update_episode_indices()

        self._check_data_validity()
        self._create_index_pool()

    def _check_data_validity(self):

        store_episode_frame_nums = self.zarr_store.attrs.get("episode_frame_nums")
        if store_episode_frame_nums is not None and isinstance(
            store_episode_frame_nums, dict
        ):
            self.episode_frame_nums: dict[int, int] = {}
            for episode_idx in self.used_episode_indices:
                if str(episode_idx) not in store_episode_frame_nums:
                    break
                self.episode_frame_nums[episode_idx] = store_episode_frame_nums[
                    str(episode_idx)
                ]
                self.episode_valid_indices_min[episode_idx] = (
                    self.max_history_length - self.history_padding_length
                )
                self.episode_valid_indices_max[episode_idx] = (
                    store_episode_frame_nums[str(episode_idx)]
                    + self.future_padding_length
                    - self.max_future_length
                )
            else:

                print(
                    "Data completeness is already checked for all used episodes. Skipping data completeness check."
                )
                self.avg_frame_num = float(np.mean(list(self.episode_frame_nums.values())))
                self.std_frame_num = float(np.std(list(self.episode_frame_nums.values())))
                self.max_frame_num = int(np.max(list(self.episode_frame_nums.values())))
                self.min_frame_num = int(np.min(list(self.episode_frame_nums.values())))
                print(f"Frame num stats: avg: {self.avg_frame_num: .1f}, std: {self.std_frame_num: .1f}, max: {self.max_frame_num}, min: {self.min_frame_num}")
                
                return

        print("Checking data completeness...")
        for episode_idx in tqdm.tqdm(
            self.used_episode_indices
        ):  # Check data completeness
            if f"episode_{episode_idx}" not in self.zarr_store:
                raise ValueError(
                    f"Episode {episode_idx} not found in zarr store {self.zarr_path}. Please check the zarr store format."
                )
            episode_group = self.zarr_store[f"episode_{episode_idx}"]
            first_entry_name = list(self.source_data_meta.keys())[0]
            episode_frame_num = len(episode_group[first_entry_name])
            self.episode_frame_nums[episode_idx] = episode_frame_num
            self.episode_valid_indices_min[episode_idx] = (
                self.max_history_length - self.history_padding_length
            )
            self.episode_valid_indices_max[episode_idx] = (
                episode_frame_num + self.future_padding_length - self.max_future_length
            )

            for entry_meta in self.source_data_meta.values():
                source_name = entry_meta.name
                name = entry_meta.name
                if source_name not in episode_group:
                    raise ValueError(
                        f"Entry {source_name} not found in episode {episode_idx} of zarr store {self.zarr_path}."
                    )
                if not isinstance(episode_group[source_name], zarr.Array):
                    raise ValueError(
                        f"Entry {source_name} in zarr store {self.zarr_path}/episode_{episode_idx} is not an array."
                    )
                if episode_group[source_name][0].shape != entry_meta.shape:
                    raise ValueError(
                        f"Source shape of entry {source_name} in zarr store {self.zarr_path}/episode_{episode_idx}: {episode_group[source_name][0].shape} does not match shape in data_meta: {entry_meta.shape}."
                    )
                if isinstance(episode_group[source_name][0], np.ndarray):
                    data_slice = episode_group[source_name][0]
                    if data_slice.shape != entry_meta.shape:
                        raise ValueError(
                            f"Shape of entry {name} in zarr store {self.zarr_path}/episode_{episode_idx}: {data_slice.shape} does not match shape in data_meta: {entry_meta.shape}."
                        )
                if len(episode_group[source_name]) != episode_frame_num:
                    raise ValueError(
                        f"Length of entry {source_name} in zarr store {self.zarr_path}/episode_{episode_idx}: {len(episode_group[source_name])} does not match episode frame number {episode_frame_num} of {first_entry_name}."
                    )

        temp_store = zarr.open(self.zarr_path, mode="a")
        if "episode_frame_nums" not in temp_store.attrs:
            temp_store.attrs["episode_frame_nums"] = {
                str(episode_idx): self.episode_frame_nums[episode_idx]
                for episode_idx in self.used_episode_indices
            }
        else:
            store_episode_frame_nums = temp_store.attrs["episode_frame_nums"]
            store_episode_frame_nums.update(
                {
                    str(episode_idx): self.episode_frame_nums[episode_idx]
                    for episode_idx in self.used_episode_indices
                }
            )
            temp_store.attrs["episode_frame_nums"] = store_episode_frame_nums
        print("Data completeness is checked.")
        
        self.avg_frame_num = float(np.mean(list(self.episode_frame_nums.values())))
        self.std_frame_num = float(np.std(list(self.episode_frame_nums.values())))
        self.max_frame_num = int(np.max(list(self.episode_frame_nums.values())))
        self.min_frame_num = int(np.min(list(self.episode_frame_nums.values())))
        print(f"Frame num stats: avg: {self.avg_frame_num: .1f}, std: {self.std_frame_num: .1f}, max: {self.max_frame_num}, min: {self.min_frame_num}")
        

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

        source_data_dict: dict[str, Any] = {}
        episode_store = cast(zarr.Group, self.zarr_store[f"episode_{episode_idx}"])
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
                continue
            indices = [traj_idx + i for i in entry_meta.include_indices]
            if entry_meta.rand_idx_offset_max > 0:
                indices = [i + self.rng.integers(-entry_meta.rand_idx_offset_max, entry_meta.rand_idx_offset_max + 1) for i in indices]
            # Crop the indices to the valid range. Will introduce padding if the indices are out of range.
            indices = [
                (0 if i < 0 else episode_length - 1 if i >= episode_length else i)
                for i in indices
            ]
            source_data_dict[entry_meta.name] = episode_store[entry_meta.name][indices]

        processed_data_dict = self._process_source_data(source_data_dict)

        torch_data_dict: dict[str, torch.Tensor] = {}
        torch_data_dict["episode_idx"] = torch.tensor([episode_idx])
        torch_data_dict["traj_idx"] = torch.tensor([traj_idx])
        for entry_meta in self.output_data_meta.values():
            if (
                output_entry_names is not None
                and len(output_entry_names) > 0
                and entry_meta.name not in output_entry_names
            ):
                continue
            processed_data = processed_data_dict[entry_meta.name]
            if isinstance(processed_data, np.ndarray):
                if entry_meta.data_type == "image":
                    processed_data = self.process_image_data(
                        processed_data
                    )  # -> (..., C, H, W), float32 (0~1)
                processed_data = torch.from_numpy(processed_data.astype(np.float32))


            torch_data_dict[entry_meta.name] = processed_data

        """
        torch_data_dict (example):
            "robot0_10d": (length, 10),
            "robot0_wrist_camera": (length, 3, image_size, image_size),
            "third_person_camera": (length, 3, image_size, image_size),
            "action0_10d": (length, 10),
            "traj_idx": (1),
            "episode_idx": (1),
        """
        return torch_data_dict

    def __getitem__(self, idx: int):
        episode_idx, traj_idx = self.index_pool[idx]

        torch_data_dict = self._get_single_traj_data(episode_idx, traj_idx)

        if self.normalizer is not None:
            torch_data_dict = self.normalizer.normalize(torch_data_dict)

        torch_data_dict = self.transforms.apply(torch_data_dict)

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
    ) -> dict[str, torch.Tensor]:


        if sample_num == -1:
            sample_num = len(self.index_pool)

        if sampled_indices is None:
            # sampled_indices = np.random.default_rng(self.seed).choice(
            sampled_indices = self.rng.choice(
                len(self.index_pool), min(sample_num, len(self.index_pool)), replace=False
            )
        else:
            assert len(sampled_indices) == sample_num, f"sampled_indices should be of length {sample_num}, but got {len(sampled_indices)}"

        trajs: list[batch_type] = []
        print(f"Sampling {sample_num} data from {len(self.index_pool)} trajectories.")
        for idx in tqdm.tqdm(sampled_indices):
            episode_idx, start_idx = self.index_pool[idx]

            traj = self._get_single_traj_data(episode_idx, start_idx, output_entry_names)
            trajs.append(traj)

        sampled_data_dict: batch_type = aggregate_batch(trajs, aggregate_fn=torch.stack)

        return sampled_data_dict









