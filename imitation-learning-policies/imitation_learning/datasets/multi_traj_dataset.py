from functools import partial
from typing import Any

import numpy as np
import torch
import tqdm
import numpy.typing as npt

from imitation_learning.common.datatypes import batch_type
from imitation_learning.datasets.base_dataset import BaseDataset
from robot_utils.torch_utils import aggregate_batch


class MultiTrajDataset(BaseDataset):
    """
    Dataset that loads multiple trajectories from a same episode. Can be applied to both episodic and aggregated datasets (e.g. UMI).
    Use multi-inheritance to call it with EpisodicDataset or AggregatedDataset.
    """

    def __init__(
        self,
        traj_num: int,
        traj_interval_min: int,
        traj_interval_max: int,
        split_dataloader_cfg: dict[str, Any] | None = None,
        episode_starting_idx_max: int | None = None,
        **kwargs,
    ):
        traj_interval_min *= kwargs["down_sample_steps"]
        traj_interval_max *= kwargs["down_sample_steps"]
        super().__init__(**kwargs) # Should call the __init__ method from EpisodicDataset or AggregatedDataset. Please manage the MRO sequence properly.

        assert (
            self.starting_percentile_min == 0.0
        ), f"Minimum starting percentile should be 0.0 for multi-trajectory dataset, but got {self.starting_percentile_min}."
        # assert (
        #     self.index_pool_size_per_episode > 0
        # ), "Index pool size per episode must be specified."

        self.traj_num: int = traj_num
        self.traj_interval_min: int = traj_interval_min
        self.traj_interval_max: int = traj_interval_max
        
        self.split_dataloader_cfg: dict[str, Any] | None = split_dataloader_cfg

        if self.traj_interval_min > self.traj_interval_max:
            raise ValueError(
                f"traj_interval_min {self.traj_interval_min} is larger than traj_interval_max {self.traj_interval_max}."
            )

        self.episode_starting_idx_max: int | None = episode_starting_idx_max

        self.overall_index_pool: dict[int, list[int]] = {}
        self.init_overall_index_pool()
        self.resample_index_pool()

        """
        index_pool has self.store_episode_num * self.used_episode_ratio * self.index_pool_size_per_episode items.
        Each item contains a tuple of (episode_idx, indices), where indices is a list of self.traj_num indices, 
        where each index means the 0 index of this trajectory in an episode.
        """

    def init_overall_index_pool(self):
        """
        Initialize the index pool for the overall dataset based on episode length.
        """
        self.overall_index_pool = {}
        for episode_idx in self.used_episode_indices:
            episode_length = self.episode_frame_nums[episode_idx]
            

            if self.episode_starting_idx_max is not None:
                episode_starting_idx_max = self.episode_starting_idx_max
            else:
                middle_start_idx = episode_length - (self.traj_num - 1) * self.traj_interval_min
                episode_starting_idx_max = int(episode_length * self.starting_percentile_max)
                episode_starting_idx_max = max(min(middle_start_idx, episode_starting_idx_max), self.traj_interval_max)
                # Starting index should be
                # 1. At least self.traj_interval_max so that all of the timesteps will be sampled
                # 2. At most episode_length * starting_percentile_max for manual control
                # 3. At most middle_start_idx so that there shouldn't be too many padding trajectories in the end
                
            self.overall_index_pool[episode_idx] = list(range(episode_starting_idx_max))

      
            # middle_start_idx = episode_length - (self.traj_num - 1) * self.traj_interval_min
            # middle_start_idx = max(middle_start_idx, self.traj_interval_max)
            # episode_starting_idx_max = int(episode_length * self.starting_percentile_max)
            # self.overall_index_pool[episode_idx] = list(range(min(episode_starting_idx_max, middle_start_idx)))

    def repeat_dataset(self, repeat_num: float | None = None):
        if repeat_num is not None:
            self.repeat_dataset_num: float = repeat_num
        self.resample_index_pool()
      

    def resample_index_pool(self):
        self.index_pool = []
        # episode_index_size = int(self.index_pool_size_per_episode * self.repeat_dataset_num)
        for episode_idx in self.used_episode_indices:
            # if episode_idx not in self.overall_index_pool:
            #     continue
            # else:gg
            #     print(f"Overall index pool size for episode {episode_idx}: {len(self.overall_index_pool[episode_idx])}")
            if self.index_pool_size_per_episode > 0:
                episode_index_size = int(
                    self.index_pool_size_per_episode * self.repeat_dataset_num * self.episode_frame_nums[episode_idx] / self.avg_frame_num
                )
            elif self.index_pool_size_per_episode == -1:
                episode_index_size = self.episode_frame_nums[episode_idx] * self.repeat_dataset_num
            else:
                raise ValueError(f"index_pool_size_per_episode {self.index_pool_size_per_episode} is invalid. Must be -1 or a positive integer.")

            # Revert to the last version
            start_indices = self.rng.choice(
                self.overall_index_pool[episode_idx],
                size=episode_index_size,
                replace=True,
            )
            # if episode_index_size <= len(self.overall_index_pool[episode_idx]):
            #     start_indices = self.rng.choice(
            #         self.overall_index_pool[episode_idx],
            #         size=episode_index_size,
            #         replace=False,
            #     )
            # else: # If the episode is too short, we need to sample with replacement
            #     start_indices = copy.deepcopy(self.overall_index_pool[episode_idx])
            #     start_indices.extend(self.rng.choice(
            #         start_indices,
            #         size=episode_index_size - len(self.overall_index_pool[episode_idx]),
            #         replace=True,
            #     ))
            self.index_pool.extend(
                [(episode_idx, int(start_idx)) for start_idx in start_indices]
            )

        # assert len(self.index_pool) == episode_index_size * len(
        #     self.used_episode_indices
        # ), f"Index pool size {len(self.index_pool)} does not match the expected size {episode_index_size * len(self.used_episode_indices)}"

    def _create_index_pool(self):
        """Index pool should be created through init_overall_index_pool and resample_index_pool"""
        pass
        

    def _sample_multi_traj_indices(self, episode_idx: int, start_idx: int):
        """
        Given the start index of the first trajectory, sample the indices of the subsequent trajectories.
        """
        indices: list[int] = []
        next_idx = start_idx
        indices.append(next_idx)
        for i in range(self.traj_num - 1):
            next_idx = next_idx + int(self.rng.integers(
                low=self.traj_interval_min, high=self.traj_interval_max + 1
            ))
            indices.append(next_idx)
        return indices

    def split_unused_episodes(
        self,
        remaining_ratio: float = 1.0,
        other_used_episode_indices: list[int] | None = None,
    ):
        unused_dataset = super().split_unused_episodes(
            remaining_ratio, other_used_episode_indices
        )
        unused_dataset.init_overall_index_pool()
        unused_dataset.resample_index_pool()
        if self.split_dataloader_cfg is not None:
            unused_dataset.dataloader_cfg = self.split_dataloader_cfg
        return unused_dataset

    def sample_data(
        self,
        output_entry_names: list[str],
        sample_num: int,
        augment_data: bool,
        normalize_data: bool,
        sampled_indices: npt.NDArray[np.int64] | None = None,
    ) -> batch_type:
        if sampled_indices is None:
            sampled_indices = self.rng.choice(
                len(self.index_pool), min(sample_num, len(self.index_pool)), replace=False
            )
        else:
            assert len(sampled_indices) == sample_num, f"sampled_indices should be of length {sample_num}, but got {len(sampled_indices)}"

        samples = []
        print(f"Sampling {sample_num} data from {len(self.index_pool)} trajectories.")
        for idx in tqdm.tqdm(sampled_indices):
            episode_idx, start_idx = self.index_pool[idx]
            zero_indices = self._sample_multi_traj_indices(episode_idx, start_idx)
            trajs: list[batch_type] = []
            valid_min = self.episode_valid_indices_min[episode_idx]  # Inclusive
            valid_max = self.episode_valid_indices_max[episode_idx]  # Exclusive

            for zero_index in zero_indices:
                traj = self._get_single_traj_data(
                    episode_idx, zero_index, output_entry_names
                )
                trajs.append(traj)

            sample_data_dict = aggregate_batch(trajs, aggregate_fn=torch.stack)

            if normalize_data:
                assert self.normalizer is not None, "Normalizer is not set."
                sample_data_dict = self.normalizer.normalize(sample_data_dict)

            if augment_data:
                sample_data_dict = self.transforms.apply(
                    sample_data_dict, consistent_on_batch=True
                )

            samples.append(sample_data_dict)

        all_samples_data_dict: batch_type = aggregate_batch(
            samples, aggregate_fn=torch.stack
        )

        return all_samples_data_dict

    def __getitem__(self, idx: int):
        episode_idx, start_idx = self.index_pool[idx]
        zero_indices = self._sample_multi_traj_indices(episode_idx, start_idx)
        trajs: list[batch_type] = []
        valid_min = self.episode_valid_indices_min[episode_idx]  # Inclusive
        valid_max = self.episode_valid_indices_max[episode_idx]  # Exclusive

        for zero_index in zero_indices:
            traj = self._get_single_traj_data(episode_idx, zero_index)
            if zero_index >= valid_max:
                traj["entire_traj_is_padding"] = torch.tensor(True)
            else:
                traj["entire_traj_is_padding"] = torch.tensor(False)

            trajs.append(traj)

        output_data_dict: batch_type = aggregate_batch(trajs, aggregate_fn=partial(torch.stack, dim=0))
        """
        output_data_dict (example):
            # local_cond
            "robot0_10d": (traj_num, length, 10),
            # global_cond
            "robot0_camera_images": (traj_num, length, 3, 256, 256),
            # output
            "action0_10d": (traj_num, length, 10),
            # meta
            "traj_idx": (traj_num),
            "episode_idx": (traj_num),
            "entire_traj_is_padding": (traj_num),
            "variance": (traj_num), # Optional
        """

        # for k, v in output_data_dict.items():
        #     print(f"dataloader 0 {k}: {v.shape}")


        # for k, v in output_data_dict.items():
        #     print(f"dataloader 1 {k}: {v.shape}")       
        # Batch size here is the `traj_num` dimension, which should be consistent

        if self.normalizer is not None:
            output_data_dict = self.normalizer.normalize(output_data_dict)
            
        # for k, v in output_data_dict.items():
        #     print(f"dataloader 2 {k}: {v.shape}")
        output_data_dict = self.transforms.apply(
            output_data_dict, consistent_on_batch=True
        )

        return output_data_dict
