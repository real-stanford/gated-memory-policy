import copy
import json
import multiprocessing
import os
import subprocess
import time
from typing import Any, Optional, Union, cast

import numpy as np
import numpy.typing as npt
import torch
import zarr
import dill
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from imitation_learning.common.dataclasses import (DataMeta, SourceDataMeta,
                                                   construct_data_meta_dict,
                                                   construct_source_data_meta)
from imitation_learning.common.datatypes import batch_type
from imitation_learning.datasets.normalizer import FixedNormalizer
from imitation_learning.datasets.transforms import BaseTransforms
from robot_utils.torch_utils import torch_load
from robot_utils.torch_utils import is_main_process

# torch.set_num_threads(1)  # To disable torch multithreading for dataloader


class BaseDataset(Dataset[batch_type]):
    """
    Base class for all datasets.
    """

    def __init__(
        self,
        root_dir: str, 
        # The folder that contains all the zarr stores
        # data path should be: root_dir/name/episode_data.zarr or root_dir/name.zarr
        name: str,
        robot_num: int,
        compressed_dir: str, # The folder that contains the lz4 compressed data (compressed_dir/name.lz4)
        # If dataset is not found in root_dir, the program will extract the data from compressed_dir
        include_episode_num: int,
        include_episode_indices: list[int],
        used_episode_ratio: float,
        index_pool_size_per_episode: int,
        history_padding_length: int,
        future_padding_length: int,
        seed: int,
        source_data_meta: Union[dict[str, dict[str, Any]], DictConfig],
        output_data_meta: Union[dict[str, dict[str, Any]], DictConfig],
        dataloader_cfg: dict[str, Any],
        starting_percentile_max: float,
        starting_percentile_min: float,
        apply_image_augmentation_in_cpu: bool,
        use_relative_pose: bool,
        use_relative_gripper_width: bool,
        normalizer_sample_num: int,
        normalizer_dir: str, # Or directly pass the normalizer path (ended with normalizer.json)
        repeat_dataset_num: int,
        random_split_dataset: bool = True, # If false, will not use random split for the dataset
        down_sample_steps: int = 1, # For compatibility with previous checkpoints
        statistics_data_path: str = "", # For memory gate training which needs an additional file
        **unused_kwargs,
    ):
        print(f"BaseDataset unused_kwargs: {unused_kwargs}")

        self.down_sample_steps: int = down_sample_steps
        history_padding_length = (
            history_padding_length * down_sample_steps
        )
        future_padding_length = (
            future_padding_length * down_sample_steps
        )
        for meta in source_data_meta.values():
            meta["include_indices"] = [
                i * down_sample_steps for i in meta["include_indices"]
            ]

        assert (
            isinstance(robot_num, int) and robot_num >= 1
        ), f"robot_num must be an integer greater than 0, but got {robot_num}."
        self.robot_num: int = robot_num
        self.name: str = name
        print(f"Dataset name: {self.name}")

        self.include_episode_num: int = include_episode_num
        self.include_episode_indices: list[int] = include_episode_indices
        self.used_episode_ratio: float = used_episode_ratio
        self.random_split_dataset: bool = random_split_dataset
        self.index_pool_size_per_episode: int = index_pool_size_per_episode
        self.history_padding_length: int = history_padding_length
        self.future_padding_length: int = future_padding_length
        self.seed: int = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.dataloader_cfg: dict[str, Any] = dataloader_cfg
        self.starting_percentile_max: float = starting_percentile_max
        self.starting_percentile_min: float = starting_percentile_min
        self.apply_image_augmentation_in_cpu: bool = apply_image_augmentation_in_cpu
        self.use_relative_pose: bool = use_relative_pose
        self.use_relative_gripper_width: bool = use_relative_gripper_width
        self.sim_config_str: str = ""
        self.episode_frame_nums: dict[int, int] = {}

        # Load data store (zarr by default, overridden by subclasses)
        self._load_data_store(root_dir, name, compressed_dir)

        assert len(source_data_meta) > 0, "source_data_meta is empty."
        self.source_data_meta: dict[str, SourceDataMeta] = construct_source_data_meta(
            source_data_meta
        )

        assert len(output_data_meta) > 0, "output_data_meta is empty."
        self.output_data_meta: dict[str, DataMeta] = construct_data_meta_dict(
            output_data_meta
        )

        self.max_history_length: int = max(
            0,
            -min(
                entry_meta.include_indices[0]
                for entry_meta in self.source_data_meta.values()
            ),
        )
        self.max_future_length: int = max(
            0,
            max(
                entry_meta.include_indices[-1]
                for entry_meta in self.source_data_meta.values()
            ),
        )

        if self.history_padding_length > self.max_history_length:
            raise ValueError(
                f"history_padding_length {self.history_padding_length} is larger than max_history_length {self.max_history_length}. This may cause ambiguity in the data."
            )

        self.normalizer: Optional[FixedNormalizer] = None
        self.normalizer_dir: str = normalizer_dir
        if normalizer_dir is not None and normalizer_dir != "":
            if not normalizer_dir.endswith(".json"):
                normalizer_path = os.path.join(
                    normalizer_dir, f"{self.name}_normalizer.json"
                )
            else:
                normalizer_path = normalizer_dir
            if os.path.exists(normalizer_path):
                print(f"Loading normalizer from {normalizer_path}.")
                self.normalizer = FixedNormalizer(self.output_data_meta)
                self.normalizer.from_dict(json.load(open(normalizer_path)))
                self.normalizer.to(torch.device("cpu"))
                print(f"Normalizer loaded from {normalizer_path}.")
                print(f"Normalizer: {self.normalizer.as_dict('list')}")
        self.normalizer_sample_num: int = normalizer_sample_num

        self.store_episode_num: int
        self.used_episode_indices: list[int]
        self.used_episode_num: int

        self.episode_valid_indices_min: dict[int, int]
        self.episode_valid_indices_max: dict[int, int]

        self.index_pool: list[tuple[int, int]] = []
        """
        index_pool has self.store_episode_num * self.used_episode_ratio * self.index_pool_size_per_episode items.
        Each item contains a tuple of (episode_idx, index), where index means the 0 index of this trajectory in an episode.
        """

        self.avg_frame_num: float
        self.std_frame_num: float
        self.max_frame_num: int
        self.min_frame_num: int
        self.episode_valid_indices_min: dict[int, int] = {}
        self.episode_valid_indices_max: dict[int, int] = {}  # Exclusive

        self.transforms: BaseTransforms = BaseTransforms(
            self.output_data_meta, self.apply_image_augmentation_in_cpu, self.seed
        )

        self.repeat_dataset_num: float = repeat_dataset_num

        self.statistics_data: None | dict[int, dict[str, torch.Tensor]] = None
        ### For memory gate training ###
        if statistics_data_path != "":
            self.statistics_data = torch_load(statistics_data_path, pickle_module=dill, weights_only=False)
            print(f"Statistics data loaded from {statistics_data_path}")
            assert self.statistics_data is not None
            print(f"{self.statistics_data.keys()=}")

    def _load_data_store(self, root_dir: str, name: str, compressed_dir: str):
        """
        Load the backing data store. By default, finds and opens a zarr store.
        Subclasses can override this to load from other formats (e.g., parquet).

        Must set self.episode_frame_nums for all episodes by the end.
        """
        os.makedirs(root_dir, exist_ok=True)

        if name.endswith(".zarr"):
            name = name.replace(".zarr", "")

        if is_main_process():
            if os.path.exists(os.path.join(root_dir, name, "episode_data.zarr")):
                zarr_path = os.path.join(root_dir, name, "episode_data.zarr")
            elif os.path.exists(os.path.join(root_dir, name + ".zarr")):
                zarr_path = os.path.join(root_dir, name + ".zarr")
            elif os.path.exists(os.path.join(root_dir, name)):
                zarr_path = os.path.join(root_dir, name)
            else:
                if "/" in name:
                    parent_dir = os.path.dirname(name)
                    os.makedirs(os.path.join(root_dir, parent_dir), exist_ok=True)
                file = open(os.path.join(root_dir, f"{name}.lock"), "w")
                file.close()
                print(f"Dataset {name} not found in {root_dir}")
                print(f"Checked paths: {os.path.join(root_dir, name, 'episode_data.zarr')}, {os.path.join(root_dir, name + '.zarr')}, {os.path.join(root_dir, name)}")
                lz4_path = os.path.join(compressed_dir, name + '.tar.lz4')
                lz4_zarr_path = os.path.join(compressed_dir, name + '.zarr.tar.lz4')
                if os.path.exists(lz4_path):
                    # Why this doesn't work???
                    # print(f"Extracting dataset {name} from {compressed_dir} to {root_dir}")
                    # os.makedirs(f"{root_dir}/{name}", exist_ok=True)
                    # extract_cmd =f"lz4 -d -c {lz4_path} | tar xf - -C {root_dir}/{name} --strip-components=1"
                    # print(f"extract_cmd: {extract_cmd}")
                    # subprocess.run(extract_cmd, cwd=root_dir, shell=True, check=True)
                    print(f"Extracting dataset {name} from {compressed_dir} to {root_dir}")
                    working_dir = os.getcwd()
                    subprocess.run([
                        f"cd {working_dir} && lz4 -d -c {lz4_path} | tar xf - -C {root_dir}"
                        # f"lz4 -d -c {lz4_path} | tar xf - -C {root_dir}/{name} --strip-components=1"
                    ],
                    cwd=root_dir,
                    shell=True,
                    check=True,
                    )
                elif os.path.exists(lz4_zarr_path):
                    print(f"Extracting dataset {name} from {compressed_dir} to {root_dir}")
                    subprocess.run(f"mkdir -p {root_dir}/{name}.zarr", shell=True)
                    extract_cmd = f"lz4 -d -c {lz4_zarr_path} | tar xf - -C {root_dir}/{name}.zarr --strip-components=1"
                    print(f"extract_cmd: {extract_cmd}")
                    subprocess.run(extract_cmd, cwd=root_dir, shell=True, check=True)
                else:
                    zip_path = os.path.join(compressed_dir, name + '.zarr.zip')
                    if os.path.exists(zip_path):
                        print(f"Extracting dataset {name} from {compressed_dir} to {root_dir}")
                        subprocess.run([
                            f"unzip -q {zip_path} -d {root_dir}{name}.zarr"
                        ],
                        cwd=root_dir,
                        shell=True,
                        check=True,
                        )
                    else:
                        raise FileNotFoundError(f"Dataset {name} not found in {compressed_dir} and {root_dir}")

                print(f"Dataset {name} extracted to {root_dir}")
                os.remove(os.path.join(root_dir, f"{name}.lock"))

        # Use wait_for_main_process() will lead to memory leak on GPU 0
        # Will manually create a lock file to wait for the main process to extract dataset

        if os.path.exists(os.path.join(root_dir, name, "episode_data.zarr")):
            zarr_path = os.path.join(root_dir, name, "episode_data.zarr")
        elif os.path.exists(os.path.join(root_dir, name + ".zarr")):
            zarr_path = os.path.join(root_dir, name + ".zarr")
        elif os.path.exists(os.path.join(root_dir, name)):
            zarr_path = os.path.join(root_dir, name)
        else:
            while True:
                print(f"Waiting for the main process to extract dataset...")
                time.sleep(10) # Wait for the main process to extract dataset
                if os.path.exists(os.path.join(root_dir, f"{name}.lock")):
                    print(f"Find lock file: {os.path.join(root_dir, f'{name}.lock')}. Waiting for the main process to extract dataset...")
                    continue

                if os.path.exists(os.path.join(root_dir, name, "episode_data.zarr")):
                    zarr_path = os.path.join(root_dir, name, "episode_data.zarr")
                    break
                elif os.path.exists(os.path.join(root_dir, name + ".zarr")):
                    zarr_path = os.path.join(root_dir, name + ".zarr")
                    break
                elif os.path.exists(os.path.join(root_dir, name)):
                    zarr_path = os.path.join(root_dir, name)
                    break

        while os.path.exists(os.path.join(root_dir, f"{name}.lock")):
            time.sleep(1)

        self.zarr_path: str = zarr_path

        if os.path.exists(f"{self.zarr_path}/../sim_config.yaml"):
            print(f"sim_config.yaml found in {self.zarr_path}/../")
            sim_config_dict = OmegaConf.load(f"{self.zarr_path}/../sim_config.yaml")
            self.sim_config_str = OmegaConf.to_yaml(sim_config_dict, resolve=False)

        print(f"{self.zarr_path=}")
        zarr_store = zarr.open(self.zarr_path, mode="r")
        assert isinstance(
            zarr_store, zarr.Group
        ), f"zarr store {self.zarr_path} is not a group."
        self.zarr_store: zarr.Group = zarr_store
        """
        self.statistics_data: dict[int, dict[str, torch.Tensor]] = {
            episode_idx: {
                "with_mem_variances": torch.Tensor(frame_num),
                "no_mem_variances": torch.Tensor(frame_num),
                "with_mem_errors": torch.Tensor(frame_num),
                "no_mem_errors": torch.Tensor(frame_num),
            }
        """

    def _check_data_validity(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _get_single_traj_data(self, episode_idx: int, traj_idx: int, output_entry_names: list[str] | None = None):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _create_index_pool(self):

        self.index_pool = []
        # rng = np.random.default_rng(self.seed)
        for episode_idx in self.used_episode_indices:

            valid_idx_min = self.episode_valid_indices_min[episode_idx]
            valid_idx_max = self.episode_valid_indices_max[episode_idx]
            # valid_idx_min <= sample_idx < valid_idx_max

            zero_idx_max = valid_idx_min + int(
                (valid_idx_max - valid_idx_min) * self.starting_percentile_max
            )  # Exclusive
            zero_idx_min = valid_idx_min + int(
                (valid_idx_max - valid_idx_min) * self.starting_percentile_min
            )  # Inclusive

            if self.index_pool_size_per_episode == -1:
                index_pool_size = zero_idx_max - zero_idx_min
            else:
                assert (
                    self.index_pool_size_per_episode > 0
                ), f"index_pool_size_per_episode must be positive or -1, but got {self.index_pool_size_per_episode}."
                index_pool_size = self.index_pool_size_per_episode

            if index_pool_size >= zero_idx_max - zero_idx_min:
                indices = np.arange(zero_idx_min, zero_idx_max)
                random_indices = self.rng.choice(
                    range(zero_idx_min, zero_idx_max),
                    size=index_pool_size - (zero_idx_max - zero_idx_min),
                    replace=True,
                )
                indices = np.concatenate([indices, random_indices])
            else:
                indices = self.rng.choice(
                    range(zero_idx_min, zero_idx_max),
                    size=index_pool_size,
                    replace=False,
                )
            indices = np.sort(indices)
            for index in indices:
                if self.statistics_data is not None:
                    if (
                        episode_idx not in self.statistics_data or \
                        "with_mem_errors" not in self.statistics_data[episode_idx] or \
                        index >= len(self.statistics_data[episode_idx]["with_mem_errors"]) or \
                        torch.isnan(self.statistics_data[episode_idx]["with_mem_errors"][index]).any()
                    ):
                        continue

                self.index_pool.append((episode_idx, index))

    def _update_episode_indices(self):

        if len(self.include_episode_indices) > 0:
            print(
                f"Dataset {self.name}: Using specified episode indices: {self.include_episode_indices}."
            )
            self.include_episode_num: int = len(self.include_episode_indices)
            for episode_idx in self.include_episode_indices:
                assert (
                    episode_idx < self.store_episode_num
                ), f"episode_idx {episode_idx} is out of range. Max is {self.store_episode_num}."
        else:
            if self.include_episode_num > 0:
                assert (
                    self.include_episode_num <= self.store_episode_num
                ), f"include_episode_num {self.include_episode_num} is greater than the number of episodes {self.store_episode_num}."
                self.include_episode_indices = np.random.default_rng(self.seed).choice(
                # self.include_episode_indices = self.rng.choice(
                    self.store_episode_num, size=self.include_episode_num, replace=False
                ).tolist()
                print(
                    f"Dataset {self.name}: Using {self.include_episode_num} episodes from {self.store_episode_num} episodes: {self.include_episode_indices}"
                )
            elif self.include_episode_num == -1:
                self.include_episode_num = self.store_episode_num
                self.include_episode_indices = list(range(self.include_episode_num))
                print(
                    f"Dataset {self.name}: Using all {self.include_episode_num} episodes from {self.store_episode_num}"
                )
            else:
                raise ValueError(
                    f"include_episode_num {self.include_episode_num} is invalid. Must be -1 or a positive integer."
                )

        self.include_episode_indices = sorted(self.include_episode_indices)

        if self.random_split_dataset:
            self.used_episode_indices: list[int] = cast(
                list[int],
                np.random.default_rng(self.seed).choice(
                # self.rng.choice(
                    self.include_episode_indices,
                    size=int(self.include_episode_num * self.used_episode_ratio),
                    replace=False,
                ).tolist(),
            )
        else:
            self.used_episode_indices = self.include_episode_indices[:int(self.include_episode_num * self.used_episode_ratio)]
        self.used_episode_indices = sorted(self.used_episode_indices)
        print(f"Dataset {self.name}: Used episode indices: {self.used_episode_indices}")
        self.used_episode_num: int = len(self.used_episode_indices)

    def repeat_dataset(self, repeat_num: float | None = None):
        if repeat_num is None:
            repeat_num = self.repeat_dataset_num
        else:
            self.repeat_dataset_num = repeat_num
        index_pool_size = len(self.index_pool)
        repeated_size = int(index_pool_size * repeat_num)
        # repeated_indices = np.random.default_rng(self.seed).choice(
        repeated_indices = self.rng.choice(
            range(index_pool_size),
            size=repeated_size,
            replace=True,
        )
        self.index_pool = [self.index_pool[i] for i in repeated_indices]

    def trim_dataset_episodes(self, remaining_episode_num: int | None = None, remaining_episode_indices: list[int] | None = None):
        assert remaining_episode_num is not None or remaining_episode_indices is not None
        assert remaining_episode_num is None or remaining_episode_indices is None

        size_before_trimming = len(self.index_pool)

        if remaining_episode_num is not None:
            current_used_episodes = set([episode_idx for episode_idx, _ in self.index_pool])
            current_episode_num = len(current_used_episodes)
            if current_episode_num <= remaining_episode_num:
                return
            remaining_episodes = sorted(np.random.default_rng(self.seed).choice(
                list(current_used_episodes),
                size=remaining_episode_num,
                replace=False,
            ).tolist())
        else:
            remaining_episodes = sorted(remaining_episode_indices)
        # print(f"self.index_pool: {self.index_pool}, remaining_episodes: {remaining_episodes}")
        self.index_pool = [index for index in self.index_pool if index[0] in remaining_episodes]
        print(f"Remaining episodes after trimming: {remaining_episodes}")
        print(f"Index pool size trimmed from {size_before_trimming} to {len(self.index_pool)}")

    def split_unused_episodes(
        self,
        remaining_ratio: float = 1.0,
        other_used_episode_indices: Optional[list[int]] = None,
    ):
        """
        Split unused episodes from the included episodes.
        """
        print(
            f"Splitting unused data with remaining ratio {remaining_ratio} and other used episode ids {other_used_episode_indices}."
        )
        unused_dataset = copy.deepcopy(self)
        unused_dataset.rng = np.random.default_rng(unused_dataset.seed)
        if other_used_episode_indices is None:
            other_used_episode_indices = []
        unused_episode_indices = [
            episode_idx
            for episode_idx in self.include_episode_indices
            if episode_idx not in self.used_episode_indices
            and episode_idx not in other_used_episode_indices
        ]
        unused_dataset.used_episode_indices = cast(
            list[int],
            np.random.default_rng(self.seed).choice(
            # self.rng.choice(
                unused_episode_indices,
                size=int(len(unused_episode_indices) * remaining_ratio),
                replace=False,
            ).tolist(),
        )
        unused_dataset.used_episode_indices = sorted(
            unused_dataset.used_episode_indices
        )
        unused_dataset.used_episode_ratio = len(
            unused_dataset.used_episode_indices
        ) / len(unused_dataset.include_episode_indices)
        unused_dataset._check_data_validity()
        unused_dataset._create_index_pool()
        assert (
            len(unused_dataset) >= 1
        ), f"Splitted dataset {unused_dataset.name} has no data. Please check the used_data_ratio and the overall dataset size"
        print(f"Splitted dataset {unused_dataset.name}: Used episode indices: {unused_dataset.used_episode_indices}")
        return unused_dataset

    def get_dataloader(self):
        return DataLoader(self, **self.dataloader_cfg)

    def sample_data(
        self,
        output_entry_names: list[str],
        sample_num: int,
        augment_data: bool,
        normalize_data: bool,
        sampled_indices: npt.NDArray[np.int64] | None = None,
    ) -> batch_type:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def calc_stats_worker_single_process(self, start_idx: int, end_idx: int, entry_names: list[str]):
        """
        return: 
            stats: dict[str, dict[str, torch.Tensor]] = {
                entry_name: {
                    "min": torch.Tensor(batch_size, data_shape),
                    "max": torch.Tensor(batch_size, data_shape),
                }
            }
        """
        data: dict[str, torch.Tensor] = self.sample_data(
            sample_num=end_idx - start_idx, 
            output_entry_names=entry_names, 
            augment_data=False, 
            normalize_data=False, 
            sampled_indices=np.arange(start_idx, end_idx)
        )
        stats: dict[str, dict[str, torch.Tensor]] = {}
        for entry_name in entry_names:
            stats[entry_name] = {
                "min": torch.min(data[entry_name], dim=1).values, # (batch_size, *)
                "max": torch.max(data[entry_name], dim=1).values, # (batch_size, *)
            }
        return stats

    def calc_stats(self, entry_names: list[str], process_num: int=0):
        """
        return: 
            stats: dict[str, dict[str, torch.Tensor]] = {
                entry_name: {
                    "min": torch.Tensor(batch_size, data_shape),
                    "max": torch.Tensor(batch_size, data_shape),
                }
            }
        """
        # assert process_num > 0
        if process_num == 0:
            process_num = multiprocessing.cpu_count() - 2
        print(f"Calculating stats with {process_num} processes.")

        index_pool_size = len(self.index_pool)
        indices_per_process = index_pool_size // process_num
        start_indices = np.arange(0, index_pool_size, indices_per_process)[:process_num]
        end_indices = start_indices + indices_per_process
        end_indices[-1] = index_pool_size

        with multiprocessing.Pool(process_num) as pool:
            results = pool.starmap(self.calc_stats_worker_single_process, zip(start_indices, end_indices, [entry_names] * process_num))
        
        stats: dict[str, dict[str, torch.Tensor]] = {}

        for result in results:
            for entry_name in entry_names:
                if entry_name not in stats:
                    stats[entry_name] = {
                        "min": result[entry_name]["min"],
                        "max": result[entry_name]["max"],
                    }
                else:
                    stats[entry_name]["min"] = torch.cat((stats[entry_name]["min"], result[entry_name]["min"]), dim=0)
                    stats[entry_name]["max"] = torch.cat((stats[entry_name]["max"], result[entry_name]["max"]), dim=0)

        return stats
        

    def fit_normalizer(self, stats_path: str | None = None, quantile: float = 1.0) -> FixedNormalizer:

        assert 0.90 <= quantile <= 1.0, "Quantile must be between 0.90 and 1.0"

        if quantile < 1.0:
            print(f"fitting normalizer with quantile: {quantile}")
            for entry_meta in self.output_data_meta.values():
                if entry_meta.normalizer != "identity":
                    assert entry_meta.normalizer == "range_clip", f"Normalizer for {entry_meta.name} must be range_clip when quantile < 1.0, but got {entry_meta.normalizer}"

        print(f"Fitting normalizer for {self.name}.")
        self.normalizer = FixedNormalizer(self.output_data_meta)

        normalize_entries = [
            entry_meta.name
            for entry_meta in self.output_data_meta.values()
            if entry_meta.normalizer != "identity"
        ]

        if stats_path is not None:
            stats = cast(dict[str, dict[str, torch.Tensor]], torch_load(stats_path, pickle_module=dill))
        else:
            stats = self.calc_stats(
                entry_names=normalize_entries,
            )

        aggregated_stats: dict[str, dict[str, torch.Tensor]] = {}
        for entry_name in normalize_entries:
            quantile_min = torch.quantile(stats[entry_name]["min"], (1 - quantile)/2, dim=0)
            quantile_max = torch.quantile(stats[entry_name]["max"], (1 + quantile)/2, dim=0)
            aggregated_stats[entry_name] = {
                "min": quantile_min,
                "max": quantile_max,
            }
            
        print(f"Aggregated stats: {aggregated_stats}")

        self.normalizer.from_dict(aggregated_stats)


        self.normalizer.to(torch.device("cpu"))

        normalizer_state_dict = self.normalizer.as_dict("list")

        if self.normalizer_dir != "":
            os.makedirs(self.normalizer_dir, exist_ok=True)
            normalizer_path = os.path.join(
                self.normalizer_dir, f"{self.name}_normalizer.json"
            )
            with open(normalizer_path, "w") as f:
                json.dump(normalizer_state_dict, f)
            print(f"Normalizer dict saved to {normalizer_path}.")

        return self.normalizer

    def process_image_data(self, data: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        if (
            data.shape[-1] <= 4
        ):  # (..., H, W, C) where the color dimension is usually a small number (1 (grayscale), 3 (RGB), or 4 (RGBD))
            dims = len(data.shape)
            data = data.transpose((*range(dims - 3), -1, -3, -2))  # (..., C, H, W)
        if data.dtype == np.uint8:
            return (data / 255.0).astype(np.float32)
        return data.astype(np.float32)

    def __len__(self) -> int:
        return len(self.index_pool)

    def __getitem__(self, idx: int) -> batch_type:
        raise NotImplementedError("This method should be implemented in subclasses.")
