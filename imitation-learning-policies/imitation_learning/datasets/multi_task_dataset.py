from copy import deepcopy
from typing import Any, cast

import hydra
from imitation_learning.common.datatypes import batch_type
from imitation_learning.datasets.base_dataset import BaseDataset
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
from omegaconf import OmegaConf

from imitation_learning.datasets.normalizer import FixedNormalizer

class MultiTaskDataset(Dataset[batch_type]):
    def __init__(
        self,
        name: str,
        dataset_names: str,
        sub_dataset_target: str,
        dataset_configs: dict[str, dict[str, Any]],
        **base_config: dict[str, Any],
    ):
        """
        name: the name of the multi-task dataset
        dataset_names: if select certain sub-datasets, should be the name of multiple sub-datasets connected by ","; or use "dataset_name/*" to select all tasks in a dataset; or use "*" to select all datasets & tasks in the dataset
        root_dir, compressed_dir, normalizer_dir should be the same for all datasets
        dataset_configs: {
            "dataset_name_1": {
                "sample_ratio": 1.0,
                "override_config_1": "value_1",
                "override_config_2": "value_2",
                ...
            },
            ...
        }
        """

        self.name: str = name

        self.dataloader_cfg: dict[str, Any] = base_config["dataloader_cfg"]

        if isinstance(dataset_configs, DictConfig):
            dataset_configs = cast(dict[str, dict[str, Any]], OmegaConf.to_container(dataset_configs))

        dataset_names_str: str = dataset_names

        dataset_names_list: list[str] = []
            
        if " " in dataset_names_str:
            dataset_names_str = dataset_names_str.replace(" ", "")  # Remove spaces
        if ";" in dataset_names_str:
            dataset_names_str = dataset_names_str.replace(";", ",")
        if "," in dataset_names_str:
            dataset_names_list = dataset_names_str.split(",")
        else:
            dataset_names_list = [dataset_names_str]

        used_dataset_configs = {}
        for name in dataset_names_list:
            if "/*" in name:
                dataset_name = name.replace("*", "")
                matched_dataset_names = [name for name in dataset_configs.keys() if name.startswith(dataset_name)]
                used_dataset_configs.update({name: dataset_configs[name] for name in matched_dataset_names})
            elif name == "*": # Use all datasets & tasks in the dataset
                used_dataset_configs.update(dataset_configs)
            else:
                used_dataset_configs[name] = dataset_configs[name]

        assert len(used_dataset_configs) >= 1, "At least one dataset is required"
            
        print(f"included datasets: {list(used_dataset_configs.keys())}")
        self.dataset_configs: dict[str, dict[str, Any]] = used_dataset_configs
        assert len(self.dataset_configs) >= 1, "At least one dataset is required"
        if isinstance(base_config, DictConfig):
            base_config = cast(dict[str, Any], OmegaConf.to_container(base_config))
        self.base_config: dict[str, Any] = base_config

        self.sample_ratios: dict[str, float] = {
            name: config.pop("sample_ratio") for name, config in self.dataset_configs.items()
        }
        # TODO: sample index pool based on sample_ratios

        self.datasets: dict[str, BaseDataset] = {}
        for dataset_name, dataset_config in self.dataset_configs.items():
            print(f"Initializing dataset: {dataset_name}")
            config = deepcopy(self.base_config)
            config.update(deepcopy(dataset_config))
            config["name"] = dataset_name
            config["_target_"] = sub_dataset_target
            self.datasets[dataset_name] = hydra.utils.instantiate(config)

        self.index_pool: list[tuple[str, int]] = []
        """
        First value: dataset index
        Second value: data index in the corresponding dataset
        """
        self._create_index_pool()

    @property
    def normalizer(self) -> FixedNormalizer | None:
        return next(iter(self.datasets.values())).normalizer

    def _create_index_pool(self):
        self.index_pool = []
        for dataset_name, dataset in self.datasets.items():
            self.index_pool.extend((dataset_name, i) for i in range(len(dataset)))
            print(f"Dataset {dataset_name} index pool size: {len(dataset)}")
        
        print(f"Index pool size: {len(self.index_pool)}")

    def __len__(self):
        return len(self.index_pool)

    def __getitem__(self, idx: int) -> batch_type:
        dataset_name, data_idx = self.index_pool[idx]
        data = self.datasets[dataset_name][data_idx]
        data["dataset_name"] = dataset_name
        return data
    
    def split_unused_episodes(
        self,
        remaining_ratio: float = 1.0,
        other_used_episode_indices: list[int] | None = None,
    ):
        unused_dataset = deepcopy(self)
        unused_dataset.index_pool = []
        unused_dataset.datasets = {}

        for dataset_name, dataset in self.datasets.items():
            unused_dataset.datasets[dataset_name] = dataset.split_unused_episodes(
                remaining_ratio, other_used_episode_indices
            )
        unused_dataset._create_index_pool()

        return unused_dataset

    def repeat_dataset(self):
        for dataset_name, dataset in self.datasets.items():
            dataset.repeat_dataset()
        self._create_index_pool()

    def get_dataloader(self):
        return DataLoader(self, **self.dataloader_cfg)
    

