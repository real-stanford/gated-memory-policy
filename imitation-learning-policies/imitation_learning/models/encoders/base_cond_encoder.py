from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from imitation_learning.common.dataclasses import DataMeta, construct_data_meta_dict


class BaseEncoder(nn.Module):
    def __init__(
        self,
        data_meta: DictConfig | dict[str, dict[str, Any]],
        name: str,
        data_entry_names: list[str],
    ):
        super().__init__()
        data_meta_: dict[str, DataMeta] = construct_data_meta_dict(data_meta)

        self.cond_meta = {}
        for data_entry_name in data_entry_names:
            if data_entry_name in data_meta_:
                self.cond_meta[data_entry_name] = data_meta_[data_entry_name]
            else:
                print(f"BaseEncoder Warning: {data_entry_name} not found in data_meta")

        self.encoder_dict: nn.ModuleDict
        self.name: str = name
        self.data_entry_names: list[str] = data_entry_names
        self.feature_dim: int
        self.token_num: int

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Each tensor in obs: (batch_size, traj_len, ...)
        return: (batch_size, token_num, feature_dim)
        """
        raise NotImplementedError("This method should be implemented by the subclass")
