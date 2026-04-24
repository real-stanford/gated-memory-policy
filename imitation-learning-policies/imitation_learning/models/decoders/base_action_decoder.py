from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from imitation_learning.common.dataclasses import DataMeta, construct_data_meta_dict


class BaseActionDecoder(nn.Module):
    def __init__(
        self,
        data_meta: dict[str, dict[str, Any]] | DictConfig,
        name: str,
        data_entry_names: list[str],
    ):
        super().__init__()
        data_meta_: dict[str, DataMeta] = construct_data_meta_dict(data_meta)
        self.data_entry_names: list[str] = data_entry_names
        self.action_meta: dict[str, DataMeta] = {
            k: v for k, v in data_meta_.items() if k in data_entry_names
        }
        self.decoder_dict: dict[str, nn.Module] = {}
        self.name: str = name
        self.data_shape: tuple[int, ...]  # For input data
        self.token_num: int
        self.latent_dim: int

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def encode(self, action_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
