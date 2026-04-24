from math import prod
from typing import Any

import torch
import torch.nn as nn

from imitation_learning.common.dataclasses import DataMeta
from imitation_learning.models.common.modules import Projector


class LowDimIdentityEncoder(nn.Module):
    def __init__(self, data_meta: dict[str, Any] | DataMeta):
        super().__init__()
        if isinstance(data_meta, dict):
            data_meta = DataMeta(**data_meta)
        self.data_meta = data_meta
        self.feature_dim = prod(self.data_meta.shape) * self.data_meta.length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, traj_len, *shape = x.shape
        assert (
            traj_len == self.data_meta.length
        ), f"traj_len: {traj_len}, self.data_meta.length: {self.data_meta.length} are not consistent"
        assert tuple(shape) == tuple(
            self.data_meta.shape
        ), f"x.shape[2:]: {tuple(shape)}, self.data_meta.shape: {tuple(self.data_meta.shape)}"
        x = x.reshape(batch_size, -1)
        return x


class LowDimProjector(nn.Module):
    def __init__(
        self,
        data_meta: dict[str, Any] | DataMeta,
        feature_dim: int,
        projector_type: str,
    ):
        super().__init__()
        if isinstance(data_meta, dict):
            data_meta = DataMeta(**data_meta)
        self.data_meta: DataMeta = data_meta
        self.feature_dim: int = feature_dim
        self.token_num: int = 1
        self.input_dim: int = prod(self.data_meta.shape)
        if projector_type == "identity":
            assert (
                self.input_dim == self.feature_dim
            ), f"mismatched size in data shape: {data_meta.shape}, feature_dim: {feature_dim} for identity projector"
            self.model = nn.Identity()
        else:
            self.model = Projector(projector_type, self.input_dim, self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, trajectory_len, *data_shape)
        return: (batch_size, trajectory_len, token_num=1, feature_dim)
        """
        batch_size, traj_len, *shape = x.shape
        assert (
            traj_len == self.data_meta.length
        ), f"traj_len: {traj_len}, self.data_meta.length: {self.data_meta.length} are not consistent"
        assert tuple(shape) == tuple(
            self.data_meta.shape
        ), f"x.shape[2:]: {shape}, self.data_meta.shape: {self.data_meta.shape}"
        x = x.reshape(batch_size, traj_len, self.input_dim)
        x = self.model(x)  # (batch_size, trajectory_len, self.feature_dim)
        x = x.reshape(batch_size, traj_len, self.token_num, self.feature_dim)
        return x
