from typing import Any

import torch
from torch import nn

from imitation_learning.common.dataclasses import DataMeta


class LowDimIdentityDecoder(nn.Module):
    def __init__(self, data_meta: dict[str, Any] | DataMeta):
        super().__init__()
        if isinstance(data_meta, dict):
            data_meta = DataMeta(**data_meta)
        self.data_meta: DataMeta = data_meta
        self.token_num: int = self.data_meta.length
        assert len(self.data_meta.shape) == 1, "data_meta.shape should be 1D"
        self.latent_dim: int = self.data_meta.shape[0]
        self.data_shape: tuple[int, ...] = self.data_meta.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: (batch_size, traj_len, *data_shape) or (batch_size, traj_num, traj_len, *data_shape)
        output: (batch_size, traj_len, *data_shape) or (batch_size, traj_num, traj_len, *data_shape)
        For identity decoder, we use *latent_shape = (traj_len, *data_shape)
        """
        data_ndim = len(self.data_shape)
        assert (
            x.shape[-data_ndim:] == self.data_shape
        ), f"x.shape[-data_ndim:] ({x.shape[-data_ndim:]}) is not equal to (traj_len, *data_shape): {self.data_shape}"
        return x
