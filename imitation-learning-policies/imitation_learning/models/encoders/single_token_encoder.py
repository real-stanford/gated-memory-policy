from typing import Callable

import torch
import torch.nn as nn

from imitation_learning.models.encoders.base_cond_encoder import BaseEncoder
from imitation_learning.models.encoders.image_encoders import BaseImageEncoder
from imitation_learning.models.encoders.low_dim_encoders import \
    LowDimIdentityEncoder


class SingleTokenEncoder(BaseEncoder):
    """Will concatenate all observation features in feature dimension into a single token"""

    def __init__(
        self,
        image_encoder_partial: Callable[..., BaseImageEncoder],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.feature_dim: int = 0
        self.encoder_dict: nn.ModuleDict = nn.ModuleDict()
        for meta in self.cond_meta.values():
            if meta.data_type == "low_dim":
                self.encoder_dict[meta.name] = LowDimIdentityEncoder(meta)
            elif meta.data_type == "image":
                self.encoder_dict[meta.name] = image_encoder_partial(image_meta=meta)
            else:
                raise ValueError(f"Unknown obs type: {meta.data_type}")
        for meta in self.cond_meta.values():
            encoder = self.encoder_dict[meta.name]
            self.feature_dim += encoder.feature_dim * meta.length
        self.token_num: int = 1

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Each tensor in obs: (batch_size, traj_len, ...)
        return: (batch_size, 1, feature_dim)
        """
        features: list[torch.Tensor] = []
        for meta in self.cond_meta.values():
            feature: torch.Tensor = self.encoder_dict[meta.name](obs[meta.name])
            if meta.data_type == "image":
                assert (
                    len(feature.shape) == 4
                ), f"feature.shape: {feature.shape}, feature.shape should be 4"
                assert (
                    feature.shape[2] == 1
                ), f"feature.shape[2]: {feature.shape[2]}, feature should only have 1 token per image"
                feature = feature.squeeze(2)
                batch_size, traj_len, feature_dim = feature.shape
                feature = feature.reshape(batch_size, feature_dim * traj_len)
            features.append(feature)

        aggregated_feature = torch.cat(features, dim=-1)
        return aggregated_feature.unsqueeze(1)  # (batch_size, 1, feature_dim)
