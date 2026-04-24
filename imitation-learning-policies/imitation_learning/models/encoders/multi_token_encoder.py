from typing import Callable

import einops
import torch
import torch.nn as nn

from imitation_learning.models.encoders.base_cond_encoder import BaseEncoder
from imitation_learning.models.encoders.image_encoders import BaseImageEncoder
from imitation_learning.models.encoders.low_dim_encoders import LowDimProjector


class MultiTokenEncoder(BaseEncoder):
    """Will concatenate in token dimension. All encoders should have the same feature dimension"""

    def __init__(
        self,
        image_encoder_partial: Callable[..., BaseImageEncoder],
        projector_type: str = "",
        feature_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.token_num: int = 0

        # Handle empty cond_meta (e.g., when proprio is disabled via empty data_entry_names)
        if len(self.cond_meta) == 0:
            print(f"{self.name}: No data entries, creating empty encoder with token_num=0")
            self.encoder_dict: nn.ModuleDict = nn.ModuleDict()
            self.feature_dim: int = feature_dim if feature_dim is not None else 0
            return

        # Check if all entries have length=0 (e.g., when proprio_length=0)
        total_effective_length = sum(meta.length for meta in self.cond_meta.values())
        if total_effective_length == 0:
            print(f"{self.name}: All data entries have length=0, creating empty encoder with token_num=0")
            self.encoder_dict = nn.ModuleDict()
            self.feature_dim = feature_dim if feature_dim is not None else 0
            # Clear cond_meta and data_entry_names so forward() knows to return empty tensor
            # and so that the policy doesn't try to fetch data for these entries
            self.cond_meta = {}
            self.data_entry_names = []
            return

        self.encoder_dict = nn.ModuleDict()
        for meta in self.cond_meta.values():
            if meta.data_type == "image":
                self.encoder_dict[meta.name] = image_encoder_partial(image_meta=meta)
                if feature_dim is None:
                    feature_dim = self.encoder_dict[meta.name].feature_dim
                else:
                    assert (
                        feature_dim == self.encoder_dict[meta.name].feature_dim
                    ), f"feature dim mismatch: {meta.name} {feature_dim} vs {self.encoder_dict[meta.name].feature_dim}"

        print(
            self.name,
            "feature_dim",
            feature_dim,
            "data_entry_names",
            self.data_entry_names,
        )
        if feature_dim is None:
            raise ValueError("feature_dim must be provided if not using image encoder")
        self.feature_dim: int = feature_dim

        for meta in self.cond_meta.values():
            if meta.data_type == "low_dim":
                self.encoder_dict[meta.name] = LowDimProjector(
                    meta,
                    feature_dim=feature_dim,
                    projector_type=projector_type,
                )
            elif meta.data_type == "image":
                pass
            else:
                raise ValueError(f"Unknown obs type: {meta.data_type}")

        for meta in self.cond_meta.values():
            encoder = self.encoder_dict[meta.name]
            self.token_num += encoder.token_num * meta.length

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Each tensor in obs: (batch_size, traj_len, ...)
        or (batch_size, traj_num, traj_len, ...)

        # Or if obs is already encoded into image features, it should be (batch_size, traj_num, traj_len * token_num, feature_dim)
        # Then we just concatenate the features and return
        """
        
        features: list[torch.Tensor] = []
        for meta in self.cond_meta.values():
            encoder = self.encoder_dict[meta.name]

            feature_key = f"{meta.name}_feature"
            if feature_key in obs and hasattr(encoder, "feature_aggregation") and encoder.feature_aggregation == "map": # Only use cached features if the feature aggregation is "map"
                assert obs[feature_key].shape[-1] == self.feature_dim, f"{feature_key}.shape: {obs[feature_key].shape}, {self.feature_dim=}"
                assert obs[feature_key].shape[-2] == meta.length, f"{feature_key}.shape: {obs[feature_key].shape}, {meta.length=}"
                # print(f"obs[feature_key].shape: {obs[feature_key].shape}")
                features.append(obs[feature_key])
                continue
            
            if obs[meta.name].ndim == len(meta.shape) + 3:
                # (batch_size, traj_num, traj_len, ...)
                assert (
                    meta.length == obs[meta.name].shape[2]
                ), f"{meta.name}: traj_len({obs[meta.name].shape[2]}) != meta.length({meta.length}). Data shape: {obs[meta.name].shape}, meta.shape: {meta.shape}"
                assert meta.shape == tuple(
                    obs[meta.name].shape[3:]
                ), f"{meta.name}: meta.shape({meta.shape}) != data.shape({obs[meta.name].shape[3:]})"
            elif obs[meta.name].ndim == len(meta.shape) + 2:
                # (batch_size, traj_len, ...)
                assert (
                    meta.length == obs[meta.name].shape[1]
                ), f"{meta.name}: traj_len({obs[meta.name].shape[1]}) != meta.length({meta.length}). Data shape: {obs[meta.name].shape}, meta.shape: {meta.shape}"
                assert meta.shape == tuple(
                    obs[meta.name].shape[2:]
                ), f"{meta.name}: meta.shape({meta.shape}) != data.shape({obs[meta.name].shape[2:]})"
            else:
                raise ValueError(
                    f"Unknown data shape: {obs[meta.name].shape} for {meta.name}"
                )
            feature: torch.Tensor = encoder(obs[meta.name])
            feature = einops.rearrange(
                feature,
                "... traj_len token_num feature_dim -> ... (traj_len token_num) feature_dim",
            )
            # (batch_size, traj_len * token_num, feature_dim)
            # (batch_size, traj_num, traj_len * token_num, feature_dim)
            features.append(feature)

        result = torch.cat(
            features, dim=-2
        )  # (batch_size, sum(traj_len * token_num), feature_dim)
        # or (batch_size, traj_num, sum(traj_len * token_num), feature_dim)
        assert result.shape[-2] == self.token_num, f"{result.shape=}, {self.token_num=}"
        assert (
            result.shape[-1] == self.feature_dim
        ), f"{result.shape=}, {self.feature_dim=}"
        return result
