import numpy as np
import dill
import torch
import torch.nn as nn

from imitation_learning.common.dataclasses import DataMeta, construct_data_meta
from imitation_learning.common.datatypes import batch_type
from imitation_learning.models.encoders.image_encoders import BaseImageEncoder
from omegaconf import DictConfig

from robot_utils.torch_utils import torch_load


class MemoryGate(nn.Module):
    def __init__(
        self,
        image_encoder: BaseImageEncoder,
        device: torch.device,
        hidden_dim: int,
        freeze_mlp: bool,
        cutoff_threshold: float,
        proprio_meta: DictConfig | None,
        ckpt_path: str = "",
        loss_fn_name: str = "bce",
        fixed_output_val: bool | None = None, # For ablation study. True will always output 1, False will always output 0
        **unused_kwargs,
    ):
        """
        Currently only support one set of images and one set of proprio
        """
        print(f"MemoryGate unused kwargs: {unused_kwargs}")
        super().__init__()

        self.image_encoder: BaseImageEncoder = image_encoder
        img_length = self.image_encoder.image_meta.length
        img_feature_dim = self.image_encoder.feature_dim

        if proprio_meta is not None and int(proprio_meta["length"]) > 0:    

            self.proprio_meta: DataMeta | None = construct_data_meta(proprio_meta)
            self.use_proprio: bool = self.proprio_meta is not None
            self.variance_feature_dim: int = (
                img_feature_dim * img_length + (0 if self.proprio_meta is None else int(np.prod(self.proprio_meta.shape)) * self.proprio_meta.length)
            )
        else:
            self.proprio_meta = None
            self.use_proprio = False
            self.variance_feature_dim = img_feature_dim * img_length

        # self.variance_feature_dim = img_feature_dim * img_length

        assert (
            self.image_encoder.feature_aggregation == "cls"
            or self.image_encoder.feature_aggregation == "map"
        ), f"Currently only support cls (for CLIP) or map (for SigLIP) aggregation"

        hidden_dims = [
            hidden_dim,
            hidden_dim // 2,
            hidden_dim // 4,
        ]


        self.mlp: nn.Module = nn.Sequential(
            nn.Linear(
                self.variance_feature_dim,
                hidden_dims[0],
            ),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dims[2], 1),
            nn.Sigmoid(),
        )

        self.device: torch.device = device
        
        self.loss_fn_name: str = loss_fn_name
        if loss_fn_name == "bce":
            self.loss_fn: nn.Module = nn.BCELoss()
        elif loss_fn_name == "mse":
            self.loss_fn: nn.Module = nn.MSELoss()
        else:
            raise ValueError(f"Invalid loss function: {loss_fn_name}. Only support bce (for binary cross entropy) or mse (for mean squared error) loss function.")

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"MemoryGate trainable parameters: {trainable_params}, total parameters: {total_params}"
        )
        if freeze_mlp:
            assert self.image_encoder.frozen, "image_encoder must be frozen when freeze_mlp is True"

            if ckpt_path is None or ckpt_path == "":
                print("Warning: No memory gate checkpoint path provided. Will not load weights.")
            else:
                ckpt = torch_load(ckpt_path, pickle_module=dill, weights_only=False)
                print(f"Loaded memory gate checkpoint from {ckpt_path}")

                state_dict = ckpt["model_state_dict"]
                self.load_state_dict(state_dict)
            for param in self.mlp.parameters():
                param.requires_grad = False
        self.cutoff_threshold: float = cutoff_threshold
        self.fixed_output_val: bool | None = fixed_output_val

    def to(self, device: torch.device, **kwargs):
        self.device = device
        super().to(device, **kwargs)
        return self

    def get_gate_value(
        self, batch: batch_type
    ) -> torch.Tensor:
        """
        batch:
            images: (batch_size, trajectory_len, 3, image_size, image_size)
            proprio: (batch_size, trajectory_len, proprio_dim)
        return: (batch_size, )
        """
        features = self.get_gate_feature(batch)
        gate = self.mlp(features).squeeze(-1)  # (batch_size,)
        # gate = (
        #     gate.squeeze(-1) ** 2
        # )  # (batch_size, ) # Square the gate to make it non-negative
        gate = torch.where(
            gate < self.cutoff_threshold, torch.zeros_like(gate), gate
        )
        if self.fixed_output_val is not None:
            gate = self.fixed_output_val * torch.ones_like(gate)
            
        return gate

    def get_gate_feature(
        self, batch: batch_type
    ) -> torch.Tensor:
        """
        batch:
            images: (batch_size, trajectory_len, 3, image_size, image_size)
            proprio: (batch_size, trajectory_len, proprio_dim)
        return: (batch_size, )
        """

        images = batch[self.image_encoder.image_meta.name]
        if self.proprio_meta is not None and self.proprio_meta.length > 0:
            proprio = batch[self.proprio_meta.name]
            batch_size, trajectory_len, *proprio_shape = proprio.shape
            assert tuple(proprio_shape) == self.proprio_meta.shape, f"proprio_shape: {proprio_shape}, should be {self.proprio_meta.shape}"
        else:
            proprio = None

        batch_size, trajectory_len, *image_shape = images.shape
        assert (
            tuple(image_shape) == self.image_encoder.image_meta.shape
        ), f"image_shape: {image_shape}, image_shape should be {self.image_encoder.image_meta.shape}"

        features = self.image_encoder(
            images
        )  # (batch_size, trajectory_len, 1, feature_dim)
        features = features.reshape(
            batch_size, -1
        )  # (batch_size, feature_dim * trajectory_len)
        if self.use_proprio:
            assert proprio is not None, "proprio is None when use_proprio is True"
            proprio = proprio.reshape(batch_size, -1)
            features = torch.cat([features, proprio], dim=-1)

        return features

    # Will not be trained separately.
    def compute_loss(
        self, batch: batch_type
    ) -> torch.Tensor:
        """
        batch:
            images: (batch_size, trajectory_len, 3, image_size, image_size)
            proprio: (batch_size, trajectory_len, proprio_dim)
            gate_label: (batch_size, 1)
        """
        gate_value = self.get_gate_value(batch)
        loss = self.loss_fn(gate_value, batch["gate_label"].squeeze())
        return loss

    def forward(
        self, batch: batch_type
    ) -> torch.Tensor:
        return self.compute_loss(batch)
