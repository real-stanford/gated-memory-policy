
import torch
import torch.nn as nn

from imitation_learning.common.datatypes import batch_type
from imitation_learning.models.decoders.base_action_decoder import \
    BaseActionDecoder
from imitation_learning.models.encoders.base_cond_encoder import BaseEncoder


class BasePolicy(nn.Module):
    def __init__(
        self,
        global_cond_encoder: BaseEncoder,
        local_cond_encoder: BaseEncoder | None,
        action_decoder: BaseActionDecoder,
        device: torch.device,
        seed: int,
        **kwargs,
    ):
        super().__init__()
        self.global_cond_encoder: BaseEncoder = global_cond_encoder
        # If local_cond_encoder has no data entries (e.g., proprio disabled), treat it as None
        if local_cond_encoder is not None and len(local_cond_encoder.cond_meta) == 0:
            print("Warning: local_cond_encoder has no data entries, setting to None")
            local_cond_encoder = None
        self.local_cond_encoder: BaseEncoder | None = local_cond_encoder
        self.action_decoder: BaseActionDecoder = action_decoder
        self.device: torch.device = device
        self.seed: int = seed
        torch.manual_seed(seed)
        self.torch_rng: torch.Generator = torch.Generator(device=device)
        print(f"BasePolicy unused kwargs: {kwargs}")

    def to(self, device: torch.device, **kwargs):
        self.device = device
        self.torch_rng = torch.Generator(device=device).manual_seed(self.seed)
        super().to(device, **kwargs)
        return self

    def predict_action(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """
        # Example input:
        normalized_batch:
            "robot0_wrist_camera": (batch_size, data_length, 3, image_size, image_size)
            "robot0_10d": (batch_size, data_length, 8)
        return:
            "action0_10d": (batch_size, data_length, 8)
            "future_img_features": (batch_size, data_length, feature_dim) # Optional
        """
        raise NotImplementedError()

    def compute_loss(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        raise NotImplementedError()

    def forward(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """To be compatible with DataDistributedParallel"""
        return self.compute_loss(normalized_batch)

    def reset(self):
        raise NotImplementedError()

    def reset_rng(self):
        self.torch_rng = torch.Generator(device=self.device).manual_seed(self.seed)
