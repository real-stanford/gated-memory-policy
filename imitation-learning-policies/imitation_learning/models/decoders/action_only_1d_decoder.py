
import torch

from imitation_learning.models.decoders.base_action_decoder import \
    BaseActionDecoder
from imitation_learning.models.decoders.low_dim_decoders import \
    LowDimIdentityDecoder


class ActionOnly1DDecoder(BaseActionDecoder):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(name="action_only_1d_decoder", **kwargs)
        self.traj_len: int = next(iter(self.action_meta.values())).length
        self.token_num: int = self.traj_len
        self.latent_dim: int = sum(meta.shape[0] for meta in self.action_meta.values())
        self.data_shape: tuple[int, ...] = (
            self.traj_len,
            self.latent_dim,
        )

        for meta in self.action_meta.values():
            assert len(meta.shape) == 1, f"Action {meta.name} is not 1D: {meta.shape}"
            assert (
                meta.length == self.traj_len
            ), f"Action trajectory length is inconsistent: {meta.length} ({meta.name}) vs {self.traj_len} ({meta.name})"
            if meta.data_type == "low_dim":
                decoder = LowDimIdentityDecoder(meta)
                self.decoder_dict[meta.name] = decoder
                assert (
                    len(meta.shape) == 1
                ), f"LowDim data entry {meta.name} is not 1D: {meta.shape}"
                assert (
                    decoder.latent_dim == meta.shape[0]
                ), f"Decoder {meta.name} latent dimension is inconsistent with action shape: {decoder.latent_dim} vs {meta.shape[0]}"
            else:
                raise ValueError(f"Unknown action type: {meta.data_type}")

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        latent: (batch_size, traj_len, action_dim) or (batch_size, traj_num, traj_len, action_dim)
        """
        if latent.ndim == 3:
            start_idx = 0
            assert latent.shape[1:] == (
                self.traj_len,
                self.latent_dim,
            ), f"latent.shape[1:]: {latent.shape[1:]} is not equal to {(self.traj_len, self.latent_dim)}"
            action_dict: dict[str, torch.Tensor] = {}
            for meta in self.action_meta.values():
                dim = meta.shape[0]
                decoded_action = self.decoder_dict[meta.name](
                    latent[:, :, start_idx : start_idx + dim]
                )
                action_dict[meta.name] = decoded_action
                start_idx += dim

        elif latent.ndim == 4:
            start_idx = 0
            assert latent.shape[2:] == (
                self.traj_len,
                self.latent_dim,
            ), f"latent.shape[2:]: {latent.shape[2:]} is not equal to {(self.traj_len, self.latent_dim)}"
            action_dict: dict[str, torch.Tensor] = {}
            for meta in self.action_meta.values():
                dim = meta.shape[0]
                decoded_action = self.decoder_dict[meta.name](
                    latent[:, :, :, start_idx : start_idx + dim]
                )
                action_dict[meta.name] = decoded_action
                start_idx += dim
        else:
            raise ValueError(f"Unknown latent shape: {latent.shape}")
        return action_dict

    def encode(self, action_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        action = torch.cat(
            [action_dict[meta.name] for meta in self.action_meta.values()], dim=-1
        )  # (batch_size, traj_len, action_dim)
        assert action.shape[1:] == (
            self.traj_len,
            self.latent_dim,
        ), f"action.shape: {action.shape} is not equal to {(self.traj_len, self.latent_dim)}"
        return action
