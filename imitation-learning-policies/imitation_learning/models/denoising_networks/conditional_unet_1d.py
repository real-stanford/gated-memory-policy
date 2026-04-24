"""Addapted from diffusion policy codebase (https://github.com/real-stanford/diffusion_policy)"""

import einops
import torch
import torch.nn as nn

from imitation_learning.models.denoising_networks.base_denoising_network import \
    BaseDenoisingNetwork
from imitation_learning.models.denoising_networks.modules import (
    Conv1dBlock, Downsample1d, Rearrange, SinusoidalPosEmb, Upsample1d)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int,
        group_num: int,
        cond_predict_scale: bool,
    ):
        super().__init__()

        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                Conv1dBlock(
                    in_channels, out_channels, kernel_size, group_num=group_num
                ),
                Conv1dBlock(
                    out_channels, out_channels, kernel_size, group_num=group_num
                ),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder: nn.Module = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(BaseDenoisingNetwork):
    def __init__(
        self,
        diffusion_step_embed_dim: int,
        down_sample_channel_nums: list[int],
        kernel_size: int,
        group_num: int,
        cond_predict_scale: bool,
        **kwargs,
    ):
        kwargs["name"] = "conditional_unet_1d"
        super().__init__(**kwargs)

        self.diffusion_step_encoder: nn.Module = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        self.cond_dim: int = self.global_cond_dim + diffusion_step_embed_dim

        self.down_modules: nn.ModuleList = nn.ModuleList()

        input_channel_num = (
            self.action_dim
        )  # each action dimension (x, y, z, ...) is regarded as a separate channel

        all_channel_nums = [input_channel_num] + list(down_sample_channel_nums)
        for idx, (channel_in, channel_out) in enumerate(
            zip(all_channel_nums[:-1], all_channel_nums[1:])
        ):
            _ = self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            channel_in,
                            channel_out,
                            cond_dim=self.cond_dim,
                            kernel_size=kernel_size,
                            group_num=group_num,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            channel_out,
                            channel_out,
                            cond_dim=self.cond_dim,
                            kernel_size=kernel_size,
                            group_num=group_num,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Downsample1d(channel_out),
                    ]
                )
            )

        self.mid_modules: nn.ModuleList = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    all_channel_nums[-1],
                    all_channel_nums[-1],
                    cond_dim=self.cond_dim,
                    kernel_size=kernel_size,
                    group_num=group_num,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    all_channel_nums[-1],
                    all_channel_nums[-1],
                    cond_dim=self.cond_dim,
                    kernel_size=kernel_size,
                    group_num=group_num,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )

        self.up_modules: nn.ModuleList = nn.ModuleList()
        for idx, (channel_in, channel_out) in enumerate(
            zip(
                reversed(all_channel_nums[1:]),
                reversed(all_channel_nums[:-1]),
            )
        ):
            _ = self.up_modules.append(
                nn.ModuleList(
                    [
                        Upsample1d(channel_in),
                        ConditionalResidualBlock1D(
                            channel_in * 2,
                            channel_out,
                            cond_dim=self.cond_dim,
                            kernel_size=kernel_size,
                            group_num=group_num,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            channel_out,
                            channel_out,
                            cond_dim=self.cond_dim,
                            kernel_size=kernel_size,
                            group_num=group_num,
                            cond_predict_scale=cond_predict_scale,
                        ),
                    ]
                )
            )

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"ConditionalUnet1D trainable parameters: {trainable_params}")

    def forward(
        self,
        data_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        data_dict: {
            "noisy_action": (B, traj_length, action_dim),
            "step": (B,),
            "global_cond": (B, global_cond_token_num, cond_dim),
        """
        trajectory = data_dict["noisy_action"]
        diffusion_step = data_dict["step"]
        global_cond = data_dict["global_cond"]

        step_feature = self.diffusion_step_encoder(diffusion_step)
        if len(global_cond.shape) == 3:
            assert (
                global_cond.shape[1] == 1
            ), "cond_token_num should be 1 for ConditionalUnet1D"
            global_cond = global_cond[:, 0, :]
        global_feature = torch.cat([step_feature, global_cond], dim=-1)

        h: list[torch.Tensor] = []
        x = einops.rearrange(
            trajectory,
            "batch_size traj_length action_dim -> batch_size action_dim traj_length",
        )
        # 1d convolution should be applied to the last (trajectory) dimension
        # each action dimension is regarded as a separate channel

        for module in self.down_modules:
            assert isinstance(module, nn.ModuleList)
            resnet, resnet2, downsample = module
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            assert isinstance(mid_module, nn.Module)
            x = mid_module(x, global_feature)

        for up_module in self.up_modules:
            assert isinstance(up_module, nn.ModuleList)
            upsample, resnet, resnet2 = up_module
            x = upsample(x)  # (B, C_in, T) -> (B, C_in, 2T)
            x = torch.cat((x, h.pop()), dim=1)  # (B, C_in, 2T) -> (B, 2C_in, 2T)
            x = resnet(x, global_feature)  # (B, 2C_in, 2T) -> (B, C_out, 2T)
            x = resnet2(x, global_feature)  # (B, C_out, 2T) -> (B, C_out, 2T)

        trajectory = einops.rearrange(
            x, "batch_size action_dim traj_length -> batch_size traj_length action_dim"
        )
        return trajectory  # (batch_size, traj_length, action_dim)
