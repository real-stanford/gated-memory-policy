"""Code adapted from https://github.com/thu-ml/RoboticsDiffusionTransformer"""


import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import (Attention, Mlp, RmsNorm)

from imitation_learning.models.common.modules import Projector
from imitation_learning.models.common.pos_embeddings import (
    get_1d_sincos_pos_embed_from_grid, get_nd_sincos_pos_embed_from_grid)
from imitation_learning.models.denoising_networks.base_denoising_network import \
    BaseDenoisingNetwork
from imitation_learning.models.denoising_networks.modules import (
    CrossAttention, SinusoidalPosEmb)


class ConditionalTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, head_num: int):
        super().__init__()

        self.hidden_dim: int = hidden_dim
        self.head_num: int = head_num

        self.norm1: nn.Module = RmsNorm(hidden_dim, eps=1e-6)
        self.attn: Attention = Attention(
            dim=hidden_dim,
            num_heads=head_num,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
        )
        self.cross_attn: CrossAttention = CrossAttention(
            dim=hidden_dim,
            head_num=head_num,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
        )

        self.norm2 = RmsNorm(hidden_dim, eps=1e-6)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ff = Mlp(
            in_features=hidden_dim, hidden_features=hidden_dim, act_layer=nn.GELU
        )

        self.norm3 = RmsNorm(hidden_dim, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        global_cond: torch.Tensor,
        global_cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), global_cond, global_cond_mask)
        x = x + self.ff(self.norm3(x))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()

        self.norm = RmsNorm(hidden_dim, eps=1e-6)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.linear = Mlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim,
            out_features=output_dim,
            act_layer=approx_gelu,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class ConditionalTransformer(BaseDenoisingNetwork):
    def __init__(
        self,
        head_num: int,
        layer_num: int,
        hidden_dim: int,
        projector_type: str,
        global_cond_pos_emb_type: str="2d",
        **kwargs,
    ):
        if "name" not in kwargs:
            kwargs["name"] = "conditional_transformer"
        super().__init__(**kwargs)

        self.head_num: int = head_num
        self.layer_num: int = layer_num
        self.hidden_dim: int = hidden_dim
        self.projector_type: str = projector_type

        assert global_cond_pos_emb_type in ["1d", "2d"], f"global_cond_pos_emb_type must be 1d or 2d, but got {global_cond_pos_emb_type}"
        self.global_cond_pos_emb_type: str = global_cond_pos_emb_type

        self.mask_token: nn.Parameter = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )  # will replace the local condition or trajectory / time embedding

        self.diffusion_step_encoder: nn.Module = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.action_projector: nn.Module = Projector(
            projector_type, self.action_dim, hidden_dim
        )

        self.global_cond_projector: nn.Module = Projector(
            projector_type, self.global_cond_dim, hidden_dim
        )

        self.local_cond_projector: nn.Module = Projector(
            projector_type, self.local_cond_dim, hidden_dim
        )

        # Will be initialized by sin-cos embedding
        self.input_pos_embedding: nn.Parameter = nn.Parameter(
            torch.zeros(
                1,
                1
                + self.local_cond_token_num
                + self.action_token_num,
                hidden_dim,
            )  # +1 for the diffusion step
        )

        self.global_cond_pos_embedding: nn.Parameter = nn.Parameter(
            torch.zeros(1, self.global_cond_token_num, hidden_dim)
        )

        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                ConditionalTransformerBlock(hidden_dim, head_num)
                for _ in range(layer_num)
            ]
        )

        self.action_final_layer: FinalLayer = FinalLayer(hidden_dim, self.action_dim)


        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        input_pos_embedding = get_1d_sincos_pos_embed_from_grid(
            embed_dim=self.hidden_dim,
            pos=np.arange(
                1
                + self.local_cond_token_num
                + self.action_token_num
            ),
        )  # (local_cond_token_num + action_token_num + 1, hidden_dim)
        self.input_pos_embedding.data.copy_(
            torch.from_numpy(input_pos_embedding).unsqueeze(0)
        )

        if self.global_cond_pos_emb_type == "2d":
            for i in range(1, 10):  
                image_size = np.sqrt(self.global_cond_token_num // i)
                print(f"image_size: {image_size}")
                if not image_size.is_integer():
                    continue
                image_size = int(image_size)
                image_num = self.global_cond_token_num // (image_size * image_size)
                # Using image patches as tokens
                if self.global_cond_token_num > 1:
                    global_cond_pos_embedding = get_nd_sincos_pos_embed_from_grid(
                            grid_sizes=(image_num, image_size, image_size),
                            embed_dim=self.hidden_dim,
                        )  # (global_cond_token_num, hidden_dim)
                else:
                    global_cond_pos_embedding = np.random.randn(1, self.hidden_dim)
                break
            else:
                raise ValueError(f"Cannot find a valid image size for 2d position embedding. global_cond_token_num: {self.global_cond_token_num}")
                
        elif self.global_cond_pos_emb_type == "1d":
            # Using aggregated features as tokens
            global_cond_pos_embedding = get_1d_sincos_pos_embed_from_grid(
                embed_dim=self.hidden_dim,
                pos=np.arange(self.global_cond_token_num),
            )
        else:
            raise ValueError(f"global_cond_pos_emb_type must be 1d or 2d, but got {self.global_cond_pos_emb_type}")

        global_cond_pos_embedding = global_cond_pos_embedding.reshape(
            1, self.global_cond_token_num, self.hidden_dim
        )
        self.global_cond_pos_embedding.data.copy_(
            torch.from_numpy(global_cond_pos_embedding)
        )

        # # Initialize timestep and control freq embedding MLP
        # nn.init.normal_(self.diffusion_step_encoder[1].weight, std=0.02)
        # nn.init.normal_(self.diffusion_step_encoder[3].weight, std=0.02)

        # # Initialize the final layer: zero-out the final linear layer
        # nn.init.constant_(self.action_final_layer.ffn_final.fc2.weight, 0)
        # nn.init.constant_(self.action_final_layer.ffn_final.fc2.bias, 0)

    def _project_to_latent_space(
        self, data_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        This function will project the raw actions, local condition, and global condition to the hidden_dim space
        Will also append the diffusion step, local condition, action, and future img features to the input
        data_dict: {
            "noisy_action": (batch, action_token_num, action_dim),
            "step": (batch,),
            "global_cond": (batch, global_cond_token_num, cond_dim),
            "local_cond": (batch, local_cond_token_num, cond_dim),
            "global_cond_mask": (batch, global_cond_token_num), # optional, 0 means masked, 1 means visible
            "local_cond_mask": (batch, local_cond_token_num), # optional, 0 means masked, 1 means visible
            "noisy_traj_mask": (batch, action_token_num), # optional, 0 means masked, 1 means visible
            "step_mask": (batch,), # optional, 0 means masked, 1 means visible
        }
        input_dict: {
            "x": (batch, 1(diffusion_step) + local_cond_token_num + action_token_num, hidden_dim),
            "global_cond": (batch, global_cond_token_num, hidden_dim),
            "global_cond_mask": (batch, global_cond_token_num), # optional, 0 means masked, 1 means visible
        }
        """
        acceptable_keys = [
            "noisy_action",
            "step",
            "global_cond",
            "local_cond",
            "global_cond_mask",
            "local_cond_mask",
            "noisy_traj_mask",
            "step_mask",
        ]
        for key in data_dict.keys():
            assert key in acceptable_keys, f"key: {key} is not in acceptable_keys"

        trajectory = data_dict["noisy_action"]
        diffusion_step = data_dict["step"]
        global_cond = data_dict["global_cond"]
        if "local_cond" in data_dict:
            local_cond = data_dict["local_cond"]
        else:
            local_cond = None

        traj_len = trajectory.shape[1]

        x = self.action_projector(trajectory)  # (batch, action_token_num, hidden_dim)

        if "noisy_traj_mask" in data_dict:
            x[data_dict["noisy_traj_mask"] == 0] = self.mask_token.type_as(x)

        gc = self.global_cond_projector(
            global_cond
        )  # (batch, global_cond_token_num, hidden_dim)

        if local_cond is not None:
            lc = self.local_cond_projector(
                local_cond
            )  # (batch, local_cond_token_num, hidden_dim
            # Masking
            if "local_cond_mask" in data_dict:
                lc[data_dict["local_cond_mask"] == 0] = self.mask_token.type_as(lc)
            x = torch.cat(
                [lc, x], dim=1
            )  # (batch, local_cond_token_num + action_token_num, hidden_dim)

        t = self.diffusion_step_encoder(diffusion_step).unsqueeze(
            1
        )  # (batch, 1, hidden_dim) or (1, 1, hidden_dim)
        if t.shape[0] == 1:
            t = t.expand(trajectory.shape[0], -1, -1)  # (batch, 1, hidden_dim)

        if "step_mask" in data_dict:
            t[data_dict["step_mask"] == 0] = self.mask_token.type_as(t)

        x = torch.cat([t, x], dim=1)  # (batch, traj_length + 1, hidden_dim)

        x = x + self.input_pos_embedding
        gc = gc + self.global_cond_pos_embedding

        input_dict = {
            "x": x,
            "global_cond": gc,
        }

        if "global_cond_mask" in data_dict:
            input_dict["global_cond_mask"] = data_dict["global_cond_mask"]

        return input_dict

    def _run_blocks(self, input_dict: dict[str, torch.Tensor]):
        for i, block in enumerate(self.blocks):
            input_dict["x"] = block(**input_dict)
        return input_dict

    def _get_results(self, input_dict: dict[str, torch.Tensor]):

        token_num = input_dict["x"].shape[1]

        action_token_start_idx = (
            token_num - self.action_token_num
        )
        action_token_end_idx = token_num
        action_tokens = input_dict["x"][
            :, action_token_start_idx:action_token_end_idx, :
        ]

        action = self.action_final_layer.forward(
            action_tokens
        )  # (batch, traj_length, action_dim)

        result: dict[str, torch.Tensor] = {
            "action": action,
        }

        return result

    def forward(
        self,
        data_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        input_dict = self._project_to_latent_space(data_dict)
        input_dict = self._run_blocks(input_dict)
        return self._get_results(input_dict)
