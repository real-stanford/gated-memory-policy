import copy
import einops
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, RmsNorm

from imitation_learning.models.common.modules import Projector
from imitation_learning.models.common.pos_embeddings import \
    get_1d_sincos_pos_embed_from_grid
from imitation_learning.models.denoising_networks.conditional_transformer import (
    ConditionalTransformer, ConditionalTransformerBlock)
from imitation_learning.models.denoising_networks.modules import \
    HistoryCrossAttention


class BinaryGatingSTEv1(torch.autograd.Function):
    """
    Straight-through estimator for binary gating.
    Backward assumes y = x * gate
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, gate)
        # Return 1.0 if x > 0.5, else 0.0
        return x * (gate > 0.5).float()


    @staticmethod
    def backward(ctx, grad_outputs):
        # Identity gradient: let the MLP think it's continuous
        x, gate = ctx.saved_tensors
        grad_x = grad_outputs * gate
        grad_gate = grad_outputs * x
        return grad_x, grad_gate


class BinaryGatingSTEv2(torch.autograd.Function):
    """
    Straight-through estimator for binary gating.
    Backward uses binary mask to compute gradient of x
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        mask = (gate > 0.5).float()
        ctx.save_for_backward(x, mask)
        # Return 1.0 if x > 0.5, else 0.0
        return x * mask


    @staticmethod
    def backward(ctx, grad_outputs):
        # Identity gradient: let the MLP think it's continuous
        x, mask = ctx.saved_tensors
        grad_x = grad_outputs * mask
        grad_gate = grad_outputs * x
        return grad_x, grad_gate

class BinaryGatingSTEv3(torch.autograd.Function):
    """
    Straight-through estimator for binary gating.
    Backward uses binary mask
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        mask = (gate > 0.5).float()
        ctx.save_for_backward(x, mask)
        return x * mask


    @staticmethod
    def backward(ctx, grad_outputs):
        # Identity gradient: let the MLP think it's continuous
        x, mask = ctx.saved_tensors
        grad_x = grad_outputs * mask
        grad_gate = grad_outputs
        return grad_x, grad_gate


class MemoryTransformerBlock(ConditionalTransformerBlock):
    def __init__(
        self,
        hidden_dim: int,
        head_num: int,
        history_attention_type: str,
        input_token_num: int,
        skip_history_attn: bool,  # For ablation study
        add_additional_self_attn: bool,
        ssmax_scaling_param: float | None,
        binary_gating: bool = True, # To be compatible with previous checkpoints. Should be overridden in the future configs
        straight_through: str = "", # Whether to use straight-through estimator for binary gating
    ):
        super().__init__(hidden_dim, head_num)

        self.binary_gating: bool = binary_gating

        self.history_cross_attn: HistoryCrossAttention = HistoryCrossAttention(
            dim=hidden_dim,
            head_num=head_num,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
            attention_type=history_attention_type,
            ssmax_scaling_param=ssmax_scaling_param,
        )
        self.history_norm1: nn.Module = RmsNorm(hidden_dim, eps=1e-6)
        self.history_norm2: nn.Module = RmsNorm(hidden_dim, eps=1e-6)
        self.add_additional_self_attn:bool = add_additional_self_attn
        if self.add_additional_self_attn:
            self.history_norm3: nn.Module = RmsNorm(hidden_dim, eps=1e-6)
            self.history_self_attn: Attention = Attention(
                dim=hidden_dim,
                num_heads=head_num,
                qkv_bias=True,
                qk_norm=True,
                norm_layer=RmsNorm,
            )
        self.input_token_num: int = input_token_num
        self.straight_through: str = straight_through
        if self.straight_through != "":
            assert self.binary_gating, "Straight-through estimator for binary gating is only supported when binary gating is enabled"

        self.skip_history_attn: bool = skip_history_attn

    def forward(
        self,
        x: torch.Tensor,
        global_cond: torch.Tensor,
        global_cond_mask: torch.Tensor | None = None,
        history_latents: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        memory_gate_val: torch.Tensor | None = None,
        step: torch.Tensor | None = None,
        record_data_dict: dict[str, list[torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """
        x: (batch_size, token_num, hidden_dim)
        global_cond: (batch_size, cond_token_num, hidden_dim)
        global_cond_mask: (batch_size, cond_token_num)
        history_latents: (batch_size, history_len, action_token_num, hidden_dim)
        history_mask: (batch_size, history_len)
        memory_gate_val: (batch_size,)
        step: (batch_size,)
        """
        assert (
            x.shape[1] == self.input_token_num
        ), f"x.shape[1] ({x.shape[1]}) must be equal to input_token_num ({self.input_token_num})"
        attn: torch.Tensor = self.attn(self.norm1(x))
        x = x + attn
        cross_attn: torch.Tensor = self.cross_attn(
            self.norm2(x), global_cond, global_cond_mask
        )
        x = x + cross_attn

        # print(f"{memory_gate_val=}")

        if history_latents is not None and not self.skip_history_attn:
            # TODO: automatically skip history calculation if memory gate val is 0

            history_attention: torch.Tensor = self.history_cross_attn(
                self.history_norm1(x),
                history_latents,
                history_mask,
                record_data_dict,
            )  # (batch_size, token_num, hidden_dim)

            if memory_gate_val is not None:
                # history_attention = history_attention * memory_gate_val[:, None, None]
                if self.binary_gating:
                    if self.straight_through == "v1":
                        # Only this version works for the memory gate training
                        gated_history_attention = BinaryGatingSTEv1.apply(history_attention, memory_gate_val[:, None, None])
                    elif self.straight_through == "v2":
                        gated_history_attention = BinaryGatingSTEv2.apply(history_attention, memory_gate_val[:, None, None])
                    elif self.straight_through == "v3":
                        gated_history_attention = BinaryGatingSTEv3.apply(history_attention, memory_gate_val[:, None, None])
                    else:
                        gated_history_attention = history_attention * (memory_gate_val[:, None, None] > 0.5).float()
                    assert isinstance(gated_history_attention, torch.Tensor)
                    assert gated_history_attention.shape == history_attention.shape
                    history_attention = gated_history_attention
                else:
                    history_attention = history_attention * memory_gate_val[:, None, None]
                    
                # if record_data_dict is not None and "memory_gate_val" in record_data_dict:
                #     record_data_dict["memory_gate_val"].append(memory_gate_val.clone()) # Should not be detached for regularization

            x = x + history_attention
            
            # if memory_gate_val is None:
            #     history_attention: torch.Tensor = self.history_cross_attn(
            #         self.history_norm1(x),
            #         history_latents,
            #         history_mask,
            #         record_data_dict,
            #     )  # (batch_size, token_num, hidden_dim)
            #     x = x + history_attention

            # else:
            #     # history_attention = history_attention * memory_gate_val[:, None, None]
            #     if self.binary_gating:

            #         # In training, calculate history attention normally for gate training 
            #         if torch.torch.is_grad_enabled() and self.straight_through != "":
            #             history_attention: torch.Tensor = self.history_cross_attn(
            #                 self.history_norm1(x),
            #                 history_latents,
            #                 history_mask,
            #                 record_data_dict,
            #             )  # (batch_size, token_num, hidden_dim)
            #             if self.straight_through == "v1":
            #                 # Only this version works for the memory gate training
            #                 gated_history_attention = BinaryGatingSTEv1.apply(history_attention, memory_gate_val[:, None, None])
            #             elif self.straight_through == "v2":
            #                 gated_history_attention = BinaryGatingSTEv2.apply(history_attention, memory_gate_val[:, None, None])
            #             elif self.straight_through == "v3":
            #                 gated_history_attention = BinaryGatingSTEv3.apply(history_attention, memory_gate_val[:, None, None])
            #             else:
            #                 gated_history_attention = history_attention * (memory_gate_val[:, None, None] > 0.5).float()
            #             assert isinstance(gated_history_attention, torch.Tensor)
            #             assert gated_history_attention.shape == history_attention.shape
            #             history_attention = gated_history_attention
            #             x = x + history_attention
            #         else:
            #             # In inference mode, skip history attention calculation if not necessary
            #             binarized_memory_gate_val = (memory_gate_val > 0.5).bool()
            #             # if sum(binarized_memory_gate_val) > 0:
            #             #     gated_history_attention: torch.Tensor = self.history_cross_attn(
            #             #         self.history_norm1(x[binarized_memory_gate_val]),
            #             #         history_latents[binarized_memory_gate_val],
            #             #         history_mask[binarized_memory_gate_val] if history_mask is not None else None,
            #             #         record_data_dict,
            #             #     )
            #             #     x = x + gated_history_attention
            #             # else:
            #             #     # all memory gate values are 0, skip history attention calculation
            #             #     pass

            #     else:
            #         history_attention: torch.Tensor = self.history_cross_attn(
            #             self.history_norm1(x),
            #             history_latents,
            #             history_mask,
            #             record_data_dict,
            #         )  # (batch_size, token_num, hidden_dim)

            #         history_attention = history_attention * memory_gate_val[:, None, None]
            #         x = x + history_attention
                    
            #     if record_data_dict is not None and "memory_gate_val" in record_data_dict:
            #         record_data_dict["memory_gate_val"].append(memory_gate_val.clone()) # Should not be detached for regularization


        if self.add_additional_self_attn:
            x = x + self.history_self_attn(self.history_norm3(x))

        x = x + self.ff(self.norm3(x))

        return x


class MemoryTransformer(ConditionalTransformer):
    def __init__(
        self,
        max_history_len: int,
        freeze_non_history_modules: bool,
        history_attention_type: str,
        record_data_entries: list[str],  # For debugging
        ssmax_scaling_param: float | None,
        include_action_history: bool,
        history_action_num_per_chunk: int,
        skip_history_attn: bool,  # For ablation study
        add_memory_gate_token: bool = False, # For compatibility with previous checkpoints
        binary_gating: bool = True, # For compatibility with previous checkpoints
        straight_through: str = "", # "v1", "v2", "v3", or ""
        add_additional_self_attn: bool = True,
        history_img_features_dim: int = 0,
        history_img_features_token_num: int = 0,
        **kwargs,
    ):
        kwargs["name"] = "memory_transformer"

        super().__init__(**kwargs)

        self.binary_gating: bool = binary_gating
        self.skip_history_attn: bool = skip_history_attn

        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                MemoryTransformerBlock(
                    hidden_dim=self.hidden_dim,
                    head_num=self.head_num,
                    history_attention_type=history_attention_type,
                    input_token_num=self.input_pos_embedding.shape[1] + (1 if add_memory_gate_token else 0),
                    skip_history_attn=skip_history_attn,
                    add_additional_self_attn=add_additional_self_attn,
                    ssmax_scaling_param=ssmax_scaling_param,
                    binary_gating=binary_gating,
                    straight_through=straight_through,
                )
                for _ in range(self.layer_num)
            ]
        )

        self.max_history_len: int = max_history_len

        self.record_data_entries: list[str] = record_data_entries
        self.recorded_data_dict: dict[str, list[torch.Tensor]] = {}

        self.history_time_embedding: nn.Parameter = nn.Parameter(
            torch.zeros(
                1, max_history_len, self.hidden_dim
            )  # (batch, max_history_len, hidden_dim)
        )

        assert history_action_num_per_chunk > 0, "history_action_num_per_chunk must be greater than 0"
        assert history_action_num_per_chunk <= self.action_token_num, "history_action_num_per_chunk must be less than or equal to action_token_num"

        self.history_action_num_per_chunk: int = history_action_num_per_chunk

        self.initialize_memory_weights()

        if freeze_non_history_modules:
            for name, param in self.named_parameters():
                if "history" not in name:
                    param.requires_grad = False

        self.history_img_features_dim: int = history_img_features_dim
        self.history_img_features_token_num: int = history_img_features_token_num
        print(
            f"history_img_features_dim: {self.history_img_features_dim}, history_img_features_token_num: {self.history_img_features_token_num}"
        )
        self.history_img_features_projector: nn.Module | None = None
        if (
            self.history_img_features_dim > 0
            and self.history_img_features_token_num > 0
        ):
            self.history_img_features_projector = Projector(
                self.projector_type, self.history_img_features_dim, self.hidden_dim
            )

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"MemoryTransformer trainable parameters: {trainable_params}, total parameters: {total_params}"
        )
        self.include_action_history: bool = include_action_history
        assert (
            self.include_action_history or self.history_img_features_dim > 0
        ), "At least one of include_action_history or history_img_features_dim must be True"
        
        if add_memory_gate_token:
            assert binary_gating, "Memory gate token is only supported when binary gating is enabled"
            self.memory_gate_tokens: nn.Parameter | None = nn.Parameter(
                torch.randn(2, self.hidden_dim)
            ) # Two tokens: [0] when memory gate == 0, [1] when memory gate == 1
        else:
            self.memory_gate_tokens = None

    def set_skip_history_attn(self, skip_history_attn: bool):
        self.skip_history_attn = skip_history_attn
        for block in self.blocks:
            assert isinstance(block, MemoryTransformerBlock)
            block.skip_history_attn = skip_history_attn

    def initialize_memory_weights(self):
        # Initialize transformer
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.blocks.apply(_basic_init)

        history_time_embedding = get_1d_sincos_pos_embed_from_grid(
            embed_dim=self.hidden_dim,
            pos=np.arange(self.max_history_len),
        )  # (max_history_len, hidden_dim)
        self.history_time_embedding.data.copy_(
            torch.from_numpy(history_time_embedding).unsqueeze(0)
        )  # (1, max_history_len, hidden_dim)

    def _project_to_latent_space(
        self, data_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        This function will manage all the history related data obtain history latents
        Will also call the super class's _process_data_dict to process the rest of the data
        data_dict: {
            "noisy_action": (batch, traj_length, action_dim),
            "step": (batch,),
            "global_cond": (batch, global_cond_token_num, global_cond_dim),
            "local_cond": (batch, local_cond_token_num, local_cond_dim),
            "global_cond_mask": (batch, global_cond_token_num),
            "local_cond_mask": (batch, local_cond_token_num),
            "history_noisy_actions": (batch, history_len, history_action_num_per_chunk, action_dim),
            "history_mask": (batch, history_len),
            "history_img_features": (batch, history_len, history_img_features_token_num, history_img_features_dim),
            "memory_gate_val": (batch,),
        }

        input_dict: {
            "x": (batch, 1(memory gate token, optional) + 1(denoising step) + local_cond_token_num + action_token_num, hidden_dim),
            "global_cond": (batch, global_cond_token_num, hidden_dim),
            "global_cond_mask": (batch, global_cond_token_num), # optional
            "history_latents": (batch, history_len, action_token_num, hidden_dim),
            "history_mask": (batch, history_len), # optional
            "memory_gate_val": (batch,), # optional
            "step": (batch,),
        }
        """
        if "history_noisy_actions" in data_dict:
            history_noisy_actions = data_dict.pop("history_noisy_actions")

            history_len = history_noisy_actions.shape[1]
            assert (
                history_len <= self.max_history_len
            ), "history_len must be less than or equal to max_history_len"
            # Apply relative time embeddings to the history latents
            history_latents_list: list[torch.Tensor] = []
            if self.include_action_history:
                action_latents: torch.Tensor = self.action_projector(
                    history_noisy_actions
                )  # (batch, history_len, history_action_num_per_chunk, hidden_dim)
                start_idx = (
                    self.input_pos_embedding.shape[1]
                    - self.action_token_num
                )
                end_idx = (
                    self.input_pos_embedding.shape[1]
                    - self.action_token_num
                    + self.history_action_num_per_chunk
                )
                action_latents = (
                    action_latents
                    + self.input_pos_embedding[:, None, start_idx:end_idx, :]
                )
                history_latents_list.append(action_latents)

            if self.history_img_features_projector is not None and self.max_history_len > 0:
                history_img_features = data_dict.pop("history_img_features")
                # print(f"{torch.allclose(history_img_features, history_img_features[0])=}")
                history_img_features_latent = self.history_img_features_projector(
                    history_img_features
                )  # (batch, history_len, history_img_features_token_num, hidden_dim)
                history_latents_list.append(history_img_features_latent)
                # (batch, history_len, action_token_num + history_img_features_token_num, hidden_dim)

            history_latents = torch.cat(history_latents_list, dim=2)
            history_latents = (
                history_latents + self.history_time_embedding[:, -history_len:, None, :]
            )
            input_dict = {}
            if "history_mask" in data_dict:
                input_dict["history_mask"] = data_dict.pop("history_mask")
            if "memory_gate_val" in data_dict:
                input_dict["memory_gate_val"] = data_dict.pop("memory_gate_val")
            input_dict["step"] = data_dict["step"]
            input_dict["history_latents"] = history_latents
            input_dict.update(super()._project_to_latent_space(data_dict))

            if self.memory_gate_tokens is not None:
                assert "memory_gate_val" in input_dict, "memory_gate_val must be in input_dict when memory gate tokens are used"
                binary_gate_val = (input_dict["memory_gate_val"] > 0.5).int()
                batch_memory_gate_tokens = self.memory_gate_tokens[binary_gate_val, None, :] # (batch, 1, hidden_dim)
                input_dict["x"] = torch.cat([batch_memory_gate_tokens, input_dict["x"]], dim=1)
                # print(f"{input_dict['x'].shape=}")

            return input_dict

        else:
            input_dict = super()._project_to_latent_space(data_dict)

        return input_dict

    def _project_to_latent_space_multi_traj(
        self, data_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Will generate history latents from noisy actions
        Will also call the super class's _process_data_dict to process the rest of the data
        This function is only called during training when multiple supervised trajectories are provided at the same time
        data_dict: {
            # Same as no-history version
            "noisy_action": (batch, traj_num, data_length, action_dim),
            "step": (batch,),
            "global_cond": (batch, traj_num, global_cond_token_num, global_cond_dim),
            "local_cond": (batch, traj_num, local_cond_token_num, local_cond_dim),
            "global_cond_mask": (batch, traj_num, global_cond_token_num),
            "local_cond_mask": (batch, traj_num, local_cond_token_num),
            "noisy_traj_mask": (batch, traj_num, data_length),
            "step_mask": (batch, traj_num),
            # History-specific
            "history_noisy_actions": (batch, traj_num, history_action_num_per_chunk, action_dim), # Optional
            "history_img_features": (batch, traj_num, data_length, feature_dim), # Optional
            "history_mask": (batch, traj_num),
            "memory_gate_val": (batch, traj_num), # Optional
            "training_traj_indices": (batch, max_training_traj_num), # Optional
        }

        return: {
            "x": (batch*traj_num, local_cond_token_num + action_token_num + 1(denoising step) + 1(memory gate token, optional), hidden_dim),
            "global_cond": (batch*traj_num, global_cond_token_num, hidden_dim),
            "global_cond_mask": (batch*traj_num, global_cond_token_num),

            "step": (batch*traj_num,),
            "history_latents": (batch*traj_num, max_history_len, token_num, hidden_dim),
            "history_mask": (batch*traj_num, max_history_len),
            "memory_gate_val": (batch*traj_num,),
        }
        if self.max_training_traj_num > 0, traj_num will be replaced by self.max_training_traj_num if exceeds
        """

        traj_num = data_dict["noisy_action"].shape[1]
        batch_size = data_dict["noisy_action"].shape[0]
        device = data_dict["noisy_action"].device

        assert (
            traj_num >= self.max_history_len + 1
        ), f"To make sure all the history time embeddings are trained,traj_num ({traj_num}) must be at least equal to max_history_len + 1 ({self.max_history_len + 1}) in each training step"
        assert (
            traj_num == data_dict["history_noisy_actions"].shape[1]
        ), f"history_noisy_actions ({data_dict['history_noisy_actions'].shape}) must have the same trajectory number (dimension 1) as noisy_action ({traj_num})"

        # Step 1: Project history features on ALL trajectories (needed for history window construction)
        history_latents_list: list[torch.Tensor] = []

        if self.include_action_history:
            projected_history_actions = self.action_projector(
                data_dict["history_noisy_actions"]
            )  # (batch, traj_num, history_action_num_per_chunk, hidden_dim)
            start_idx = (
                self.input_pos_embedding.shape[1]
                - self.action_token_num
            )
            end_idx = (
                self.input_pos_embedding.shape[1]
                - self.action_token_num
                + self.history_action_num_per_chunk
            )
            projected_history_actions = (
                projected_history_actions
                + self.input_pos_embedding[:, None, start_idx:end_idx, :]
            )
            history_latents_list.append(projected_history_actions)

        if self.history_img_features_projector is not None:
            projected_history_img_features_latent = self.history_img_features_projector(
                data_dict["history_img_features"]
            )  # (batch, traj_num, traj_length, hidden_dim)
            history_latents_list.append(projected_history_img_features_latent)

        history_latents = torch.cat(history_latents_list, dim=2)
        # (batch, traj_num, token_num, hidden_dim)

        # Step 2: Determine effective trajectories and sample non-history tensors early
        has_training_traj_indices = "training_traj_indices" in data_dict
        if has_training_traj_indices:
            traj_indices = data_dict["training_traj_indices"]  # (batch, effective_traj_num)
            effective_traj_num = traj_indices.shape[1]
        else:
            traj_indices = torch.arange(traj_num, device=device).unsqueeze(0).expand(batch_size, -1)
            effective_traj_num = traj_num

        reshaped_data_dict: dict[str, torch.Tensor] = {}
        non_history_keys = [
            "noisy_action",
            "global_cond",
            "local_cond",
            "global_cond_mask",
            "local_cond_mask",
            "noisy_traj_mask",
            "step_mask",
        ]
        for key in non_history_keys:
            if key in data_dict:
                tensor = data_dict[key]
                if has_training_traj_indices:
                    batch_idx_2d = torch.arange(batch_size, device=device)[:, None]
                    tensor = tensor[batch_idx_2d, traj_indices]
                reshaped_data_dict[key] = einops.rearrange(
                    tensor, "batch traj_num ... -> (batch traj_num) ..."
                )

        reshaped_data_dict["step"] = (
            data_dict["step"].repeat(1, effective_traj_num).view(-1)
        )

        # Step 3: Project non-history tensors (only effective trajectories go through the projectors)
        input_dict = super()._project_to_latent_space(reshaped_data_dict)
        input_dict["step"] = reshaped_data_dict["step"]

        # Step 4: Construct history windows via vectorized gather (replaces Python for-loop)
        # For trajectory index i, history window position k (0..max_history_len-1)
        # maps to source index: i - max_history_len + k
        window_offsets = torch.arange(self.max_history_len, device=device)
        # source_indices[b, j, k] = traj_indices[b, j] - max_history_len + k
        source_indices = traj_indices[:, :, None] - self.max_history_len + window_offsets[None, None, :]
        # (batch, effective_traj_num, max_history_len)
        valid_mask = source_indices >= 0
        source_indices_clamped = source_indices.clamp(min=0)

        batch_idx_3d = torch.arange(batch_size, device=device)[:, None, None]
        merged_history_latents = history_latents[batch_idx_3d, source_indices_clamped]
        # (batch, effective_traj_num, max_history_len, token_num, hidden_dim)
        merged_history_latents = merged_history_latents * valid_mask[:, :, :, None, None]

        if "history_mask" in data_dict:
            full_history_mask = data_dict["history_mask"]  # (batch, traj_num)
            merged_history_masks = full_history_mask[batch_idx_3d, source_indices_clamped]
            # (batch, effective_traj_num, max_history_len)
            merged_history_masks = merged_history_masks & valid_mask
        else:
            merged_history_masks = valid_mask

        input_dict["history_latents"] = einops.rearrange(
            merged_history_latents, "batch traj_num ... -> (batch traj_num) ... "
        )  # (batch*effective_traj_num, max_history_len, token_num, hidden_dim)

        input_dict["history_latents"] = (
            input_dict["history_latents"] + self.history_time_embedding[:, :, None, :]
        )
        input_dict["history_mask"] = einops.rearrange(
            merged_history_masks, "batch traj_num ... -> (batch traj_num) ... "
        )

        # Step 5: Memory gate (sample if needed)
        if "memory_gate_val" in data_dict:
            memory_gate_val = data_dict["memory_gate_val"]
            if has_training_traj_indices:
                batch_idx_2d = torch.arange(batch_size, device=device)[:, None]
                memory_gate_val = memory_gate_val[batch_idx_2d, traj_indices]
            input_dict["memory_gate_val"] = einops.rearrange(
                memory_gate_val, "batch traj_num ... -> (batch traj_num) ..."
            )
        if self.memory_gate_tokens is not None:
            assert "memory_gate_val" in input_dict, "memory_gate_val must be in input_dict when memory gate tokens are used"
            binary_gate_val = (input_dict["memory_gate_val"] > 0.5).int()
            batch_memory_gate_tokens = self.memory_gate_tokens[binary_gate_val, None, :] # (batch, 1, hidden_dim)
            input_dict["x"] = torch.cat([batch_memory_gate_tokens, input_dict["x"]], dim=1)

        return input_dict

    def parallel_forward(
        self, data_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Only used when training with multiple ground-truth trajectories
        data_dict: {
            "noisy_action": (batch, traj_num, traj_length, action_dim),
            "step": (batch,),
            "global_cond": (batch, traj_num, global_cond_token_num, global_cond_dim),
            "local_cond": (batch, traj_num, local_cond_token_num, local_cond_dim),
            "history_noisy_actions": (batch, traj_num, history_action_num_per_chunk, action_dim),
            "history_mask": (batch, traj_num),
            "memory_gate_val": (batch, traj_num),
            "training_traj_indices": (batch, max_training_traj_num), # Optional, if provided, will only use the indexed trajectories for training
        }
        return: {
            "action": (batch, traj_num, traj_length, action_dim),
        }
        if "training_traj_indices" in data_dict, traj_num will be replaced by max_training_traj_num
        """

        input_dict = self._project_to_latent_space_multi_traj(data_dict)
        parallel_results = self._get_results(self._run_blocks(input_dict))

        batch_size = data_dict["noisy_action"].shape[0]
        if "training_traj_indices" in data_dict:
            traj_num = data_dict["training_traj_indices"].shape[1]
        else:
            # raise RuntimeError("training_traj_indices is not provided")
            traj_num = data_dict["noisy_action"].shape[1]
        results: dict[str, torch.Tensor] = {
            "action": einops.rearrange(
                parallel_results["action"],
                "(batch traj_num) ... -> batch traj_num ...",
                batch=batch_size,
                traj_num=traj_num,
            ),
        }
        return results

    def _run_blocks(self, input_dict: dict[str, torch.Tensor]):


        if len(self.record_data_entries) > 0:
            record_data_dict: dict[str, list[torch.Tensor]] | None = {
                entry: [] for entry in self.record_data_entries
            }
            if "memory_gate_val" in input_dict and "memory_gate_val" in record_data_dict:
                record_data_dict["memory_gate_val"].append(input_dict["memory_gate_val"].clone())
        else:
            record_data_dict = None

        for i, block in enumerate(self.blocks):
            input_dict["x"] = block(record_data_dict=record_data_dict, **input_dict)

        if record_data_dict is not None:
            self.recorded_data_dict = record_data_dict

        return input_dict

    def forward(
        self,
        data_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        data_dict = copy.deepcopy(data_dict)  # Avoid modifying the original data_dict

        input_dict = self._project_to_latent_space(data_dict)
        input_dict = self._run_blocks(input_dict)
        return self._get_results(input_dict)
