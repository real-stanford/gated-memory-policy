import copy
from functools import partial
from typing import Any, cast

import einops
import torch
import torch.nn.functional as F

from imitation_learning.common.datatypes import batch_type
from imitation_learning.models.common.memory_gate import \
    MemoryGate
from imitation_learning.models.denoising_networks.memory_transformer import \
    MemoryTransformer
from imitation_learning.models.encoders.multi_token_encoder import \
    MultiTokenEncoder
from imitation_learning.policies.base_denoising_policy import BaseDenoisingPolicy
from torch._dynamo.eval_frame import OptimizedModule
from robot_utils.data_utils import dict_apply
from robot_utils.torch_utils import aggregate_batch, split_batch

import cv2
import numpy as np
import time

class HistoryDenoisingPolicy(BaseDenoisingPolicy):
    def __init__(
        self,
        skip_memory: bool,
        history_mask_max_prob: float,
        history_img_feature_encoder: MultiTokenEncoder | None,
        action_no_error_range: tuple[int, int],
        train_history_action_noise_level: str,
        eval_history_action_noise_level: str,
        history_action_num_per_chunk: int,
        # add_noise_to_history_img_features: bool,
        memory_gate: MemoryGate | None = None,
        max_training_traj_num: int = -1,
        **kwargs,
    ):

        if history_img_feature_encoder is not None:
            kwargs["denoising_network_partial"] = partial(
                kwargs["denoising_network_partial"],
                history_img_features_dim=history_img_feature_encoder.feature_dim,
                history_img_features_token_num=history_img_feature_encoder.token_num,
                history_action_num_per_chunk=history_action_num_per_chunk,
            )
        if memory_gate is not None and isinstance(memory_gate, MemoryGate):
            kwargs["denoising_network_partial"] = partial(
                kwargs["denoising_network_partial"],
            )

        super().__init__(**kwargs)

        self.action_no_error_range: tuple[int, ...] = tuple(action_no_error_range)
        """
        If any training action between the two values is error, will not backprop the loss
        """
        assert self.action_no_error_range[1] > self.action_no_error_range[0] >= 0


        self.history_noisy_actions_dict: dict[int, list[list[torch.Tensor]]] = {}
        """
        episode_idx -> list [num_history] of lists [num_inference_steps] of tensors [noisy_history_action, shape: (action_length, action_dim)]
        history_mask_max_prob: [0, 1], higher means more history latents will be masked
        Since the action diffusion is not in latent space, we directly store the noisy action as latents
        In the future, we can also store image features/latents in the buffer
        """
        self.history_img_features_dict: dict[int, list[torch.Tensor]] = {}
        """
        episode_idx -> list [num_history] of tensors [history_img_features, shape: (history_img_features_length, img_length*history_img_features_token_num, history_img_features_dim)]
        """
        assert (
            0 <= history_mask_max_prob <= 1
        ), f"history_mask_max_prob must be in [0, 1], but got {history_mask_max_prob}"
        self.history_mask_max_prob: float = history_mask_max_prob

        if not isinstance(memory_gate, MemoryGate):
            self.memory_gate: MemoryGate | None = None
            print(f"No memory gate provided. Got {memory_gate}")
        else:
            self.memory_gate: MemoryGate | None = memory_gate

        self.skip_memory: bool = skip_memory
        """
        If True, will not pass history latents to the denoising network
        """
        if skip_memory:
            self.enable_skip_memory()

        self.recorded_data_dicts: dict[int, list[dict[str, torch.Tensor]]] = {}
        """
        episode_idx -> list of dicts
        each item in the list:
            "history_cross_attention": (diffusion_step_num, transformer_layer_num, head_num, token_num, history_len*token_num)
            "memory_gate_val": (diffusion_step_num, transformer_layer_num, input_token_num)
        """

        self.history_img_feature_encoder: MultiTokenEncoder | None = (
            history_img_feature_encoder
        )

        self.history_action_num_per_chunk: int = history_action_num_per_chunk
        """
        Number of history actions to be stored in the buffer. Should be the number of executed actions in one chunk.
        """

        assert train_history_action_noise_level in ["last_step", "none", "random"]
        assert eval_history_action_noise_level in ["last_step", "none", "random"]
        self.train_history_action_noise_level: str = train_history_action_noise_level
        self.eval_history_action_noise_level: str = eval_history_action_noise_level
        """
        last_step: use one-step less noisy history action as condition
        none: use clean history action as condition
        random: use random noise levey history action as condition
        """


        self.max_training_traj_num: int = max_training_traj_num
        """
        Maximum number of trajectories to be used for training. If -1, will use all the trajectories. Otherwise, will sample a random subset of trajectories.
        This is used when there are too many trajectories in the dataloader (say 150+) to save memory.
        """

    def enable_skip_memory(self):
        self.skip_memory = True
        print(f"Setting skip_memory to {self.skip_memory}")
        for key, params in self.denoising_network.named_parameters():
            if "history" in key:
                if self.skip_memory:
                    params.requires_grad = False

        if self.memory_gate is not None:
            for key, params in self.memory_gate.named_parameters():
                if self.skip_memory:
                    params.requires_grad = False

    # ================================ Inference =================================

    def predict_action(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """
        Single-trajectory input:
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
                "robot0_10d": (batch_size, traj_length, 8)
                "episode_idx": (batch_size,)
            return:
                "action0_10d": (batch_size, traj_length, 8)

        Multi-trajectory input: (only available when skip_memory is False)
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size)
                "robot0_10d": (batch_size, traj_num, traj_length, 8)
            return:
                "action0_10d": (batch_size, traj_num, traj_length, 8)
        """

        meta = next(iter(self.global_cond_encoder.cond_meta.values()))
        if meta.name not in normalized_batch:
            input_shape = normalized_batch[f"{meta.name}_feature"].shape
            expected_shape = torch.Size([self.global_cond_encoder.feature_dim])
        else:
            input_shape = normalized_batch[meta.name].shape
            expected_shape = meta.shape

        if self.skip_memory:
            assert (
                len(input_shape) - len(expected_shape) == 2
            ), "Please make sure you are using single-trajectory dataset when skip_memory is True"
            return super().predict_action(normalized_batch)
        
        if len(input_shape) - len(expected_shape) == 2:
            return self.predict_single_traj(normalized_batch)

        elif len(input_shape) - len(expected_shape) == 3:
            return self.predict_multi_traj(normalized_batch)

        else:
            raise ValueError(
                f"Unexpected input shape: {input_shape}, expected shape: {expected_shape}"
            )

    def predict_single_traj(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """
        normalized_batch:
            "robot0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
            "robot0_10d": (batch_size, traj_length, 8)
            "third_person_camera": (batch_size, traj_length, 3, image_size, image_size) # For table-bin scenario
            "episode_idx": (batch_size,)
        return:
            "action0_10d": (batch_size, traj_length, 8)
        """

        assert (
            "action" not in normalized_batch
        ), "Please exclude the batch `action` for evaluation"

        data_dict, _ = self._encode_input_add_noise(normalized_batch, mode="eval")

        if self.memory_gate is not None:
            memory_gate_val = (
                self.memory_gate.get_gate_value(normalized_batch)
            ) # (batch_size, )
            bs = memory_gate_val.shape[0]
            binarized_memory_gate_val = (memory_gate_val > 0.5).bool()

            assert isinstance(self.denoising_network, MemoryTransformer)

            if not torch.torch.torch.is_grad_enabled() \
                and self.denoising_network.binary_gating \
                and bs == 1 \
                and sum(binarized_memory_gate_val) == 0 \
                and "history_cross_attention" not in self.denoising_network.record_data_entries:
                self.denoising_network.set_skip_history_attn(True)
            else:
                self.denoising_network.set_skip_history_attn(False)

        else:
            memory_gate_val = None

        new_history_action_dict: dict[int, list[torch.Tensor]] = {}
        batch_size, traj_length, action_dim = data_dict["noisy_action"].shape
        assert batch_size == len(
            normalized_batch["episode_idx"]
        ), f"Please make sure the batch size {batch_size} in data_dict['trajectory'].shape: {data_dict['trajectory'].shape}, is the same as the number of episodes {len(normalized_batch['episode_idx'])}"


        if isinstance(self.denoising_network, OptimizedModule):
            # After torch compile: Just fix the type of the denoising network for type checking.
            self.denoising_network = cast(MemoryTransformer, cast(Any, self.denoising_network))
        else:
            assert isinstance(self.denoising_network, MemoryTransformer)
        max_history_len: int = self.denoising_network.max_history_len

        # history_img_features is invariant to the diffusion step
        if self.history_img_feature_encoder is not None and max_history_len > 0:
            history_img_features = torch.zeros(
                (
                    batch_size,
                    max_history_len,
                    self.history_img_feature_encoder.token_num,
                    self.history_img_feature_encoder.feature_dim,
                ),
                device=self.device,
            )
            for idx, episode_idx in enumerate(normalized_batch["episode_idx"]):
                if int(episode_idx) in self.history_img_features_dict.keys():
                    history_len = len(self.history_img_features_dict[int(episode_idx)])
                    history_img_features[idx, -history_len:] = torch.stack(
                        self.history_img_features_dict[int(episode_idx)], dim=0
                    )

        recorded_data_dicts: list[dict[str, torch.Tensor]] = []

        for k, t in enumerate(self.noise_scheduler.get_inference_timesteps()):
            history_noisy_actions = torch.zeros(
                (
                    batch_size,
                    max_history_len,
                    self.history_action_num_per_chunk,
                    action_dim
                ),
                device=self.device,
            ) # (batch_size, max_history_len, history_action_num_per_chunk, action_dim)

            history_mask = torch.zeros(
                (batch_size, max_history_len),
                device=self.device,
                dtype=torch.bool,
            )

            # print(f"{self.history_noisy_actions_dict.keys()=}")
            # print(f"{normalized_batch['episode_idx']=}")

            for l, episode_idx in enumerate(normalized_batch["episode_idx"]):
                if int(episode_idx) in self.history_noisy_actions_dict.keys():

                    if self.eval_history_action_noise_level == "none":
                        diffusion_step_idx = -1
                    elif self.eval_history_action_noise_level == "random":
                        rand_idx = int(torch.randint(0, len(self.noise_scheduler.get_inference_timesteps()), (1,)).item())
                        diffusion_step_idx = rand_idx
                    elif self.eval_history_action_noise_level == "last_step":
                        diffusion_step_idx = k
                    else:
                        raise ValueError(f"Invalid history action noise level: {self.eval_history_action_noise_level}")

                    history_len = len(self.history_noisy_actions_dict[int(episode_idx)])
                    stacked_history_noisy_action = torch.stack(
                        [
                            self.history_noisy_actions_dict[int(episode_idx)][i][
                                diffusion_step_idx
                            ]
                            for i in range(history_len)
                        ],
                        dim=0,
                    )  # (history_len, token_num, hidden_dim)

                    history_noisy_actions[l, -history_len:] = (
                        stacked_history_noisy_action
                    )

                    history_mask[l, -history_len:] = 1

            # These keys need to be overridden every time before the denoising network is called
            # Since the denoising network will pop the keys after the forward pass
            data_dict["history_noisy_actions"] = history_noisy_actions
            data_dict["history_mask"] = history_mask
            if memory_gate_val is not None: 
                data_dict["memory_gate_val"] = memory_gate_val

                
            data_dict["step"] = (
                torch.ones((batch_size,), device=self.device) * t
            )

            if self.history_img_feature_encoder is not None and max_history_len > 0:
                noise_ratio = t / self.noise_scheduler.train_step_num
                data_dict["history_img_features"] = history_img_features

            # if self.mask_in_eval:
            #     self._add_random_masks(data_dict)

            model_output = self.denoising_network.forward(data_dict)
            if len(self.denoising_network.record_data_entries) > 0:
                # print(f"{self.denoising_network.record_data_entries=}, {self.denoising_network.recorded_data_dict=}")
                merged_data_dict = dict_apply(
                    self.denoising_network.recorded_data_dict,
                    lambda x: torch.stack(x, dim=1).detach(),
                )
                recorded_data_dicts.append(copy.deepcopy(merged_data_dict))

            data_dict["noisy_action"] = self.noise_scheduler.step(
                model_output["action"],
                int(t),
                data_dict["noisy_action"],
            )

            for l, episode_idx in enumerate(normalized_batch["episode_idx"]):
                if int(episode_idx) not in new_history_action_dict.keys():
                    new_history_action_dict[int(episode_idx)] = []
                new_history_action_dict[int(episode_idx)].append(
                    data_dict["noisy_action"][l, :self.history_action_num_per_chunk].detach().clone()
                )

        if max_history_len == 0: # For ablation study
            output = self.action_decoder.forward(data_dict["noisy_action"])
            return output  # (batch_size, traj_length, action_dim)

        # ================================ Update history buffer ================================

        for episode_idx, history_action in new_history_action_dict.items():
            # History: list [num_inference_steps] of tensors [noisy_history_action, shape: (action_length, action_dim)]
            if episode_idx not in self.history_noisy_actions_dict.keys():
                self.history_noisy_actions_dict[episode_idx] = []
            self.history_noisy_actions_dict[episode_idx].append(history_action)

            while (
                len(self.history_noisy_actions_dict[episode_idx])
                > self.denoising_network.max_history_len
            ):
                self.history_noisy_actions_dict[episode_idx].pop(0)

        
        if self.history_img_feature_encoder is not None:
            img_dict = {}
            for k in self.history_img_feature_encoder.data_entry_names:
                if k in normalized_batch:
                    img_dict[k] = normalized_batch[k]
                elif f"{k}_feature" in normalized_batch:
                    img_dict[f"{k}_feature"] = normalized_batch[f"{k}_feature"]
                else:
                    raise ValueError(f"Key {k} not found in normalized_batch")
            new_history_img_features = self.history_img_feature_encoder.forward(
                img_dict
            )
            # (batch_size, data_length*img_feature_token_num=1, history_img_features_dim)
            new_history_img_features = new_history_img_features.reshape(
                batch_size, -1, self.history_img_feature_encoder.feature_dim
            )  # (batch_size, data_length * img_feature_token_num, history_img_features_dim)

            for episode_idx, history_img_features in zip(
                normalized_batch["episode_idx"], new_history_img_features
            ):
                episode_idx = int(episode_idx)
                if episode_idx not in self.history_img_features_dict.keys():
                    self.history_img_features_dict[episode_idx] = []
                self.history_img_features_dict[episode_idx].append(
                    history_img_features.detach().clone()
                )

                while (
                    len(self.history_img_features_dict[episode_idx])
                    > self.denoising_network.max_history_len
                ):
                    self.history_img_features_dict[episode_idx].pop(0)


        if len(recorded_data_dicts) > 0:
            merged_data_dict: dict[str, torch.Tensor] = {}
            merged_data_dict = aggregate_batch(
                recorded_data_dicts, partial(torch.stack, dim=1)
            )
            # "history_cross_attention": (batch_size, diffusion_step_num, transformer_layer_num, head_num, token_num, history_len*token_num)
            splitted_data_dicts = split_batch(
                merged_data_dict, partial(torch.unbind, dim=0)
            )
            for k, splitted_data_dict in enumerate(splitted_data_dicts):
                episode_idx = normalized_batch["episode_idx"][k]
                splitted_data_dict = dict_apply(
                    splitted_data_dict, lambda x: x.detach().clone().cpu()
                )
                if int(episode_idx) not in self.recorded_data_dicts.keys():
                    self.recorded_data_dicts[int(episode_idx)] = []
                self.recorded_data_dicts[int(episode_idx)].append(splitted_data_dict)

        output = self.action_decoder.forward(data_dict["noisy_action"])

        return output  # (batch_size, traj_length, action_dim)

    def predict_multi_traj(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """
        Used when the batch contains multiple trajectories in the same episode.
        This function is only used when running validation with multiple ground-truth trajectories.

        normalized_batch:
            "robot0_wrist_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size)
            "robot0_10d": (batch_size, traj_num, traj_length, 8)
            "third_person_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size) # For table-bin scenario
            "episode_idx": (batch_size) # Need to be overridden
        return:
            "action0_10d": (batch_size, traj_num, traj_length, 8)
        """
        traj_num_dim_idx = 1  # batch_size is 0

        # Use single trajectory prediction to iteratively predict all trajectories
        self.reset()
        actions: list[dict[str, torch.Tensor]] = []
        if "variance_temperature" in normalized_batch:
            normalized_batch.pop(
                "variance_temperature"
            )  # Remove variance_temperature from meta

        batch_size = normalized_batch["episode_idx"].shape[0]
        traj_num = normalized_batch["episode_idx"].shape[1]


        for batch in split_batch(
            normalized_batch,
            partial(torch.unbind, dim=traj_num_dim_idx),
        ):  # Along the traj_num dimension
            """
            batch:
                "robot0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
                "robot0_10d": (batch_size, traj_length, 8)
                "third_person_camera": (batch_size, traj_length, 3, image_size, image_size) # For table-bin scenario
                "episode_idx": (batch_size, )
            """
            batch["episode_idx"] = torch.arange(batch_size, device=self.device) # Override episode idx so the history can be correctly recorded
            actions.append(
                self.predict_single_traj(batch)
            )  # (batch_size, traj_length, action_dim)

        return aggregate_batch(
            actions, partial(torch.stack, dim=traj_num_dim_idx)
        )  # (batch_size, traj_num, traj_length, action_dim)


    # ================================ Training =================================

    def _encode_input_multi_traj(
        self, normalized_batch: batch_type
    ) -> tuple[batch_type, batch_type]:
        """
        Should be called only when training history cross-attention modules
        args:
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_num, data_length, 3, image_size, image_size)
                "robot0_wrist_camera_feature": (batch_size, traj_num, data_length, 768) [Optional]
                "robot0_10d": (batch_size, traj_num, data_length, 10)
                "action0_10d": (batch_size, traj_num, data_length, 10)
                "future_0_wrist_camera": (batch_size, traj_num, data_length, 3, image_size, image_size)
                "third_person_camera": (batch_size, traj_num, data_length, 3, image_size, image_size) # For table-bin scenario
        return:
            data_dict:
                "global_cond": (batch_size, traj_num, token_num, global_cond_dim)
                "local_cond": (batch_size, traj_num, token_num, local_cond_dim)
                "noisy_action": (batch_size, traj_num, data_length, action_dim) # Noisy action latents
                "history_noisy_actions": (batch_size, traj_num, history_action_num_per_chunk, action_dim) # History latents, the noise will be 1-inference-step less than "noisy_action", to match the inference scenarios
                "history_img_features": (batch_size, traj_num, token_num, history_img_features_dim) # History image features
                "history_noisy_future_img_features": (batch_size, traj_num, token_num, feature_dim)
                "memory_gate_val": (batch_size, traj_num)
                "step": (batch_size,)
            target:
                "action": (batch_size, traj_num, data_length, 8)
        """

        batch_size = next(iter(normalized_batch.values())).shape[0]
        traj_num = next(iter(normalized_batch.values())).shape[1]
        data_dict: dict[str, torch.Tensor] = {}

        global_cond_dict = {
            k: v
            for k, v in normalized_batch.items()
            if k in self.global_cond_encoder.data_entry_names
        }

        global_cond_dict_feature = {
            k: v
            for k, v in normalized_batch.items()
            if "feature" in k and k.replace("_feature", "") in self.global_cond_encoder.data_entry_names
        }
        global_cond_dict.update(global_cond_dict_feature)

        data_dict["global_cond"] = einops.rearrange(
            self.global_cond_encoder.forward(
                dict_apply(
                    global_cond_dict,
                    lambda x: einops.rearrange(x, "b t ... -> (b t) ..."),
                )
            ),
            "(b t) ... -> b t ...",
            b=batch_size,
        )

        target: dict[str, torch.Tensor] = {}

        if self.local_cond_encoder is not None:
            local_cond_dict = {
                k: v
                for k, v in normalized_batch.items()
                if k in self.local_cond_encoder.data_entry_names
            }
            data_dict["local_cond"] = einops.rearrange(
                self.local_cond_encoder.forward(
                    dict_apply(
                        local_cond_dict,
                        lambda x: einops.rearrange(x, "b t ... -> (b t) ..."),
                    )
                ),
                "(b t) ... -> b t ...",
                b=batch_size,
            )

        train_timesteps: int = self.noise_scheduler.train_step_num
        inference_timesteps = self.noise_scheduler.inference_step_num
        step_ratio = train_timesteps // inference_timesteps

        data_dict["step"] = self.noise_scheduler.sample_training_timesteps(
            batch_size=batch_size,
            device=self.device,
            generator=self.torch_rng,
        )

        trajectory = einops.rearrange(
            self.action_decoder.encode(
                dict_apply(
                    {
                        k: normalized_batch[k]
                        for k in self.action_decoder.data_entry_names
                    },
                    lambda x: einops.rearrange(x, "b t ... -> (b t) ..."),
                )
            ),
            "(b t) ... -> b t ...",
            b=batch_size,
        ) # (batch_size, traj_num, traj_length, action_dim)

        action_noise = torch.randn_like(
            trajectory,
        )  # (batch_size, traj_num, traj_length, action_dim)
        data_dict["noisy_action"], target["action"] = self.noise_scheduler.get_noisy_action_and_target(
            trajectory,
            action_noise,
            data_dict["step"],
        )

        history_latent_diffusion_step = self.noise_scheduler.get_less_noisy_timesteps(data_dict["step"])

        if self.train_history_action_noise_level == "none":
            data_dict["history_noisy_actions"] = trajectory
        elif self.train_history_action_noise_level == "random":

            flattened_traj = einops.rearrange(
                trajectory,
                "b t ... -> (b t) ...",
            )
            history_action_noise = torch.randn_like(
                flattened_traj,
            )
            rand_timesteps = torch.randint(
                0,
                train_timesteps,
                (batch_size * traj_num,),
                device=self.device,
                generator=self.torch_rng,
            )
            data_dict["history_noisy_actions"] = einops.rearrange(
                self.noise_scheduler.get_noisy_action_and_target(
                    flattened_traj,
                    history_action_noise,
                    cast(torch.IntTensor, rand_timesteps),
                )[0],
                "(b t) ... -> b t ...",
                b=batch_size,
            )
            
        elif self.train_history_action_noise_level == "last_step":
            history_action_noise = torch.randn_like(
                trajectory,
            )
            data_dict["history_noisy_actions"], _ = self.noise_scheduler.get_noisy_action_and_target(
                trajectory,
                history_action_noise,
                cast(torch.IntTensor, history_latent_diffusion_step),
            )
        else:
            raise ValueError(f"Unknown history action noise level: {self.train_history_action_noise_level}")

        # Truncate the history noisy actions to the number of history actions per chunk
        data_dict["history_noisy_actions"] = data_dict["history_noisy_actions"][:, :, :self.history_action_num_per_chunk]

        if self.memory_gate is not None:
            normalized_batch_without_text = {
                k: v
                for k, v in normalized_batch.items()
                if not isinstance(v[0], str)
            }
            flattened_data_dict = dict_apply(
                normalized_batch_without_text,
                lambda x: einops.rearrange(x, "b t ... -> (b t) ..."),
            )
            val = self.memory_gate.get_gate_value(
                flattened_data_dict
            )
            data_dict["memory_gate_val"] = einops.rearrange(
                val,
                "(b t) ... -> b t ...",
                b=batch_size,
            ) # (batch_size, traj_num) 
            # print(f"Memory gate val: {data_dict['memory_gate_val']}, {data_dict['memory_gate_val'].shape}, \ntraj_idx: {normalized_batch['traj_idx']}, {normalized_batch['traj_idx'].shape}")

        if self.history_img_feature_encoder is not None:
            img_dict = {
                k: normalized_batch[k]
                for k in self.history_img_feature_encoder.data_entry_names if k in normalized_batch
            }
            img_feature_dict = {
                f"{k}_feature": normalized_batch[f"{k}_feature"]
                for k in self.history_img_feature_encoder.data_entry_names if f"{k}_feature" in normalized_batch
            }
            img_dict.update(img_feature_dict)
            data_dict["history_img_features"] = (
                self.history_img_feature_encoder.forward(img_dict)
            )  # (batch_size, traj_num, img_num*img_feature_token_num, history_img_features_dim)


        data_dict["entire_traj_is_padding"] = normalized_batch["entire_traj_is_padding"]

        if self.max_training_traj_num > 0:
            valid_traj_indices: list[torch.Tensor] = []
            for i in range(batch_size):
                valid_traj_num = int(torch.sum(~data_dict["entire_traj_is_padding"][i]))
                assert not torch.any(data_dict["entire_traj_is_padding"][i, :valid_traj_num]), f"entire_traj_is_padding must be False for the first few trajectories, but got {data_dict['entire_traj_is_padding'][i, :valid_traj_num]}"
                sampled_traj_indices = torch.randint(0, valid_traj_num, (self.max_training_traj_num,), device=data_dict["entire_traj_is_padding"].device)
                # aggregated_indices = sampled_traj_indices + i * self.max_training_traj_num
                valid_traj_indices.append(sampled_traj_indices)
            all_valid_traj_indices = torch.stack(valid_traj_indices, dim=0)
            # print(f"{all_valid_traj_indices.shape=}")
            data_dict["training_traj_indices"] = all_valid_traj_indices # (batch_size, max_training_traj_num)

            batch_idx = torch.arange(batch_size, device=self.device)
            for k, v in target.items():
                target[k] = target[k][batch_idx, all_valid_traj_indices]
                # print(f"{k}: {target[k].shape}")
            # data_dict will be processed in MemoryTransformer._project_to_latent_space_multi_traj
            
        return data_dict, target



    def compute_loss(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """
        If self.skip_memory, will directly call the method in the superclass:
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
                "robot0_wrist_camera_feature": (batch_size, traj_length, 768) [Optional]
                "robot0_10d": (batch_size, traj_length, 8)
                "action0_10d": (batch_size, traj_length, 8)
                "future_0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
                "third_person_camera": (batch_size, traj_length, 3, image_size, image_size) # For table-bin scenario
                "action_is_error": (batch_size, traj_length)
                "action_is_critical": (batch_size, traj_length)

        If not self.skip_memory:
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size)
                "robot0_wrist_camera_feature": (batch_size, traj_num, traj_length, 768) [Optional]
                "robot0_10d": (batch_size, traj_num, traj_length, 8)
                "action0_10d": (batch_size, traj_num, traj_length, 8)
                "future_0_wrist_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size)
                "third_person_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size) # For table-bin scenario
                "entire_traj_is_padding": (batch_size, traj_num)
                "action_is_error": (batch_size, traj_num, traj_length)
                "action_is_critical": (batch_size, traj_num, traj_length) # Optional
        """
        if self.skip_memory:
            return super().compute_loss(normalized_batch)

        # print(f"normalized_batch keys: {normalized_batch.keys()}")

        loss = {}

        self.shared_model_manager.clear_cache() # Clear the cache before every forward pass


        # print(normalized_batch["robot0_wrist_camera"][0,0].min(), normalized_batch["robot0_wrist_camera"][0,0].max())
        # img = normalized_batch["robot0_wrist_camera"][0,0].cpu().numpy()
        # cv2_img = img.squeeze(0).transpose(1, 2, 0)  # [image_size, image_size, 3]
        # cv2_img = (cv2_img) * 255
        # cv2_img = cv2.cvtColor(cv2_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"robot0_wrist_camera.png", cv2_img)

        # third_person_camera = normalized_batch["third_person_camera"][0,0].cpu().numpy()
        # print(third_person_camera.min(), third_person_camera.max())
        # cv2_img = third_person_camera.squeeze(0).transpose(1, 2, 0)  # [image_size, image_size, 3]
        # cv2_img = (cv2_img) * 255
        # cv2_img = cv2.cvtColor(cv2_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"third_person_camera.png", cv2_img)

        # exit()

        action_key_names = self.action_decoder.data_entry_names
        action_traj_length = normalized_batch[action_key_names[0]].shape[2]

        global_cond_key_names = self.global_cond_encoder.data_entry_names
        global_cond_valid_key_name = global_cond_key_names[0]
        if f"{global_cond_valid_key_name}_feature" in normalized_batch:
            global_cond_valid_key_name = f"{global_cond_valid_key_name}_feature"

        traj_num = normalized_batch[global_cond_valid_key_name].shape[1]
        batch_size = normalized_batch[global_cond_valid_key_name].shape[0]

        if "local_cond" in normalized_batch and len(normalized_batch["local_cond"]) > 0:
            assert (batch_size, traj_num) == next(
                iter(normalized_batch["local_cond"].values())
            ).shape[:2], "Please make sure you are using multi-trajectory dataset"

        assert (batch_size, traj_num) == normalized_batch[action_key_names[0]].shape[
            :2
        ], f"Please make sure you are using multi-trajectory dataset. (batch_size: {batch_size}, traj_num: {traj_num}, action_shape: {normalized_batch[action_key_names[0]].shape})"

        data_dict, target = self._encode_input_multi_traj(normalized_batch)

        # self._add_random_masks(data_dict)
        
        if isinstance(self.denoising_network, OptimizedModule):
            # After torch compile: Just fix the type of the denoising network for type checking.
            self.denoising_network = cast(MemoryTransformer, cast(Any, self.denoising_network))
        else:
            assert isinstance(
                self.denoising_network, MemoryTransformer
            ), "MemoryTransformer is required for memory-based policy"

        model_output: dict[str, torch.Tensor] = self.denoising_network.parallel_forward(
            data_dict
        )
        if "memory_gate_val" in self.denoising_network.recorded_data_dict:
            memory_gate_val = self.denoising_network.recorded_data_dict[
                "memory_gate_val"
            ]  # (batch_size, traj_num, transformer_layer_num, input_token_num)

            if len(memory_gate_val) > 0:
                memory_gate_val = einops.reduce(
                    memory_gate_val, "l (b t)-> b t", "mean", b=batch_size, t=traj_num
                )  # (batch_size, traj_num)
                memory_gate_val = memory_gate_val * (
                    ~normalized_batch["entire_traj_is_padding"]
                )  # Do not compute loss for padding trajectories
            else:
                memory_gate_val = None
        else:
            memory_gate_val = None

        critical_memory_gate_val = None

        action_loss = F.mse_loss(
            model_output["action"], target["action"], reduction="none"
        )

        action_loss = einops.reduce(
            action_loss, "b t ... -> b t", "mean"
        )  # mean over all dimensions except batch and traj_num # (batch_size, traj_num)
        if self.max_training_traj_num <= 0:
            action_loss = action_loss * (
                ~normalized_batch["entire_traj_is_padding"]
            )  # Do not compute loss for padding trajectories

        # img_feature_loss = None

        if "action_is_error" in normalized_batch and self.max_training_traj_num <= 0:
            traj_error_mask = torch.zeros(action_traj_length, device=self.device)
            traj_error_mask[
                self.action_no_error_range[0] : self.action_no_error_range[1]
            ] = 1
            traj_is_error = (
                normalized_batch["action_is_error"] * traj_error_mask[None, None, :]
            )  # (batch_size, traj_num, traj_length)
            traj_is_error = einops.reduce(
                traj_is_error, "b t ... -> b t", "any"
            )  # (batch_size, traj_num)
            action_loss = action_loss * (~traj_is_error)

            # if img_feature_loss is not None:
            #     img_feature_loss = img_feature_loss * (~traj_is_error)
            if memory_gate_val is not None:
                memory_gate_val = memory_gate_val * (~traj_is_error)

        critical_action_loss = None
        if "action_is_critical" in normalized_batch and self.max_training_traj_num <= 0:
            # Is based on the previous filtered loss (action_is_error and entire_traj_is_padding)
            single_action_is_critical = normalized_batch["action_is_critical"]
            # (batch_size, traj_num, traj_length)
            traj_action_is_critical = torch.any(
                single_action_is_critical, dim=2
            ).squeeze(
                -1
            )  # (batch_size, traj_num)
            critical_action_loss = action_loss * traj_action_is_critical

            if critical_action_loss.sum() > 0:
                loss["critical_action"] = (
                    critical_action_loss.sum() / (critical_action_loss != 0).sum()
                )
            else:
                loss["critical_action"] = critical_action_loss.sum()

            if memory_gate_val is not None:
                critical_memory_gate_val = memory_gate_val * traj_action_is_critical
                valid_mask = traj_action_is_critical * (action_loss != 0)
                if valid_mask.sum() > 0:
                    loss["critical_memory_gate_val"] = (
                        critical_memory_gate_val.sum()
                        / valid_mask.sum()
                    )
                    loss["critical_binary_memory_gate_val"] = (critical_memory_gate_val > 0.5).sum() / valid_mask.sum()
                else:
                    loss["critical_memory_gate_val"] = critical_memory_gate_val.sum()
                    loss["critical_binary_memory_gate_val"] = (critical_memory_gate_val > 0.5).sum()

        if memory_gate_val is not None:
            valid_num = (action_loss != 0).sum() if (action_loss != 0).sum() > 0 else memory_gate_val.sum()
            loss["memory_gate_val"] = memory_gate_val.sum() / valid_num
            loss["binary_memory_gate_val"] = (memory_gate_val > 0.5).sum() / valid_num
            
        if (action_loss != 0).sum() == 0:
            loss["action"] = action_loss.sum() / action_loss.numel()
        else:
            loss["action"] = action_loss.sum() / (action_loss != 0).sum()



        return loss

    def reset(self):
        super().reset()
        self.history_noisy_actions_dict = {}
        self.history_img_features_dict = {}
        self.recorded_data_dicts = {}