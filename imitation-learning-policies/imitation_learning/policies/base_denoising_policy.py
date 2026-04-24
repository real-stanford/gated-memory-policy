from typing import Callable, Literal, cast

import torch
from imitation_learning.common.datatypes import batch_type
from imitation_learning.models.common.noise_scheduler import BaseNoiseScheduler
from imitation_learning.models.denoising_networks.base_denoising_network import BaseDenoisingNetwork
from imitation_learning.policies.base_policy import BasePolicy
from imitation_learning.models.encoders.image_encoders import SharedModelManager
import torch.nn.functional as F

class BaseDenoisingPolicy(BasePolicy):
    def __init__(
        self,
        shared_model_manager: SharedModelManager,
        denoising_network_partial: Callable[..., BaseDenoisingNetwork],
        noise_scheduler: BaseNoiseScheduler,
        input_perturb: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shared_model_manager: SharedModelManager = shared_model_manager
        print(f"action_decoder.latent_dim: {self.action_decoder.latent_dim}")
        print(f"action_decoder.token_num: {self.action_decoder.token_num}")
        print(
            f"global_cond_encoder.feature_dim: {self.global_cond_encoder.feature_dim}"
        )
        print(f"global_cond_encoder.token_num: {self.global_cond_encoder.token_num}")
        denoising_network_kwargs = {
            "action_dim": self.action_decoder.latent_dim,
            "action_token_num": self.action_decoder.token_num,
            "global_cond_dim": self.global_cond_encoder.feature_dim,
            "global_cond_token_num": self.global_cond_encoder.token_num,
        }

        if self.local_cond_encoder is not None:
            print(f"local_cond_encoder.token_num: {self.local_cond_encoder.token_num}")
            denoising_network_kwargs["local_cond_dim"] = (
                self.local_cond_encoder.feature_dim
            )
            denoising_network_kwargs["local_cond_token_num"] = (
                self.local_cond_encoder.token_num
            )

        
        self.denoising_network: BaseDenoisingNetwork = denoising_network_partial(
            **denoising_network_kwargs
        )

        # self.denoising_network = cast(BaseDenoisingNetwork, torch.compile(self.denoising_network))

        self.noise_scheduler: BaseNoiseScheduler = noise_scheduler

        self.input_perturb: float = input_perturb

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"BaseDenoisingPolicy trainable parameters: {trainable_params}, total parameters: {total_params}"
        )
        
    def compile(self):
        self.denoising_network = cast(BaseDenoisingNetwork, torch.compile(self.denoising_network))
        
    def _encode_input_add_noise(
        self, normalized_batch: batch_type, mode: Literal["train", "eval"]
    ) -> tuple[batch_type, batch_type]:
        """
        args:
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
                "robot0_10d": (batch_size, traj_length, 10)
                "action0_10d": (batch_size, traj_length, 10) (training only)
        return:
            data_dict:
                "global_cond": (batch_size, token_num, global_cond_dim)
                "local_cond": (batch_size, token_num, local_cond_dim)
                "noisy_action": (batch_size, traj_length, action_dim)
                "step": (batch_size,)
            target:
                "action": (batch_size, traj_length, action_dim) (training only)
        """
        batch_size = next(iter(normalized_batch.values())).shape[0]
        data_dict: batch_type = {}
        normalized_global_cond = {}
        for k in self.global_cond_encoder.data_entry_names:
            if k in normalized_batch:
                normalized_global_cond[k] = normalized_batch[k]
            elif f"{k}_feature" in normalized_batch:
                normalized_global_cond[f"{k}_feature"] = normalized_batch[f"{k}_feature"]
            else:
                raise ValueError(f"Key {k} not found in normalized_batch")

        data_dict["global_cond"] = self.global_cond_encoder.forward(
            normalized_global_cond
        )

        if self.local_cond_encoder is not None:
            normalized_local_cond = {
                k: normalized_batch[k] for k in self.local_cond_encoder.data_entry_names
            }
            data_dict["local_cond"] = self.local_cond_encoder.forward(
                normalized_local_cond
            )
            assert (
                len(data_dict["local_cond"].shape) == 3
            ), f"local_cond.shape: {data_dict['local_cond'].shape}, expected: (batch_size, token_num, {self.local_cond_encoder.feature_dim})"

        target: batch_type = {}

        if mode == "train":  # For training
            normalized_action = {
                k: normalized_batch[k] for k in self.action_decoder.data_entry_names
            }
            action_traj_BLD = self.action_decoder.encode(normalized_action)
            action_noise_BLD = torch.randn(
                action_traj_BLD.shape,
                device=self.device,
                generator=self.torch_rng,
            )  # (batch_size, traj_length, action_dim)

            if self.input_perturb > 0:
                # input perturbation by adding additonal noise to alleviate exposure bias
                # reference: https://github.com/forever208/DDPM-IP
                action_noise_BLD = action_noise_BLD + self.input_perturb * torch.randn(
                    action_traj_BLD.shape,
                    device=self.device,
                    generator=self.torch_rng,
                )

            # Clean action is after step 0, pure gaussian noise is at step train_step_num - 1
            timestep_B = self.noise_scheduler.sample_training_timesteps(
                batch_size=action_traj_BLD.shape[0],
                device=self.device,
                generator=self.torch_rng,
            )

            noisy_action_traj_BLD, target["action"] = self.noise_scheduler.get_noisy_action_and_target(
                action_traj_BLD, action_noise_BLD, timestep_B
            )  # (batch_size, traj_length, action_dim)

            data_dict["noisy_action"] = noisy_action_traj_BLD
            data_dict["step"] = timestep_B

        elif mode == "eval":  # For evaluation / rollout
            data_dict["noisy_action"] = torch.randn(
                size=(
                    batch_size,
                    self.action_decoder.token_num,
                    self.action_decoder.latent_dim,
                ),
                device=self.device,
                generator=self.torch_rng,
            )
            ## The first step for evaluation should not be 0. This is just placeholders.
            data_dict["step"] = torch.zeros((batch_size,), device=self.device)
            
        return data_dict, target


    def compute_loss(self, normalized_batch: batch_type) -> batch_type:
        """
        normalized_batch:
            "robot0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
            "robot0_10d": (batch_size, traj_length, 10)
            "action0_10d": (batch_size, traj_length, 10)
        """
        self.shared_model_manager.clear_cache() # Clear the cache before every forward pass

        data_dict, target = self._encode_input_add_noise(normalized_batch, mode="train")
        # self._add_random_masks(data_dict)

        model_output = self.denoising_network.forward(data_dict)

        action_loss = F.mse_loss(
            model_output["action"], target["action"], reduction="none"
        )

        loss = {
            "action": action_loss.mean()
        }

        return loss


    def predict_action(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """
        normalized_batch:
            "robot0_wrist_camera": (batch_size, data_length, 3, image_size, image_size)
            "robot0_10d": (batch_size, data_length, 10)
        return:
            "action0_10d": (batch_size, data_length, 10)
        """
        
        data_dict, _ = self._encode_input_add_noise(normalized_batch, mode="eval")

        for t in self.noise_scheduler.get_inference_timesteps():
            data_dict["step"] = torch.ones_like(data_dict["step"]) * t

            # if self.mask_in_eval:
            #     self._add_random_masks(data_dict)

            model_output = self.denoising_network.forward(data_dict)
            data_dict["noisy_action"] = self.noise_scheduler.step(
                model_output["action"],
                int(t),
                data_dict["noisy_action"],
            )

        output = self.action_decoder.forward(data_dict["noisy_action"])

        return output

    def reset(self):
        self.shared_model_manager.clear_cache() # Clear the cache before every forward pass
        print(f"Resetting policy")
        self.denoising_network.reset()

    def __del__(self):
        SharedModelManager.reset()
