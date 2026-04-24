from typing import cast
from diffusers.schedulers.scheduling_ddim import DDIMScheduler, DDIMSchedulerOutput
import torch

class BaseNoiseScheduler:
    def __init__(
        self,
        train_step_num: int,
        inference_step_num: int,
    ):
        self.train_step_num: int = train_step_num
        self.inference_step_num: int = inference_step_num

        assert self.train_step_num % self.inference_step_num == 0, "train_step_num must be divisible by inference_step_num"

    def sample_training_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Should be implemented by the subclass")

    def get_less_noisy_timesteps(
        self,
        current_timestep_B: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Should be implemented by the subclass")

    def get_noisy_action_and_target(
        self,
        clean_action: torch.Tensor, # (batch_size, traj_length, action_dim) or (batch_size, traj_num, traj_length, action_dim)
        noise: torch.Tensor, # (batch_size, traj_length, action_dim) or (batch_size, traj_num, traj_length, action_dim)
        timestep_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        raise NotImplementedError("Should be implemented by the subclass")

    def get_inference_timesteps(self) -> list[int]:
        raise NotImplementedError("Should be implemented by the subclass")

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        noisy_action: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Should be implemented by the subclass")


class DDIMNoiseScheduler(BaseNoiseScheduler):
    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        clip_sample: bool,
        set_alpha_to_one: bool,
        steps_offset: int,
        prediction_type: str,
        train_step_num: int,
        inference_step_num: int,
    ):
        super().__init__(train_step_num, inference_step_num)

        assert prediction_type in ["epsilon", "sample"], "prediction_type must be either 'epsilon' or 'sample'"
        self.prediction_type: str = prediction_type

        self.scheduler = DDIMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
            prediction_type=prediction_type,
            num_train_timesteps=train_step_num,
        )

    def sample_training_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.randint(
            0, self.train_step_num, (batch_size,), device=device, generator=generator,
        )

    def get_less_noisy_timesteps(
        self,
        current_timestep_B: torch.Tensor,
    ) -> torch.Tensor:
        step_size = int(self.train_step_num // self.inference_step_num)
        return torch.clamp(current_timestep_B - step_size, min=0)
        

    def get_noisy_action_and_target(
        self,
        clean_action: torch.Tensor, # (batch_size, traj_length, action_dim) or (batch_size, traj_num, traj_length, action_dim)
        noise: torch.Tensor, # (batch_size, traj_length, action_dim) or (batch_size, traj_num, traj_length, action_dim)
        timestep_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        timestep_B = cast(torch.IntTensor, timestep_B)
        noisy_action = self.scheduler.add_noise(clean_action, noise, timestep_B)
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = clean_action
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        return noisy_action, target


    def get_inference_timesteps(self) -> list[int]:
        # DDIM scheduler is in decreasing order

        self.scheduler.set_timesteps(self.inference_step_num)
        return self.scheduler.timesteps.tolist()

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        noisy_action: torch.Tensor,
    ) -> torch.Tensor:
        ddim_output_action = self.scheduler.step(model_output, timestep, noisy_action)

        assert isinstance(ddim_output_action, DDIMSchedulerOutput)
        return ddim_output_action.prev_sample



class FlowNoiseScheduler(BaseNoiseScheduler):
    """
    Matching step: 0: clean action; train_step_num: pure gaussian noise
    The denoising step increases from 0 to train_step_num - step_size
    """
    
    def sample_training_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.randint(0, self.train_step_num, (batch_size,), device=device, generator=generator)

    def get_less_noisy_timesteps(
        self,
        current_timestep_B: torch.Tensor,
    ) -> torch.Tensor:
        step_size = int(self.train_step_num // self.inference_step_num)
        return torch.clamp(current_timestep_B + step_size, max=self.train_step_num)

    def get_noisy_action_and_target(
        self,
        clean_action: torch.Tensor,# (batch_size, traj_length, action_dim) or (batch_size, traj_num, traj_length, action_dim)
        noise: torch.Tensor,# (batch_size, traj_length, action_dim) or (batch_size, traj_num, traj_length, action_dim)
        timestep_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep_B = cast(torch.IntTensor, timestep_B)
        if clean_action.dim() == 3:
            timestep_B = timestep_B[:, None, None]
        else:
            timestep_B = timestep_B[:, None, None, None]
        noisy_action = clean_action * timestep_B / self.train_step_num + \
            noise * (1 - timestep_B / self.train_step_num)
        target = clean_action - noise
        return noisy_action, target
    
    def get_inference_timesteps(self) -> list[int]:
        step_size = int(self.train_step_num // self.inference_step_num)
        # return [i * step_size for i in range(self.inference_step_num - 1, -1, -1)]
        return [i * step_size for i in range(0, self.inference_step_num)]

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        noisy_action: torch.Tensor,
    ) -> torch.Tensor:
        return noisy_action + model_output / self.inference_step_num