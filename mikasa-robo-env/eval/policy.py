"""Policy abstraction, checkpoint loading, and observation conversion."""

from __future__ import annotations

import os
import sys
from typing import Protocol, runtime_checkable

import torch
import yaml

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_root = os.path.dirname(_repo_root)
sys.path.insert(0, os.path.join(_project_root, "imitation-learning-policies"))
sys.path.insert(0, os.path.join(_project_root, "robot-utils", "src"))

from omegaconf import OmegaConf
import hydra

from imitation_learning.common.dataclasses import construct_data_meta_dict
from imitation_learning.datasets.normalizer import FixedNormalizer
from imitation_learning.policies.base_policy import BasePolicy
from robot_utils.config_utils import register_resolvers

register_resolvers()

# Actions per prediction step; matches training chunk size

# Policy interface

@runtime_checkable
class Policy(Protocol):
    """Unified interface for local checkpoint and remote RPC policies.

    Eval loops depend only on this interface — no isinstance checks needed.
    """

    action_horizon: int
    no_proprio: bool

    def predict_action(self, obs: dict[str, torch.Tensor], num_envs: int, device: torch.device) -> torch.Tensor:
        """Env obs -> action chunk (B, action_horizon, 8)."""
        ...

    def reset(self) -> None: ...
    def report_progress(self, step: int, total: int) -> None: ...
    def report_results(self, results: dict) -> None: ...
    def cleanup(self) -> None: ...


class LocalPolicy:
    """Wraps a checkpoint-loaded BasePolicy + FixedNormalizer into the Policy interface.

    Handles obs conversion, normalization, inference, and unnormalization internally.
    """

    def __init__(
        self,
        raw_policy: BasePolicy,
        normalizer: FixedNormalizer,
        device: torch.device,
        action_horizon: int,
        no_proprio: bool = False,
    ):
        self._policy = raw_policy
        self._normalizer = normalizer
        self._device = device
        self.no_proprio = no_proprio
        self.action_horizon = action_horizon

    @torch.no_grad()
    def predict_action(self, obs: dict[str, torch.Tensor], num_envs: int, device: torch.device) -> torch.Tensor:
        batch = obs_to_policy_input(obs, num_envs, device, no_proprio=self.no_proprio)
        batch = self._normalizer.normalize(batch)
        action_dict = self._policy.predict_action(batch)
        action_dict = self._normalizer.unnormalize(action_dict)
        # action0_8d = action-space output, robot0_8d = state-space (direct-qpos)
        output_key = "action0_8d" if "action0_8d" in action_dict else "robot0_8d"
        return action_dict[output_key][:, :self.action_horizon, :]

    def reset(self) -> None:
        self._policy.reset()

    def report_progress(self, step: int, total: int) -> None:
        pass

    def report_results(self, results: dict) -> None:
        pass

    def cleanup(self) -> None:
        pass


# Checkpoint loading

def load_policy(checkpoint_path: str, device: torch.device) -> tuple[BasePolicy, FixedNormalizer, int]:
    """Load raw policy + normalizer from a training checkpoint.

    Returns the unwrapped objects — wrap with LocalPolicy for eval loop use.
    """
    print(f"loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg = OmegaConf.create(ckpt["cfg_str_unresolved"])
    config_path = checkpoint_path.replace(".ckpt", ".yaml")
    with open(config_path, "w") as f:
        yaml.dump(OmegaConf.to_container(cfg), f)
    print(f"saved config to: {config_path}")

    policy = hydra.utils.instantiate(cfg["workspace"]["model"])
    assert isinstance(policy, BasePolicy)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.to(device)
    policy.eval()
    policy.reset()

    data_meta = construct_data_meta_dict(cfg["workspace"]["train_dataset"]["output_data_meta"])
    normalizer = FixedNormalizer(data_meta)
    normalizer.to(device)
    normalizer.load_state_dict(ckpt["normalizer_state_dict"])

    action_horizon = cfg["workspace"]["model"]["action_length"]
    assert isinstance(action_horizon, int)

    return policy, normalizer, action_horizon


# Observation conversion

def obs_to_policy_input(
    obs: dict[str, torch.Tensor],
    num_envs: int,
    device: torch.device,
    no_proprio: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert env observation dict to policy input format.

    Input:  rgb (B,H,W,6) uint8, joints (B,25) float
    Output: two camera tensors (B,1,3,H,W) float [0,1], optional proprio (B,1,8),
            episode_idx (B,) for memory transformer history tracking.
    """
    from kornia.geometry.transform import resize

    rgb = obs["rgb"]
    joints = obs["joints"]

    # (B,1,3,H,W) float [0,1]; env renders at 128x128, policy expects 256x256
    third_person = rgb[..., :3].float().div_(255.0).permute(0, 3, 1, 2).unsqueeze(1)
    wrist_cam = rgb[..., 3:].float().div_(255.0).permute(0, 3, 1, 2).unsqueeze(1)

    if third_person.shape[-1] != 256:
        third_person = resize(third_person.squeeze(1), (256, 256), interpolation="bilinear").unsqueeze(1)
        wrist_cam = resize(wrist_cam.squeeze(1), (256, 256), interpolation="bilinear").unsqueeze(1)

    result = {
        "third_person_camera": third_person.to(device),
        "robot0_wrist_camera": wrist_cam.to(device),
        "episode_idx": torch.arange(num_envs, device=device),
    }
    if not no_proprio:
        proprio = torch.cat([joints[:, 7:14], joints[:, 14:15]], dim=-1).unsqueeze(1)
        result["robot0_8d"] = proprio.to(device)
    return result
