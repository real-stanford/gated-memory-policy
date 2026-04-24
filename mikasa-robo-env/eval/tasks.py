"""Task registry, environment creation, and control mode utilities."""

import math
import os
import sys
from dataclasses import dataclass, field

import gymnasium as gym
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_root = os.path.dirname(_repo_root)
sys.path.insert(0, os.path.join(_project_root, "imitation-learning-policies"))
sys.path.insert(0, os.path.join(_project_root, "robot-utils", "src"))

import mani_skill.envs  # noqa: F401 — registers ManiSkill envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from mikasa_robo_suite.memory_envs import *  # noqa: F403 — registers MIKASA envs
from mikasa_robo_suite.utils.wrappers import *  # noqa: F403

from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper

import warnings
warnings.filterwarnings(
    "ignore", message=".*env\\.\\w+ to get variables from other wrappers is deprecated.*"
)


# Task registry
# Add new tasks here. Wrappers applied in order:
#   _BASE_WRAPPERS -> extra_wrappers -> FlattenRGBD -> FlattenAction -> VecEnv

WrapperSpec = tuple[type, dict]

_BASE_WRAPPERS: list[WrapperSpec] = [
    (StateOnlyTensorToDictWrapper, {}),
    (InitialZeroActionWrapper, {"n_initial_steps": 0}),
    (RenderStepInfoWrapper, {}),
    (RenderRewardInfoWrapper, {}),
]


@dataclass
class TaskConfig:
    episode_steps: int
    extra_wrappers: list[WrapperSpec] = field(default_factory=list)


TASK_REGISTRY: dict[str, TaskConfig] = {
    # shell game (90 steps)
    "ShellGameTouch-v0": TaskConfig(episode_steps=90, extra_wrappers=[(ShellGameRenderCupInfoWrapper, {})]),
    "ShellGamePush-v0":  TaskConfig(episode_steps=90, extra_wrappers=[(ShellGameRenderCupInfoWrapper, {})]),
    "ShellGamePick-v0":  TaskConfig(episode_steps=90, extra_wrappers=[(ShellGameRenderCupInfoWrapper, {})]),
    # intercept (90 steps)
    "InterceptSlow-v0":       TaskConfig(episode_steps=90),
    "InterceptMedium-v0":     TaskConfig(episode_steps=90),
    "InterceptFast-v0":       TaskConfig(episode_steps=90),
    "InterceptGrabSlow-v0":   TaskConfig(episode_steps=90),
    "InterceptGrabMedium-v0": TaskConfig(episode_steps=90),
    "InterceptGrabFast-v0":   TaskConfig(episode_steps=90),
    # rotate (90 steps)
    "RotateStrictPos-v0":     TaskConfig(episode_steps=90, extra_wrappers=[(RotateRenderAngleInfoWrapper, {})]),
    "RotateStrictPosNeg-v0":  TaskConfig(episode_steps=90, extra_wrappers=[(RotateRenderAngleInfoWrapper, {})]),
    "RotateLenientPos-v0":    TaskConfig(episode_steps=90, extra_wrappers=[(RotateRenderAngleInfoWrapper, {})]),
    "RotateLenientPosNeg-v0": TaskConfig(episode_steps=90, extra_wrappers=[(RotateRenderAngleInfoWrapper, {})]),
    # take it back (180 steps)
    "TakeItBack-v0": TaskConfig(episode_steps=180),
    # remember color (60 steps)
    "RememberColor3-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberColorInfoWrapper, {})]),
    "RememberColor5-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberColorInfoWrapper, {})]),
    "RememberColor6-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberColorInfoWrapper, {})]),
    "RememberColor9-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberColorInfoWrapper, {})]),
    # remember shape (60 steps)
    "RememberShape3-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberShapeInfoWrapper, {})]),
    "RememberShape5-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberShapeInfoWrapper, {})]),
    "RememberShape9-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberShapeInfoWrapper, {})]),
    # remember shape + color (60 steps)
    "RememberShapeAndColor3x2-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberShapeAndColorInfoWrapper, {})]),
    "RememberShapeAndColor3x3-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberShapeAndColorInfoWrapper, {})]),
    "RememberShapeAndColor5x3-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberShapeAndColorInfoWrapper, {})]),
    # color sequence tasks (120 steps)
    "BunchOfColors3-v0": TaskConfig(episode_steps=120),
    "BunchOfColors5-v0": TaskConfig(episode_steps=120),
    "BunchOfColors7-v0": TaskConfig(episode_steps=120),
    "SeqOfColors3-v0":   TaskConfig(episode_steps=120),
    "SeqOfColors5-v0":   TaskConfig(episode_steps=120),
    "SeqOfColors7-v0":   TaskConfig(episode_steps=120),
    "ChainOfColors3-v0": TaskConfig(episode_steps=120),
    "ChainOfColors5-v0": TaskConfig(episode_steps=120),
    "ChainOfColors7-v0": TaskConfig(episode_steps=120),
}


# Environment creation

def make_eval_env(
    env_id: str,
    task_cfg: TaskConfig,
    num_envs: int,
    num_eval_steps: int,
    capture_video: bool,
    output_dir: str,
    seed: int,
    reward_mode: str,
    control_mode: str = "pd_joint_delta_pos",
) -> tuple[ManiSkillVectorEnv, gym.Env]:
    """Create GPU-vectorized env with the full wrapper stack.

    Returns (vec_env, inner_env). inner_env.render() includes wrapper overlays;
    vec_env.render() bypasses them.
    """
    env = gym.make(
        env_id, num_envs=num_envs, reconfiguration_freq=1,
        obs_mode="rgb", control_mode=control_mode,
        render_mode="all" if capture_video else "rgb_array",
        sim_backend="gpu", reward_mode=reward_mode,
    )

    for wrapper_cls, wrapper_kwargs in _BASE_WRAPPERS + task_cfg.extra_wrappers:
        env = wrapper_cls(env, **wrapper_kwargs)

    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=False, joints=True)

    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    inner_env = env
    vec_env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)
    return vec_env, inner_env


# Direct-qpos teleportation

def teleport_and_step(env, predicted_8d: torch.Tensor):
    """Teleport robot to predicted joint positions and advance env (no physics).

    Bypasses the PD controller entirely: sets qpos directly, zeros velocity,
    and replaces scene.step with GPU sync so gravity/forces don't apply.

    predicted_8d: (B, 8) — 7 arm qpos + 1 gripper qpos.
    Panda has 9 DOF (7 arm + 2 finger), so the single gripper value is mirrored.
    """
    base_env = env.unwrapped
    robot = base_env.agent.robot

    # Build full 9-DOF qpos from predicted 8D
    qpos = robot.get_qpos()
    qpos[:, :7] = predicted_8d[:, :7]
    qpos[:, 7:9] = predicted_8d[:, 7:8].expand(-1, 2)

    # Teleport: set position + zero velocity, sync GPU
    robot.set_qpos(qpos)
    robot.set_qvel(torch.zeros_like(qpos))
    base_env.scene._gpu_apply_all()
    base_env.scene.px.gpu_update_articulation_kinematics()
    base_env.scene._gpu_fetch_all()

    # Fix shape visibility for memory tasks: during hiding phase, shapes are at
    # z=1000. Without physics, evaluate() would capture that as original_poses
    # and never bring them back. Reset to initial table-level positions first.
    initial_poses = getattr(base_env, "initial_poses", None)
    if initial_poses is not None:
        for actor_dict in [getattr(base_env, "cubes", {}), getattr(base_env, "shapes", {})]:
            for key in initial_poses:
                if key in actor_dict:
                    pose = actor_dict[key].pose.raw_pose.clone()
                    pose[:, :3] = initial_poses[key]
                    actor_dict[key].pose = pose
        base_env.scene._gpu_apply_all()
        base_env.scene._gpu_fetch_all()

    # Replace physics step with GPU sync only
    scene = base_env.scene
    original_step = scene.step

    def _sync_only():
        scene._gpu_apply_all()
        scene.px.gpu_update_articulation_kinematics()
        scene._gpu_fetch_all()

    scene.step = _sync_only
    try:
        result = env.step(None)
    finally:
        scene.step = original_step

    return result


# Control mode and eval step helpers

def resolve_control_mode(abs_joint_pos: bool, direct_qpos: bool) -> tuple[str, str]:
    """Map CLI flags to (control_mode, mode_label)."""
    if abs_joint_pos:
        return "pd_joint_pos", "abs-joint-pos"
    if direct_qpos:
        return "pd_joint_delta_pos", "direct-qpos"
    return "pd_joint_delta_pos", "eval"


def compute_eval_steps(
    num_eval_steps: int | None,
    num_eval_episodes: int | None,
    num_envs: int,
    episode_steps: int,
) -> int:
    """Resolve total env.step() calls from config."""
    if num_eval_steps is not None:
        return num_eval_steps
    if num_eval_episodes is not None:
        return math.ceil(num_eval_episodes / num_envs) * episode_steps
    return episode_steps
