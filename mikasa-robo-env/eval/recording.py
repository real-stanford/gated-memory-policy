"""Episode recording, saving, and frame annotation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import cv2
import imageio
import numpy as np
import torch


# Episode data container

@dataclass
class EpisodeData:
    """Single episode's recorded data, returned by EpisodeBuffers.flush()."""

    rgb: list[np.ndarray] = field(default_factory=list)
    joints: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    render_frames: list[np.ndarray] = field(default_factory=list)


# Per-env episode buffers

class EpisodeBuffers:
    """Rolling per-env buffers. Call append_*() every step, flush() on episode end."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self._rgb: list[list] = [[] for _ in range(num_envs)]
        self._joints: list[list] = [[] for _ in range(num_envs)]
        self._actions: list[list] = [[] for _ in range(num_envs)]
        self._rewards: list[list] = [[] for _ in range(num_envs)]
        self._render: list[list] = [[] for _ in range(num_envs)]

    def append_obs(self, obs: dict[str, torch.Tensor], render_frame=None):
        rgb_cpu = obs["rgb"].cpu().numpy()
        joints_cpu = obs["joints"].cpu().numpy()
        for i in range(self.num_envs):
            self._rgb[i].append(rgb_cpu[i])
            self._joints[i].append(joints_cpu[i])
        if render_frame is not None:
            if isinstance(render_frame, torch.Tensor):
                render_frame = render_frame.cpu().numpy()
            for i in range(self.num_envs):
                self._render[i].append(render_frame[i])

    def append_action(self, action: torch.Tensor):
        action_cpu = action.cpu().numpy()
        for i in range(self.num_envs):
            self._actions[i].append(action_cpu[i])

    def append_reward(self, reward: torch.Tensor):
        reward_cpu = reward.cpu().numpy()
        for i in range(self.num_envs):
            self._rewards[i].append(float(reward_cpu[i]))

    def flush(self, env_idx: int) -> EpisodeData:
        """Return collected data for one env and clear its buffers."""
        data = EpisodeData(
            rgb=list(self._rgb[env_idx]),
            joints=list(self._joints[env_idx]),
            actions=list(self._actions[env_idx]),
            rewards=list(self._rewards[env_idx]),
            render_frames=list(self._render[env_idx]),
        )
        self._rgb[env_idx].clear()
        self._joints[env_idx].clear()
        self._actions[env_idx].clear()
        self._rewards[env_idx].clear()
        self._render[env_idx].clear()
        return data


# Episode saving

def save_episode(
    episode_dir: str,
    data: EpisodeData,
    success: bool,
    metrics: dict,
    fps: int = 30,
):
    """Save episode to disk: episode.npz + annotated videos + metrics.json."""
    os.makedirs(episode_dir, exist_ok=True)
    T = len(data.rgb)

    rgb = np.stack(data.rgb)
    joints = np.stack(data.joints)
    action = np.stack(data.actions)
    reward = np.array(data.rewards, dtype=np.float32)
    success_arr = np.full(T, int(success), dtype=np.int64)
    done = np.zeros(T, dtype=np.int64)
    done[-1] = 1

    np.savez_compressed(
        os.path.join(episode_dir, "episode.npz"),
        rgb=rgb, joints=joints, action=action,
        reward=reward, success=success_arr, done=done,
    )

    # Side-by-side two-camera obs video
    obs_frames = np.concatenate([rgb[..., :3], rgb[..., 3:]], axis=2)
    obs_annotated = _annotate_frames(obs_frames.copy(), action, reward, font_scale=0.35, thickness=1)
    _save_video(episode_dir, "obs", obs_annotated, fps)

    # Render video (wrapper overlays already applied; add action text)
    if data.render_frames:
        render_arr = np.stack(data.render_frames)
        render_annotated = _annotate_action_on_frames(render_arr.copy(), action, font_scale=0.7, thickness=2)
        _save_video(episode_dir, "render", render_annotated, fps)

    with open(os.path.join(episode_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


# Episode completion handling

def handle_episode_completions(
    infos: dict,
    buffers: EpisodeBuffers,
    task_dir: str,
    metrics: dict[str, list],
    num_episodes: int,
    env=None,
) -> int:
    """Process completed episodes from ManiSkillVectorEnv, save to disk.

    Returns updated episode count. ManiSkillVectorEnv sets infos["_final_info"]
    as a (B,) bool mask and infos["final_info"]["episode"] as metric tensors.
    """
    if "final_info" not in infos:
        return num_episodes

    mask = infos["_final_info"]
    ep_data = infos["final_info"]["episode"]

    for k, v in ep_data.items():
        metrics[k].append(v)

    for i in range(buffers.num_envs):
        if not mask[i]:
            continue

        ep_metrics = {k: float(v[i]) for k, v in ep_data.items()}
        success = ep_metrics.get("success_once", 0.0) > 0.5
        tag = "success" if success else "failure"
        ep_metrics["episode_idx"] = num_episodes

        if env is not None:
            seeds = getattr(env.unwrapped, "_episode_seed", None)
            if seeds is not None:
                ep_metrics["env_seed"] = int(seeds[i])

        data = buffers.flush(i)
        save_episode(
            episode_dir=os.path.join(task_dir, f"episode_{num_episodes:04d}_{tag}"),
            data=data,
            success=success,
            metrics=ep_metrics,
        )
        num_episodes += 1

    return num_episodes


# Frame annotation helpers

def _put_text(frame, text, pos, font_scale=0.5, thickness=1, color=(255, 255, 255)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def _annotate_frames(frames, actions, rewards, font_scale=0.35, thickness=1):
    """Overlay step, reward, and action on obs side-by-side frames (256x512)."""
    for i in range(len(frames)):
        y = 15
        _put_text(frames[i], f"step: {i}", (5, y), font_scale, thickness)
        y += 18
        _put_text(frames[i], f"rew: {rewards[i]:.3f}", (5, y), font_scale, thickness)
        y += 18
        act = actions[i]
        _put_text(frames[i], f"act: [{', '.join(f'{a:.2f}' for a in act)}]", (5, y), font_scale, thickness)
    return frames


def _annotate_action_on_frames(frames, actions, font_scale=0.5, thickness=1):
    """Overlay action text on render frames (512x512). Other overlays already from wrappers."""
    for i in range(len(frames)):
        act = actions[i]
        _put_text(frames[i], f"act[0:4]: [{', '.join(f'{a:.2f}' for a in act[:4])}]", (10, 150), font_scale, thickness)
        _put_text(frames[i], f"act[4:8]: [{', '.join(f'{a:.2f}' for a in act[4:])}]", (10, 175), font_scale, thickness)
    return frames


def _save_video(episode_dir, name, frames, fps):
    """Save annotated video as MP4."""
    imageio.mimsave(os.path.join(episode_dir, f"video_{name}.mp4"), frames, fps=fps)
