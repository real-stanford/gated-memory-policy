"""Evaluate imitation learning policies on MIKASA-Robo environments.

Modes:
  default:       policy predicts 8D joint deltas, executed via pd_joint_delta_pos PD controller.
  abs-joint-pos: policy predicts absolute target qpos, executed via pd_joint_pos PD controller.
  direct-qpos:   policy predicts absolute joint positions, robot is teleported (no physics).

Usage:
  python mikasa_eval.py --env-id ShellGameTouch-v0 --checkpoint path/to/ckpt --num-envs 16
  python mikasa_eval.py --env-id RememberColor3-v0 --checkpoint path/to/ckpt --abs-joint-pos
  python mikasa_eval.py --env-id RememberShape3-v0 --checkpoint path/to/ckpt --direct-qpos
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from tasks import TASK_REGISTRY, make_eval_env, teleport_and_step, resolve_control_mode, compute_eval_steps
from recording import EpisodeBuffers, handle_episode_completions


# Eval loops

def run_eval(env, inner_env, policy, num_envs, num_eval_steps, device, seed,
             output_dir, env_id, task_dir=None):
    """Standard eval: predict action chunk -> execute -> repeat. Saves episodes on completion."""
    if task_dir is None:
        task_dir = os.path.join(output_dir, env_id)
    os.makedirs(task_dir, exist_ok=True)

    policy.reset()
    obs, _ = env.reset(seed=seed)
    buffers = EpisodeBuffers(num_envs)
    metrics = defaultdict(list)
    num_episodes = 0
    completed = np.zeros(num_envs, dtype=bool)
    step = 0
    pbar = tqdm(total=num_eval_steps, desc="eval")

    while step < num_eval_steps:
        action_chunk = policy.predict_action(obs, num_envs, device)

        for t in range(policy.action_horizon):
            if step >= num_eval_steps:
                break

            render_frame = inner_env.render()
            buffers.append_obs(obs, render_frame=render_frame)
            action = action_chunk[:, t, :]
            buffers.append_action(action)

            obs, reward, _term, _trunc, infos = env.step(action)
            buffers.append_reward(reward)
            step += 1
            pbar.update(1)
            policy.report_progress(step, num_eval_steps)

            if "final_info" in infos:
                num_episodes = handle_episode_completions(
                    infos, buffers, task_dir, metrics, num_episodes, env=env,
                )
                completed |= infos["_final_info"].cpu().numpy()
                if completed.all():
                    policy.reset()
                    completed[:] = False
                    break

    pbar.close()
    return _aggregate_results(metrics, num_episodes, task_dir)


def run_direct_qpos(env, inner_env, policy, num_envs, num_eval_steps, device, seed,
                    output_dir, env_id, task_dir=None):
    """Direct-qpos eval: policy predicts absolute joint positions, robot is teleported."""
    if task_dir is None:
        task_dir = os.path.join(output_dir, env_id)
    os.makedirs(task_dir, exist_ok=True)

    policy.reset()
    obs, _ = env.reset(seed=seed)
    buffers = EpisodeBuffers(num_envs)
    metrics = defaultdict(list)
    num_episodes = 0
    completed = np.zeros(num_envs, dtype=bool)
    step = 0
    pbar = tqdm(total=num_eval_steps, desc="direct-qpos eval")

    while step < num_eval_steps:
        action_chunk = policy.predict_action(obs, num_envs, device)

        for t in range(policy.action_horizon):
            if step >= num_eval_steps:
                break

            render_frame = inner_env.render()
            buffers.append_obs(obs, render_frame=render_frame)
            predicted_qpos = action_chunk[:, t, :]
            buffers.append_action(predicted_qpos)

            obs, reward, _term, _trunc, infos = teleport_and_step(env, predicted_qpos)
            buffers.append_reward(reward)
            step += 1
            pbar.update(1)
            policy.report_progress(step, num_eval_steps)

            if "final_info" in infos:
                num_episodes = handle_episode_completions(
                    infos, buffers, task_dir, metrics, num_episodes, env=env,
                )
                completed |= infos["_final_info"].cpu().numpy()
                if completed.all():
                    policy.reset()
                    completed[:] = False
                    break

    pbar.close()
    return _aggregate_results(metrics, num_episodes, task_dir)


def _aggregate_results(metrics, num_episodes, task_dir):
    """Compute mean metrics and save summary."""
    results = {"num_episodes": num_episodes}
    for k, v in metrics.items():
        results[k] = torch.stack(v).float().mean().item()

    with open(os.path.join(task_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {num_episodes} episodes to {task_dir}/")
    return results


# CLI

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate IL policy on MIKASA-Robo")
    parser.add_argument("--env-id", type=str, default=None, choices=list(TASK_REGISTRY.keys()),
                        help="task env id (auto-detected when using --remote-server)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to .ckpt file (not needed with --remote-server)")
    parser.add_argument("--remote-server", type=str, default=None,
                        help="tcp://host:port of RemotePolicyServer (overrides --checkpoint)")
    parser.add_argument("--progress-file", type=str, default=None,
                        help="path to write eval progress JSON (for orchestrator)")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-eval-steps", type=int, default=None,
                        help="total env.step() calls (overrides --num-eval-episodes)")
    parser.add_argument("--num-eval-episodes", type=int, default=None,
                        help="target number of episodes (rounds up to fill parallel envs)")
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-mode", type=str, default="normalized_dense")
    parser.add_argument("--no-proprio", action="store_true",
                        help="vision-only: omit robot0_8d from policy input")
    parser.add_argument("--abs-joint-pos", action="store_true",
                        help="policy predicts absolute target qpos via pd_joint_pos controller")
    parser.add_argument("--direct-qpos", action="store_true",
                        help="policy predicts absolute qpos, robot is teleported (no physics)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load policy
    if args.remote_server:
        from remote_policy_client import RemotePolicyClient
        policy = RemotePolicyClient(args.remote_server, progress_file=args.progress_file)
        if args.env_id is None:
            args.env_id = policy.env_id
        args.no_proprio = policy.no_proprio
    else:
        if args.checkpoint is None or args.env_id is None:
            raise ValueError("--checkpoint and --env-id are required without --remote-server")
        from policy import LocalPolicy, load_policy
        raw_policy, normalizer, action_horizon = load_policy(args.checkpoint, device)
        policy = LocalPolicy(raw_policy, normalizer, device, action_horizon, no_proprio=args.no_proprio)

    # Setup
    task_cfg = TASK_REGISTRY[args.env_id]

    if args.abs_joint_pos and args.direct_qpos:
        raise ValueError("--abs-joint-pos and --direct-qpos are mutually exclusive")

    control_mode, mode_label = resolve_control_mode(args.abs_joint_pos, args.direct_qpos)
    num_eval_steps = compute_eval_steps(
        args.num_eval_steps, args.num_eval_episodes, args.num_envs, task_cfg.episode_steps,
    )

    env, inner_env = make_eval_env(
        env_id=args.env_id, task_cfg=task_cfg, num_envs=args.num_envs,
        num_eval_steps=num_eval_steps, capture_video=args.capture_video,
        output_dir=args.output_dir, seed=args.seed, reward_mode=args.reward_mode,
        control_mode=control_mode,
    )

    # Run eval
    print(f"\n{mode_label}: {args.env_id} | {num_eval_steps} steps | {args.num_envs} envs")

    try:
        if args.direct_qpos:
            results = run_direct_qpos(
                env, inner_env, policy, args.num_envs, num_eval_steps, device, args.seed,
                output_dir=args.output_dir, env_id=args.env_id,
            )
        else:
            results = run_eval(
                env, inner_env, policy, args.num_envs, num_eval_steps, device, args.seed,
                output_dir=args.output_dir, env_id=args.env_id,
            )
    finally:
        env.close()

    # Report
    n = int(results.pop("num_episodes"))
    print(f"\n{mode_label}: {n} episodes in {num_eval_steps} steps "
          f"({num_eval_steps * args.num_envs} transitions across {args.num_envs} envs)")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    results["num_episodes"] = n
    policy.report_results(results)
    policy.cleanup()


if __name__ == "__main__":
    main()
