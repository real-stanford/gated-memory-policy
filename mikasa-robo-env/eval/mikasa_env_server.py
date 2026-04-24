"""MIKASA remote evaluation server for distributed eval.

Runs in its own conda env (with gymnasium / ManiSkill). Connects to the
policy server via robotmq, waits for checkpoint-loaded signals, evaluates
each checkpoint, and posts results back.

    Terminal 1 (imitation):  python scripts/run_remote_policy_server.py
    Terminal 2 (mikasa):     bash shell_scripts/serve_mikasa_env.sh [GPU_ID] [PORT]
    Terminal 3 (imitation):  python scripts/run_mikasa_eval.py
"""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, fields

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import robotmq
import torch
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<cyan>{time:HH:mm:ss}</cyan> {message}")

from tasks import TASK_REGISTRY, make_eval_env, resolve_control_mode, compute_eval_steps
from mikasa_eval import run_eval, run_direct_qpos
from remote_policy_client import RemotePolicyClient

_CFG_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".eval_config_cache.json")

_METRIC_RENAME = {
    "success_once": "success_rate",
    "success_at_end": "success_rate_at_end",
    "return": "mean_return",
    "episode_len": "mean_episode_len",
    "reward": "mean_reward",
}


# Eval config

@dataclass
class EvalConfig:
    """Per-checkpoint eval parameters (received from orchestrator via robotmq)."""

    num_envs: int = 16
    num_eval_steps: int | None = None
    num_eval_episodes: int | None = 100
    capture_video: bool = False
    seed: int = 0
    abs_joint_pos: bool = False
    direct_qpos: bool = False
    reward_mode: str = "normalized_dense"
    output_dir: str | None = None
    ckpt_path: str | None = None
    epoch: int | None = None
    run_name: str | None = None

    _field_names: set | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "EvalConfig":
        if cls._field_names is None:
            cls._field_names = {f.name for f in fields(cls) if f.name != "_field_names"}
        return cls(**{k: v for k, v in d.items() if k in cls._field_names})

    def save(self, path: str = _CFG_CACHE) -> None:
        d = asdict(self)
        d.pop("_field_names", None)
        with open(path, "w") as f:
            json.dump(d, f)

    @classmethod
    def load(cls, path: str = _CFG_CACHE) -> "EvalConfig | None":
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return cls.from_dict(json.load(f))


# Robotmq helpers

def _policy_ready(endpoint: str) -> bool:
    """Probe the policy server — True if a checkpoint is currently loaded."""
    try:
        probe = robotmq.RMQClient("mikasa_env_probe", endpoint)
        raw = probe.request_with_data("policy_config", robotmq.serialize(True), timeout_s=3)
        result = robotmq.deserialize(raw)
        return isinstance(result, dict) and "workspace" in result
    except Exception:
        return False


def _wait_for_policy(endpoint: str, timeout: int = 30) -> None:
    """Block until the policy server has a checkpoint loaded."""
    for _ in range(timeout):
        if _policy_ready(endpoint):
            return
        time.sleep(1)
    raise ConnectionError("policy server never loaded checkpoint")


def _wait_for_topic(client: robotmq.RMQClient, topic: str) -> None:
    """Block until topic has data, then pop it."""
    while True:
        status = client.get_topic_status(topic, timeout_s=1)
        if status == -2:
            raise ConnectionError("policy server disconnected")
        if status > 0:
            break
        time.sleep(1)
    client.pop_data(topic, n=1)


def _get_eval_config(client: robotmq.RMQClient) -> EvalConfig:
    """Try robotmq queue first, then disk cache, then defaults."""
    raw, _ = client.pop_data("eval_config", n=1)
    if raw:
        cfg = EvalConfig.from_dict(robotmq.deserialize(raw[0]))
        cfg.save()
        return cfg

    cached = EvalConfig.load()
    if cached:
        logger.info("using cached eval config")
        return cached

    return EvalConfig()


def _build_task_dir(output_dir: str, env_id: str, cfg: EvalConfig) -> str:
    """Build per-checkpoint output path: output_dir/env_id/ckpt_str/."""
    if cfg.run_name and cfg.epoch is not None:
        ckpt_str = f"{cfg.run_name}_epoch{cfg.epoch}"
    elif cfg.ckpt_path:
        ckpt_str = os.path.splitext(os.path.basename(cfg.ckpt_path))[0]
    else:
        return os.path.join(output_dir, env_id)
    return os.path.join(output_dir, env_id, ckpt_str)


# Main loop

_log_file_dir: str | None = None
_log_file_id: int | None = None


def _ensure_log_file(output_dir: str) -> None:
    """Point the file log sink at output_dir (re-adds only when dir changes)."""
    global _log_file_dir, _log_file_id
    if _log_file_dir == output_dir:
        return
    if _log_file_id is not None:
        logger.remove(_log_file_id)
    _log_file_id = logger.add(
        os.path.join(output_dir, "env_server.log"),
        format="{time:HH:mm:ss} {message}",
    )
    _log_file_dir = output_dir


def run_server(policy_server_address: str, default_output_dir: str) -> None:
    client = robotmq.RMQClient("mikasa_env_server", policy_server_address)
    logger.info(f"connected to {policy_server_address}")

    # If the policy server already has a checkpoint (we were killed mid-eval), resume
    resumed = False
    if _policy_ready(policy_server_address):
        logger.info("resuming — policy server already has checkpoint loaded")
        resumed = True

    try:
        while True:
            if not resumed:
                logger.info("waiting for checkpoint...")
                try:
                    _wait_for_topic(client, "new_checkpoint_loaded")
                except ConnectionError:
                    logger.warning("policy server gone, retrying in 5s...")
                    time.sleep(5)
                    continue
            resumed = False

            cfg = _get_eval_config(client)
            output_dir = cfg.output_dir or default_output_dir
            os.makedirs(output_dir, exist_ok=True)
            _ensure_log_file(output_dir)

            try:
                _eval_one_checkpoint(policy_server_address, output_dir, cfg)
            except ConnectionError:
                logger.error("policy server disconnected during eval")
            except Exception:
                traceback.print_exc()
                try:
                    p = RemotePolicyClient(policy_server_address)
                    p.report_results({"error": True})
                    p.cleanup()
                except Exception:
                    pass
    except KeyboardInterrupt:
        logger.info("interrupted, shutting down")


# Per-checkpoint evaluation

def _eval_one_checkpoint(
    policy_server_address: str, output_dir: str, cfg: EvalConfig,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _wait_for_policy(policy_server_address)
    policy = RemotePolicyClient(policy_server_address)

    env_id = policy.env_id
    task_cfg = TASK_REGISTRY[env_id]
    control_mode, mode_label = resolve_control_mode(cfg.abs_joint_pos, cfg.direct_qpos)
    num_eval_steps = compute_eval_steps(
        cfg.num_eval_steps, cfg.num_eval_episodes, cfg.num_envs, task_cfg.episode_steps,
    )
    task_dir = _build_task_dir(output_dir, env_id, cfg)

    logger.info(f"{mode_label}: {env_id} | {num_eval_steps} steps | {cfg.num_envs} envs | {task_dir}")

    env, inner_env = make_eval_env(
        env_id=env_id, task_cfg=task_cfg, num_envs=cfg.num_envs,
        num_eval_steps=num_eval_steps, capture_video=cfg.capture_video,
        output_dir=task_dir, seed=cfg.seed, reward_mode=cfg.reward_mode,
        control_mode=control_mode,
    )

    try:
        if cfg.direct_qpos:
            results = run_direct_qpos(
                env, inner_env, policy, cfg.num_envs, num_eval_steps,
                device, cfg.seed, output_dir=output_dir, env_id=env_id,
                task_dir=task_dir,
            )
        else:
            results = run_eval(
                env, inner_env, policy, cfg.num_envs, num_eval_steps,
                device, cfg.seed, output_dir=output_dir, env_id=env_id,
                task_dir=task_dir,
            )
    finally:
        env.close()

    # Rename metrics for orchestrator compatibility
    n = int(results.pop("num_episodes"))
    results = {_METRIC_RENAME.get(k, k): v for k, v in results.items()}

    logger.info(f"done: {n} episodes")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.4f}")

    results["num_episodes"] = n
    policy.report_results(results)
    policy.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIKASA remote env server")
    parser.add_argument("--policy-server", default="tcp://localhost:18765")
    parser.add_argument("--output-dir", default="eval_results")
    args = parser.parse_args()
    run_server(args.policy_server, args.output_dir)
