"""Orchestrate sequential MIKASA checkpoint evaluation.

Submits checkpoints to a running PolicyServer and sends eval config
to a running MikasaEnvServer via robotmq.  Collects results.

Usage:
    python scripts/run_mikasa_eval.py
    python scripts/run_mikasa_eval.py +evaluator.eval.num_envs=8
"""

import glob
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import dill
import numpy as np
import robotmq
import sys
from loguru import logger

import hydra
from omegaconf import DictConfig, OmegaConf
from robot_utils.torch_utils import torch_load

# clean log format: just time + message
logger.remove()
logger.add(sys.stderr, format="<cyan>{time:HH:mm:ss}</cyan> {message}")


@dataclass
class EvalConfig:
    """Eval parameters sent to the env server via robotmq per checkpoint."""
    num_envs: int = 16
    num_eval_steps: int | None = None
    num_eval_episodes: int | None = 100
    capture_video: bool = False
    seed: int = 0
    abs_joint_pos: bool = False
    direct_qpos: bool = False


@dataclass
class MikasaCheckpointEvaluator:
    """Sweeps checkpoints, submitting each to the policy server and
    forwarding eval config to the env server."""

    ckpt_root_dir: str
    port: int = 18765
    timeout: float = 600
    resume_from: str | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)

    def run(self) -> None:
        root = self.ckpt_root_dir.rstrip("/")
        queue = self._discover_checkpoints(root)
        if not queue:
            logger.info(f"no checkpoints found in {root}")
            return
        logger.info(f"found {len(queue)} checkpoints in {root}")

        # resume or fresh run
        if self.resume_from:
            self.output_dir = self.resume_from
            all_results = self._load_prior_results()
            completed = {r["ckpt_path"] for r in all_results}
            queue = [info for info in queue if info["ckpt_path"] not in completed]
            logger.info(f"resuming from {self.output_dir}, {len(completed)} done, {len(queue)} remaining")
        else:
            timestr = datetime.now().strftime("%Y-%m-%d_%H-%M")
            self.output_dir = os.path.join("eval_output", timestr)
            all_results = []

        os.makedirs(self.output_dir, exist_ok=True)
        logger.add(
            os.path.join(self.output_dir, "orchestrator.log"),
            format="{time:HH:mm:ss} {message}",
        )
        logger.info(f"output dir: {self.output_dir}")

        if not queue:
            logger.info("all checkpoints already evaluated")
            return

        endpoint = f"tcp://localhost:{self.port}"
        client = robotmq.RMQClient("mikasa_ckpt_orchestrator", endpoint)
        client.pop_data("rollout_results", n=0)  # clear stale

        try:
            for i, info in enumerate(queue):
                logger.info(f"[{i+1}/{len(queue)}] {info['ckpt_path']} (epoch {info['epoch']})")

                self._send_eval_config(client, info)
                self._submit(client, info)

                result = self._wait_for_results(client)
                if result is not None and not result.get("error"):
                    result.update(ckpt_path=info["ckpt_path"], epoch=info["epoch"], run_name=info["run_name"])
                    all_results.append(result)
                    # save after each checkpoint so progress is never lost
                    self._export(all_results, os.path.join(self.output_dir, "mikasa_eval_results.json"))
                    logger.info(f"  done: {result.get('success_rate', result.get('success_once', 'N/A'))}")
                else:
                    logger.info(f"  skipped (failure)")

        except KeyboardInterrupt:
            logger.info("interrupted")

        if all_results:
            out = os.path.join(self.output_dir, "mikasa_eval_results.json")
            self._export(all_results, out)

    def _load_prior_results(self) -> list[dict]:
        path = os.path.join(self.output_dir, "mikasa_eval_results.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return []

    # Checkpoint discovery

    def _discover_checkpoints(self, root: str) -> list[dict]:
        return self._glob_checkpoints(root)

    def _load_manifest(self, root: str) -> list[dict]:
        assert self.ckpt_manifest is not None
        with open(self.ckpt_manifest) as f:
            lines = [l.strip() for l in f if l.strip()]
        queue = []
        for rel_path in lines:
            path = os.path.join(root, rel_path)
            if not os.path.exists(path):
                logger.warning(f"{path} not found, skipping")
                continue
            if os.path.isdir(path):
                ckpts = sorted(glob.glob(os.path.join(path, "**/*.ckpt"), recursive=True))
                for c in ckpts:
                    queue.append(self._parse_ckpt_path(c))
            else:
                queue.append(self._parse_ckpt_path(path))
        return queue

    def _glob_checkpoints(self, root: str) -> list[dict]:
        paths = sorted(glob.glob(os.path.join(root, "**/*.ckpt"), recursive=True))
        return [self._parse_ckpt_path(p) for p in paths]

    @staticmethod
    def _parse_ckpt_path(path: str) -> dict:
        ckpt = torch_load(path, map_location="cpu", pickle_module=dill)
        cfg = OmegaConf.create(ckpt["cfg_str_unresolved"])
        return {
            "ckpt_path": path,
            "epoch": int(ckpt["epoch"]),
            "run_name": cfg["run_name"],
            "time_tag": f"{cfg['date_str']}-{cfg['time_str']}",
            "task_name": cfg["task_name"],
            "policy_name": cfg["policy_name"],
            "project_name": cfg["project_name"],
            "train_server_name": "localhost",
            "completed_timestamp": time.time(),
        }

    # Robotmq communication

    @staticmethod
    def _submit(client: robotmq.RMQClient, info: dict) -> None:
        status = client.get_topic_status("available_checkpoints", timeout_s=5)
        if status == -2:
            raise ConnectionError("Policy server is not connected")
        client.put_data("available_checkpoints", robotmq.serialize(info))
        logger.info(f"  submitted checkpoint to policy server")

    def _send_eval_config(self, client: robotmq.RMQClient, ckpt_info: dict) -> None:
        from omegaconf import OmegaConf
        eval_dict = {
            **OmegaConf.to_container(self.eval, resolve=True),  # type: ignore[arg-type]
            "output_dir": os.path.abspath(self.output_dir),
            "ckpt_path": ckpt_info["ckpt_path"],
            "epoch": ckpt_info["epoch"],
            "run_name": ckpt_info["run_name"],
        }
        client.put_data("eval_config", robotmq.serialize(eval_dict))

    def _wait_for_results(self, client: robotmq.RMQClient) -> dict | None:
        start = time.time()
        while time.time() - start < self.timeout:
            status = client.get_topic_status("rollout_results", timeout_s=2)
            if status == -2:
                logger.warning("policy server disconnected, aborting wait")
                return None
            raw, _ = client.peek_data("rollout_results", n=1)
            if len(raw) > 0:
                client.pop_data("rollout_results", n=1)
                return robotmq.deserialize(raw[0])
            time.sleep(2)
        logger.warning(f"timed out after {self.timeout}s")
        return None

    # Export

    @staticmethod
    def _export(results: list[dict], path: str) -> None:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        rates = [r["success_rate"] for r in results if "success_rate" in r]
        if rates:
            logger.info(f"mean success: {np.mean(rates):.3f}, max: {np.max(rates):.3f}")
        logger.info(f"exported {len(results)} results to {path}")


@hydra.main(config_path="../imitation_learning/configs", config_name="eval_mikasa", version_base=None)
def main(cfg: DictConfig):
    evaluator = hydra.utils.instantiate(cfg.evaluator)
    evaluator.run()


if __name__ == "__main__":
    main()