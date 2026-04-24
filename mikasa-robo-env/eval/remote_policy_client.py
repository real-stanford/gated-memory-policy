"""Remote policy client for distributed eval via robotmq."""

import json
import os
import re
import tempfile

import robotmq
import torch

from tasks import TASK_REGISTRY
from policy import obs_to_policy_input


def _resolve_env_id(name: str) -> str:
    """Convert hydra task name to MIKASA env id if needed.

    e.g. mikasa_remember_color_3 -> RememberColor3-v0
    """
    if name in TASK_REGISTRY:
        return name
    converted = re.sub(r"^mikasa_", "", name)
    converted = re.sub(r"_([a-z])", lambda m: m.group(1).upper(), converted)
    converted = re.sub(r"_(\d)", r"\1", converted)
    converted = converted[0].upper() + converted[1:] + "-v0"
    return converted


class RemotePolicyClient:
    """RPC client wrapping robotmq calls to a remote policy server.

    Conforms to the Policy protocol: predict_action, reset, report_progress,
    report_results, cleanup, plus action_horizon and no_proprio attributes.
    """

    def __init__(
        self,
        policy_server_endpoint: str,
        timeout_s: int = 30,
        progress_file: str | None = None,
    ) -> None:
        self.client = robotmq.RMQClient("mikasa_eval_client", policy_server_endpoint)
        self.timeout_s = timeout_s

        if progress_file is None:
            fd, self.progress_file = tempfile.mkstemp(prefix="mikasa_eval_progress_", suffix=".json")
            os.close(fd)
        else:
            self.progress_file = progress_file

        self.policy_config = self._get_policy_config()
        self.action_horizon: int = self.policy_config["workspace"]["model"]["action_length"]
        self.env_id: str = _resolve_env_id(self.policy_config["workspace"]["train_dataset"]["name"])
        self.no_proprio: bool = int(self.policy_config["workspace"]["model"]["proprio_length"]) == 0

    def _get_policy_config(self) -> dict:
        return robotmq.deserialize(
            self.client.request_with_data("policy_config", robotmq.serialize(True))
        )

    def predict_action(self, obs: dict, num_envs: int, device: torch.device) -> torch.Tensor:
        """Send obs to remote policy server, receive action chunk (B, action_horizon, 8)."""
        batch = obs_to_policy_input(obs, num_envs, device, self.no_proprio)

        # Serialize: cameras as uint8, everything else as float
        send_dict = {}
        for k, v in batch.items():
            if k.endswith("camera"):
                send_dict[k] = (v * 255).byte().cpu().numpy()
            else:
                send_dict[k] = v.cpu().numpy()

        raw = self.client.request_with_data(
            "policy_inference", robotmq.serialize(send_dict), timeout_s=self.timeout_s,
        )
        result = robotmq.deserialize(raw)
        if isinstance(result, str):
            raise RuntimeError(f"Policy server error: {result}")

        # action0_8d = action-space output, robot0_8d = state-space (direct-qpos)
        output_key = "action0_8d" if "action0_8d" in result else "robot0_8d"
        return torch.from_numpy(result[output_key][:, :self.action_horizon, :]).to(device)

    def reset(self) -> None:
        self.client.request_with_data("policy_reset", robotmq.serialize(True))

    def report_progress(self, step: int, total: int) -> None:
        """Write eval progress to tmp file for orchestrator to read."""
        with open(self.progress_file, "w") as f:
            json.dump({"step": step, "total": total}, f)

    def report_results(self, results_dict: dict) -> None:
        self.client.put_data("rollout_results", robotmq.serialize(results_dict))
        self.client.request_with_data("done_rollout", robotmq.serialize(results_dict))

    def cleanup(self) -> None:
        """Remove tmp progress file."""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
