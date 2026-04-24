import copy
import time
from typing import Any

import robotmq

from imitation_learning.envs.base_env import BaseEnv


class RemoteEnv(BaseEnv):
    def __init__(
        self,
        server_endpoint: str,
        train_server_name: str,
        env_id: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_endpoint: str = server_endpoint
        self.train_server_name: str = train_server_name
        self.prev_epoch: int = -1
        self.prev_result_epoch: int = -1
        self.env_id: int = env_id
        self.client: robotmq.RMQClient = robotmq.RMQClient(
            client_name=f"{self.run_name}_{self.env_id}", server_endpoint=self.server_endpoint
        )
        self.rollout_running: bool = False

    def reset(self) -> None:
        self.rollout_running: bool = False
        self.prev_epoch: int = -1
        self.prev_result_epoch: int = -1

    def start_rollout(
        self,
        epoch: int,
        ckpt_path: str,
        enforce_increasing_epoch: bool = True,
        load_weights_only: bool = False,
        task_name: str = "",
        policy_name: str = "",
        exclude_timestamp: bool = False,
        record_history_attention: bool = False,
    ) -> None:

        if enforce_increasing_epoch:
            assert epoch > self.prev_epoch, "Epoch must be increasing"
        self.prev_epoch = epoch

        status = self.client.get_topic_status("available_checkpoints", timeout_s=1)

        if status == -2:
            print(
                f"Server at {self.server_endpoint} is not connected. Please start the server and try again."
            )
            return
        elif status == -1:
            print(
                f"Topic `available_checkpoints` is not available. Please create the topic on the server."
            )
            return
        else:
            ckpt_info_dict: dict[str, Any] = {
                "ckpt_path": ckpt_path,
                "train_server_name": self.train_server_name,
                "record_history_attention": record_history_attention,
                "load_weights_only": load_weights_only,
            }
            if load_weights_only:
                assert task_name and policy_name, "Task name and policy name must be provided when loading weights only"
                ckpt_info_dict["task_name"] = task_name
                ckpt_info_dict["policy_name"] = policy_name
            if not exclude_timestamp:
                ckpt_info_dict["completed_timestamp"] = time.time()
            ckpt_info = robotmq.serialize(ckpt_info_dict)
            self.client.put_data("available_checkpoints", ckpt_info)
            self.rollout_running = True

    def fetch_results(self, check_run_name: bool = True, enforce_increasing_epoch: bool = True) -> list[dict[str, Any]]:
        status = self.client.get_topic_status("available_checkpoints", timeout_s=1)

        if status == -2:
            print(
                f"Server at {self.server_endpoint} is not connected. Please start the server and try again."
            )
            return []
        elif status == -1:
            print(
                f"Topic `available_checkpoints` is not available. Please create the topic on the server."
            )
            return []
        else:
            if enforce_increasing_epoch:
                raw_results, timestamps = self.client.peek_data(
                    "rollout_results", n=0
                )
            else:
                # Consume all results
                raw_results, timestamps = self.client.pop_data(
                    "rollout_results", n=0
                )
            # earliest results are at the beginning of the queue


        results: list[dict[str, Any]] = []
        for result in raw_results:
            result_dict: dict[str, Any] = robotmq.deserialize(result)
            assert isinstance(result_dict, dict)
            if not check_run_name or (
                result_dict["run_name"] == self.run_name
                and result_dict["time_tag"] == self.time_tag
            ):
                result_dict.pop("run_name")
                result_dict.pop("time_tag")
                if not enforce_increasing_epoch or result_dict["epoch"] > self.prev_result_epoch:
                    results.append(copy.deepcopy(result_dict))
                    self.prev_result_epoch = result_dict["epoch"]

        if len(results) > 0:
            self.rollout_running = False

        return results

    def clear_results_buffer(self):
        self.client.pop_data("rollout_results", n=0)
