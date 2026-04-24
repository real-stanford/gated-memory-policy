import gc
import os
import subprocess
import time
from typing import Any, cast

import dill
import hydra
import numpy as np
import numpy.typing as npt
import robotmq
import torch
from omegaconf import DictConfig, OmegaConf
import cv2
from imitation_learning.common.dataclasses import construct_data_meta_dict
from imitation_learning.datasets.normalizer import FixedNormalizer
from imitation_learning.policies.base_policy import BasePolicy
from imitation_learning.policies.history_denoising_policy import HistoryDenoisingPolicy
from imitation_learning.utils.config_utils import compose_hydra_config
from robot_utils.data_utils import dict_apply
from robot_utils.image_utils import resize_with_padding
from robot_utils.torch_utils import torch_load, torch_save
from robot_utils.logging_utils import echo_exception
from robot_utils.pose_utils import convert_10d_to_batch, convert_batch_to_10d
from hydra.core.global_hydra import GlobalHydra
from imitation_learning.utils.config_utils import remove_keys_from_config

class PolicyServer:
    def __init__(
        self,
        server_endpoint: str,
        device: str,
        wait_ckpt_writing_time_s: int,
    ):
        self.device: torch.device = torch.device(device)
        self.server_endpoint: str = server_endpoint
        self.server: robotmq.RMQServer = robotmq.RMQServer(
            "policy_server", server_endpoint
        )
        self.wait_ckpt_writing_time_s: int = wait_ckpt_writing_time_s
        print(f"Remote policy server initialized on {server_endpoint}")
        message_remaining_time_s = 3600  # 1 hour

        # Communication with trainer
        self.server.add_topic("available_checkpoints", message_remaining_time_s)
        self.server.add_topic("rollout_results", message_remaining_time_s)

        # Communication with simulation environment
        self.server.add_topic("new_checkpoint_loaded", message_remaining_time_s)
        self.server.add_topic("eval_config", message_remaining_time_s)
        # Synchronous communication topics: only keep 1 minute for response
        self.server.add_topic("policy_config", message_remaining_time_s=60)
        self.server.add_topic("policy_reset", message_remaining_time_s=60)
        # Use shared memory to speed up data transfer # Shared memory is unstable in SLURM, don't know why
        # self.server.add_shared_memory_topic( 
        #     "policy_inference",
        #     message_remaining_time_s=60,
        #     shared_memory_size_gb=0.5,
        # )
        self.server.add_topic("policy_inference", message_remaining_time_s=60)
        self.server.add_topic("done_rollout", message_remaining_time_s=60)
        self.server.add_topic("export_recorded_data", message_remaining_time_s=60)

        self.policy: BasePolicy | None = None
        self.policy_cfg: DictConfig | dict[str, Any] | None = None
        self.ckpt_path: str | None = None
        self.normalizer: FixedNormalizer | None = None
        self.robot_num: int | None = None
        self.debug_cnt: int = 0

    def _load_ckpt(self, ckpt_info_dict: dict[str, Any]) -> None:
        """
        ckpt_info_dict:
            run_name: str
            time_tag: str
            epoch: int
            ckpt_path: str
            train_server_name: str
            ** other configs, e.g. switch_bin_color
        """

        assert ckpt_info_dict["ckpt_path"].endswith(
            ".ckpt"
        ), "Checkpoint path must end with .ckpt"
        if not os.path.exists(ckpt_info_dict["ckpt_path"]):
            # Download the checkpoint to the local directory
            current_dir = os.getcwd()
            os.makedirs(os.path.dirname(ckpt_info_dict["ckpt_path"]), exist_ok=True)
            ckpt_path = os.path.join(current_dir, ckpt_info_dict["ckpt_path"])
            if "completed_timestamp" in ckpt_info_dict:
                completed_timestamp = ckpt_info_dict["completed_timestamp"]
                sleep_time = max(
                    0, completed_timestamp + self.wait_ckpt_writing_time_s - time.time()
                )
                print(
                    f"Waiting for {sleep_time:.1f} seconds for checkpoint to be saved"
                )
                time.sleep(sleep_time)  # Wait until the checkpoint is properly saved

            subprocess.run(
                [
                    "scp",
                    f"{ckpt_info_dict['train_server_name']}:{ckpt_path}",
                    ckpt_info_dict["ckpt_path"],
                ],
                check=True,
            )
            print(f"Downloaded checkpoint to {ckpt_info_dict['ckpt_path']}")

        ckpt = torch_load(
            ckpt_info_dict["ckpt_path"], map_location=self.device, pickle_module=dill
        )

        self.ckpt_path = ckpt_info_dict["ckpt_path"]

        self.policy_cfg = cast(DictConfig, OmegaConf.create(ckpt["cfg_str_unresolved"]))


        ## HACK: Update configs for compatibility
        if "memory_gate" in self.policy_cfg["workspace"]["model"] and "denoising_network_partial" in self.policy_cfg["workspace"]["model"] and "binary_gating" not in self.policy_cfg["workspace"]["model"]["denoising_network_partial"]:
            self.policy_cfg["workspace"]["model"]["denoising_network_partial"]["binary_gating"] = True


        if (
            "record_history_attention" in ckpt_info_dict
            and ckpt_info_dict["record_history_attention"]
        ):

            print(f"Recording history attention")
            self.policy_cfg["workspace"]["model"]["denoising_network_partial"][
                "record_data_entries"
            ].append("history_cross_attention")
            if "denoising_network_partial" in self.policy_cfg["workspace"]["model"] and "memory_gate_val" not in self.policy_cfg["workspace"]["model"]["denoising_network_partial"]["record_data_entries"]:
                self.policy_cfg["workspace"]["model"]["denoising_network_partial"]["record_data_entries"].append("memory_gate_val")
        
        ckpt_info_dict["epoch"] = ckpt["epoch"]
        ckpt_info_dict["run_name"] = self.policy_cfg["run_name"]
        ckpt_info_dict["time_tag"] = f"{self.policy_cfg['date_str']}-{self.policy_cfg['time_str']}"

        self.policy_cfg["epoch"] = ckpt_info_dict["epoch"]

        if "longhist" in self.policy_cfg["policy_name"] and "iphumi" in self.policy_cfg["task_name"]:
            # Cache image tokens to speed up policy inference
            self.policy_cfg["workspace"]["model"]["global_cond_encoder"]["image_encoder_partial"]["individual_forward"] = True 

        cfg_path = ckpt_info_dict["ckpt_path"].replace(".ckpt", ".yaml")
        try:
            with open(cfg_path, "w") as f:
                f.write(OmegaConf.to_yaml(self.policy_cfg, resolve=False, sort_keys=True))
                print(f"Exported config to {cfg_path}")
        except PermissionError:
            import tempfile
            cfg_path = os.path.join(tempfile.gettempdir(), os.path.basename(cfg_path))
            with open(cfg_path, "w") as f:
                f.write(OmegaConf.to_yaml(self.policy_cfg, resolve=False, sort_keys=True))
                print(f"Exported config to {cfg_path} (fallback due to permissions)")

        self.robot_num = cast(int, self.policy_cfg["workspace"]["train_dataset"]["robot_num"])

        if "memory_gate" in self.policy_cfg["workspace"]["model"] and self.policy_cfg["workspace"]["model"]["memory_gate"] is not None:
            if "ckpt_path" in self.policy_cfg["workspace"]["model"]["memory_gate"]:
                self.policy_cfg["workspace"]["model"]["memory_gate"]["ckpt_path"] = ""

        # self.policy_cfg["workspace"]["model"]["noise_scheduler"]["inference_step_num"] = 12

        self.policy = hydra.utils.instantiate(self.policy_cfg["workspace"]["model"])
        assert isinstance(self.policy, BasePolicy), "Policy must be a BasePolicy"

        self.policy.load_state_dict(ckpt["model_state_dict"])
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.reset()
        data_meta = construct_data_meta_dict(
            self.policy_cfg["workspace"]["train_dataset"]["output_data_meta"]
        )
        self.normalizer = FixedNormalizer(data_meta)
        self.normalizer.to(self.device)
        # self.normalizer.load_state_dict(ckpt["normalizer_state_dict"])
        self.normalizer.load_state_dict(ckpt["normalizer_state_dict"], strict=False) # HACK: strict=False to ignore the identity normalizer


        print(f"Putting data to new_checkpoint_loaded")

        ckpt_info_dict["task_name"] = self.policy_cfg["task_name"]
        ckpt_info_dict["project_name"] = self.policy_cfg["project_name"]
        ckpt_info_dict["policy_name"] = self.policy_cfg["policy_name"]


        self.server.put_data(
            topic="new_checkpoint_loaded",
            data=robotmq.serialize(ckpt_info_dict),
        )
        print(f"Checkpoint loaded. Waiting for environment requests")

    def _load_weights_only(self, cfg: DictConfig, ckpt_path: str) -> None:
        """
        To load legacy weights that has different config from the current codebase: first init a new policy with the new config, then load the weights into the new policy.
        """
        self.policy_cfg = remove_keys_from_config(cfg)
        ckpt = torch_load(ckpt_path, map_location=self.device, pickle_module=dill)

        self.robot_num = cast(int, self.policy_cfg["workspace"]["train_dataset"]["robot_num"])

        self.policy = hydra.utils.instantiate(self.policy_cfg["workspace"]["model"])
        assert self.policy is not None, "Policy must be instantiated"
        self.policy.load_state_dict(ckpt["model_state_dict"])
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.reset()
        data_meta = construct_data_meta_dict(
            self.policy_cfg["workspace"]["train_dataset"]["output_data_meta"]
        )
        self.normalizer = FixedNormalizer(data_meta)
        self.normalizer.to(self.device)
        self.normalizer.load_state_dict(ckpt["normalizer_state_dict"])

        # time_tag = datetime.datetime.now()

        ckpt_cfg = cast(DictConfig, OmegaConf.create(ckpt["cfg_str_unresolved"]))
        print(self.policy_cfg)

        ckpt_info_dict = {
            "time_tag": f"{ckpt_cfg['date_str']}-{ckpt_cfg['time_str']}",
            "run_name": ckpt_cfg["run_name"],
            "project_name": ckpt_cfg["project_name"],
            "task_name": self.policy_cfg["task_name"],
            "epoch": ckpt["epoch"],
            "ckpt_path": ckpt_path,
            "train_server_name": self.server_endpoint,
        }
        print(f"ckpt_info_dict: {ckpt_info_dict}")

        self.server.put_data(
            topic="new_checkpoint_loaded",
            data=robotmq.serialize(ckpt_info_dict),
        )
        print(f"Checkpoint loaded. Waiting for environment requests")

    def _predict_action(
        self,
        input_data_dict: dict[str, npt.NDArray[np.float32]],
        episode_idx: npt.NDArray[np.int32],
    ) -> dict[str, npt.NDArray[np.float32]]:
        """
        input_data_dict:
            robot0_wrist_camera: (batch_size, obs_history_len, 3, H, W)
            robot0_10d: (batch_size, obs_history_len, 10,)
            third_person_camera: (batch_size, obs_history_len, 3, H, W)
        return:
            action0_10d: (batch_size, 1, 10,) (Mujoco tasks)

        """

        assert self.policy is not None, "Policy must be loaded before predicting action"

        batch_data: dict[str, torch.Tensor] = {}

        batch_data = dict_apply(
            input_data_dict, lambda x: torch.from_numpy(x).float().to(self.device)
        )

        assert self.robot_num is not None, "Robot number must be set"

        for robot_idx in range(self.robot_num):
            if (
                f"robot{robot_idx}_10d" in batch_data.keys()
                and self.policy.local_cond_encoder is not None
            ):
                proprio_len = self.policy.local_cond_encoder.cond_meta[
                    f"robot{robot_idx}_10d"
                ].length
                assert batch_data[f"robot{robot_idx}_10d"].shape[1] == proprio_len, f"Robot {robot_idx} 10d shape: {batch_data[f'robot{robot_idx}_10d'].shape}, proprio_len: {proprio_len}"

        batch_data["episode_idx"] = torch.tensor(episode_idx, device=self.device)

        if self.normalizer is not None:
            batch_data = self.normalizer.normalize(batch_data)


        os.makedirs(f"debug_images", exist_ok=True)
        for k, v in batch_data.items():
            if k.endswith("camera"):
                img = v[0, 0, ...].cpu().numpy() * 255.0
                img = img.transpose(1, 2, 0)
                cv2.imwrite(f"debug_images/{self.debug_cnt}_{k}.png", img)
        self.debug_cnt += 1

        with torch.no_grad():
            normalized_action = self.policy.predict_action(batch_data)
        
        # print(f"normalized_action: {normalized_action}")
        # print("--------------------------------")   
        if self.normalizer is not None:
            unnormalized_action = self.normalizer.unnormalize(normalized_action)
        else:
            unnormalized_action = normalized_action

        action = {k: v.detach().cpu().numpy() for k, v in unnormalized_action.items()}

        # print(f"action: {action}")
        # print("--------------------------------")
        return action

    def run(self):
        while True:
            data, topic = self.server.wait_for_request(timeout_s=0.1)

            if topic == "policy_inference":
                if self.policy is None:
                    self.server.reply_request(
                        topic="policy_inference",
                        data=robotmq.serialize("Policy not loaded"),
                    )
                    continue
                try:
                    assert self.robot_num is not None, "Robot number must be set"
                    process_start_time = time.time()
                    raw_dict: dict[str, Any] = robotmq.deserialize(data)

                    if isinstance(raw_dict["episode_idx"], list):
                        raw_dict["episode_idx"] = np.array(raw_dict["episode_idx"])

                    # Means the data dict only has one episode, should be extended for the batch dimension
                    expand_data_batch_dim = False
                    if isinstance(raw_dict["episode_idx"], int) or raw_dict["episode_idx"].shape == (): 
                        expand_data_batch_dim = True
                        raw_dict["episode_idx"] = np.array([raw_dict["episode_idx"]])
                        for k in raw_dict.keys():
                            if k != "episode_idx" and isinstance(raw_dict[k], np.ndarray):
                                raw_dict[k] = raw_dict[k][np.newaxis, ...]

                    # for k in raw_dict.keys():
                    #     if isinstance(raw_dict[k], np.ndarray):
                    #         print(f"{k}: {raw_dict[k].shape}")
                    
                    # If the data does not have 10d pose, convert the eef_xyz_wxyz and gripper_width to 10d pose
                    convert_to_10d = False
                    for i in range(self.robot_num):
                        if f"robot{i}_gripper_width" in raw_dict.keys() and f"robot{i}_eef_xyz_wxyz" in raw_dict.keys():
                            convert_to_10d = True
                            if raw_dict[f"robot{i}_eef_xyz_wxyz"].size == 0 or raw_dict[f"robot{i}_gripper_width"].size == 0:
                                continue
                            raw_dict[f"robot{i}_10d"] = convert_batch_to_10d(raw_dict[f"robot{i}_eef_xyz_wxyz"], raw_dict[f"robot{i}_gripper_width"])
                    """
                    raw_dict:
                    {
                        "robot0_wrist_camera": (batch_size, obs_history_len, 3, H, W)
                        "robot0_10d": (batch_size, obs_history_len, 10,)
                        "third_person_camera": (batch_size, obs_history_len, 3, H, W)
                        "episode_idx": (batch_size,)
                    }
                    """

                    for k in raw_dict.keys():
                        if (
                            isinstance(raw_dict[k], np.ndarray)
                            and raw_dict[k].dtype == np.uint8
                        ):
                            assert "camera" in k or "image" in k, f"Only camera images are allowed to be uint8, got {k}"
                            raw_dict[k] = raw_dict[k].astype(np.float32) / 255.0
                        if "camera" in k or "image" in k:
                            if raw_dict[k].shape[-1] == 3:
                                raw_dict[k] = raw_dict[k].transpose(0, 1, 4, 2, 3) # (batch_size, obs_history_len, H, W, 3) -> (batch_size, obs_history_len, 3, H, W)
                            if k in self.policy.global_cond_encoder.cond_meta.keys() and raw_dict[k].shape[-2:] != (self.policy.global_cond_encoder.cond_meta[k].shape[-2:]):
                                print(f"Resizing {k} from {raw_dict[k].shape[-2:]} to {self.policy.global_cond_encoder.cond_meta[k].shape[-2:]}")
                                raw_dict[k] = resize_with_padding(raw_dict[k], self.policy.global_cond_encoder.cond_meta[k].shape[-2:])

                    global_cond_keys = [
                        meta.name
                        for meta in self.policy.global_cond_encoder.cond_meta.values()
                    ]
                    if self.policy.local_cond_encoder is not None:
                        local_cond_keys = [
                            meta.name
                            for meta in self.policy.local_cond_encoder.cond_meta.values()
                        ]
                    else:
                        local_cond_keys = []
                    if (
                        isinstance(self.policy, HistoryDenoisingPolicy)
                        and self.policy.history_img_feature_encoder is not None
                    ):
                        history_keys = (
                            self.policy.history_img_feature_encoder.data_entry_names
                        )
                    else:
                        history_keys = []

                    keys: list[str] = global_cond_keys + local_cond_keys + history_keys
                    keys = [k for k in keys if k in raw_dict.keys()]
                    policy_start_time = time.time()
                    actions = self._predict_action(
                        {k: raw_dict[k] for k in keys}, raw_dict["episode_idx"]
                    )

                    if convert_to_10d:
                        for i in range(self.robot_num):
                            actions[f"action{i}_eef_xyz_wxyz"], actions[f"action{i}_gripper_width"] = convert_10d_to_batch(actions[f"action{i}_10d"])

                    assert self.policy_cfg is not None, "Policy config must be set"
                    if "ptp" in self.policy_cfg["policy_name"]:
                        # Past token prediction: only return the actions indices that are >= 0
                        try:
                            action_indices = self.policy_cfg["workspace"]["model"]["action_indices"]
                        except KeyError:
                            # For compatibility with old configs
                            action_indices = self.policy_cfg["workspace"]["train_dataset"]["action_indices"]
                        zero_index = action_indices.index(0)
                        past_actions = {}
                        for k, v in actions.items():
                            if k.startswith("action"):
                                assert len(action_indices) == v.shape[1], "Action indices must match the action length"
                                actions[k] = v[:, zero_index:]
                                past_actions[k.replace("action", "past_action")] = v[:, :zero_index]
                        actions.update(past_actions)

                    if expand_data_batch_dim:
                        # Remove the batch dimension
                        for k in actions.keys():
                            actions[k] = actions[k][0]

                    print(
                        f"{self.server_endpoint}: Data processing time: {policy_start_time - process_start_time: .4f}s, Policy inference time: {time.time() - policy_start_time: .4f}s"
                    )
                    self.server.reply_request(
                        topic="policy_inference",
                        data=robotmq.serialize(actions),
                    )
                except Exception as e:
                    err_str = echo_exception()
                    print(f"Error in policy inference: {err_str}")
                    self.server.reply_request(
                        topic="policy_inference",
                        data=robotmq.serialize(err_str),
                    )
            elif topic == "policy_config":
                print("Policy config requested")
                try:
                    if isinstance(self.policy_cfg, DictConfig):
                        cfg_dict = OmegaConf.to_container(self.policy_cfg)
                    else:
                        cfg_dict = self.policy_cfg
                    self.server.reply_request(
                        topic="policy_config",
                        data=robotmq.serialize(cfg_dict),
                    )
                except Exception as e:
                    print(f"Error in loading policy config: {e}")
                    cfg_dict = {}
                    err_str = echo_exception()
                    print(f"Error in policy config: {err_str}")
                    self.server.reply_request(
                        topic="policy_config",
                        data=robotmq.serialize(err_str),
                    )
            elif topic == "policy_reset":
                if self.policy is not None:
                    self.policy.reset()
                self.server.reply_request(
                    topic="policy_reset",
                    data=robotmq.serialize(self.policy is not None),
                )
            elif topic == "done_rollout":
                rollout_result = robotmq.deserialize(data)
                self.server.reply_request(
                    topic="done_rollout",
                    data=robotmq.serialize(self.policy is not None),
                )

                if (
                    self.policy is not None
                    and hasattr(self.policy, "recorded_data_dicts")
                    and len(self.policy.recorded_data_dicts.keys()) > 0
                ):
                    self.export_recorded_data()

                if self.policy is not None:
                    self.policy.reset() # Clear data to prevent memory leak
                    print(f"Deleting policy {type(self.policy)}")
                    del self.policy

                self.policy = None
                self.policy_cfg = None
                self.ckpt_path = None
                self.normalizer = None
                self.robot_num = None
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(self.device)
                print(f"Rollout done. {rollout_result} Waiting for new checkpoint")

            elif topic == "export_recorded_data":
                file_name: str = robotmq.deserialize(data)
                export_path = self.export_recorded_data(file_name)
                self.server.reply_request(
                    topic="export_recorded_data",
                    data=robotmq.serialize(export_path),
                )

            if self.policy is None:
                ckpt_info, _ = self.server.pop_data(
                    "available_checkpoints", n=1
                )
                if len(ckpt_info) == 0:
                    time.sleep(1)
                    continue
                ckpt_info_dict: dict[str, Any] = robotmq.deserialize(ckpt_info[0])


                try:
                    if "load_weights_only" in ckpt_info_dict and ckpt_info_dict["load_weights_only"]:
                        print(f"Loading weights only from {ckpt_info_dict['ckpt_path']}")
                        GlobalHydra.instance().clear()
                        with hydra.initialize('../../imitation_learning/configs'):
                            cfg = compose_hydra_config(ckpt_info_dict["task_name"], ckpt_info_dict["policy_name"])
                        self._load_weights_only(cfg, ckpt_info_dict["ckpt_path"])
                    else:
                        print(f"Loading checkpoint {ckpt_info_dict['ckpt_path']}")
                        self._load_ckpt(ckpt_info_dict)
                except Exception as e:
                    error_str = echo_exception()
                    print(
                        f"Error loading checkpoint {ckpt_info_dict['ckpt_path']}: {error_str}"
                    )
                    continue

    def export_recorded_data(self, file_name: str | None = None):
        assert self.ckpt_path is not None, "Checkpoint path must be set"
        assert self.policy is not None, "Policy must be loaded"
        if not hasattr(self.policy, "recorded_data_dicts") or len(self.policy.recorded_data_dicts) == 0:
            print("No recorded data to export")
            return "No recorded data to export"

        run_dir = os.path.dirname(os.path.dirname(self.ckpt_path))
        ckpt_name = os.path.basename(self.ckpt_path)
        if not os.path.exists(os.path.join(run_dir, "recorded_data")):
            os.makedirs(os.path.join(run_dir, "recorded_data"))
        if file_name is None:
            data_file_name = f"{ckpt_name}.pt".replace(".ckpt", "")
        else:
            data_file_name = f"{ckpt_name}_{file_name}.pt".replace(".ckpt", "")
        export_path = os.path.join(
            run_dir,
            "recorded_data",
            data_file_name,
        )
        with open(export_path, "wb") as f:
            torch_save(self.policy.recorded_data_dicts, f, pickle_module=dill)
        current_path = os.getcwd()
        print(f"Exported recorded data to {current_path}/{export_path}")
        self.policy.recorded_data_dicts = {}
        return f"{current_path}/{export_path}"
