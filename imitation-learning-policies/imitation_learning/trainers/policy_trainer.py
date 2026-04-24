import copy
import os
from functools import partial
from typing import Any, cast

from imitation_learning.datasets.base_dataset import BaseDataset
import torch
import torch.nn.functional as F
import tqdm
from accelerate import Accelerator

from imitation_learning.common.datatypes import batch_type
from imitation_learning.envs.base_env import BaseEnv
from imitation_learning.policies.base_policy import BasePolicy
from imitation_learning.policies.history_denoising_policy import HistoryDenoisingPolicy
from imitation_learning.trainers.base_trainer import BaseTrainer
from robot_utils.torch_utils import aggregate_batch, torch_save
from imitation_learning.utils.data_utils import get_shakiness_score_torch



class PolicyTrainer(BaseTrainer):

    def __init__(
        self,
        rollout_every: int,
        rollout_env: BaseEnv | None,
        critical_action_loss_weights: list[float],
        memory_gate_loss_weight: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rollout_every: int = rollout_every
        assert rollout_every >= 0, "rollout_every must be non-negative"
        assert self.rollout_every == 0 or (
            self.rollout_every % self.checkpoint_every == 0
        ), "rollout_every must be a multiple of checkpoint_every"

        if not isinstance(rollout_env, BaseEnv):
            rollout_env = None
        self.rollout_env: BaseEnv | None = rollout_env
        self.memory_gate_loss_weight: float = memory_gate_loss_weight
        self.critical_action_loss_weights: list[float] = critical_action_loss_weights
        assert (
            len(self.critical_action_loss_weights) > 0
        ), "critical_action_loss_weights must be non-empty"

    def compute_loss(self, batch: batch_type) -> tuple[torch.Tensor, dict[str, float]]:
        assert self.model is not None

        # import cv2
        # import numpy as np
        # img = batch["third_person_camera"][0, 0].detach().cpu().numpy()
        # img = img.transpose(1, 2, 0)
        # img = img * 255.0
        # img = img.astype(np.uint8)
        # cv2.imwrite("debug_img.png", img)
        # exit()

        # print(f"{batch['traj_idx']=}")

        loss_dict = self.model(batch)
        detached_loss_dict: dict[str, float] = {}
        detached_loss_dict["train/action_loss"] = float(loss_dict["action"].detach().cpu())
        loss = loss_dict["action"]  # Will be replaced by other weighted sum

        if "memory_gate_val" in loss_dict:  
            loss += loss_dict["memory_gate_val"] * self.memory_gate_loss_weight
            detached_loss_dict["train/memory_gate_val"] = float(
                loss_dict["memory_gate_val"].detach().cpu()
            )
            detached_loss_dict["train/binary_memory_gate_val"] = float(
                loss_dict["binary_memory_gate_val"].detach().cpu()
            )

        if "critical_action" in loss_dict:
            idx = min(self.epoch, len(self.critical_action_loss_weights) - 1)
            try:
                loss += (
                    loss_dict["critical_action"]
                    * self.critical_action_loss_weights[idx]
                )
                detached_loss_dict["train/critical_action_loss"] = float(
                    loss_dict["critical_action"].detach().cpu()
                )
            except:
                print(f"{loss=}")
                print(f"{loss_dict['critical_action']=}")
                raise

        if "critical_memory_gate_val" in loss_dict:
            # Will not add to loss, but will log it
            detached_loss_dict["train/critical_memory_gate_val"] = float(
                loss_dict["critical_memory_gate_val"].detach().cpu()
            )
            detached_loss_dict["train/critical_binary_memory_gate_val"] = float(
                loss_dict["critical_binary_memory_gate_val"].detach().cpu()
            )

        detached_loss_dict["train/loss"] = float(loss.detach().cpu())

        return loss, detached_loss_dict

    def eval_model_step(
        self, step_log: dict[str, Any], train_sampling_batch: batch_type
    ):

        if self.epoch == self.num_epochs - 1:
            # There are some unknown bugs in the last epoch of validation
            return

        assert self.ema_model is not None
        assert self.val_dataloader is not None
        assert self.accelerator is not None

        eval_policy = self.ema_model.averaged_model
        assert isinstance(eval_policy, BasePolicy)

        def log_action_mse(
            step_log: "dict[str, Any]",
            category: str,
            pred_action: "dict[str, torch.Tensor]",
            gt_action: "dict[str, torch.Tensor]",
            accelerator: Accelerator,
            entire_traj_is_padding: torch.Tensor | None,  # (batch_size, traj_num)
            traj_is_error: torch.Tensor | None,  # (batch_size, traj_num)
            traj_is_critical: torch.Tensor | None,  # (batch_size, traj_num)
        ):
            single_gpu_stats: dict[str, Any] = {}
            # print(f"{entire_traj_is_padding=}, {traj_is_error=}, {traj_is_critical=}")
            for key in pred_action.keys():
                assert (
                    pred_action[key].shape == gt_action[key].shape
                ), f"{pred_action[key].shape=}, {gt_action[key].shape=}"

                mse = F.mse_loss(pred_action[key], gt_action[key], reduction="none")
                mse = mse.mean(
                    dim=(-2, -1)
                )  # Single-traj: (batch_size, ), Multi-traj: (batch_size, traj_num)

                if (
                    entire_traj_is_padding is not None
                ):  # For multi-trajectory evaluation
                    assert entire_traj_is_padding.shape == mse.shape
                    mse = mse * (~entire_traj_is_padding)

                if traj_is_error is not None:
                    assert traj_is_error.shape == mse.shape, f"{traj_is_error.shape=}, {mse.shape=}"
                    mse = mse * (~traj_is_error)

                if traj_is_critical is not None:
                    assert traj_is_critical.shape == mse.shape
                    critical_mse = mse * traj_is_critical
                    assert (mse != 0).sum() > 0, f"All {key} are 0"
                    if (critical_mse != 0).sum() == 0:
                        critical_mse = torch.zeros(1, device=critical_mse.device)
                    else:
                        critical_mse = critical_mse.sum() / (critical_mse != 0).sum()
                    single_gpu_stats[f"{category}/{key}_critical_mse"] = critical_mse

                mse = mse.sum() / (mse != 0).sum()
                single_gpu_stats[f"{category}/{key}_mse"] = mse
                single_gpu_stats[f"{category}/{key}_shakiness"] = torch.mean(
                    get_shakiness_score_torch(pred_action[key])
                )

            if accelerator.num_processes > 1:
                gathered_stats: dict[str, torch.Tensor] = {}
                for key in single_gpu_stats.keys():
                    gathered_stat = accelerator.gather(single_gpu_stats[key])
                    assert isinstance(gathered_stat, torch.Tensor)
                    gathered_stats[key] = torch.mean(gathered_stat, dim=0)

                if accelerator.is_main_process:
                    step_log.update(gathered_stats)
            else:
                step_log.update(single_gpu_stats)

        def get_traj_flags(
            batch: batch_type,
        ) -> "dict[str, torch.Tensor | None]":
            if "action_is_critical" in batch:
                traj_is_critical = batch["action_is_critical"].squeeze(-1).any(dim=-1) # (batch_size, traj_num) or (batch_size, )
            else:
                traj_is_critical = None

            if "action_is_error" in batch and isinstance(eval_policy, HistoryDenoisingPolicy):
                action_key_names = eval_policy.action_decoder.data_entry_names
                action_traj_length = batch[action_key_names[0]].shape[-2]
                traj_error_mask = torch.zeros(
                    action_traj_length,
                    device=batch["action_is_error"].device,
                )
                traj_error_mask[
                    cast(HistoryDenoisingPolicy, eval_policy)
                    .action_no_error_range[0] : cast(HistoryDenoisingPolicy, eval_policy)
                    .action_no_error_range[1]
                ] = 1
                traj_is_error = batch["action_is_error"].squeeze(-1) * traj_error_mask
                traj_is_error = traj_is_error.any(dim=-1) # (batch_size, traj_num) or (batch_size, )
            else:
                traj_is_error = None

            if "entire_traj_is_padding" in batch:
                entire_traj_is_padding = batch["entire_traj_is_padding"]
            else:
                entire_traj_is_padding = None

            if self.debug:
                print(
                    f"entire_traj_is_padding: {entire_traj_is_padding is not None}, traj_is_error: {traj_is_error is not None}, traj_is_critical: {traj_is_critical is not None}"
                )

            return {
                "entire_traj_is_padding": entire_traj_is_padding, # (batch_size, traj_num) or (batch_size, ) or None
                "traj_is_error": traj_is_error, # (batch_size, traj_num) or (batch_size, ) or None
                "traj_is_critical": traj_is_critical, # (batch_size, traj_num) or (batch_size, ) or None
            }

        if self.sample_every != 0 and (self.epoch % self.sample_every) == 0 and not self.debug:
            with torch.no_grad():
                flags = get_traj_flags(train_sampling_batch)

                gt_action = {}
                action_key_names = eval_policy.action_decoder.data_entry_names
                for key in action_key_names:
                    gt_action[key] = train_sampling_batch.pop(key)
                
                pred_action = eval_policy.predict_action(train_sampling_batch)

                log_action_mse(
                    step_log, "train_sample", pred_action, gt_action, self.accelerator, **flags
                )

        if self.val_every != 0 and (self.epoch % self.val_every) == 0:
            # The last epoch of validation is buggy. Will skip it

            with torch.no_grad():
                all_gt_actions: list[dict[str, torch.Tensor]] = []
                all_pred_actions: list[dict[str, torch.Tensor]] = []
                all_traj_flags: list[dict[str, torch.Tensor | None]] = (
                    []
                )  # (val_batch_num): (batch_size, traj_num)
                if hasattr(self.val_dataloader.base_dataloader.dataset, "resample_index_pool"):
                    self.val_dataloader.base_dataloader.dataset.resample_index_pool()
                for batch_idx, val_batch in enumerate(self.val_dataloader):
                    traj_flags = get_traj_flags(val_batch)
                    all_traj_flags.append(traj_flags)
                    gt_action = {}
                    action_key_names = eval_policy.action_decoder.data_entry_names
                    for key in action_key_names:
                        gt_action[key] = val_batch.pop(key)

                    pred_action = eval_policy.predict_action(val_batch)

                    all_gt_actions.append(gt_action)
                    all_pred_actions.append(pred_action)

                    if self.debug:
                        break

                gt_actions = aggregate_batch(all_gt_actions, partial(torch.cat, dim=0))
                pred_actions = aggregate_batch(
                    all_pred_actions, partial(torch.cat, dim=0)
                )

                flags = aggregate_batch(all_traj_flags, partial(torch.cat, dim=0))

                log_action_mse(
                    step_log, "val_sample", pred_actions, gt_actions, self.accelerator, **flags
                )

    def rollout_model_step(self, step_log: dict[str, Any]):
        assert self.accelerator is not None
        if (
            self.rollout_every != 0
            and self.epoch % self.rollout_every == 0
            and self.accelerator.is_main_process
        ):
            if self.rollout_env is not None:
                self.rollout_env.start_rollout(
                    self.epoch, self.checkpoint_manager.get_last_ckpt_path()
                )
                results = self.rollout_env.fetch_results()
                print(f"Rollout results: {results}")
                if len(results) > 0:
                    # If everything works well, there should be only one result,
                    # but in case there are multiple, use the latest one
                    result = results[-1]
                    step_log["rollout/success_rate"] = result["success_rate"]
                    step_log["rollout/epoch"] = result["epoch"]

    def eval_model(
        self, rounds: int, dataloader_names: list[str], ema_only: bool = True, use_episode_num: int = -1
    ):

        for dataloader_name in dataloader_names:
            assert dataloader_name in ["train", "val"]

        assert (
            self.cfg_str_unresolved is not None
            and self.model is not None
            and self.train_dataloader is not None
            and self.val_dataloader is not None
            and self.ema_model is not None
            and self.accelerator is not None
        )

        # To make it compatible with accelerate multi-gpu evaluation
        self.model.forward = self.model.predict_action
        self.ema_model.averaged_model.forward = (
            self.ema_model.averaged_model.predict_action
        )

        (
            self.train_dataloader,
            self.val_dataloader,
            self.model,
        ) = self.accelerator.prepare(
            self.train_dataloader,
            self.val_dataloader,
            self.model,
        )

        assert (
            self.train_dataloader is not None
            and self.val_dataloader is not None
            and self.model is not None
            and self.ema_model is not None
        )

        device: torch.device = self.model.device
        self.ema_model.to(device)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
        self.model.requires_grad_(False)
        self.model.eval()

        dataloader_dict = {
            "val": self.val_dataloader,
            "train": self.train_dataloader,
        }

        all_results = []

        for dataloader_name in dataloader_names:
            dataloader = dataloader_dict[dataloader_name]
            if use_episode_num != -1:
                dataset: BaseDataset = dataloader.base_dataloader.dataset
                dataset.trim_dataset_episodes(remaining_episode_num=use_episode_num)
                
            batch_results: list[dict[str, torch.Tensor]] = []
            with torch.no_grad():
                for round in range(rounds):
                    with tqdm.tqdm(
                        dataloader,
                        desc=f"Eval {dataloader_name}, round {round+1}/{rounds}",
                        leave=False,
                        mininterval=self.tqdm_interval_sec,
                        ncols=70,
                    ) as tepoch:
                        for i, batch in enumerate(tepoch):

                            gt_action = {}
                            for key, value in batch.items():
                                if key.startswith("action"):
                                    gt_action[key] = value

                            ema_pred_action = self.ema_model.averaged_model(batch)

                            batch_result: dict[str, torch.Tensor] = {}
                            # Calculate MSE errors for each key

                            if not ema_only:
                                policy_pred_action = self.model(batch)
                                for key in policy_pred_action.keys():
                                    policy_result_key = f"policy_{key}"
                                    assert (
                                        policy_pred_action[key].shape
                                        == gt_action[key].shape
                                    )
                                    batch_result[policy_result_key] = (
                                        policy_pred_action[key]
                                    )

                                    if (
                                        len(policy_pred_action[key].shape) == 3
                                    ):  # Single trajectory (batch_size, traj_length, action_dim)

                                        batch_result[
                                            f"{policy_result_key}_mse"
                                        ] = torch.nn.functional.mse_loss(
                                            policy_pred_action[key],
                                            gt_action[key],
                                            reduction="none",
                                        ).mean(
                                            dim=(1)
                                        )  # (batch_size, action_dim)

                                    elif (
                                        len(policy_pred_action[key].shape) == 4
                                    ):  # Multi-trajectory (batch_size, traj_num, traj_length, action_dim)
                                        entire_traj_is_padding: torch.Tensor = batch["entire_traj_is_padding"]

                                        policy_mse = torch.nn.functional.mse_loss(
                                            policy_pred_action[key],
                                            gt_action[key],
                                            reduction="none",
                                        ).mean(
                                            dim=(2)
                                        )  # (batch_size, traj_num, action_dim)
                                        policy_mse = policy_mse * (
                                            ~entire_traj_is_padding[:, :, None]
                                        )
                                        batch_result[f"{policy_result_key}_mse"] = (
                                            policy_mse
                                        )

                            for key in ema_pred_action.keys():
                                batch_result[f"gt_{key}"] = gt_action[key]
                                ema_result_key = f"ema_{key}"

                                assert (
                                    ema_pred_action[key].shape == gt_action[key].shape
                                )
                                batch_result[ema_result_key] = ema_pred_action[key]

                                if (
                                    len(ema_pred_action[key].shape) == 3
                                ):  # Single trajectory (batch_size, traj_length, action_dim)
                                    batch_result[
                                        f"{ema_result_key}_mse"
                                    ] = torch.nn.functional.mse_loss(
                                        ema_pred_action[key],
                                        gt_action[key],
                                        reduction="none",
                                    ).mean(
                                        dim=(1)
                                    )  # (batch_size, action_dim)

                                elif (
                                    len(ema_pred_action[key].shape) == 4
                                ):  # Multi-trajectory (batch_size, traj_num, traj_length, action_dim)
                                    entire_traj_is_padding: torch.Tensor = batch["entire_traj_is_padding"]
                                    batch_result["entire_traj_is_padding"] = entire_traj_is_padding
                                    ema_mse = torch.nn.functional.mse_loss(
                                        ema_pred_action[key],
                                        gt_action[key],
                                        reduction="none",
                                    ).mean(
                                        dim=(2)
                                    )  # (batch_size, traj_num, action_dim)
                                    ema_mse = ema_mse * (
                                        ~entire_traj_is_padding[:, :, None]
                                    )
                                    batch_result[f"{ema_result_key}_mse"] = ema_mse

                            batch_result["traj_idx"] = batch["traj_idx"]
                            batch_result["episode_idx"] = batch["episode_idx"]
                            batch_results.append(copy.deepcopy(batch_result))

            # Aggregate results accross gpus
            gathered_results = []

            for batch in batch_results:
                gathered_batch = {
                    key: self.accelerator.gather(val) for key, val in batch.items()
                }
                gathered_results.append(gathered_batch)

            if self.accelerator.is_main_process:
                results: dict[str, torch.Tensor] = aggregate_batch(
                    gathered_results, partial(torch.cat, dim=0)
                )
                results = {key: val.cpu() for key, val in results.items()}
                file_name = f"{dataloader_name}_results.pt"
                torch_save(
                    results,
                    os.path.join(
                        self.output_dir,
                        f"epoch_{self.epoch-1}_eval",
                        file_name,
                    ),
                )
                print(
                    f"Evaluation results saved to {os.path.join(self.output_dir, f'epoch_{self.epoch-1}_eval', file_name)}"
                )
                all_results.append(copy.deepcopy(results))
                
        if self.accelerator.is_main_process and len(dataloader_names) > 1:
            all_results = aggregate_batch(all_results, partial(torch.cat, dim=0))
            file_name = f"all_results.pt"
            torch_save(
                all_results,
                os.path.join(
                    self.output_dir, f"epoch_{self.epoch-1}_eval", file_name
                ),
            )
            print(
                f"All evaluation results saved to {os.path.join(self.output_dir, f'epoch_{self.epoch-1}_eval', file_name)}"
            )

        if self.accelerator.num_processes > 1:
            self.accelerator.wait_for_everyone()
