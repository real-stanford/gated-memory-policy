from functools import partial
from typing import Any, cast

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from imitation_learning.common.datatypes import batch_type
from imitation_learning.models.common.memory_gate import \
    MemoryGate
from imitation_learning.trainers.base_trainer import BaseTrainer
from robot_utils.torch_utils import aggregate_batch


class MemoryGateTrainer(BaseTrainer):

    def __init__(
        self, 
        error_ratio_threshold: float, 
        **kwargs):
        self.error_ratio_threshold: float = error_ratio_threshold
        self.loss_fn_name: str = ""
        super().__init__(**kwargs)

    def _add_gt_gate_labels(self, batch: batch_type):
        if "gate_label" not in batch:
            if self.loss_fn_name == "bce":
                batch["gate_label"] = (batch["with_mem_errors"] * self.error_ratio_threshold < batch["no_mem_errors"]).to(torch.float32)
            elif self.loss_fn_name == "mse":
                batch["gate_label"] = torch.clamp(batch["no_mem_errors"] / (batch["with_mem_errors"] * self.error_ratio_threshold), 0.0, 1.0)
            else:
                raise ValueError(f"Invalid loss function: {self.model.loss_fn_name}. Only support bce (for binary cross entropy) or mse (for mean squared error) loss function.")
            batch.pop("with_mem_errors")
            batch.pop("no_mem_errors")
            batch.pop("with_mem_variances")
            batch.pop("no_mem_variances")
        return batch["gate_label"]

    def init_trainer(self, **kwargs):
        super().init_trainer(**kwargs)
        assert self.model is not None
        self.loss_fn_name = self.model.loss_fn_name

    def compute_loss(self, batch: batch_type) -> tuple[torch.Tensor, dict[str, float]]:
        # gt_gate_label = batch["gate_label"]
        # batch["gate_label"] = torch.zeros_like(batch["episode_idx"], dtype=torch.float32)
        self._add_gt_gate_labels(batch)


        # print(f"{batch['gate_label']=}")

        # gt_gate_val = batch["meta"]["variance"]

        # proprio = batch["robot0_10d"]
        # images = batch["robot0_wrist_camera"]

        self.model = cast(MemoryGate, self.model)

        loss: torch.Tensor = self.model(batch)
        detached_loss_dict: dict[str, float] = {}
        detached_loss_dict["train/loss"] = float(loss.detach().cpu())
        return loss, detached_loss_dict

    def eval_model_step(
        self, step_log: dict[str, Any], train_sampling_batch: dict[str, Any]
    ):
        assert self.ema_model is not None
        assert self.val_dataloader is not None
        assert self.accelerator is not None

        eval_model = self.ema_model.averaged_model
        assert isinstance(eval_model, MemoryGate)
        self._add_gt_gate_labels(train_sampling_batch)

        def log_gate_value_mse(
            step_log: dict[str, Any],
            category: str,
            pred_gate_val: torch.Tensor,
            gt_gate_val: torch.Tensor,
            accelerator: Accelerator,
        ):
            single_gpu_stats: dict[str, Any] = {}
            assert pred_gate_val.shape == gt_gate_val.shape
            single_gpu_stats[f"{category}_gate_val_mse"] = F.mse_loss(
                pred_gate_val, gt_gate_val
            )

            gathered_stats: dict[str, torch.Tensor] = {}
            for key in single_gpu_stats.keys():
                gathered_stat = accelerator.gather(single_gpu_stats[key])
                assert isinstance(gathered_stat, torch.Tensor)
                gathered_stats[key] = torch.mean(gathered_stat, dim=0)

            if accelerator.is_main_process:
                step_log.update(gathered_stats)

            if accelerator.is_main_process:
                step_log.update(single_gpu_stats)

        if (self.epoch % self.sample_every) == 0:
            with torch.no_grad():
                pred_gate_val = eval_model.get_gate_value(
                    train_sampling_batch,
                )
                log_gate_value_mse(
                    step_log,
                    "train",
                    pred_gate_val,
                    train_sampling_batch["gate_label"],
                    self.accelerator,
                )

        if (self.epoch % self.val_every) == 0:
            with torch.no_grad():
                all_gt_gate_val = []
                all_pred_gate_val = []
                for val_batch in self.val_dataloader:
                    self._add_gt_gate_labels(val_batch)
                    pred_gate_val = eval_model.get_gate_value(
                        val_batch,
                    )
                    all_gt_gate_val.append(val_batch["gate_label"])
                    all_pred_gate_val.append(pred_gate_val)
                gt_gate_val = aggregate_batch(
                    all_gt_gate_val, partial(torch.cat, dim=0)
                )
                pred_gate_val = aggregate_batch(
                    all_pred_gate_val, partial(torch.cat, dim=0)
                )
                log_gate_value_mse(
                    step_log, "val", pred_gate_val, gt_gate_val, self.accelerator
                )
