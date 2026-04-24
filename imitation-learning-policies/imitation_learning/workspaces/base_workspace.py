from typing import Any, Callable, cast

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from imitation_learning.common.datatypes import batch_type
from imitation_learning.datasets.base_dataset import BaseDataset
from imitation_learning.datasets.multi_task_dataset import MultiTaskDataset
from imitation_learning.models.ema_model import EMAModel
from imitation_learning.trainers.base_trainer import BaseTrainer


class BaseWorkspace:
    def __init__(
        self,
        name: str,
        train_dataset: BaseDataset,
        model: nn.Module,
        ema_partial: Callable[..., EMAModel],
        trainer: BaseTrainer,
        logging_cfg: dict[str, Any] | OmegaConf,
        cfg_str_unresolved: str,
    ):
        self.trainer: BaseTrainer = trainer
        self.model: nn.Module = model
        self.name: str = name
        self.ema_partial: Callable[..., EMAModel] = ema_partial

        if isinstance(logging_cfg, OmegaConf):
            logging_cfg = cast(
                dict[str, Any], OmegaConf.to_container(logging_cfg, resolve=True)
            )
        self.logging_cfg: dict[str, Any] = logging_cfg

        self.train_dataset: BaseDataset = train_dataset

        # if self.train_dataset.normalizer is None and is_main_process():
        #     self.train_dataset.fit_normalizer()

        if not isinstance(self.train_dataset, MultiTaskDataset) and self.train_dataset.normalizer is None:
            raise ValueError("Normalizer is not found in the dataset. Please fit the normalizer first.")

        self.val_dataset: BaseDataset = train_dataset.split_unused_episodes()
        self.val_dataset.dataloader_cfg["shuffle"] = False

        if self.trainer.debug:
            self.val_dataset.dataloader_cfg["batch_size"] = (
                self.train_dataset.dataloader_cfg["batch_size"]
            )
            # Will not expand the batch size for validation

        self.train_dataset.repeat_dataset() # Only repeat the training dataset
        self.train_dataloader: DataLoader[batch_type] = (
            self.train_dataset.get_dataloader()
        )
        self.val_dataloader: DataLoader[batch_type] = self.val_dataset.get_dataloader()

        self.cfg_str_unresolved: str = cfg_str_unresolved

        self.ema_model: EMAModel = self.ema_partial(self.model)

    def train(self):
        self.trainer.init_trainer(
            cfg_str_unresolved=self.cfg_str_unresolved,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            model=self.model,
            ema_model=self.ema_model,
            logging_cfg=self.logging_cfg,
        )
        self.trainer.run()

    def load_base_ckpt(
        self, base_ckpt: dict[str, Any], ignore_keywords: list[str] | None = None
    ):
        """
        Load only the model states (not the workspace states)
        """
        if ignore_keywords is not None:
            base_ckpt["model_state_dict"] = {
                k: v
                for k, v in base_ckpt["model_state_dict"].items()
                if not any(keyword in k for keyword in ignore_keywords)
            }
            if "ema_model_state_dict" in base_ckpt:
                base_ckpt["ema_model_state_dict"] = {
                    k: v
                    for k, v in base_ckpt["ema_model_state_dict"].items()
                    if "history" not in k
                }
        self.model.load_state_dict(base_ckpt["model_state_dict"], strict=False)
        if "ema_model_state_dict" in base_ckpt:
            self.ema_model.load_state_dict(
                base_ckpt["ema_model_state_dict"], strict=False
            )
        else:
            self.ema_model.averaged_model.load_state_dict(
                base_ckpt["model_state_dict"], strict=False
            )

    def load_module_ckpt(self, module_name: str, ckpt: dict[str, Any]):
        assert hasattr(
            self.model, module_name
        ), f"Module {module_name} not found in model"
        module = getattr(self.model, module_name)
        module.load_state_dict(ckpt["model_state_dict"], strict=False)
        ema_module = getattr(self.ema_model.averaged_model, module_name)
        ema_module.load_state_dict(ckpt["ema_model_state_dict"], strict=False)

    def resume_training(self, ckpt: dict[str, Any]):
        self._load_workspace(ckpt)
        self.trainer.run()

    def _load_workspace(self, ckpt: dict[str, Any]):
        """
        Load both the model and the workspace states (normalizer, optimizer, lr_scheduler, etc.)
        """
        # Check whether the normalizer is correct
        assert (
            self.train_dataset.normalizer is not None
            and self.val_dataset.normalizer is not None
        )
        normalizer_state_dict = self.train_dataset.normalizer.state_dict()
        for key in normalizer_state_dict:
            if not torch.allclose(
                normalizer_state_dict[key], ckpt["normalizer_state_dict"][key]
            ):
                print(f"Normalizer state dict mismatch for key {key}")
                print(normalizer_state_dict)
                print(ckpt["normalizer_state_dict"])
                print("Loading normalizer state dict from ckpt")
                self.train_dataset.normalizer.load_state_dict(
                    ckpt["normalizer_state_dict"]
                )
                self.val_dataset.normalizer.load_state_dict(
                    ckpt["normalizer_state_dict"]
                )
                break

        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)

        if "ema_model_state_dict" in ckpt:
            self.ema_model.load_state_dict(ckpt["ema_model_state_dict"], strict=False)
        else:
            self.ema_model.averaged_model.load_state_dict(
                ckpt["model_state_dict"], strict=False
            )

        if "optimizer_state_dict" in ckpt:
            optimizer_state_dict = ckpt["optimizer_state_dict"]
        else:
            optimizer_state_dict = None
            print("No optimizer state dict found in ckpt")

        if "lr_scheduler_state_dict" in ckpt:
            lr_scheduler_state_dict = ckpt["lr_scheduler_state_dict"]
        else:
            lr_scheduler_state_dict = None
            print("No lr_scheduler state dict found in ckpt")

        self.trainer.init_trainer(
            cfg_str_unresolved=self.cfg_str_unresolved,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            model=self.model,
            logging_cfg=self.logging_cfg,
            optimizer_state_dict=optimizer_state_dict,
            lr_scheduler_state_dict=lr_scheduler_state_dict,
            ema_model=self.ema_model,
        )
        self.trainer.epoch = ckpt["epoch"] + 1
        self.trainer.global_step = ckpt["global_step"] + 1

    def eval_model(
        self,
        ckpt: dict[str, Any],
        rounds: int = 1,
        dataloader_names: list[str] = ["train", "val"],
        use_episode_num: int = -1,
    ):
        self._load_workspace(ckpt)
        self.trainer.eval_model(rounds, dataloader_names, use_episode_num=use_episode_num)
