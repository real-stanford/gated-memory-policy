import copy
import threading
from itertools import chain
from typing import Any, Callable, cast

import numpy as np
import torch
import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.accelerator import KwargsHandler
from accelerate.utils import broadcast
from diffusers.optimization import get_scheduler
from omegaconf import ListConfig, OmegaConf
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from imitation_learning.common.datatypes import batch_type
from imitation_learning.datasets.base_dataset import BaseDataset
from imitation_learning.models.ema_model import EMAModel
from imitation_learning.policies.base_denoising_policy import BaseDenoisingPolicy
from imitation_learning.trainers.checkpoint_manager import CheckpointManager
from robot_utils.torch_utils import aggregate_batch, exclude_params, filter_params, params, to_cpu
from robot_utils.logging_utils import (merge_param_names, print_once, print_step_log)

class BaseTrainer:
    def __init__(
        self,
        seed: int,
        optimizer_partial: Callable[..., torch.optim.Optimizer],
        lr_scheduler_name: str,
        lr_warmup_steps: int,
        num_epochs: int,
        val_every: int,
        sample_every: int,
        checkpoint_every: int,
        output_dir: str,
        tqdm_interval_sec: float,
        mixed_precision: str,
        gradient_accumulation_steps: int,
        checkpoint_manager: CheckpointManager,
        finetune_keywords: list[str],
        regular_keywords: list[
            str
        ],  # Will forcefully set the learning rate to the regular learning rate, even if it is in the finetune_keywords
        skip_saving_keywords: list[str],
        keep_saving_keywords: list[str],
        debug: bool,
        loss_temperatures: list[float],
        clip_grad_norms: list[float],  # 0 means no clipping
        finetune_lr_ratio: float,
        **unused_kwargs,
    ):
        print(f"BaseTrainer unused kwargs: {unused_kwargs}")
        
        self.output_dir: str = output_dir
        self.lr_warmup_steps: int = lr_warmup_steps
        self.lr_scheduler_name: str = lr_scheduler_name

        self.num_epochs: int = num_epochs
        self.seed: int = seed
        self.optimizer_partial: Callable[..., torch.optim.Optimizer] = optimizer_partial
        self.torch_rng: torch.Generator = torch.Generator().manual_seed(seed)
        self.np_rng: np.random.Generator = np.random.default_rng(seed)

        self.global_step: int = 0
        self.epoch: int = 0

        assert (
            val_every >= 0
        ), "val_every must be non-negative. Use 0 to disable validation."
        self.val_every: int = val_every
        self.sample_every: int = sample_every
        self.checkpoint_every: int = checkpoint_every

        self.tqdm_interval_sec: float = tqdm_interval_sec

        self._saving_thread: threading.Thread | None = None
        self.mixed_precision: str = mixed_precision
        self.gradient_accumulation_steps: int = gradient_accumulation_steps

        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: LambdaLR | None = None
        self.model: nn.Module | None = None
        self.ema_model: EMAModel | None = None
        self.train_dataloader: DataLoader[batch_type] | None = None
        self.val_dataloader: DataLoader[batch_type] | None = None
        self.logging_cfg: dict[str, Any] | None = None
        self.cfg_str_unresolved: str | None = None
        self.accelerator: Accelerator | None = None

        self.checkpoint_manager: CheckpointManager = checkpoint_manager
        self.finetune_keywords: list[str] = (
            finetune_keywords  # Keywords for parameters that should have lower learning rate
        )
        self.regular_keywords: list[str] = (
            regular_keywords  # Keywords for parameters that should have the regular learning rate
        )
        self.skip_saving_keywords: list[str] = (
            skip_saving_keywords  # Keywords for parameters that should not be saved
        )
        self.keep_saving_keywords: list[str] = (
            keep_saving_keywords  # Keywords for parameters that should be saved, even if they are part of skip_saving_keywords
        )
        self.finetune_lr_ratio: float = finetune_lr_ratio
        self.debug: bool = debug
        self.loss_temperatures: list[float] | None = loss_temperatures
        if isinstance(clip_grad_norms, ListConfig):
            clip_grad_norms = list(clip_grad_norms)
        assert all(
            clip_grad_norm >= 0 for clip_grad_norm in clip_grad_norms
        ), "clip_grad_norm must be non-negative"
        self.clip_grad_norms: list[float] = clip_grad_norms

    def init_trainer(
        self,
        train_dataloader: DataLoader[batch_type],
        val_dataloader: DataLoader[batch_type],
        model: nn.Module,
        ema_model: EMAModel,
        cfg_str_unresolved: str,
        logging_cfg: dict[str, Any] | None = None,
        optimizer_state_dict: dict[str, Any] | None = None,
        lr_scheduler_state_dict: dict[str, Any] | None = None,
    ):
        # print(f"BaseTrainer.init_trainer: Model device: {model.device}")
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.logging_cfg = logging_cfg
        self.cfg_str_unresolved = cfg_str_unresolved
        self.ema_model = ema_model
        assert (
            self.cfg_str_unresolved is not None
            and self.model is not None
            and self.train_dataloader is not None
            and self.val_dataloader is not None
        ), f"cfg_str_unresolved: {self.cfg_str_unresolved}, model: {self.model}, train_dataloader: {self.train_dataloader}, val_dataloader: {self.val_dataloader}"

        kwargs_handlers: list[KwargsHandler] = []
        kwargs_handlers.append(
            DistributedDataParallelKwargs(find_unused_parameters=True)
        )


        if self.mixed_precision != "":
            self.accelerator = Accelerator(
                log_with="wandb",
                mixed_precision=self.mixed_precision,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                kwargs_handlers=kwargs_handlers,
            )
        else:
            self.accelerator = Accelerator(
                log_with="wandb",
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                kwargs_handlers=kwargs_handlers,
            )   


        if self.logging_cfg is not None and not self.debug:
            wandb_cfg = self.logging_cfg.copy()
            wandb_cfg.pop("project")
            # wandb_cfg["dir"] = self.output_dir
            config = OmegaConf.to_container(OmegaConf.create(self.cfg_str_unresolved))
            assert type(config) == dict
            self.accelerator.init_trackers(
                project_name=self.logging_cfg["project"],
                config=config,
                init_kwargs={"wandb": wandb_cfg},
            )

        frozen_params = filter_params(
            self.model.named_parameters(), None, requires_grad=False
        )

        frozen_param_names = merge_param_names(
            [name for name, _ in frozen_params], layers=4, stop_at_numbers=True
        )
        frozen_param_str = "\n\t".join(frozen_param_names)
        print(f"Frozen params: \n\t{frozen_param_str}")

        # Print finetune param names: finetune_keywords exclude regular_keywords
        finetune_params = exclude_params(
            filter_params(
                self.model.named_parameters(),
                self.finetune_keywords,
                requires_grad=True,
            ),
            self.regular_keywords,
            requires_grad=True,
        )
        finetune_lr = (
            cast(float, self.optimizer_partial.keywords["lr"]) * self.finetune_lr_ratio
        )
        finetune_param_names = merge_param_names(
            [name for name, _ in finetune_params], layers=4, stop_at_numbers=True
        )
        finetune_param_str = "\n\t".join(finetune_param_names)
        print(
            f"Finetune keywords: {self.finetune_keywords}, lr: {finetune_lr: .2e}, params: \n\t{finetune_param_str}"
        )
        # Create the iterator again as it is consumed when printing names
        finetune_params = exclude_params(
            filter_params(
                self.model.named_parameters(),
                self.finetune_keywords,
                requires_grad=True,
            ),
            self.regular_keywords,
            requires_grad=True,
        )


        # Print regular param names
        regular_params = chain( # Is the same as adding two iterators together
            exclude_params(
                self.model.named_parameters(),
                self.finetune_keywords,
                requires_grad=True,
            ),
            filter_params(
                self.model.named_parameters(), self.regular_keywords, requires_grad=True
            ),
        )
        regular_lr = cast(float, self.optimizer_partial.keywords["lr"])
        regular_param_names = merge_param_names(
            [name for name, _ in regular_params], layers=4, stop_at_numbers=True
        )
        regular_param_str = "\n\t".join(regular_param_names)
        print(
            f"Regular keywords: {self.regular_keywords}, lr: {regular_lr: .2e}, params: \n\t{regular_param_str}"
        )
        # Create the iterator again as it is consumed when printing names
        regular_params = chain(
            exclude_params(
                self.model.named_parameters(),
                self.finetune_keywords,
                requires_grad=True,
            ),
            filter_params(
                self.model.named_parameters(), self.regular_keywords, requires_grad=True
            ),
        )

        self.optimizer = self.optimizer_partial(
            params=[
                {
                    "params": params(finetune_params),
                    "lr": finetune_lr,
                },
                {
                    "params": params(regular_params),
                    "lr": regular_lr,
                },
            ]
        )

        if optimizer_state_dict is not None:
            print(f"Loading optimizer state dict")
            self.optimizer.load_state_dict(optimizer_state_dict)


        self.lr_scheduler = get_scheduler(
            name=self.lr_scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_epochs * len(self.train_dataloader),
        )

        if lr_scheduler_state_dict is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    def run(self):
        assert (
            self.cfg_str_unresolved is not None
            and self.model is not None
            and self.train_dataloader is not None
            and self.val_dataloader is not None
            and self.optimizer is not None
            and self.lr_scheduler is not None
            and self.accelerator is not None
            and self.ema_model is not None
        )

        # print(f"BaseTrainer.run: Model device: {self.model.device}")

        (
            self.train_dataloader,
            self.val_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.train_dataloader,
            self.val_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        )

        print(f"Model type: {type(self.model)}")
        # model = cast(DistributedDataParallel, self.model)
        if not self.debug and hasattr(self.model, "module") and isinstance(self.model.module, BaseDenoisingPolicy):
            print(f"Compiling denoising network and vit models")
            self.model.module.denoising_network.compile()
            self.model.module.shared_model_manager.compile()

        assert (
            self.train_dataloader is not None
            and self.val_dataloader is not None
            and self.model is not None
            and self.optimizer is not None
            and self.lr_scheduler is not None
            and self.ema_model is not None
        )

        device: torch.device = cast(torch.device, self.model.device)
        # print(f"BaseTrainer.run: Device: {device}")

        current_epoch = self.epoch  # For resuming training
        self.ema_model.to(device)
        self.ema_model.eval()

        for epoch_idx in range(current_epoch, self.num_epochs):

            self.model.train()

            step_log = dict()

            train_sampling_batch: batch_type = {}

            detached_losses: list[dict[str, float]] = []

            # Resample the index pool for the train dataloader for each epoch to improve diversity
            if hasattr(self.train_dataloader.base_dataloader.dataset, "resample_index_pool"):
                self.train_dataloader.base_dataloader.dataset.resample_index_pool()

            with tqdm.tqdm(
                self.train_dataloader,
                desc=f"Train {self.epoch}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
                ncols=70,
            ) as tepoch:

                for batch_idx, batch in enumerate(tepoch):

                    with self.accelerator.accumulate(
                        self.model
                    ):  # For gradient accumulation
                        train_sampling_batch = batch

                        if (
                            "meta" in batch
                            and "variance" in batch["meta"]
                            and self.loss_temperatures
                        ):
                            # weighted loss update based on variance
                            if batch_idx < len(self.loss_temperatures):
                                batch["meta"]["variance_temperature"] = (
                                    self.loss_temperatures[batch_idx]
                                )
                            else:
                                batch["meta"]["variance_temperature"] = (
                                    self.loss_temperatures[-1]
                                )

                        step_log = {
                            "info/global_step": self.global_step,
                            "info/epoch": self.epoch,
                            "info/lr": max(self.lr_scheduler.get_last_lr()),
                        }

                        with self.accelerator.autocast():
                            loss, detached_loss_dict = self.compute_loss(batch)
                        step_log.update(detached_loss_dict)
                        detached_losses.append(detached_loss_dict)

                        self.accelerator.backward(loss)

                        if len(self.clip_grad_norms) > 0:
                            if self.epoch >= len(self.clip_grad_norms):
                                max_norm = self.clip_grad_norms[-1]
                            else:
                                max_norm = self.clip_grad_norms[self.epoch]
                            
                            if max_norm == 0:
                                max_norm = float("inf")
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(), max_norm=max_norm
                            )
                            assert isinstance(grad_norm, torch.Tensor)
                            step_log["info/grad_norm"] = float(grad_norm.item())
                            grad_norm_clipped = self.accelerator.clip_grad_norm_(
                                self.model.parameters(), max_norm=max_norm
                            )
                            assert isinstance(grad_norm_clipped, torch.Tensor)
                            step_log["info/clipped_grad_norm"] = float(grad_norm_clipped.item())

                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()

                        loss_cpu = loss.item()
                        tepoch.set_postfix(loss=loss_cpu, refresh=False)

                        if batch_idx != len(self.train_dataloader) - 1:
                            self.accelerator.log(step_log, step=self.global_step)
                            self.global_step += 1

                        if self.accelerator.is_main_process:
                            self.ema_model.step(
                                self.accelerator.unwrap_model(self.model)
                            )

                    if self.debug and batch_idx >= 10:
                        break

            mean_losses = aggregate_batch(
                detached_losses, aggregate_fn=lambda x: sum(x) / len(x)
            )
            for key, value in mean_losses.items():
                mean_key = key.replace("train/", "train/mean_")
                step_log[mean_key] = value

            for param in self.ema_model.parameters():
                broadcast([param.data], from_process=0)

            self.eval_model_step(step_log, train_sampling_batch)

            if (
                self.epoch % self.checkpoint_every
            ) == 0 and self.accelerator.is_main_process:
                normalizer = cast(BaseDataset, self.train_dataloader.dataset).normalizer
                assert normalizer is not None
                self.save_checkpoint(step_log)

            self.rollout_model_step(step_log)

            if self.debug and self.accelerator.is_main_process:
                # Logs might be different on different processes. Will be gathered by the main process
                print_step_log(step_log)

            self.accelerator.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
        self.accelerator.end_training()

    def save_checkpoint(
        self,
        ckpt_meta: dict[str, Any],
    ):
        assert (
            self.optimizer is not None
            and self.lr_scheduler is not None
            and self.train_dataloader is not None
            and self.model is not None
        )
        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model
        ema_model = self.ema_model

        assert model is not None and ema_model is not None

        dataset: BaseDataset = cast(BaseDataset, self.train_dataloader.dataset)
        assert dataset.normalizer is not None

        # frozen_params_keys = [k for k, v in model.named_parameters() if not v.requires_grad]
        frozen_params_keys: list[str] = []
        if self.skip_saving_keywords:
            frozen_params_keys = [
                name
                for name, _ in exclude_params(
                    filter_params(
                        model.named_parameters(),
                        self.skip_saving_keywords,
                        requires_grad=None,
                    ),
                    self.keep_saving_keywords,
                    requires_grad=None,
                )
            ]

        ema_model_trained_state_dict = {
            k: v
            for k, v in ema_model.averaged_model.state_dict().items()
            if k not in frozen_params_keys
        }
        model_trained_state_dict = {
            k: v for k, v in model.state_dict().items() if k not in frozen_params_keys
        }

        saved_params_str = "\n\t".join(
            merge_param_names(
                ema_model_trained_state_dict.keys(), layers=4, stop_at_numbers=True
            )
        )
        print_once(f"Saved params: \n\t{saved_params_str}")

        checkpoint = {
            "model_state_dict": to_cpu(ema_model_trained_state_dict),
            "normalizer_state_dict": dataset.normalizer.state_dict(),
            "cfg_str_unresolved": self.cfg_str_unresolved,
            "sim_config_str": (
                dataset.sim_config_str if hasattr(dataset, "sim_config_str") else ""
            ),
            "epoch": self.epoch,
            "global_step": self.global_step,
        }
        workspace = copy.deepcopy(checkpoint)
        workspace.update(
            {
                "model_state_dict": to_cpu(model_trained_state_dict),
                "ema_model_state_dict": to_cpu(ema_model_trained_state_dict),
                "optimizer_state_dict": to_cpu(self.optimizer.state_dict()),
                "lr_scheduler_state_dict": to_cpu(self.lr_scheduler.state_dict()),
            }
        )

        self.checkpoint_manager.save_ckpt(ckpt_meta, checkpoint, workspace)

    def compute_loss(self, batch: batch_type) -> tuple[torch.Tensor, dict[str, float]]:
        assert self.model is not None
        loss = self.model(batch)
        detached_loss_dict: dict[str, float] = {}
        detached_loss_dict["train/loss"] = float(loss.detach().cpu())
        return loss, detached_loss_dict

    def eval_model_step(
        self, step_log: dict[str, Any], train_sampling_batch: dict[str, Any]
    ):
        raise NotImplementedError("This method should be implemented by the subclass")

    def rollout_model_step(self, step_log: dict[str, Any]):
        """
        Optional
        """
        pass

    def eval_model(self, rounds: int, dataloader_names: list[str], use_episode_num: int = -1):
        raise NotImplementedError("This method should be implemented by the subclass")
