import click
import dill
import hydra
import os
import sys

import torch

from imitation_learning.datasets.normalizer import FixedNormalizer
from imitation_learning.models.common.memory_gate import MemoryGate
from imitation_learning.trainers.memory_gate_trainer import MemoryGateTrainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imitation_learning.datasets.base_dataset import BaseDataset
from robot_utils.data_utils import dict_apply
from robot_utils.torch_utils import torch_load, torch_save
from omegaconf import OmegaConf
import tqdm

def eval_memory_gate(ckpt_path: str, episode_indices: list[int] | None = None, device: str = "cuda"):
    assert os.path.exists(ckpt_path)
    assert len(episode_indices) > 0

    ckpt = torch_load(ckpt_path, pickle_module=dill, weights_only=False)
    config_str: str = ckpt["cfg_str_unresolved"]
    config = OmegaConf.create(config_str)
    with hydra.initialize('../imitation_learning/configs'):
        memory_gate: MemoryGate = hydra.utils.instantiate(config.workspace.model)
        config.workspace.train_dataset.statistics_data_path = "" # To still use all data
        config.workspace.train_dataset.used_episode_ratio = 1.0 # Use all data
        dataset: BaseDataset = hydra.utils.instantiate(config.workspace.train_dataset)
        trainer: MemoryGateTrainer = hydra.utils.instantiate(config.workspace.trainer)

    normalizer = FixedNormalizer(dataset.output_data_meta)
    normalizer.to(device)
    normalizer.load_state_dict(ckpt["normalizer_state_dict"])

    memory_gate.to(device)
    memory_gate.load_state_dict(ckpt["model_state_dict"])

    memory_gate.eval()

    dataset.trim_dataset_episodes(remaining_episode_indices=episode_indices)
    dataset.dataloader_cfg["shuffle"] = False
    dataloader = dataset.get_dataloader()

    memory_gate_vals = []
    traj_indices = []
    episode_indices = []

    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader)):
        batch = dict_apply(batch, lambda x: x.to(device))
        normalized_batch = normalizer.normalize(batch)
        # trainer._add_gt_gate_labels(normalized_batch)
        memory_gate_val = memory_gate.get_gate_value(
            normalized_batch,
        )
        memory_gate_vals.append(memory_gate_val)
        traj_indices.append(batch["traj_idx"])
        episode_indices.append(batch["episode_idx"])

    memory_gate_vals = torch.cat(memory_gate_vals, dim=0)
    traj_indices = torch.cat(traj_indices, dim=0)
    episode_indices = torch.cat(episode_indices, dim=0)

    torch_save({
        "memory_gate_vals": memory_gate_vals,
        "traj_indices": traj_indices,
        "episode_indices": episode_indices,
    }, f"{ckpt_path.replace('.ckpt', '_memory_gate_vals.pt')}")
    print(f"Saved memory gate vals to {ckpt_path.replace('.ckpt', '_memory_gate_vals.pt')}")


# def process_gate_values(ckpt_path: str):

@click.command()
@click.option("--ckpt_path", type=str, required=True)
@click.option("--episode_indices", type=int, multiple=True, default=None)
def main(ckpt_path, episode_indices):
    if episode_indices is not None:
        eval_memory_gate(ckpt_path, list(episode_indices))
    else:
        eval_memory_gate(ckpt_path)

if __name__ == "__main__":
    main()
    # python scripts/eval_memory_gate.py --ckpt_path=data/pick_and_match_color/2025-12-21/22-45-04_match_color_gate_lr1e-4/checkpoints/epoch_0_train_mean_loss_0_067.ckpt --episode_indices=\[198,254\]