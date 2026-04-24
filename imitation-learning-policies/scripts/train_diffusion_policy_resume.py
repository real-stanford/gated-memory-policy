import click
import dill
import hydra
from omegaconf import DictConfig, OmegaConf

from robot_utils.torch_utils import torch_load
from imitation_learning.workspaces.base_workspace import BaseWorkspace


@click.command()
@click.option("--ckpt_path", type=str, required=True)
def main(ckpt_path: str):
    ckpt = torch_load(ckpt_path, pickle_module=dill)

    config_str: str = ckpt["cfg_str_unresolved"]
    config = OmegaConf.create(config_str)
    OmegaConf.set_struct(config, True)
    assert type(config) == DictConfig
    # Update some configs here
    # config["workspace"]["train_dataset"]["dataloader_cfg"]["batch_size"] = 128

    config["workspace"]["cfg_str_unresolved"] = config_str
    workspace: BaseWorkspace = hydra.utils.instantiate(config["workspace"])
    workspace.resume_training(ckpt)


if __name__ == "__main__":
    main()
