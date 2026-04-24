import os
from typing import cast
import sys

from omegaconf import DictConfig, OmegaConf
import torch

from imitation_learning.utils.config_utils import compose_memory_gate_hydra_config
from robot_utils.config_utils import register_resolvers
from robot_utils.torch_utils import is_main_process
from imitation_learning.workspaces.base_workspace import BaseWorkspace
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydra

register_resolvers()

@hydra.main(
    config_path="../imitation_learning/configs",
    config_name="train_memory_gate",
    version_base=None,
)
def train_memory_gate(cfg: DictConfig):

    base_cfg = compose_memory_gate_hydra_config(cfg["task_name"])
    OmegaConf.set_struct(base_cfg, False) # Enable overrides for multi-task datasets

    # Apply overrides
    cfg = cast(DictConfig, OmegaConf.merge(base_cfg, cfg))

    # print(yaml.dump(OmegaConf.to_container(cfg, resolve=False)))
    cfg_str_unresolved = OmegaConf.to_yaml(cfg, resolve=False)
    project_name = cfg["project_name"]
    cfg_str_unresolved = cfg_str_unresolved.replace("cosmos_uva_debug2", project_name)
    cfg["workspace"]["cfg_str_unresolved"] = cfg_str_unresolved

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if is_main_process():
        if not os.path.exists(cfg.workspace.trainer.output_dir):
            os.makedirs(cfg.workspace.trainer.output_dir, exist_ok=True)
        with open(f"{cfg.workspace.trainer.output_dir}/{cfg.workspace.name}.yaml", "w") as f:
            # Ensure cfg_str_unresolved is not resolved
            resolved_cfg = cast(DictConfig, OmegaConf.to_container(OmegaConf.create(cfg), resolve=True))
            resolved_cfg["workspace"]["cfg_str_unresolved"] = cfg_str_unresolved
            f.write(OmegaConf.to_yaml(resolved_cfg, resolve=False))
            print(f"Saved config to {cfg.workspace.trainer.output_dir}/{cfg.workspace.name}.yaml")

    print(f"statistics_data_path: {cfg.workspace.train_dataset.statistics_data_path}")
    
    workspace: BaseWorkspace = hydra.utils.instantiate(cfg["workspace"])

    workspace.train()


    # statistics = torch_load(statistics_path, pickle_module=dill, weights_only=False)
    # """
    # statistics: dict[int, dict[str, torch.Tensor]] = {
    #     episode_idx: {
    #         "with_mem_variances": torch.Tensor(frame_num),
    #         "no_mem_variances": torch.Tensor(frame_num),
    #         "with_mem_errors": torch.Tensor(frame_num),
    #         "no_mem_errors": torch.Tensor(frame_num),
    #     }
    # """

    # assert cfg.dataset_type in ["train", "val"]

    

    
        




if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    # main()
    train_memory_gate()