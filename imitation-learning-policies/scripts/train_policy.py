import os
from typing import cast

import dill
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig

from imitation_learning.utils.config_utils import compose_hydra_config
from robot_utils.config_utils import register_resolvers
from robot_utils.torch_utils import torch_load, is_main_process
from imitation_learning.workspaces.base_workspace import BaseWorkspace
import torch.multiprocessing as mp
from imitation_learning.utils.config_utils import remove_keys_from_config
register_resolvers()

@hydra.main(
    config_path="../imitation_learning/configs",
    config_name="train_policy",
    version_base=None,
)
def main(cfg: DictConfig):
    mp.set_start_method('spawn', force=True)
    np.set_printoptions(precision=4, suppress=True, threshold=np.inf)
    base_ckpt = None
    
    base_cfg = compose_hydra_config(cfg["task_name"], cfg["policy_name"])

    # Apply overrides
    cfg = cast(DictConfig, OmegaConf.merge(base_cfg, cfg))  # Apply overrides

    if (
        "base_ckpt_path" in cfg
        and cfg["base_ckpt_path"] is not None
        and cfg["base_ckpt_path"] != ""
    ):
        base_ckpt = torch_load(cfg["base_ckpt_path"], pickle_module=dill)
        cfg["base_ckpt_cfg_str"] = base_ckpt["cfg_str_unresolved"] if "cfg_str_unresolved" in base_ckpt else ""
        
    if "base_ckpt_ignore_keywords" in cfg and cfg["base_ckpt_ignore_keywords"] is not None and cfg["base_ckpt_ignore_keywords"] != "":
        assert base_ckpt is not None, "base_ckpt_path is required when base_ckpt_ignore_keywords is provided"
        assert isinstance(cfg["base_ckpt_ignore_keywords"], ListConfig)
        base_ckpt_ignore_keywords: list[str] | None = list(cfg["base_ckpt_ignore_keywords"])
    else:
        base_ckpt_ignore_keywords = None

    cfg_str_unresolved = OmegaConf.to_yaml(cfg, resolve=False)
    # HACK: change logger_project_name to project_name
    # TODO: Remove this when releasing
    project_name = cfg["project_name"]
    cfg_str_unresolved = cfg_str_unresolved.replace("cosmos_uva_debug2", project_name)

    cfg["workspace"]["cfg_str_unresolved"] = cfg_str_unresolved
        
    if cfg["attach_vscode_debugger"]:
        print(f"Rank: {int(os.environ.get('RANK', 0))}")
        if int(os.environ.get("RANK", 0)) == 0:
            import debugpy
            debugpy.listen(5678)
            print("Waiting for VSCode debugger to attach at port 5678")
            debugpy.wait_for_client()
            print("VSCode debugger attached")

        # wait_for_main_process()
        

    if is_main_process():
        if not os.path.exists(cfg.workspace.trainer.output_dir):
            os.makedirs(cfg.workspace.trainer.output_dir, exist_ok=True)
        with open(f"{cfg.workspace.trainer.output_dir}/{cfg.workspace.name}.yaml", "w") as f:
            # Ensure cfg_str_unresolved is not resolved
            resolved_cfg = cast(DictConfig, OmegaConf.to_container(OmegaConf.create(cfg), resolve=True))
            resolved_cfg["workspace"]["cfg_str_unresolved"] = cfg_str_unresolved
            f.write(OmegaConf.to_yaml(resolved_cfg, resolve=False))
            print(f"Saved config to {cfg.workspace.trainer.output_dir}/{cfg.workspace.name}.yaml")
        
    # To control initialization
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    cfg = remove_keys_from_config(cfg)

    workspace: BaseWorkspace = hydra.utils.instantiate(cfg["workspace"])

    if base_ckpt is not None:
        try:
            print(f"Loading base base_ckpt from {cfg['base_ckpt_path']}")
            workspace.load_base_ckpt(base_ckpt, ignore_keywords=base_ckpt_ignore_keywords)
        except Exception as e:
            print(f"Error loading base base_ckpt: {e}. Please include 'base_ckpt_ignore_keywords' as a ListConfig in the config.")

    # if "memory_gate_ckpt_path" in cfg and cfg["memory_gate_ckpt_path"] is not None and cfg["memory_gate_ckpt_path"] != "":
    #     try:
    #         memory_gate_ckpt = torch_load(cfg["memory_gate_ckpt_path"], pickle_module=dill)
    #         workspace.load_module_ckpt("memory_gate", memory_gate_ckpt)
    #     except Exception as e:
    #         raise e

    workspace.train()


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
