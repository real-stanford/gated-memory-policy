import json
import dill
import hydra
from hydra.core.global_hydra import GlobalHydra
import click 
import os
import torch 
from imitation_learning.common.dataclasses import construct_data_meta_dict
from imitation_learning.datasets.base_dataset import BaseDataset
from imitation_learning.datasets.multi_traj_dataset import MultiTrajDataset
from imitation_learning.datasets.normalizer import FixedNormalizer
from imitation_learning.utils.config_utils import compose_hydra_config
from robot_utils.config_utils import register_resolvers
from robot_utils.torch_utils import torch_save

register_resolvers()

@click.command()
@click.argument("dataset_path", type=str)
@click.argument("task_name", type=str)
@click.option("--policy_name", type=str, default="diffusion_transformer")
@click.option("--stats_only", type=bool, is_flag=True, default=False)
@click.option("--stats_path", type=str, default=None)
@click.option("--quantile", type=float, default=0.99)
def fit_dataset_normalizer(dataset_path: str, task_name: str, policy_name: str, stats_only: bool, stats_path: str | None, quantile: float):
    GlobalHydra.instance().clear()
    os.environ["HYDRA_FULL_ERROR"] = "1"
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)
    print(os.getcwd())

    with hydra.initialize('../imitation_learning/configs'):
        cfg = compose_hydra_config(task_name, policy_name)

    dataset_dir = os.path.dirname(dataset_path)
    dataset_name = os.path.basename(dataset_path)
    if "." in dataset_name:
        dataset_name = dataset_name.split(".")[0]

    cfg["workspace"]["train_dataset"]["root_dir"] = dataset_dir
    cfg["workspace"]["train_dataset"]["name"] = dataset_name
    cfg["workspace"]["train_dataset"]["normalizer_dir"] = dataset_dir
    cfg["workspace"]["train_dataset"]["compressed_dir"] = dataset_dir

    print(f"{cfg['workspace']['train_dataset']}")

    dataset: BaseDataset = hydra.utils.instantiate(cfg["workspace"]["train_dataset"])

    print(f"Done instantiating dataset")
    
    
    if isinstance(dataset, MultiTrajDataset):
        print(dataset.overall_index_pool)


    if stats_only:
        entry_names = [
            entry_meta.name
            for entry_meta in dataset.output_data_meta.values()
            if entry_meta.normalizer != "identity"
        ]
        stats = dataset.calc_stats(entry_names=entry_names)
        # print(stats)
        stats_path = os.path.join(dataset_dir, f"{dataset_name}_stats.pth")
        torch_save(stats, stats_path, pickle_module=dill)
        print(f"Stats saved to {stats_path}")

    else:
        dataset.fit_normalizer(stats_path=stats_path, quantile=quantile)



if __name__ == "__main__":
    fit_dataset_normalizer()



