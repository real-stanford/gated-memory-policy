import os

import click
import dill
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from robot_utils.torch_utils import torch_load
from imitation_learning.workspaces.base_workspace import BaseWorkspace

def eval_model(
    ckpt_path: str,
    rounds: int,
    dataset_dir: str,
    eval_episode_num: int,
    index_pool_size_per_episode: int,
    dataset_type: str,
):
    np.set_printoptions(precision=3, suppress=True)
    ckpt = torch_load(ckpt_path, pickle_module=dill, weights_only=False)

    config_str: str = ckpt["cfg_str_unresolved"]
    config = OmegaConf.create(config_str)
    OmegaConf.set_struct(config, True) # Will set the config to strict mode
    assert type(config) == DictConfig

    output_dir = os.path.dirname(ckpt_path)

    config["workspace"]["trainer"]["output_dir"] = output_dir
    config["workspace"]["cfg_str_unresolved"] = config_str
    config["workspace"]["logging_cfg"] = None  # Disable wandb logging

    if "memory" in config["policy_name"]:
        # config["workspace"]["train_dataset"]["starting_percentile_max"] = 0.1 # For place_back / match_color. Now should be the default value
        # batch_size = 16
        batch_size = 8
        # Only start with the first 8 frames to make sure the history length is always full
        config["workspace"]["train_dataset"]["starting_percentile_max"] = 0

        dataset_traj_max_nums = {
            "pick_and_place_back": 16,
            "pick_and_match_color": 16,
            "push_cube": 80,
            "fling_cloth": 55,
            "robomimic_tool_hang_ph": 90,
            "robomimic_square_ph": 45,
            "robomimic_square_mh": 75,
            "robomimic_transport_ph": 90,
            "robomimic_transport_mh": 90,
            "iphumi_place_back": 50,
            "real_world_iterative_casting": 40
        }
        if config["task_name"] in dataset_traj_max_nums:
            config["workspace"]["train_dataset"]["traj_num"] = dataset_traj_max_nums[config["task_name"]]
        else:
            raise ValueError(f"Unknown task name: {config['task_name']}")
        config["workspace"]["train_dataset"]["traj_interval_min"] = config["workspace"]["train_dataset"]["traj_interval_max"] 
        # Reduce randomness during evaluation
            
    else:
        batch_size = 128
    config["workspace"]["train_dataset"]["dataloader_cfg"]["batch_size"] = batch_size
    config["workspace"]["train_dataset"]["dataloader_cfg"]["num_workers"] = 16

    if "split_dataloader_cfg" in config["workspace"]["train_dataset"]:
        config["workspace"]["train_dataset"]["split_dataloader_cfg"]["batch_size"] = batch_size
        config["workspace"]["train_dataset"]["split_dataloader_cfg"]["num_workers"] = 16

    # assert repeat_dataset == 1, "Repeat dataset is not supported for error distribution calculation"
    config["workspace"]["train_dataset"]["repeat_dataset_num"] = 1 # Always use only once to control the number of samples
    if dataset_dir != "":
        config["workspace"]["train_dataset"]["root_dir"] = dataset_dir
        config["workspace"]["train_dataset"]["compressed_dir"] = dataset_dir
        config["workspace"]["train_dataset"]["normalizer_dir"] = dataset_dir

    # config["workspace"]["train_dataset"]["eval_episode_num"] = eval_episode_num
    config["workspace"]["train_dataset"]["index_pool_size_per_episode"] = index_pool_size_per_episode

    # if "third_person_camera" in config["workspace"]["train_dataset"]["output_data_meta"]:
    #     config["workspace"]["train_dataset"]["output_data_meta"]["third_person_camera"]["augmentation"] = [
    #         {
    #             "name": "Resize",
    #             "size": [256, 256],
    #             "antialias": True,
    #         }
    #     ]
    # if "robot0_eye_in_hand_image" in config["workspace"]["train_dataset"]["output_data_meta"]:
    #     config["workspace"]["train_dataset"]["output_data_meta"]["robot0_eye_in_hand_image"]["augmentation"] = [
    #         {
    #             "name": "Resize",
    #             "size": [256, 256],
    #             "antialias": True,
    #         }
    #     ]
    # config["workspace"]["train_dataset"]["output_data_meta"]["robot0_10d"]["augmentation"] = []
    # dataloader_names = ["train"]
    # dataloader_names = ["train", "val"]
    dataloader_names = [dataset_type]
    dir_name = f"epoch_{ckpt['epoch']}_eval"
    trainer_dir_name = f"epoch_{ckpt['epoch']}_eval"
    trainer_dir = os.path.join(
        config["workspace"]["trainer"]["output_dir"], trainer_dir_name
    )

    if os.path.exists(trainer_dir):
        os.rename(
            trainer_dir,
            os.path.join(config["workspace"]["trainer"]["output_dir"], dir_name),
        )
    file_name_0 = f"{dataloader_names[0]}_results"
    # file_name_all = f"all_results"
    print(f"file_name_0: {file_name_0}")
    # print(f"file_name_all: {file_name_all}")
    # output_dir = config["workspace"]["trainer"]["output_dir"]

    pt_path = os.path.join(output_dir, dir_name, f"{file_name_0}.pt")
    print(pt_path)
    if not os.path.exists(
        pt_path 
    ):
    # if True:
        workspace: BaseWorkspace = hydra.utils.instantiate(config["workspace"])
        # normalizer_dict = workspace.train_dataset.normalizer.as_dict("list")
        # json.dump(normalizer_dict, open(os.path.join(workspace.train_dataset.normalizer_dir, f'{workspace.train_dataset.name}_normalizer_updated.json'), 'w'))
        workspace.eval_model(ckpt, rounds, dataloader_names, use_episode_num=eval_episode_num)


    # unnormalized_data = workspace.train_dataset.normalizer.unnormalize(workspace.train_dataset[0])
        # print(f"Normalizer state dict saved to {os.path.join(output_dir, dir_name, f'normalizer_state_dict.pt')}")
        # exit()
    # print(f"{unnormalized_data['action0_10d'][..., 0]=}")
    # exit()

    # When using multiple gpus to evaluate, the could be repeated data in the eval results, but the total number should not exceed batch size for each gpu.
    # epoch = ckpt["epoch"]
    # for dataloader_name in dataloader_names:
    #     file_name = f"{dataloader_name}_results"
    #     eval_results = torch_load(
    #         os.path.join(output_dir, dir_name, f"{file_name}.pt"), pickle_module=dill
    #     )
    #     # print({key: val.shape for key, val in eval_results.items()})
    #     print(f"{dataloader_name}:")
    #     if "MultiTraj" in config["workspace"]["train_dataset"]["_target_"]:
    #         multi_traj = True
    #     elif "SingleTraj" in config["workspace"]["train_dataset"]["_target_"]:
    #         multi_traj = False
    #     else:
    #         raise ValueError(
    #             f"Unknown dataset type: {config['workspace']['train_dataset']['_target_']}"
    #         )
    #     plot_err_distribution(
    #         eval_results, os.path.join(output_dir, dir_name, f"{file_name}.png"), multi_traj
    #     )


@click.command()
@click.argument("ckpt_path", type=str, required=True)
@click.option("--rounds", type=int, default=1)
@click.option("--dataset_dir", type=str, default="")
@click.option("--eval_episode_num", type=int, default=-1)
@click.option("--index_pool_size_per_episode", type=int, default=1)
@click.option("--dataset_type", type=str, default="val")
def main(ckpt_path, rounds, dataset_dir, eval_episode_num, index_pool_size_per_episode, dataset_type):
    eval_model(ckpt_path, rounds, dataset_dir, eval_episode_num, index_pool_size_per_episode, dataset_type)


if __name__ == "__main__":
    main()
