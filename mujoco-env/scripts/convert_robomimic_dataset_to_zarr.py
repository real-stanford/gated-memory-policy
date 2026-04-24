# Specific for robomimic datasets
import gc
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor

import click
import h5py
import numpy as np
import zarr

from env.utils.robomimic_util import RobomimicAbsoluteActionConverter
from robot_utils.logging_utils import echo_exception


@click.command()
@click.argument("robomimic_dataset_dir", type=str)
@click.argument("task_names", type=str)
@click.argument("dataset_types", type=str)
@click.option(
    "--override", is_flag=True, default=False, help="Override the existing dataset"
)
def convert_robomimic_dataset(
    robomimic_dataset_dir: str, task_names: str, dataset_types: str, override: bool
):

    # for task_name in ["can", "lift", "square"]:
    # for task_name in ["tool_hang"]:
    for task_name in task_names.split(","):

        for dataset_type in dataset_types.split(","):
            for dataset_name in ["image_v15.hdf5"]:
                hdf5_path = os.path.join(task_name, dataset_type, dataset_name)
                dataset_name = convert_hdf5_to_zarr(
                    robomimic_dataset_dir, hdf5_path, override
                )
                print(dataset_name)
                if os.path.exists(f"{robomimic_dataset_dir}/{dataset_name}.tar.lz4"):
                    if override:
                        print(f"Overriding {dataset_name} (which already exists)")
                    else:
                        print(f"Skipping {dataset_name} because it already exists")
                        continue

                subprocess.run(
                    f"source ~/.zshrc && compress {robomimic_dataset_dir}/{dataset_name}",
                    shell=True,
                    executable="/usr/bin/zsh",
                )
                # subprocess.run(["mv", f"{robomimic_dataset_dir}/{dataset_name}.tar.lz4", output_dir])


def convert_episodes(
    dataset_dir: str, hdf5_path: str, episode_keys: list[str], zarr_group: zarr.Group
):

    converter = RobomimicAbsoluteActionConverter(f"{dataset_dir}/{hdf5_path}")

    try:

        with h5py.File(f"{dataset_dir}/{hdf5_path}", "r") as f:
            all_data = f["data"]
            assert isinstance(all_data, h5py.Group)
            for episode_key in episode_keys:
                episode_idx: int = int(episode_key.split("_")[1])
                # episode_key = f"episode_{episode_idx}"
                abs_actions = converter.convert_idx(episode_idx)
                print(f"{episode_idx=}")

                # To debug, use the following (will be much slower)
                # abs_actions, info = converter.convert_and_eval_idx(episode_idx)
                # print(f"Converting episode {episode_idx}, {info=}")

                episode_group = zarr_group.create_group(f"episode_{episode_idx}")
                episode_data = all_data[episode_key]
                assert isinstance(episode_data, h5py.Group)
                episode_group.create_dataset(
                    "delta_actions", data=episode_data["actions"], compression=None
                )
                episode_group.create_dataset(
                    "abs_actions", data=abs_actions, compression=None
                )

                obs = episode_data["obs"]
                assert isinstance(obs, h5py.Group)
                episode_group.create_dataset(
                    "robot0_eef_pos", data=obs["robot0_eef_pos"], compression=None
                )
                episode_group.create_dataset(
                    "robot0_eef_quat", data=obs["robot0_eef_quat"], compression=None
                )
                episode_group.create_dataset(
                    "robot0_gripper_qpos",
                    data=obs["robot0_gripper_qpos"],
                    compression=None,
                )

                if "robot1_eef_pos" in obs.keys():
                    episode_group.create_dataset(
                        "robot1_eef_pos", data=obs["robot1_eef_pos"], compression=None
                    )
                    episode_group.create_dataset(
                        "robot1_eef_quat", data=obs["robot1_eef_quat"], compression=None
                    )
                    episode_group.create_dataset(
                        "robot1_gripper_qpos",
                        data=obs["robot1_gripper_qpos"],
                        compression=None,
                    )

                for img_keys in [
                    "agentview_image",
                    "sideview_image",
                    "shouldercamera0_image",
                    "shouldercamera1_image",
                    "robot0_eye_in_hand_image",
                    "robot1_eye_in_hand_image",
                ]:
                    if img_keys not in obs.keys():
                        continue

                    img_data = np.array(obs[img_keys])

                    if img_data.dtype != np.uint8 and np.max(img_data) <= 1.0:
                        img_data = (img_data * 255).astype(np.uint8)
                    # assert isinstance(img_data, h5py.Dataset)
                    data_shape = img_data.shape
                    chunk_size = tuple([10] + list(data_shape[-3:]))
                    episode_group.create_dataset(
                        img_keys, data=img_data, chunks=chunk_size
                    )

            del converter
            gc.collect()
    except Exception as e:
        print(echo_exception())
        raise e


def convert_hdf5_to_zarr(dataset_dir: str, hdf5_path: str, override: bool):
    dataset_name = "_".join(hdf5_path.split("/")[-3:-1])
    dataset_name = "robomimic_" + dataset_name
    os.makedirs(dataset_dir, exist_ok=True)
    zarr_path = os.path.join(dataset_dir, dataset_name, "episode_data.zarr")
    if os.path.exists(zarr_path):
        if override:
            print(f"Overriding {dataset_name} (which already exists)")
            shutil.rmtree(zarr_path)
        else:
            print(f"Skipping {dataset_name} because it already exists")
            return dataset_name
    zarr_store = zarr.DirectoryStore(zarr_path)
    zarr_group = zarr.group(zarr_store)

    worker_num = 20

    with h5py.File(f"{dataset_dir}/{hdf5_path}", "r") as f:
        all_data = f["data"]
        assert isinstance(all_data, h5py.Group)
        keys = list(all_data.keys())

    print(f"{len(keys)=}")

    with ProcessPoolExecutor(max_workers=worker_num) as executor:
        key_num_per_worker = len(keys) // worker_num
        for i in range(worker_num):
            episode_keys = keys[
                i * key_num_per_worker : min((i + 1) * key_num_per_worker, len(keys))
            ]
            executor.submit(
                convert_episodes, dataset_dir, hdf5_path, episode_keys, zarr_group
            )
        # executor.shutdown(wait=True)

    # Single process for debugging
    # convert_episodes(dataset_dir, hdf5_path, keys, zarr_group)

    return dataset_name


if __name__ == "__main__":
    convert_robomimic_dataset()
