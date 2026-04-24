import os
import subprocess
from typing import Any, Union, cast

import numpy as np
import numpy.typing as npt
import zarr
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from env.modules.common import data_buffer_type, robot_data_type





def flatten_episode_data(episode_data: dict[str, Any]) -> dict[str, Any]:
    """
    This is not a good implementation. There should be better ways to do this without hardcoding the keys.
    Before flattening:
    {
        "robots_obs": [
            [ # timestep 0
                { # robot 0
                    "wrist_camera": [...],
                    "tcp_xyz_wxyz": [...],
                    ...
                },
                ...
            ],
            ...
        ],
        "env_objs_obs": [
            [ # timestep 0
                { # env
                    "third_person_camera": [...],
                    "timestamp": [...],
                    ...
                },
                { # obj 0
                    "object_pose_xyz_wxyz": [...],
                    "bin_id": [...],
                    "tcp_relative_poses_xyz_wxyz": [...],
                    ...
                },
                ...
            ],
            ...
        ],
        "executed_actions": [
            [ # timestep 0
                { # robot 0
                    "tcp_xyz_wxyz": [...],
                    "gripper_width": [...],
                    ...
                },
                ...
            ],
            ...
        ],
    }
    After flattening:
    {
        "robot0_wrist_camera": [...],
        "robot0_tcp_xyz_wxyz": [...],
        "third_person_camera": [...],
        "timestamp": [...],
        "obj0_object_pose_xyz_wxyz": [...],
        "obj0_bin_id": [...],
        "obj0_tcp_relative_poses_xyz_wxyz": [...],
        ...
        "action_tcp_xyz_wxyz": [...],
        "action_gripper_width": [...],
        ...
        "predicted_trajs_0_tcp_xyz_wxyz": [...],
        "predicted_trajs_0_gripper_width": [...],
        ...
        "reward": bool,
        "episode_config": {...},
        "episode_length": int,
    }
    """
    # TODO: automatically parse the data structure

    flattened_data: dict[str, Any] = {}

    if (
        "robots_obs" in episode_data
        and isinstance(episode_data["robots_obs"], list)
        and len(episode_data["robots_obs"]) > 0
    ):
        # Find a single_robots_obs that has the most keys:
        max_keys = 0
        for single_robots_obs in episode_data["robots_obs"]:
            if len(single_robots_obs[0].keys()) > max_keys:
                max_keys = len(single_robots_obs[0].keys())
                max_keys_single_robots_obs = single_robots_obs
        single_robots_obs = max_keys_single_robots_obs[0]
        assert isinstance(
            single_robots_obs, dict
        ), f"single_robots_obs: {single_robots_obs}, type: {type(single_robots_obs)}"
        robot_num = len(episode_data["robots_obs"][-1])

        robots_obs_keys = [
            ## MuJoCo
            "arm_qpos",
            "arm_qvel",
            "arm_qacc",
            "tcp_xyz_wxyz",
            "gripper_width",
            "wrist_camera",
            ## Robomimic
            "eye_in_hand_image",
            "eef_pos",
            "eef_quat",
            "gripper_qpos",
        ]
        robots_obs_static_keys = [
            "name"
        ]  # The values are the same in the entire episode
        # Initialize flattened data keys
        for robot_id in range(robot_num):
            for key, value in single_robots_obs.items():
                if key in robots_obs_static_keys:  # Initialize
                    flattened_data[f"robot{robot_id}_{key}"] = value
                elif isinstance(value, np.ndarray):
                    flattened_data[f"robot{robot_id}_{key}"] = []
                else:
                    raise ValueError(
                        f"Got unexpected key {key} with value {value}, type {type(value)}"
                    )
        for obs_cnt, robots_obs in enumerate(episode_data["robots_obs"]):
            for robot_id in range(robot_num):
                for key, value in robots_obs[robot_id].items():
                    if key in robots_obs_keys:
                        flattened_data[f"robot{robot_id}_{key}"].append(value)

    if (
        "env_objs_obs" in episode_data
        and isinstance(episode_data["env_objs_obs"], list)
        and len(episode_data["env_objs_obs"]) > 0
    ):
        max_keys = 0
        for single_env_objs_obs in episode_data["env_objs_obs"]:
            if len(single_env_objs_obs[0].keys()) > max_keys:
                max_keys = len(single_env_objs_obs[0].keys())
                max_keys_single_env_objs_obs = single_env_objs_obs
        single_env_objs_obs = max_keys_single_env_objs_obs
        obj_num = (
            len(single_env_objs_obs) - 1
        )  # The first observation is from the environment

        env_objs_obs_keys: list[str] = [
            "third_person_camera",
            "visualization_camera",
            "timestamp",
            "bin_materials",  # For TableBin
            "zoom_in_camera",  # For TableLines
            ## Robomimic
            "agentview_image",
            "sideview_image",
            "shouldercamera0_image",
            "shouldercamera1_image",
            "done",
            "reward",
        ]
        env_objs_obs_static_keys: list[str] = [
            "name",
            "bin_center_xyz",  # for TableBin
            "line_center_xyz",  # for TableLines
            "box_center_xyz",  # for TableLines
        ]

        for key, value in single_env_objs_obs[0].items():
            if key in env_objs_obs_static_keys:
                flattened_data[f"{key}"] = value
            elif key in env_objs_obs_keys:
                flattened_data[f"{key}"] = []
            else:
                raise ValueError(
                    f"Got unexpected key {key} with value {value}, type {type(value)}"
                )

        obj_obs_keys: list[str] = [
            "object_pose_xyz_wxyz",
            "object_vel_xyz_xyz",
            "bin_id",
            "tcp_relative_poses_xyz_wxyz",
            "pos_xyz",  # For cloth
            "2d_convex_hull_area",  # For cloth
        ]
        obj_obs_static_keys: list[str] = ["name"]
        for obj_id in range(obj_num):
            for key, value in single_env_objs_obs[1].items():
                if key in obj_obs_static_keys:
                    flattened_data[f"obj{obj_id}_{key}"] = value
                elif key in obj_obs_keys:
                    flattened_data[f"obj{obj_id}_{key}"] = []
                else:
                    raise ValueError(
                        f"Got unexpected key {key} with value {value}, type {type(value)}"
                    )

        for obs_cnt, env_objs_obs in enumerate(episode_data["env_objs_obs"]):
            for key, value in env_objs_obs[0].items():
                if key in env_objs_obs_keys:
                    flattened_data[f"{key}"].append(value)
            for obj_id in range(obj_num):
                for key, value in env_objs_obs[obj_id + 1].items():
                    if key in obj_obs_keys:
                        flattened_data[f"obj{obj_id}_{key}"].append(value)

    for action_key_name in ["executed_actions", "history_actions"]:
        if (
            action_key_name in episode_data
            and isinstance(episode_data[action_key_name], list)
            and len(episode_data[action_key_name]) > 0
        ):
            assert isinstance(episode_data[action_key_name][0], list)

            robot_num = len(episode_data[action_key_name][0])
            action_keys = [
                "tcp_xyz_wxyz",
                "gripper_width",
                "is_error",
                "is_critical",
                # For robomimic
                "delta_pos_xyz",
                "delta_rot_rpy",
                "delta_gripper_qpos",
                "pos_xyz",
                "ori_xyz",
            ]
            action_static_keys = ["name"]
            for robot_id in range(robot_num):
                single_robot_action: robot_data_type = episode_data[action_key_name][0][
                    robot_id
                ]
                assert isinstance(single_robot_action, dict)
                for key, value in single_robot_action.items():
                    if key in action_static_keys:
                        flattened_data[f"action{robot_id}_{key}"] = value
                    elif key in action_keys:
                        flattened_data[f"action{robot_id}_{key}"] = []
                    else:
                        raise ValueError(
                            f"Got unexpected key {key} with value {value}, type {type(value)}"
                        )
            for action_cnt, executed_actions in enumerate(
                episode_data[action_key_name]
            ):
                for robot_id in range(robot_num):
                    for key, value in executed_actions[robot_id].items():
                        if key in action_keys:
                            flattened_data[f"action{robot_id}_{key}"].append(value)
    if (
        "predicted_trajs" in episode_data
        and isinstance(episode_data["predicted_trajs"], list)
        and len(episode_data["predicted_trajs"]) > 0
    ):
        assert isinstance(episode_data["predicted_trajs"][0], tuple)
        flattened_data["predicted_trajs_time_steps"] = []
        action_keys = ["tcp_xyz_wxyz", "gripper_width"]
        action_static_keys = ["name"]
        robot_num = len(episode_data["predicted_trajs"][0][1][0])
        for robot_id in range(robot_num):
            for key in action_keys:
                flattened_data[f"predicted_trajs_{robot_id}_{key}"] = []
            for key in action_static_keys:
                flattened_data[f"predicted_trajs_{robot_id}_{key}"] = episode_data[
                    "predicted_trajs"
                ][0][1][0][0][key]

        for time_step, predicted_action_traj in episode_data["predicted_trajs"]:
            flattened_data["predicted_trajs_time_steps"].append(time_step)
            predicted_action_traj: data_buffer_type
            single_traj = {}
            for robot_id in range(robot_num):
                single_traj[robot_id] = {}
                for key in action_keys:
                    single_traj[robot_id][key] = []
            for action_id, action in enumerate(predicted_action_traj):
                for robot_id in range(robot_num):
                    for key, value in action[robot_id].items():
                        if key in action_keys:
                            single_traj[robot_id][key].append(value)
            for robot_id in range(robot_num):
                for key in action_keys:
                    if key in single_traj[robot_id]:
                        flattened_data[f"predicted_trajs_{robot_id}_{key}"].append(
                            single_traj[robot_id][key]
                        )

    attributes = ["final_reward", "episode_config", "episode_length", "is_successful"]
    for attribute in attributes:
        if attribute in episode_data:
            flattened_data[attribute] = episode_data[attribute]

    for key, value in flattened_data.items():
        if isinstance(value, list):
            flattened_data[key] = np.array(value)

    return flattened_data


def convert_to_list(
    data: Union[npt.NDArray[Any], dict[str, Any], list[Any]],
) -> Union[list[Any], dict[str, Any]]:
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: convert_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_list(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_list(item) for item in data)
    elif isinstance(data, ListConfig):
        container = OmegaConf.to_container(data)
        return [convert_to_list(item) for item in container]
    return data


def merge_episode_data(
    data_storage_dir: str, zarr_file_name: str = "episode_data.zarr"
):
    episode_root: zarr.Group = cast(
        zarr.Group, zarr.open(f"{data_storage_dir}/{zarr_file_name}", mode="r")
    )
    merged_root: zarr.Group = cast(
        zarr.Group, zarr.open(f"{data_storage_dir}/merged_data.zarr", mode="w")
    )
    merged_data = merged_root.create_group("data")
    merged_meta = merged_root.create_group("meta")
    episode_ends = []
    for episode_idx, episode_group in tqdm(episode_root.items()):
        episode_attrs = {}

        episode_meta = merged_meta.create_group(f"{episode_idx}")
        for key, value in episode_group.attrs.items():
            episode_meta.attrs[key] = value
        episode_length = episode_group.attrs["episode_length"]
        if not episode_ends:
            episode_ends.append(episode_length)
        else:
            episode_ends.append(episode_ends[-1] + episode_length)

        for key, value in episode_group.items():
            if key not in merged_data:
                merged_data.create_dataset(key, data=value, compression="blosc")
            else:
                merged_data_dataset = cast(zarr.Array, merged_data[key])
                merged_data_dataset.append(value)
    merged_meta.create_dataset("episode_ends", data=episode_ends, compression="blosc")


def compress_data(data_storage_dir: str):
    cpu_cores = os.cpu_count()
    subprocess.run(
        f"tar cf - {data_storage_dir} | pigz -p {cpu_cores} > {data_storage_dir}.tar.gz",
        shell=True,
    )
