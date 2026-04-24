from typing import Any, cast

import hydra
import numpy as np
import numpy.typing as npt
from transforms3d import quaternions

from env.modules.agents.heuristic_agents.pick_and_place_back_agent import \
    PickAndPlaceBackAgent
from env.modules.envs.table_bin_1robot_1object import TableBin1Robot1Object
from env.modules.scenes.table_bin import TableBin
from env.modules.tasks.base_task import BaseTask
from env.modules.tasks.parallel_task import ParallelTask
from robot_utils.config_utils import enable_hydra_target
from env.utils.pose_utils import (get_random_4poses_convex_combination,
                                  get_random_convex_combination)
from robot_utils.logging_utils import print_once


def _process_episode_config(
    self: "PickAndPlaceBack | PickAndPlaceBackParallel", episode_config: dict[str, Any]
) -> dict[str, Any]:

    # Only set from task config if not already specified in episode_config (e.g. hardcoded per-episode in rollout script)
    if "object_init_bin_ids" not in episode_config:
        episode_config["object_init_bin_ids"] = self.object_init_bin_ids
    episode_config["delta_bin_id"] = self.delta_bin_id
    episode_config["delay_init_bin_materials"] = self.delay_init_bin_materials
    episode_config["init_bin_materials"] = self.init_bin_materials
    episode_config["final_bin_materials"] = self.final_bin_materials
    episode_config["shuffle_bin_material"] = self.shuffle_bin_material

    if hasattr(self, 'rand_delay_s_range') and self.rand_delay_s_range is not None:
        episode_config["rand_delay_s_range"] = self.rand_delay_s_range
    else:
        episode_config["rand_delay_s_range"] = [0.0, 0.0]

    if "seed" not in episode_config:
        episode_config["seed"] = 0
        print_once("Warning: seed is not provided, will be set to 0")

    self.rng = np.random.default_rng(episode_config["seed"])

    # Sample random delay for rand_delay variant
    if "rand_delay_s" not in episode_config:
        rand_delay_s_range = episode_config.get("rand_delay_s_range", [0.0, 0.0])
        episode_config["rand_delay_s"] = float(
            self.rng.uniform(rand_delay_s_range[0], rand_delay_s_range[1])
        )

    assert isinstance(self.env, TableBin1Robot1Object)
    # Currently only one object is supported
    obj_num = 1
    if "object_poses_xyz_wxyz" not in episode_config:
        if (
            "object_init_bin_ids" not in episode_config
            or len(episode_config["object_init_bin_ids"]) == 0
        ):
            episode_config["object_init_bin_ids"] = self.rng.choice(
                list(range(4)), size=obj_num, replace=False
            )
        episode_config["object_poses_xyz_wxyz"] = []
        for i, bin_id in enumerate(episode_config["object_init_bin_ids"]):
            bin_pose_xyz_wxyz = cast(
                TableBin, self.env.scene
            ).sample_random_pose_xyz_wxyz(
                bin_id,
                self.env.objects[i].radius,
                self.env.objects[i].center_height,
                self.env.objects[i].rotation_angle_range,
                self.rng,
            )
        episode_config["object_poses_xyz_wxyz"].append(bin_pose_xyz_wxyz)
    if "robot_init_tcp_poses_xyz_wxyz" not in episode_config:
        episode_config["robot_init_tcp_poses_xyz_wxyz"] = [
            get_random_4poses_convex_combination(
                self.robot_init_tcp_poses_candidates_xyz_wxyz[0], self.rng
            )
        ]
    if "robot_init_gripper_width" not in episode_config:
        episode_config["robot_init_gripper_width"] = [
            get_random_convex_combination(
                self.robot_init_gripper_width_candidates[0], self.rng
            )
        ]

    if (
        "shuffle_bin_material" in episode_config
        and episode_config["shuffle_bin_material"]
    ):
        bin_num = 4
        if (
            "init_bin_materials" not in episode_config
            or len(episode_config["init_bin_materials"]) == 0
        ):
            shuffle_bin_materials = (
                self.rng.choice(
                    list(range(1, bin_num + 1)),
                    size=bin_num,
                    replace=False,
                )
            ).tolist()
            episode_config["init_bin_materials"] = [0] * 4
            for bin_id, material_id in zip(list(range(bin_num)), shuffle_bin_materials):
                episode_config["init_bin_materials"][bin_id] = material_id

        if (
            "final_bin_materials" not in episode_config
            or len(episode_config["final_bin_materials"]) == 0
        ):

            shuffle_bin_materials = (
                self.rng.choice(
                    list(range(1, bin_num + 1)),
                    size=bin_num,
                    replace=False,
                )
            ).tolist()
            episode_config["final_bin_materials"] = [0] * 4
            for bin_id, material_id in zip(
                list(range(bin_num)),
                shuffle_bin_materials,
            ):
                episode_config["final_bin_materials"][bin_id] = material_id

    else:
        # Don't shuffle bin materials
        if (
            "init_bin_materials" not in episode_config
            or len(episode_config["init_bin_materials"]) == 0
        ):
            episode_config["init_bin_materials"] = self.rng.choice(
                list(range(1, 5)),
                size=4,
                replace=False,
            ).tolist()

        if (
            "final_bin_materials" not in episode_config
            or len(episode_config["final_bin_materials"]) == 0
        ):
            episode_config["final_bin_materials"] = episode_config[
                "init_bin_materials"
            ].copy()
        else:
            assert (
                episode_config["final_bin_materials"]
                == episode_config["init_bin_materials"]
            ), "final_bin_materials should be the same as init_bin_materials if not shuffle bin material"

    if "delta_bin_id" in episode_config and episode_config["delta_bin_id"] != 0:
        assert isinstance(self.agent, PickAndPlaceBackAgent)
        episode_config["delta_bin_id"] = self.agent.delta_bin_id
        episode_config["match_bin_material"] = False

    return episode_config


class PickAndPlaceBack(BaseTask):
    def __init__(
        self,
        shuffle_bin_material: bool,
        robot_init_tcp_poses_candidates_xyz_wxyz: list[npt.NDArray[np.float64]],
        robot_init_gripper_width_candidates: list[npt.NDArray[np.float64]],
        object_init_bin_ids: list[int],  # Empty list means random picking from all bins
        delta_bin_id: int,
        delay_init_bin_materials: bool,
        init_bin_materials: list[
            int
        ],  # Empty list means randomly picking from all materials
        final_bin_materials: list[
            int
        ],  # Empty list means randomly picking from all materials
        rand_delay_s_range: list[float] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        """
        -------------
        |  0  |  1  |
        -------------
        |  2  |  3  |
        -------------
        robot_arm here
        """
        self.shuffle_bin_material: bool = shuffle_bin_material

        self.robot_init_tcp_poses_candidates_xyz_wxyz: list[npt.NDArray[np.float64]] = [
            np.array(pose) for pose in robot_init_tcp_poses_candidates_xyz_wxyz
        ]

        self.robot_init_gripper_width_candidates: list[npt.NDArray[np.float64]] = [
            np.array(width) for width in robot_init_gripper_width_candidates
        ]

        self.object_init_bin_ids: list[int] = object_init_bin_ids
        self.delta_bin_id: int = delta_bin_id
        self.delay_init_bin_materials: bool = delay_init_bin_materials
        self.init_bin_materials: list[int] = init_bin_materials
        self.final_bin_materials: list[int] = final_bin_materials
        self.rand_delay_s_range: list[float] | None = rand_delay_s_range

    @property
    def customized_obs_dict(self) -> dict[str, Any]:

        customized_obs_dict: dict[str, Any] = {}

        robot_pose = self.env.robots[0].tcp_xyz_wxyz
        object_pose = self.env.objects[0].pose_xyz_wxyz
        tcp_relative_pose_to_item = np.zeros(7)
        tcp_relative_pose_to_item[:3] = robot_pose[:3] - object_pose[:3]
        tcp_relative_pose_to_item[3:] = quaternions.qmult(
            quaternions.qinverse(object_pose[3:]), robot_pose[3:]
        )
        customized_obs_dict["tcp_relative_pose_to_item"] = tcp_relative_pose_to_item

        tcp_relative_pose_to_bin_center = np.zeros(7)
        scene = cast(TableBin, self.env.scene)
        bin_id = scene.get_bin_id(object_pose[:3], ignore_height=True)
        if bin_id != -1:
            tcp_relative_pose_to_bin_center[:3] = robot_pose[
                :3
            ] - scene.get_bin_center_xyz(bin_id)
            tcp_relative_pose_to_bin_center[3:] = robot_pose[3:]
            customized_obs_dict["tcp_relative_pose_to_bin_center"] = (
                tcp_relative_pose_to_bin_center
            )

        return customized_obs_dict


PickAndPlaceBack._process_episode_config = _process_episode_config


class PickAndPlaceBackParallel(ParallelTask):
    def __init__(
        self,
        shuffle_bin_material: bool,
        robot_init_tcp_poses_candidates_xyz_wxyz: list[npt.NDArray[np.float64]],
        robot_init_gripper_width_candidates: list[npt.NDArray[np.float64]],
        object_init_bin_ids: list[int],  # Empty list means random picking from all bins
        delta_bin_id: int,
        delay_init_bin_materials: bool,
        init_bin_materials: list[
            int
        ],  # Empty list means randomly picking from all materials
        final_bin_materials: list[
            int
        ],  # Empty list means randomly picking from all materials
        rand_delay_s_range: list[float] | None = None,
        **kwargs,
    ):

        self.shuffle_bin_material: bool = shuffle_bin_material

        self.robot_init_tcp_poses_candidates_xyz_wxyz: list[npt.NDArray[np.float64]] = [
            np.array(pose) for pose in robot_init_tcp_poses_candidates_xyz_wxyz
        ]

        self.robot_init_gripper_width_candidates: list[npt.NDArray[np.float64]] = [
            np.array(width) for width in robot_init_gripper_width_candidates
        ]

        self.object_init_bin_ids: list[int] = object_init_bin_ids
        self.delta_bin_id: int = delta_bin_id
        self.delay_init_bin_materials: bool = delay_init_bin_materials
        self.init_bin_materials: list[int] = init_bin_materials
        self.final_bin_materials: list[int] = final_bin_materials
        self.rand_delay_s_range: list[float] | None = rand_delay_s_range

        super().__init__(**kwargs)

        """
        -------------
        |  0  |  1  |
        -------------
        |  2  |  3  |
        -------------
        robot_arm here
        """
        # To enable calculations that is specific to the environment in _process_episode_config,
        self.env: TableBin1Robot1Object = hydra.utils.instantiate(
            enable_hydra_target(self.env_cfg)
        )


PickAndPlaceBackParallel._process_episode_config = _process_episode_config
