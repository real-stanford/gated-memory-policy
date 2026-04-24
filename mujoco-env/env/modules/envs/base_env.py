from typing import Any, cast

import numpy as np
import numpy.typing as npt
from dm_control import mjcf
from dm_control.mjcf.element import _AttachableElement
from dm_control.mjcf.physics import Physics
from mujoco import MjData, MjModel

from env.modules.common import castf64, robot_data_type
from env.modules.judges.base_judge import BaseJudge
from env.modules.objects.base_deformable_object import BaseDeformableObject
from env.modules.objects.base_object import BaseObject
from env.modules.objects.base_rigid_object import BaseRigidObject
from env.modules.robots.base_robot import BaseRobot
from env.modules.scenes.base_scene import BaseScene
from loguru import logger


class BaseEnv:
    def __init__(
        self,
        scene: BaseScene,
        judge: BaseJudge,
        robots: list[BaseRobot],
        robot_base_poses_xyz_wxyz: list[list[float]],
        robot_init_tcp_poses_xyz_wxyz: "list[list[float]] | None",
        robot_init_gripper_width: list[list[float]],
        objects: list[BaseObject],
        object_poses_xyz_wxyz: list[list[float]],
        control_freq: float,
        wait_until_stable: bool,
        **kwargs,  # To accept extra arguments from hydra
    ):
        logger.info(f"BaseEnv redundant kwargs: {kwargs}")
        self.scene: BaseScene = scene
        self.judge: BaseJudge = judge
        self.robots: list[BaseRobot] = robots
        assert len(robot_base_poses_xyz_wxyz) == len(robots) > 0
        assert len(object_poses_xyz_wxyz) == len(objects)
        self.robot_base_poses_xyz_wxyz: list[npt.NDArray[np.float64]] = [
            np.array(pose) for pose in robot_base_poses_xyz_wxyz
        ]

        if robot_init_tcp_poses_xyz_wxyz is not None:
            self.robot_init_tcp_poses_xyz_wxyz: (
                "list[npt.NDArray[np.float64]] | None"
            ) = [np.array(pose) for pose in robot_init_tcp_poses_xyz_wxyz]
        else:
            self.robot_init_tcp_poses_xyz_wxyz = None

        self.robot_init_gripper_width: list[npt.NDArray[np.float64]] = [
            np.array(width) for width in robot_init_gripper_width
        ]
        self.objects: list[BaseObject] = objects
        self.object_poses_xyz_wxyz: list[npt.NDArray[np.float64]] = [
            np.array(pose) for pose in object_poses_xyz_wxyz
        ]

        self.control_freq: float = (
            control_freq  # The frequency of the agent control loop
        )

        self.mjcf_model: "mjcf.RootElement | None" = None
        self.physics: "Physics | None" = None

        self.mujoco_dt: float | None = None

        self.episode_current_timestamp: float = 0.0

        self.wait_until_stable: bool = wait_until_stable

    def init_simulator(self):
        self.mjcf_model = self._load_mjcf_models()
        self.physics = Physics.from_mjcf_model(self.mjcf_model)

        # logger.info(self.mjcf_model.to_xml_string())
        # print_all_attrs(self.physics.model)

        assert self.physics is not None
        self.mujoco_dt = self.model.opt.timestep
        self.scene.set_mujoco_ref(self.physics)
        for idx, robot in enumerate(self.robots):
            robot.set_mujoco_ref(
                self.physics,
                self.robot_base_poses_xyz_wxyz[idx],
            )
        for obj in self.objects:
            obj.set_mujoco_ref(self.physics)
        self.reset()

    def _load_mjcf_models(self):
        world_model = self.scene.mjcf_model

        assert isinstance(world_model.worldbody, _AttachableElement)
        for robot, attachment_pose_xyz_wxyz in zip(
            self.robots, self.robot_base_poses_xyz_wxyz
        ):
            robot_site = world_model.worldbody.add(
                "site",
                name=f"{robot.arm_name}_site",
                pos=attachment_pose_xyz_wxyz[:3],
                quat=attachment_pose_xyz_wxyz[3:],
            )
            assert isinstance(robot_site, _AttachableElement)
            _ = robot_site.attach(robot.mjcf_model)

        for obj, obj_pose_xyz_wxyz in zip(self.objects, self.object_poses_xyz_wxyz):
            obj_site = world_model.worldbody.add(
                "site",
                name=f"{obj.name}_site",
                pos=obj_pose_xyz_wxyz[:3],
                quat=obj_pose_xyz_wxyz[3:],
                group=3,
            )
            assert isinstance(obj_site, _AttachableElement)

            if isinstance(obj, BaseRigidObject):
                _ = obj_site.attach(obj.mjcf_model).add("freejoint")
            elif isinstance(obj, BaseDeformableObject):
                _ = obj_site.attach(obj.mjcf_model)

            else:
                raise ValueError(f"Unknown object type: {type(obj)}")

        return world_model

    def step(self, actions: list[robot_data_type] | None, render_image: bool = True):
        # step_start_time = time.time()
        if actions is not None:
            assert (
                self.physics is not None
            ), "Simulator is not initialized. Please call init_simulator() first."
            assert len(actions) == len(self.robots)

            current_robots_ctrl = []
            target_robots_ctrl = []
            for idx, (robot, action) in enumerate(zip(self.robots, actions)):
                robot_tcp_xyz_wxyz = cast(
                    npt.NDArray[np.float64], action["tcp_xyz_wxyz"]
                )
                target_robots_ctrl.append(robot.solve_ik(robot_tcp_xyz_wxyz))
                gripper_width = cast(npt.NDArray[np.float64], action["gripper_width"])
                if isinstance(gripper_width, float):
                    gripper_width = gripper_width
                    robot.set_gripper_movement_cmd(gripper_width)
                else:
                    robot.set_gripper_movement_cmd(gripper_width[0])
                current_robots_ctrl.append(self.data.ctrl[robot.arm_ctrl_ids])
                robot.set_last_action(robot_tcp_xyz_wxyz)

            assert self.mujoco_dt is not None
            total_steps = int(1 / self.control_freq / self.mujoco_dt)
            for i in range(total_steps):

                for robot, current_ctrl, target_ctrl in zip(
                    self.robots, current_robots_ctrl, target_robots_ctrl
                ):
                    self.data.ctrl[robot.arm_ctrl_ids] = (
                        current_ctrl + (target_ctrl - current_ctrl) * i / total_steps
                    )
                # physics_step_start_time = time.time()
                # with self.physics.suppress_physics_errors():
                self.physics.step()
                # physics_step_end_time = time.time()
                # physics_step_time = physics_step_end_time - physics_step_start_time
                # logger.info(f"Physics step time: {physics_step_time}")

            self.episode_current_timestamp += 1.0 / self.control_freq

        obs = {
            "robots_obs": self._get_robots_obs(render_image),
            "env_objs_obs": self._get_env_objs_obs(render_image),
        }

        # logger.info(
        #     f"BaseEnv step: {obs['robots_obs'][0]['tcp_xyz_wxyz'][1]:.3f}, {obs['robots_obs'][1]['tcp_xyz_wxyz'][1]:.3f}"
        # )
        self.judge.update(**obs)
        reward = self.judge.get_reward()
        done = self.judge.get_done()
        info = {
            "judge_states": self.judge.get_states(),
            "executed_action": self._get_executed_action(),
            "timestamp": self.episode_current_timestamp,
        }

        # left_tracking_err = np.linalg.norm(
        #     info["executed_action"][0]["tcp_xyz_wxyz"][3:]
        #     - obs["robots_obs"][0]["tcp_xyz_wxyz"][3:]
        # )
        # right_tracking_err = np.linalg.norm(
        #     info["executed_action"][1]["tcp_xyz_wxyz"][3:]
        #     - obs["robots_obs"][1]["tcp_xyz_wxyz"][3:]
        # )

        # logger.info(f"BaseEnv step: {left_tracking_err:.3f}, {right_tracking_err:.3f}")

        # step_end_time = time.time()
        # step_time = step_end_time - step_start_time
        # logger.info(f"Step time: {step_time}")

        return (obs, reward, done, info)

    def _get_robots_obs(self, render_image: bool) -> list[robot_data_type]:
        robots_obs: list[robot_data_type] = []
        for robot in self.robots:
            robots_obs.append(robot.get_obs(render_image))
        return robots_obs

    def _get_executed_action(
        self,
    ) -> list[robot_data_type]:
        init_action: list[robot_data_type] = []
        for robot in self.robots:
            init_action.append(robot._get_executed_action())
        return init_action

    def _get_env_objs_obs(self, render_image: bool) -> list[robot_data_type]:
        raise NotImplementedError

    @property
    def model(self) -> MjModel:

        assert (
            self.physics is not None
        ), "Simulator is not initialized. Please call init_simulator() first."
        return cast(MjModel, self.physics.model.ptr)

    @property
    def data(self) -> MjData:
        assert (
            self.physics is not None
        ), "Simulator is not initialized. Please call init_simulator() first."
        return cast(MjData, self.physics.data.ptr)

    def reset(self, episode_config: dict[str, Any] | None = None):
        assert (
            self.physics is not None
        ), "Simulator is not initialized. Please call init_simulator() first."

        for object, object_pose_xyz_wxyz in zip(
            self.objects, self.object_poses_xyz_wxyz
        ):
            if isinstance(object, BaseRigidObject):
                object.set_pose_xyz_wxyz(object_pose_xyz_wxyz)
            elif isinstance(object, BaseDeformableObject):
                object.reset_grid_pos()

        self.reset_robot_joints()

        self.episode_current_timestamp = 0.0
        self.judge.reset(episode_config)

        obs = {
            "robots_obs": self._get_robots_obs(render_image=True),
            "env_objs_obs": self._get_env_objs_obs(render_image=True),
        }
        info = {
            "executed_action": self._get_executed_action(),
            "timestamp": self.episode_current_timestamp,
        }
        return (obs, info)

    def reset_robot_joints(self):

        for i, robot in enumerate(self.robots):
            robot.set_joint_state(
                qpos=robot.arm_init_qpos,
                gripper_width=np.array([robot.gripper_width_m]),
            )

        assert self.mujoco_dt is not None
        assert (
            self.physics is not None
        ), "Simulator is not initialized. Please call init_simulator() first."

        if self.robot_init_tcp_poses_xyz_wxyz is None:
            self.robot_init_tcp_poses_xyz_wxyz = []
            for i, robot in enumerate(self.robots):
                self.robot_init_tcp_poses_xyz_wxyz.append(robot.tcp_xyz_wxyz)

        else:
            for i, robot in enumerate(self.robots):
                target_robot_ctrl = robot.solve_ik(
                    self.robot_init_tcp_poses_xyz_wxyz[i]
                )
                robot.set_joint_state(
                    qpos=target_robot_ctrl,
                    gripper_width=self.robot_init_gripper_width[i],
                )
                robot.set_last_action(self.robot_init_tcp_poses_xyz_wxyz[i])

        if self.wait_until_stable:
            self._wait_until_stable()

    def _wait_until_stable(self, vel_threshold: float = 2e-5, max_steps: int = 1000):
        step_num = 0
        assert self.physics is not None
        prev_qpos = castf64(self.data.qpos.copy())
        while True:
            for i in range(10):
                self.physics.step()
            new_qpos = castf64(self.data.qpos.copy())
            avg_vel = np.linalg.norm((new_qpos - prev_qpos) / len(new_qpos))
            if avg_vel < vel_threshold:
                logger.info(f"Environment is stable after {step_num} steps.")
                break

            prev_qpos = new_qpos.copy()
            step_num += 1
            if step_num > max_steps:
                raise RuntimeError(
                    f"Environment is not stable after {max_steps} steps. {self.data.qvel=}. Please check the simulation setup. Please make sure the initial robot pose is not colliding with the environment."
                )
