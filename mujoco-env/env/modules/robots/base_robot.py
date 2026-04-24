from typing import TYPE_CHECKING, Optional, cast

import numpy as np
import numpy.typing as npt
from dm_control import mjcf
from dm_control.mjcf.element import _AttachableElement
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import enums

from robot_utils.pose_utils import get_relative_pose

if TYPE_CHECKING:
    from dm_control.mjcf.physics import Physics

import mink
from mujoco import MjData, MjModel, mj_name2id, mjtFrame, mjtObj, mju_mat2Quat

from env.modules.common import robot_data_type
from loguru import logger


class BaseRobot:
    def __init__(
        self,
        arm_name: str,
        arm_mjcf_path: str,
        arm_home_qpos: list[float],
        arm_joint_names: list[str],
        arm_actuator_names: list[str],
        arm_max_speed: list[float],
        attachment_site: str,
        gripper_name: str,
        gripper_mjcf_path: str,
        gripper_width_m: float,
        gripper_joint_names: list[str],
        gripper_actuator_names: list[str],
        gripper_tcp_site: str,
        camera_names: list[str],
        camera_resolution_hw: tuple[int, int],
        ik_solver: str = "quadprog",
        ik_dt: float = 0.002,
        ik_max_iter: int = 20,
        ik_vel_threshold: float = 1e-4,
    ):

        self.arm_name: str = arm_name
        self.arm_mjcf_path: str = arm_mjcf_path
        self.attachment_site: str = attachment_site
        self.arm_init_qpos: npt.NDArray[np.float64] = np.array(arm_home_qpos)
        self.arm_joint_names: list[str] = arm_joint_names
        self.arm_actuator_names: list[str] = arm_actuator_names
        self.arm_max_speed: npt.NDArray[np.float64] = np.array(arm_max_speed)
        self.gripper_name: str = gripper_name
        self.gripper_mjcf_path: str = gripper_mjcf_path
        self.gripper_width_m: float = gripper_width_m
        self.gripper_joint_names: list[str] = gripper_joint_names
        assert len(gripper_joint_names) == 2, "Only support two finger gripper for now"
        self.gripper_actuator_names: list[str] = gripper_actuator_names
        self.gripper_tcp_site: str = gripper_tcp_site

        self.physics: Physics
        self.camera_names: list[str] = camera_names
        self.mjcf_model: mjcf.RootElement = self._load_mjcf_models()
        self.arm_joint_ids: list[int] = []
        self.arm_ctrl_ids: list[int] = []
        self.gripper_joint_ids: list[int] = []
        self.gripper_ctrl_ids: list[int] = []

        self.camera_ids: list[int] = []
        self.camera_resolution_hw: tuple[int, int] = camera_resolution_hw

        self.tcp_site_id: int

        # Mink related
        self.mink_target_mocap_name: str = f"{self.arm_name}_{self.gripper_name}_target"
        self.mink_tcp_site_id: int
        self.arm_task: mink.FrameTask
        self.mink_configuration: mink.Configuration
        self.mink_physics: (
            Physics  # Setup a copy of mujoco model and data specificly for mink
        )
        # So the ik computation complexity will not be affected by other objects (especially for deformable objects)
        self.mink_arm_joint_ids: list[int] = []
        self.mink_arm_ctrl_ids: list[int] = []
        self.mink_limits: list[mink.Limit] = []
        self.mink_tasks: list[mink.Task] = []

        self.ik_solver: str = ik_solver
        self.ik_max_iter: int = ik_max_iter
        self.ik_dt: float = ik_dt
        self.ik_vel_threshold: float = ik_vel_threshold

        self.robot_base_pose_xyz_wxyz: npt.NDArray[np.float64]
        self.latest_action_tcp_xyz_wxyz: Optional[npt.NDArray[np.float64]]
        self._setup_mink()

    def _load_mjcf_models(self):
        arm_mjcf_model = mjcf.from_path(self.arm_mjcf_path)
        arm_mjcf_model.root.model = self.arm_name
        # logger.info(arm_mjcf_model.root.model)
        del arm_mjcf_model.keyframe
        gripper_mjcf_model = mjcf.from_path(self.gripper_mjcf_path)
        attachment_site = arm_mjcf_model.find("site", self.attachment_site)
        if attachment_site is None or not isinstance(
            attachment_site, _AttachableElement
        ):
            raise ValueError(
                f"Attachment site {self.attachment_site} not found in robot MJCF model {self.arm_mjcf_path}"
            )
        attachment_site.attach(gripper_mjcf_model)

        return arm_mjcf_model

    def _setup_mink(
        self,
    ):
        # Create a copy of the mujoco model for mink
        mink_mjcf_model = self._load_mjcf_models()
        assert isinstance(mink_mjcf_model.worldbody, _AttachableElement)
        mocap_body = mink_mjcf_model.worldbody.add(
            "body", name=self.mink_target_mocap_name, mocap=True
        )
        _ = mocap_body.add(
            "geom",
            type="box",
            size=[0.005, 0.005, 0.005],
            contype=0,
            conaffinity=0,
            rgba=[1, 1, 1, 0.2],
        )
        self.mink_physics = cast(
            "Physics", mjcf.Physics.from_mjcf_model(mink_mjcf_model)
        )

        self.mink_arm_joint_ids = [
            mj_name2id(
                self.mink_model,
                int(mjtObj.mjOBJ_JOINT),
                f"{joint_name}",
            )
            for joint_name in self.arm_joint_names
        ]
        self.mink_arm_ctrl_ids = [
            mj_name2id(
                self.mink_model,
                int(mjtObj.mjOBJ_ACTUATOR),
                f"{actuator_name}",
            )
            for actuator_name in self.arm_actuator_names
        ]
        self.mink_configuration = mink.Configuration(model=self.mink_model)
        self.mink_limits.extend(
            [
                mink.ConfigurationLimit(model=self.mink_configuration.model),
                mink.VelocityLimit(
                    model=self.mink_configuration.model,
                    velocities={
                        f"{joint_name}": max_speed
                        for joint_name, max_speed in zip(
                            self.arm_joint_names, self.arm_max_speed
                        )
                    },
                ),
            ]
        )
        self.arm_task = mink.FrameTask(
            frame_name=f"{self.gripper_name}/{self.gripper_tcp_site}",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.arm_task.set_target(
            mink.SE3.from_mocap_name(
                self.mink_model,
                self.mink_data,
                self.mink_target_mocap_name,
            )
        )

        self.mink_tasks.append(self.arm_task)

        self.mink_tcp_site_id = mj_name2id(
            self.mink_model,
            int(mjtObj.mjOBJ_SITE),
            f"{self.gripper_name}/{self.gripper_tcp_site}",
        )
        if self.mink_tcp_site_id == -1:
            raise ValueError(
                f"Site {self.gripper_tcp_site} not found in robot {self.arm_name}"
            )

    def solve_ik(self, tcp_xyz_wxyz: npt.NDArray[np.float64]):
        """
        tcp_xyz_wxyz is in the global frame. Will be converted to the robot base frame according to robot_base_pose_xyz_wxyz
        """

        tcp_relative_to_base = get_relative_pose(
            tcp_xyz_wxyz, self.robot_base_pose_xyz_wxyz
        )

        # Copy joint data from the large model to the mink model
        self.mink_data.qpos[self.mink_arm_joint_ids] = self.data.qpos[
            self.arm_joint_ids
        ]
        self.mink_data.ctrl[self.mink_arm_ctrl_ids] = self.data.ctrl[self.arm_ctrl_ids]
        self.mink_data.qvel[self.mink_arm_joint_ids] = self.data.qvel[
            self.arm_joint_ids
        ]
        self.mink_data.qacc[self.mink_arm_joint_ids] = 0.0
        self.mink_data.qfrc_applied[self.mink_arm_joint_ids] = 0.0

        mocap_id = self.mink_model.body(self.mink_target_mocap_name).mocapid[0]
        self.mink_data.mocap_pos[mocap_id] = tcp_relative_to_base[:3]
        self.mink_data.mocap_quat[mocap_id] = tcp_relative_to_base[3:]

        if tcp_relative_to_base[3] < 0.0:
            logger.info(f"Warning: tcp_xyz_wxyz: {tcp_relative_to_base} has negative w")
            tcp_relative_to_base[3:] = -tcp_relative_to_base[3:]

        self.arm_task.set_target(
            mink.SE3.from_mocap_name(
                self.mink_model,
                self.mink_data,
                self.mink_target_mocap_name,
            )
        )

        self.mink_configuration.update(self.mink_data.qpos)
        for i in range(self.ik_max_iter):
            vel = mink.solve_ik(
                self.mink_configuration,
                self.mink_tasks,
                self.ik_dt,
                solver=self.ik_solver,
                damping=1e-3,
            )
            self.mink_configuration.integrate_inplace(vel, self.ik_dt)
            if np.linalg.norm(vel) < self.ik_vel_threshold:
                break

        return self.mink_configuration.q[self.mink_arm_joint_ids].copy()

    def set_mujoco_ref(
        self,
        physics: "Physics",
        robot_base_pose_xyz_wxyz: npt.NDArray[np.float64],
    ):
        self.robot_base_pose_xyz_wxyz = robot_base_pose_xyz_wxyz.copy()
        self.physics = physics
        self.arm_joint_ids = []
        self.gripper_joint_ids = []
        self.arm_ctrl_ids = []
        self.gripper_ctrl_ids = []

        for joint_name in self.arm_joint_names:
            joint_id = mj_name2id(
                self.model, int(mjtObj.mjOBJ_JOINT), f"{self.arm_name}/{joint_name}"
            )
            if joint_id == -1:
                raise ValueError(
                    f"Joint {joint_name} not found in robot {self.arm_name}"
                )
            self.arm_joint_ids.append(joint_id)

        for actuator_name in self.arm_actuator_names:
            ctrl_id = mj_name2id(
                self.model,
                int(mjtObj.mjOBJ_ACTUATOR),
                f"{self.arm_name}/{actuator_name}",
            )
            if ctrl_id == -1:
                logger.info(
                    f"Control {actuator_name} not found in robot {self.arm_name}"
                )
            self.arm_ctrl_ids.append(ctrl_id)

        for joint_name in self.gripper_joint_names:
            joint_id = mj_name2id(
                self.model,
                int(mjtObj.mjOBJ_JOINT),
                f"{self.arm_name}/{self.gripper_name}/{joint_name}",
            )
            if joint_id == -1:
                raise ValueError(
                    f"Joint {joint_name} not found in robot {self.arm_name}/{self.gripper_name}"
                )
            self.gripper_joint_ids.append(joint_id)

        for actuator_name in self.gripper_actuator_names:
            ctrl_id = mj_name2id(
                self.model,
                int(mjtObj.mjOBJ_ACTUATOR),
                f"{self.arm_name}/{self.gripper_name}/{actuator_name}",
            )
            if ctrl_id == -1:
                logger.info(
                    f"Control {actuator_name} not found in robot {self.arm_name}/{self.gripper_name}"
                )
            self.gripper_ctrl_ids.append(ctrl_id)

        for camera_name in self.camera_names:
            camera_id = mj_name2id(
                self.model,
                int(mjtObj.mjOBJ_CAMERA),
                f"{self.arm_name}/{self.gripper_name}/{camera_name}",
            )
            if camera_id == -1:
                raise ValueError(
                    f"Camera {camera_name} not found in robot {self.arm_name}/{self.gripper_name}"
                )
            self.camera_ids.append(camera_id)

        self.tcp_site_id = mj_name2id(
            self.model,
            int(mjtObj.mjOBJ_SITE),
            f"{self.arm_name}/{self.gripper_name}/{self.gripper_tcp_site}",
        )
        if self.tcp_site_id == -1:
            raise ValueError(
                f"Site {self.gripper_tcp_site} not found in robot {self.arm_name}"
            )

        logger.info(
            f"{self.arm_name}/{self.gripper_name} robot joint ids: {self.arm_joint_ids} eef joint ids: {self.gripper_joint_ids} camera ids: {self.camera_ids}",
            f"arm_ctrl_ids: {self.arm_ctrl_ids} gripper_ctrl_ids: {self.gripper_ctrl_ids}, tcp_site_id: {self.tcp_site_id}",
        )

    def set_arm_init_qpos(self, qpos: npt.NDArray[np.float64]):
        self.data.qpos[self.arm_joint_ids] = qpos

    def set_last_action(self, tcp_xyz_wxyz: npt.NDArray[np.float64]):
        self.latest_action_tcp_xyz_wxyz = tcp_xyz_wxyz.copy()

    def set_joint_state(
        self,
        qpos: npt.NDArray[np.float64],
        qvel: Optional[npt.NDArray[np.float64]] = None,
        gripper_width: Optional[npt.NDArray[np.float64]] = None,
    ):
        # Usually used for resetting

        self.data.qpos[self.arm_joint_ids] = qpos
        self.data.ctrl[self.arm_ctrl_ids] = qpos

        if qvel is not None:
            self.data.qvel[self.arm_joint_ids] = qvel
        else:
            self.data.qvel[self.arm_joint_ids] = 0.0
        self.data.qacc[self.arm_joint_ids] = 0.0
        self.data.qfrc_applied[self.arm_joint_ids] = 0.0

        if gripper_width is not None:
            self.gripper_width_m = gripper_width[0]
            self.data.qpos[self.gripper_joint_ids] = gripper_width / 2
            self.data.ctrl[self.gripper_ctrl_ids] = gripper_width / 2

        self.data.qvel[self.gripper_joint_ids] = 0.0
        self.data.qacc[self.gripper_joint_ids] = 0.0
        self.data.qfrc_applied[self.gripper_joint_ids] = 0.0

    def set_gripper_movement_cmd(
        self,
        gripper_width_m: float,
    ):
        self.data.ctrl[self.gripper_ctrl_ids] = gripper_width_m / 2
        self.gripper_width_m = gripper_width_m

    @property
    def mink_model(self) -> MjModel:
        return self.mink_physics.model.ptr

    @property
    def mink_data(self) -> MjData:
        return self.mink_physics.data.ptr

    @property
    def model(self) -> MjModel:
        return self.physics.model.ptr

    @property
    def data(self) -> MjData:
        return self.physics.data.ptr

    @property
    def tcp_xyz_wxyz(self) -> npt.NDArray[np.float64]:
        mat: npt.NDArray[np.float64] = self.data.site_xmat[self.tcp_site_id]
        pose = np.zeros(7)
        pose[:3] = self.data.site_xpos[self.tcp_site_id]
        mju_mat2Quat(pose[3:], mat)  # type: ignore
        pose[3:] = pose[3:] / np.linalg.norm(pose[3:])
        if pose[3] < 0:
            pose[3:] = -pose[3:]

        return pose

    @property
    def camera_images(self) -> dict[str, npt.NDArray[np.uint8]]:
        assert self.physics is not None
        assert len(self.camera_ids) > 0, "No camera ids set for robot"
        scene_option = wrapper.MjvOption()
        scene_option.frame = mjtFrame.mjFRAME_SITE.value
        scene_option.frame = enums.mjtFrame.mjFRAME_NONE
        images: dict[str, npt.NDArray[np.uint8]] = {}
        for camera_id, camera_name in zip(self.camera_ids, self.camera_names):
            raw_image = self.physics.render(
                height=self.camera_resolution_hw[0],
                width=self.camera_resolution_hw[1],
                camera_id=camera_id,
                scene_option=scene_option,
            )
            assert raw_image.dtype == np.uint8
            image = cast(npt.NDArray[np.uint8], raw_image)
            images[camera_name] = image
        return images

    def get_obs(self, render_image: bool) -> robot_data_type:
        obs: robot_data_type = {
            "name": f"{self.arm_name}_{self.gripper_name}",
            "arm_qpos": self.data.qpos[self.arm_joint_ids].copy(),
            "arm_qvel": self.data.qvel[self.arm_joint_ids].copy(),
            "arm_qacc": self.data.qacc[self.arm_joint_ids].copy(),
            "tcp_xyz_wxyz": self.tcp_xyz_wxyz,
            "gripper_width": np.array(
                [
                    self.data.qpos[self.gripper_joint_ids[0]]
                    + self.data.qpos[self.gripper_joint_ids[1]]
                ]
            ),
        }
        if render_image and self.camera_ids:
            obs.update(self.camera_images)
        return obs

    def _get_executed_action(self) -> robot_data_type:
        if self.latest_action_tcp_xyz_wxyz is None:
            self.latest_action_tcp_xyz_wxyz = self.tcp_xyz_wxyz
        return {
            "name": f"{self.arm_name}_{self.gripper_name}",
            "tcp_xyz_wxyz": self.latest_action_tcp_xyz_wxyz,
            "gripper_width": np.array([self.gripper_width_m]),
        }
