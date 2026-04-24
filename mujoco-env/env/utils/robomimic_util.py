# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/common/robomimic_util.py

import copy

import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from scipy.spatial.transform import Rotation


class RobomimicAbsoluteActionConverter:
    def __init__(self, dataset_path):

        modality_mapping = {
            "low_dim": [
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_joint_pos",
            ],
            "rgb": [
                "robot0_eye_in_hand_image",
                "robot0_shoulder_cam_image",
            ],
        }

        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        abs_env_meta = copy.deepcopy(env_meta)

        # This is deprecated (for ik controller)
        # abs_env_meta['env_kwargs']['controller_configs']['body_parts']['right']['control_delta'] = False
        abs_env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"][
            "input_type"
        ] = "absolute"

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        assert len(env.env.robots) in (1, 2)

        abs_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        # assert not abs_env.env.robots[0].composite_controller.use_delta
        # logger.info(f"{abs_env.env.robots[0].part_controllers['right'].input_type=}")

        self.env = env
        self.abs_env = abs_env
        self.file = h5py.File(dataset_path, "r")

    def __len__(self):
        return len(self.file["data"])

    def convert_actions(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        # in case of multi robot
        # reshape (N,14) to (N,2,8)
        # or (N,7) to (N,1,8)
        stacked_actions = actions.reshape(*actions.shape[:-1], -1, 7)

        env = self.env
        # generate abs actions

        action_goal_xyz_xyz = np.zeros(
            stacked_actions.shape[:-1] + (6,), dtype=stacked_actions.dtype
        )

        action_gripper = stacked_actions[..., [-1]]
        part_name = "right"
        for i in range(len(states)):
            _ = env.reset_to({"states": states[i]})

            env.step(stacked_actions[i].reshape(-1))

            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.env.robots):

                # read pos and ori from robots
                controller = robot.part_controllers[part_name]
                # Orientation is in axis-angle format

                # Directly setting controller doesn't work
                # robot.control(stacked_actions[i,idx],policy_step=True)
                # scaled_delta = controller.scale_action(stacked_actions[i,idx, :6])
                # action_goal_xyz_xyz[i,idx] = controller.delta_to_abs_action(scaled_delta, goal_update_mode="achieved")
                # controller.set_goal(stacked_actions[i,idx, :6])

                abs_pos = controller.goal_pos.copy()
                abs_rot = Rotation.from_matrix(controller.goal_ori).as_rotvec().copy()
                action_goal_xyz_xyz[i, idx] = np.concatenate(
                    [abs_pos, abs_rot], axis=-1
                )

        stacked_abs_actions = np.concatenate(
            [action_goal_xyz_xyz, action_gripper], axis=-1
        )
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_idx(self, idx):
        file = self.file
        demo = file[f"data/demo_{idx}"]
        # input
        states = demo["states"][:]
        actions = demo["actions"][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)
        return abs_actions

    def convert_and_eval_idx(self, idx):
        env = self.env
        abs_env = self.abs_env
        file = self.file
        # first step have high error for some reason, not representative
        eval_skip_steps = 1

        demo = file[f"data/demo_{idx}"]
        # input
        states = demo["states"][:]
        actions = demo["actions"][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)

        # verify
        robot0_eef_pos = demo["obs"]["robot0_eef_pos"][:]
        robot0_eef_quat = demo["obs"]["robot0_eef_quat"][:]

        delta_error_info = self.evaluate_rollout_error(
            env,
            states,
            actions,
            robot0_eef_pos,
            robot0_eef_quat,
            metric_skip_steps=eval_skip_steps,
        )
        abs_error_info = self.evaluate_rollout_error(
            abs_env,
            states,
            abs_actions,
            robot0_eef_pos,
            robot0_eef_quat,
            metric_skip_steps=eval_skip_steps,
        )

        info = {"delta_max_error": delta_error_info, "abs_max_error": abs_error_info}
        return abs_actions, info

    @staticmethod
    def evaluate_rollout_error(
        env, states, actions, robot0_eef_pos, robot0_eef_quat, metric_skip_steps=1
    ):
        # first step have high error for some reason, not representative

        # evaluate abs actions
        rollout_next_states = list()
        rollout_next_eef_pos = list()
        rollout_next_eef_quat = list()
        obs = env.reset_to({"states": states[0]})
        for i in range(len(states)):

            obs = env.reset_to({"states": states[i]})
            obs, reward, done, info = env.step(actions[i])
            obs = env.get_observation()

            rollout_next_states.append(env.get_state()["states"])
            rollout_next_eef_pos.append(obs["robot0_eef_pos"])
            rollout_next_eef_quat.append(obs["robot0_eef_quat"])
        rollout_next_states = np.array(rollout_next_states)
        rollout_next_eef_pos = np.array(rollout_next_eef_pos)
        rollout_next_eef_quat = np.array(rollout_next_eef_quat)

        next_state_diff = states[1:] - rollout_next_states[:-1]
        max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        next_eef_rot_diff = (
            Rotation.from_quat(robot0_eef_quat[1:])
            * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
        )
        next_eef_rot_dist = next_eef_rot_diff.magnitude()
        max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

        info = {
            "state": max_next_state_diff,
            "pos": max_next_eef_pos_dist,
            "rot": max_next_eef_rot_dist,
        }
        return info
