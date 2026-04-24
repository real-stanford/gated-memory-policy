import numpy as np
import numpy.typing as npt
import scipy.spatial.transform as st

from real_env.common.data_classes import Trajectory
from robot_utils.pose_utils import to_wxyz, to_xyzw


class JointTrajInterpolator:
    def __init__(
        self,
        init_trajectory: Trajectory,
        traj_smooth_time_s: float,
        max_joint_speed_per_s: float,
        trim_new_trajectory_time_s: float,
        vel_calc_interval_min_s: float = 0.05,
    ):
        self.dof: int = init_trajectory.data.shape[1]
        assert traj_smooth_time_s >= 0, "Update smooth time must be non-negative"
        assert max_joint_speed_per_s >= 0, "Max joint speed must be non-negative"
        self.traj_smooth_time_s: float = traj_smooth_time_s
        self.max_joint_speed_per_s: float = max_joint_speed_per_s

        self.timestamps: npt.NDArray[np.float64] = init_trajectory.timestamps
        self.pos: npt.NDArray[np.float64] = init_trajectory.data
        self.trim_new_trajectory_time_s: float = trim_new_trajectory_time_s
        self.vel_calc_interval_min_s: float = vel_calc_interval_min_s

    def update(self, new_trajectory: Trajectory, current_timestamp: float):
        assert (
            new_trajectory.data.shape[1] == self.dof
        ), f"New trajectory must have {self.dof} dimensions"

        # Trim new trajectory
        current_idx_in_new_traj = np.searchsorted(
            new_trajectory.timestamps,
            current_timestamp + self.trim_new_trajectory_time_s,
            side="right",
        )
        # new_trajectory.timestamps[current_idx_in_new_traj] is guaranteed to be greater than current_timestamp
        if current_idx_in_new_traj == len(new_trajectory.timestamps):
            print(
                f"The entire new trajectory is out of date. Current timestamp: {current_timestamp}, new trajectory timestamps: {new_trajectory.timestamps}. Interpolator is not updated."
            )
            return
        new_data = new_trajectory.data[current_idx_in_new_traj:]
        new_timestamps = new_trajectory.timestamps[current_idx_in_new_traj:]

        # Trim old trajectory
        current_pos = self.interpolate(current_timestamp)
        current_idx = np.searchsorted(self.timestamps, current_timestamp, side="right")
        if current_idx == len(self.timestamps):
            # The entire trajectory is out of date
            self.pos = np.array([current_pos])
            self.timestamps = np.array([current_timestamp])
        else:
            # TODO: add threasholed to trim all the history actions that are too close to the first new timestamp
            self.pos = np.concatenate([[current_pos], self.pos[current_idx:]])
            self.timestamps = np.concatenate(
                [[current_timestamp], self.timestamps[current_idx:]]
            )

        # Smooth new trajectory

        new_smooth_pos: list[npt.NDArray[np.float64]] = []
        for i in range(len(new_data)):

            if new_timestamps[i] - current_timestamp < self.traj_smooth_time_s:
                new_ratio = (
                    new_timestamps[i] - current_timestamp
                ) / self.traj_smooth_time_s
                old_pos = self.interpolate(new_timestamps[i])
                pos = old_pos * (1 - new_ratio) + new_data[i] * new_ratio
            else:
                pos = new_data[i]
            new_smooth_pos.append(pos)

        # Check if the new trajectory is too fast
        for i in range(len(new_timestamps)):
            if i == 0:
                last_pos = self.pos[-1]
                last_timestamp = self.timestamps[-1]
            else:
                last_pos = new_smooth_pos[i - 1]
                last_timestamp = new_timestamps[i - 1]

            pos_speed = np.linalg.norm(new_smooth_pos[i] - last_pos) / max(
                new_timestamps[i] - last_timestamp, self.vel_calc_interval_min_s
            )

            if pos_speed > self.max_joint_speed_per_s:
                print(
                    f"The new trajectory is too fast. Position speed: {pos_speed} rad/s, max position speed: {self.max_joint_speed_per_s} rad/s. Failed to update the interpolator."
                )
                # print(new_trajectory.data)
                return

        # Replace the old trajectory with the new trajectory
        remaining_idx = np.searchsorted(self.timestamps, new_timestamps[0], side="left")
        self.pos = np.concatenate([self.pos[:remaining_idx], np.array(new_smooth_pos)])
        self.timestamps = np.concatenate(
            [self.timestamps[:remaining_idx], new_timestamps]
        )
        assert len(self.pos) == len(
            self.timestamps
        ), f"len(self.pos): {len(self.pos)}, len(self.timestamps): {len(self.timestamps)}"

    def find_delta_latency(
        self,
        input_poses: npt.NDArray[np.float64],
        input_times: npt.NDArray[np.float64],
        current_timestamp: float,
        latency_start: float = -0.3,
        latency_end: float = 0.3,
        matching_dt: float = 0.05,
        latency_precision: float = 0.01,
    ) -> float:
        input_trajectory = Trajectory(data=input_poses, timestamps=input_times)
        input_interpolator = JointTrajInterpolator(
            input_trajectory, 0.0, float("inf"), trim_new_trajectory_time_s=0.0
        )

        current_pos_prev = self.interpolate(current_timestamp - matching_dt)
        current_pos_curr = self.interpolate(current_timestamp)
        current_pos_next = self.interpolate(current_timestamp + matching_dt)
        current_pos = np.stack(
            [current_pos_prev, current_pos_curr, current_pos_next], axis=0
        )

        errors = []
        latency_start = max(latency_start, input_times[0] - current_timestamp)
        latency_end = min(latency_end, input_times[-1] - current_timestamp)
        latency_range = np.arange(latency_start, latency_end, latency_precision)

        for latency in latency_range:
            latency = float(latency)

            input_pos_prev = input_interpolator.interpolate(
                current_timestamp - matching_dt + latency
            )
            input_pos_curr = input_interpolator.interpolate(current_timestamp + latency)
            input_pos_next = input_interpolator.interpolate(
                current_timestamp + matching_dt + latency
            )
            input_pos = np.stack(
                [input_pos_prev, input_pos_curr, input_pos_next], axis=0
            )

            pos_error = np.average(np.linalg.norm(input_pos - current_pos, axis=1))
            errors.append(pos_error)

        errors = np.array(errors)
        min_error_idx = np.argmin(errors)
        return float(latency_range[min_error_idx])

    def interpolate(self, timestamp: float) -> npt.NDArray[np.float64]:
        if len(self.timestamps) == 1:
            return self.pos[0]
        idx = np.searchsorted(self.timestamps, timestamp, side="right")
        if idx == len(self.timestamps):
            return self.pos[-1]
        elif idx == 0:
            return self.pos[0]
        else:
            ratio = (timestamp - self.timestamps[idx - 1]) / (
                self.timestamps[idx] - self.timestamps[idx - 1]
            )
            return self.pos[idx - 1] * (1 - ratio) + self.pos[idx] * ratio


class PoseTrajInterpolator:
    def __init__(
        self,
        init_trajectory: Trajectory,
        traj_smooth_time_s: float,
        max_pos_speed_m_per_s: float,
        max_rot_speed_rad_per_s: float,
        trim_new_trajectory_time_s: float,
        vel_calc_interval_min_s: float = 0.05,
    ):

        assert (
            init_trajectory.data.shape[1] == 7
        ), "Each action must have 7 numbers: x, y, z, qw, qx, qy, qz"

        quat_wxyz = init_trajectory.data[:, 3:7]
        quat_xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, 0:1]], axis=1)
        self.rot: st.Rotation = st.Rotation.from_quat(quat_xyzw)
        self.pos: npt.NDArray[np.float64] = init_trajectory.data[:, :3]
        self.timestamps: npt.NDArray[np.float64] = init_trajectory.timestamps

        assert traj_smooth_time_s >= 0, "Update smooth time must be non-negative"
        assert max_pos_speed_m_per_s >= 0, "Max position speed must be non-negative"
        assert max_rot_speed_rad_per_s >= 0, "Max rotation speed must be non-negative"
        self.traj_smooth_time_s: float = traj_smooth_time_s
        self.max_pos_speed_m_per_s: float = max_pos_speed_m_per_s
        self.max_rot_speed_rad_per_s: float = max_rot_speed_rad_per_s
        self.trim_new_trajectory_time_s: float = trim_new_trajectory_time_s
        self.vel_calc_interval_min_s: float = vel_calc_interval_min_s

    def update(self, new_trajectory: Trajectory, current_timestamp: float):
        # Trim new trajectory
        current_idx_in_new_traj = np.searchsorted(
            new_trajectory.timestamps,
            current_timestamp + self.trim_new_trajectory_time_s,
            side="right",
        )
        # new_trajectory.timestamps[current_idx_in_new_traj] is guaranteed to be greater than current_timestamp
        if current_idx_in_new_traj == len(new_trajectory.timestamps):
            print(
                f"The entire new trajectory is out of date. Current timestamp: {current_timestamp}, new trajectory timestamps: {new_trajectory.timestamps}. Interpolator is not updated."
            )
            return
        new_data = new_trajectory.data[current_idx_in_new_traj:]
        new_timestamps = new_trajectory.timestamps[current_idx_in_new_traj:]
        assert (
            new_timestamps[0] >= current_timestamp
        ), f"{new_timestamps[0]=}, {current_timestamp=}"

        # Trim old trajectory
        current_pos, current_rot = self.interpolate(current_timestamp)
        current_idx = np.searchsorted(self.timestamps, current_timestamp, side="right")
        if current_idx == len(self.timestamps):
            # The entire trajectory is out of date
            self.pos = np.array([current_pos])
            self.rot = current_rot
            self.timestamps = np.array([current_timestamp])
        else:
            self.pos = np.concatenate([[current_pos], self.pos[current_idx:]])
            self.rot = st.Rotation.concatenate([current_rot, self.rot[current_idx:]])
            self.timestamps = np.concatenate(
                [[current_timestamp], self.timestamps[current_idx:]]
            )

        # Smooth new trajectory
        new_smooth_rot: list[st.Rotation] = []
        new_smooth_pos: list[npt.NDArray[np.float64]] = []
        new_quat_wxyz = new_data[:, 3:7]
        new_quat_xyzw = to_xyzw(new_quat_wxyz)
        new_rot = st.Rotation.from_quat(new_quat_xyzw)
        for i in range(len(new_data)):

            if new_timestamps[i] - current_timestamp < self.traj_smooth_time_s:
                new_ratio = (
                    new_timestamps[i] - current_timestamp
                ) / self.traj_smooth_time_s
                old_pos, old_rot = self.interpolate(new_timestamps[i])
                pos = old_pos * (1 - new_ratio) + new_data[i, :3] * new_ratio
                rot_pair = st.Rotation.concatenate([old_rot, new_rot[i]])
                rot = st.Slerp([0, 1], rot_pair)(new_ratio)
            else:
                rot = new_rot[i]
                pos = new_data[i, :3]
            new_smooth_rot.append(rot)
            new_smooth_pos.append(pos)

        remaining_idx = np.searchsorted(self.timestamps, new_timestamps[0], side="left")

        # Check if the new trajectory is too fast
        for i in range(len(new_timestamps)):

            if i == 0:
                last_pos = self.pos[remaining_idx - 1]
                last_rot = self.rot if self.rot.single else self.rot[remaining_idx - 1]
                last_timestamp = self.timestamps[remaining_idx - 1]
            else:
                last_pos = new_smooth_pos[i - 1]
                last_rot = new_smooth_rot[i - 1]
                last_timestamp = new_timestamps[i - 1]

            if i == len(new_timestamps) - 1:
                next_pos = new_smooth_pos[i]
                next_rot = new_smooth_rot[i]
                next_timestamp = new_timestamps[i]
            else:
                next_pos = new_smooth_pos[i + 1]
                next_rot = new_smooth_rot[i + 1]
                next_timestamp = new_timestamps[i + 1]

            pos_speed = np.linalg.norm(next_pos - last_pos) / max(
                next_timestamp - last_timestamp, self.vel_calc_interval_min_s
            )
            if pos_speed > self.max_pos_speed_m_per_s:
                print(
                    f"The new trajectory is too fast. Position speed: {pos_speed} m/s, max position speed: {self.max_pos_speed_m_per_s} m/s, dt: {next_timestamp - last_timestamp} s. Failed to update the interpolator."
                )
                return
            rot_speed = st.Rotation.magnitude(last_rot.inv() * next_rot) / max(
                next_timestamp - last_timestamp, self.vel_calc_interval_min_s
            )
            if rot_speed > self.max_rot_speed_rad_per_s:
                print(f"{new_trajectory.data[:, 3:7]=}")
                print(f"{self.rot.as_quat()=}")
                print(f"{i=}, {next_timestamp=}, {last_timestamp=}, {rot_speed=}")
                print(
                    f"The new trajectory is too fast. Rotation speed: {rot_speed} rad/s, max rotation speed: {self.max_rot_speed_rad_per_s} rad/s, dt: {next_timestamp - last_timestamp} s. Failed to update the interpolator."
                )
                return

        # Replace the old trajectory with the new trajectory
        sliced_rot = self.rot if self.rot.single else self.rot[:remaining_idx]
        self.rot = st.Rotation.concatenate([sliced_rot, *new_smooth_rot])
        self.pos = np.concatenate([self.pos[:remaining_idx], np.array(new_smooth_pos)])
        self.timestamps = np.concatenate(
            [self.timestamps[:remaining_idx], new_timestamps]
        )
        assert (
            len(self.rot) == len(self.pos) == len(self.timestamps)
        ), f"len(self.rot): {len(self.rot)}, len(self.pos): {len(self.pos)}, len(self.timestamps): {len(self.timestamps)}"
        if len(self.timestamps) > 1:
            assert np.all(self.timestamps[1:] > self.timestamps[:-1])

        # # DEBUG: Print the updated trajectory
        # if new_trajectory.data.shape[0] > 10:
        #     print(f"Input trajectory: {new_trajectory.data[:, :]}")
        #     print(f"Input timestamps: {new_timestamps[:]}")
        #     print(f"Updated trajectory: {self.pos[:, :]}")
        #     print(f"Updated timestamps: {self.timestamps[:]}")

    def interpolate(
        self, timestamp: float
    ) -> tuple[npt.NDArray[np.float64], st.Rotation]:
        if len(self.timestamps) == 1:
            return self.pos[0], self.rot

        assert (
            len(self.timestamps) == len(self.pos) == len(self.rot)
        ), f"len(self.timestamps): {len(self.timestamps)}, len(self.pos): {len(self.pos)}, len(self.rot): {len(self.rot)}"

        idx = np.searchsorted(self.timestamps, timestamp, side="right")
        if idx == len(self.timestamps):
            return self.pos[-1], self.rot[-1]
        elif idx == 0:
            return self.pos[0], self.rot[0]
        else:
            ratio = (timestamp - self.timestamps[idx - 1]) / (
                self.timestamps[idx] - self.timestamps[idx - 1]
            )
            new_pos = self.pos[idx - 1] * (1 - ratio) + self.pos[idx] * ratio
            new_rot = st.Slerp(
                [0, 1], st.Rotation.concatenate([self.rot[idx - 1], self.rot[idx]])
            )(ratio)
            return new_pos, new_rot

    def interpolate_xyz_wxyz(self, timestamp: float) -> npt.NDArray[np.float64]:
        """
        Return the interpolated position and quaternion in the format of np.array([x, y, z, qw, qx, qy, qz])
        """
        pos, rot = self.interpolate(timestamp)
        quat_xyzw = rot.as_quat().squeeze()
        return np.concatenate([pos, to_wxyz(quat_xyzw)], axis=0)

    def find_delta_latency(
        self,
        input_poses: npt.NDArray[np.float64],
        input_times: npt.NDArray[np.float64],
        current_timestamp: float,
        latency_start: float = -0.3,
        latency_end: float = 0.3,
        matching_dt: float = 0.05,
        latency_precision: float = 0.01,
        pos_weight: float = 1.0,
        rot_weight: float = 0.1,
    ) -> float:
        """
        Find the nearest timestamp in the trajectory
        """
        pose_samples = np.zeros((3, 6))
        pose_samples[0] = self._convert_7d_to_6d(
            self.interpolate_xyz_wxyz(current_timestamp - matching_dt)
        )
        pose_samples[1] = self._convert_7d_to_6d(
            self.interpolate_xyz_wxyz(current_timestamp)
        )
        pose_samples[2] = self._convert_7d_to_6d(
            self.interpolate_xyz_wxyz(current_timestamp + matching_dt)
        )

        input_trajectory = Trajectory(data=input_poses, timestamps=input_times)
        input_interpolator = PoseTrajInterpolator(
            input_trajectory,
            0.0,
            float("inf"),
            float("inf"),
            trim_new_trajectory_time_s=0.0,
        )

        errors = []
        clipped_latency_start = max(
            latency_start, input_times[0] - current_timestamp - matching_dt
        )
        clipped_latency_end = min(
            latency_end, input_times[-1] - current_timestamp + matching_dt
        )
        # print(f"{latency_start=}, {latency_end=}, {latency_precision=}")
        latency_range = np.arange(
            clipped_latency_start, clipped_latency_end, latency_precision
        )

        pose_rot = st.Rotation.from_rotvec(pose_samples[:, 3:6])
        pose_rot_inv = pose_rot.inv()

        for latency in latency_range:
            latency = float(latency)
            input_pos = np.zeros((3, 3))
            input_rot = []

            pos0, rot0 = input_interpolator.interpolate(
                current_timestamp - matching_dt + latency
            )
            pos1, rot1 = input_interpolator.interpolate(current_timestamp + latency)
            pos2, rot2 = input_interpolator.interpolate(
                current_timestamp + matching_dt + latency
            )

            input_pos[0] = pos0
            input_pos[1] = pos1
            input_pos[2] = pos2
            input_rot = [rot0, rot1, rot2]

            pos_error = np.average(
                np.linalg.norm(pose_samples[:, :3] - input_pos, axis=1)
            )

            input_rot_combined = st.Rotation.concatenate(input_rot)
            rot_error = np.average(
                st.Rotation.magnitude(pose_rot_inv * input_rot_combined)
            )

            error = pos_error * pos_weight + rot_error * rot_weight
            errors.append(error)
        errors = np.array(errors)
        min_error_idx = np.argmin(errors)
        min_latency = float(latency_range[min_error_idx])

        # If min error is at the start, the new trajectory may be too far in the future
        # Flip the search: find where the first new trajectory point matches in the old trajectory
        if min_error_idx == 0:
            first_input_pos, first_input_rot = input_interpolator.interpolate(
                input_times[0]
            )
            first_input_6d = self._convert_7d_to_6d(
                np.concatenate(
                    [first_input_pos, to_wxyz(first_input_rot.as_quat().squeeze())]
                )
            )

            clipped_latency_start = max(
                latency_start, current_timestamp - self.timestamps[-1]
            )  # Might be a negative value
            clipped_latency_end = min(
                latency_end, min_latency
            )  # Should be lower than the original minimum latency
            if clipped_latency_end <= clipped_latency_start:
                print(
                    f"Clipped latency end is less than or equal to clipped latency start. {clipped_latency_end=}, {clipped_latency_start=}, {min_latency=}"
                )
                print(
                    f"Current timestamp: {current_timestamp} input times: {input_times[:] - current_timestamp}, existing timestamps: {self.timestamps[:] - current_timestamp}"
                )
                return min_latency

            latency_range = np.arange(
                clipped_latency_start, clipped_latency_end, latency_precision
            )

            reverse_errors = []
            for latency in latency_range:
                latency = float(latency)
                old_pos, old_rot = self.interpolate(input_times[0] - latency)
                old_6d = self._convert_7d_to_6d(
                    np.concatenate([old_pos, to_wxyz(old_rot.as_quat().squeeze())])
                )

                pos_error = np.linalg.norm(first_input_6d[:3] - old_6d[:3])
                rot_error = st.Rotation.magnitude(
                    st.Rotation.from_rotvec(first_input_6d[3:6]).inv()
                    * st.Rotation.from_rotvec(old_6d[3:6])
                )
                reverse_errors.append(pos_error * pos_weight + rot_error * rot_weight)

            min_error_idx = np.argmin(reverse_errors)
            min_latency = float(latency_range[min_error_idx])

        return min_latency

    def _convert_7d_to_6d(
        self, pose_7d: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        pos = pose_7d[:3]
        quat_wxyz = pose_7d[3:7]
        quat_xyzw = to_xyzw(quat_wxyz)
        rot = st.Rotation.from_quat(quat_xyzw)
        rot_6d = rot.as_rotvec()

        return np.concatenate([pos, rot_6d])
