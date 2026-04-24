import math
from ctypes import cast
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

# from numba import njit
from scipy.spatial import ConvexHull, Delaunay
from transforms3d import quaternions

from robot_utils.pose_utils import positive_w, get_relative_pose

# @njit(cache=True)
def qinterp(
    q1: npt.NDArray[np.float64], q2: npt.NDArray[np.float64], t: float
) -> npt.NDArray[np.float64]:
    """Spherical linear interpolation between two quaternions.

    Args:
        q1: First quaternion (wxyz format)
        q2: Second quaternion (wxyz format)
        t: Interpolation parameter (0 to 1); 0: q1, 1: q2

    Returns:
        Interpolated quaternion (wxyz format)
    """
    # Compute the cosine of the angle between the quaternions
    dot = np.dot(q1, q2)

    # If the dot product is negative, we need to negate one of the quaternions
    # to ensure we take the shortest path
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # If the quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # Calculate the angle between quaternions
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    # Compute interpolation coefficients
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    q = (s0 * q1) + (s1 * q2)

    if q[0] < 0:
        q = -q

    return q


# @njit(cache=True)


def get_random_convex_combination(
    poses_xyz_wxyz: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    # """
    # Will use the simplex method to sample a random convex combination of the poses. The sample distribution is uniform on the convex hull of all the poses.
    # poses_xyz_wxyz: (n, 7) array of poses
    # """

    # dim = 7
    # assert len(poses_xyz_wxyz.shape) == 2
    # assert poses_xyz_wxyz.shape[1] == dim

    # hull = ConvexHull(poses_xyz_wxyz)
    # simplices = hull.simplices  # (n_simplex, dim+1)

    # def simplex_volume(simplex: npt.NDArray[np.float64]) -> float:
    #     # simplex: (dim+1, dim)
    #     homogeneous_coords = np.hstack([simplex, np.ones((dim + 1, 1))])
    #     return np.abs(np.linalg.det(homogeneous_coords)) / math.factorial(dim)

    # volumes = np.array(
    #     [simplex_volume(poses_xyz_wxyz[simplex]) for simplex in simplices]
    # )  # (n_simplex,)
    # probs = volumes / volumes.sum()
    # simplex_idx = rng.choice(len(probs), size=1, p=probs)[0]

    # simplex = poses_xyz_wxyz[simplices[simplex_idx]]  # (dim+1, dim)

    # bary_coord = rng.exponential(1.0, size=dim + 1)
    # bary_coord = bary_coord / bary_coord.sum()
    # bary_coord = bary_coord[:-1]

    barycentric_coeff = rng.random(poses_xyz_wxyz.shape[0])
    barycentric_coeff = barycentric_coeff / barycentric_coeff.sum()
    pose_xyz_wxyz = (barycentric_coeff[:, np.newaxis] * poses_xyz_wxyz).sum(axis=0)

    if len(pose_xyz_wxyz) == 7:
        pose_xyz_wxyz[3:] = pose_xyz_wxyz[3:] / np.linalg.norm(pose_xyz_wxyz[3:])

    return pose_xyz_wxyz


def get_random_4poses_convex_combination(
    poses_xyz_wxyz: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    if poses_xyz_wxyz.shape[0] < 4:
        raise ValueError("There must be at least 4 poses")
    pose_indices = rng.choice(poses_xyz_wxyz.shape[0], size=4, replace=False)
    return get_random_convex_combination(poses_xyz_wxyz[pose_indices], rng)


class ActionInterpolator:
    def __init__(
        self,
        start_pose_xyz_wxyz: npt.NDArray[np.float64],
        end_pose_xyz_wxyz: npt.NDArray[np.float64],
        start_gripper_width: npt.NDArray[np.float64],
        end_gripper_width: npt.NDArray[np.float64],
    ):
        self.start_pose_xyz_wxyz: npt.NDArray[np.float64] = positive_w(
            start_pose_xyz_wxyz
        )
        self.end_pose_xyz_wxyz: npt.NDArray[np.float64] = positive_w(end_pose_xyz_wxyz)
        self.start_gripper_width: npt.NDArray[np.float64] = start_gripper_width
        self.end_gripper_width: npt.NDArray[np.float64] = end_gripper_width

    def interpolate(
        self, dt: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raise NotImplementedError

    @property
    def is_finished(self) -> bool:
        raise NotImplementedError


class LinearActionInterpolator(ActionInterpolator):
    def __init__(
        self,
        pos_speed_m_per_s: float,
        rot_speed_rad_per_s: float,
        gripper_speed_m_per_s: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pos_speed_m_per_s: float = pos_speed_m_per_s
        self.rot_speed_rad_per_s: float = rot_speed_rad_per_s
        self.gripper_speed_m_per_s: float = gripper_speed_m_per_s
        rel_pose = get_relative_pose(self.end_pose_xyz_wxyz, self.start_pose_xyz_wxyz)
        pos_max_time = float(
            np.linalg.norm(self.start_pose_xyz_wxyz[:3] - self.end_pose_xyz_wxyz[:3])
            / self.pos_speed_m_per_s
        )
        rot_angle = quaternions.quat2axangle(rel_pose[3:])[1]
        if rot_angle > np.pi:
            rot_angle = 2 * np.pi - rot_angle
        rot_max_time = rot_angle / self.rot_speed_rad_per_s
        gripper_max_time = float(
            np.abs(self.end_gripper_width - self.start_gripper_width)
            / self.gripper_speed_m_per_s
        )
        self.duration_s: float = max(pos_max_time, rot_max_time, gripper_max_time)
        self.t: float = 0.0

    def interpolate(
        self, dt: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        self.t += dt
        if self.t > self.duration_s:
            self.t = self.duration_s
            return self.end_pose_xyz_wxyz, self.end_gripper_width

        pose_xyz_wxyz = np.zeros(7, dtype=np.float64)
        pose_xyz_wxyz[:3] = (
            self.start_pose_xyz_wxyz[:3]
            + (self.end_pose_xyz_wxyz[:3] - self.start_pose_xyz_wxyz[:3])
            * self.t
            / self.duration_s
        )
        pose_xyz_wxyz[3:] = qinterp(
            self.start_pose_xyz_wxyz[3:],
            self.end_pose_xyz_wxyz[3:],
            self.t / self.duration_s,
        )
        gripper_width = (
            self.start_gripper_width
            + (self.end_gripper_width - self.start_gripper_width)
            * self.t
            / self.duration_s
        ).astype(np.float64)
        return pose_xyz_wxyz, gripper_width

    @property
    def is_finished(self) -> bool:
        return self.t >= self.duration_s


class QuadraticActionInterpolator(ActionInterpolator):
    def __init__(
        self,
        start_speed_m_per_s: float,
        final_speed_m_per_s: float,
        **kwargs,
    ):
        """
        Include a constant acceleration that ends with the target speed and target pose at the same time.
        Rotation and gripper movements are handled proportionally to the position movements.
        """
        super().__init__(**kwargs)
        self.final_speed_m_per_s: float = final_speed_m_per_s
        self.start_speed_m_per_s: float = start_speed_m_per_s
        self.t: float = 0.0
        self.distance: float = float(
            np.linalg.norm(self.end_pose_xyz_wxyz[:3] - self.start_pose_xyz_wxyz[:3])
        )
        avg_speed = (self.final_speed_m_per_s + self.start_speed_m_per_s) / 2.0
        self.duration_s: float = self.distance / avg_speed
        self.acc_m_per_s2: float = (
            self.final_speed_m_per_s - self.start_speed_m_per_s
        ) / self.duration_s

        self.current_distance: float = 0.0
        self.current_speed_m_per_s: float = self.start_speed_m_per_s

    def interpolate(
        self, dt: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        self.t += dt
        if self.t > self.duration_s:
            self.t = self.duration_s
            return self.end_pose_xyz_wxyz, self.end_gripper_width
        next_speed_m_per_s = self.current_speed_m_per_s + self.acc_m_per_s2 * dt
        self.current_distance += (
            (self.current_speed_m_per_s + next_speed_m_per_s) * dt / 2.0
        )
        self.current_speed_m_per_s = next_speed_m_per_s
        ratio = self.current_distance / self.distance

        pose_xyz_wxyz = np.zeros(7, dtype=np.float64)
        pose_xyz_wxyz[:3] = (
            self.start_pose_xyz_wxyz[:3]
            + (self.end_pose_xyz_wxyz[:3] - self.start_pose_xyz_wxyz[:3]) * ratio
        )
        pose_xyz_wxyz[3:] = qinterp(
            self.start_pose_xyz_wxyz[3:], self.end_pose_xyz_wxyz[3:], ratio
        )
        gripper_width = (
            self.start_gripper_width
            + (self.end_gripper_width - self.start_gripper_width) * ratio
        )
        return pose_xyz_wxyz, gripper_width

    @property
    def is_finished(self) -> bool:
        return self.t >= self.duration_s


class FinalSpeedActionInterpolator(ActionInterpolator):
    def __init__(
        self,
        final_speed_m_per_s: float,
        dt: float,
        final_speed_step_num: int,
        **kwargs,
    ):
        """
        Will make sure the final speed is reached in the last few steps. Assumes the initial speed is 0.
        """
        super().__init__(**kwargs)
        self.final_speed_m_per_s = final_speed_m_per_s
        self.dt = dt
        self.final_speed_step_num = final_speed_step_num
        self.step_cnt: int = 0
        self.current_pos_speed_m_per_s: float = 0.0

        self.distance: float = float(
            np.linalg.norm(self.end_pose_xyz_wxyz[:3] - self.start_pose_xyz_wxyz[:3])
        )
        # Calculate the required timestamp
        assert (
            self.dt * final_speed_step_num * final_speed_m_per_s < self.distance
        ), "the final constant speed steps are too many"
        accelerate_distance = (
            self.distance - self.dt * final_speed_step_num * final_speed_m_per_s
        )
        accelerate_duration = accelerate_distance / final_speed_m_per_s * 2
        accelerate = final_speed_m_per_s / accelerate_duration
        accelerate_step_num = int(np.ceil(accelerate_duration / self.dt))
        self.duration_s: float = (accelerate_step_num + final_speed_step_num) * self.dt
        self.step_distances = []
        speed = 0
        current_distance = 0
        for i in range(accelerate_step_num):
            if i == 0:
                delta_t = accelerate_duration - (accelerate_step_num - 1) * self.dt
                speed = accelerate * delta_t
                delta_distance = speed * delta_t / 2
            else:
                prev_speed = speed
                speed += accelerate * self.dt

                delta_distance = (prev_speed + speed) * self.dt / 2
            current_distance += delta_distance
            self.step_distances.append(current_distance)
        assert np.allclose(
            speed, final_speed_m_per_s
        ), f"speed {speed} does not reach final speed {final_speed_m_per_s}"
        for i in range(final_speed_step_num):
            current_distance += final_speed_m_per_s * self.dt
            self.step_distances.append(current_distance)
        assert np.allclose(
            current_distance, self.distance
        ), f"distance {current_distance} does not match {self.distance}"

    def interpolate(
        self, dt: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        assert (
            dt == self.dt
        ), "FinalSpeedActionInterpolator only accepts fixed step size"
        self.step_cnt += 1
        if self.step_cnt >= len(self.step_distances):
            return self.end_pose_xyz_wxyz, self.end_gripper_width
        distance = self.step_distances[self.step_cnt - 1]
        ratio = distance / self.distance
        pose_xyz_wxyz = np.zeros(7, dtype=np.float64)
        pose_xyz_wxyz[:3] = (
            self.start_pose_xyz_wxyz[:3]
            + (self.end_pose_xyz_wxyz[:3] - self.start_pose_xyz_wxyz[:3]) * ratio
        )
        pose_xyz_wxyz[3:] = qinterp(
            self.start_pose_xyz_wxyz[3:], self.end_pose_xyz_wxyz[3:], ratio
        )
        gripper_width = (
            self.start_gripper_width
            + (self.end_gripper_width - self.start_gripper_width) * ratio
        )
        return pose_xyz_wxyz, gripper_width

    @property
    def is_finished(self) -> bool:
        return self.step_cnt >= len(self.step_distances)


class MultiStepLinearInterpolator(ActionInterpolator):
    def __init__(
        self,
        pose_waypoints: npt.NDArray[np.float64],
        gripper_width_waypoints: npt.NDArray[np.float64],
        pos_speeds_m_per_s: npt.NDArray[np.float64],
        rot_speeds_rad_per_s: npt.NDArray[np.float64],
        gripper_speeds_m_per_s: npt.NDArray[np.float64],
    ):
        self.pose_waypoints: npt.NDArray[np.float64] = pose_waypoints
        self.gripper_width_waypoints: npt.NDArray[np.float64] = gripper_width_waypoints
        self.pos_speeds_m_per_s: npt.NDArray[np.float64] = pos_speeds_m_per_s
        self.rot_speeds_rad_per_s: npt.NDArray[np.float64] = rot_speeds_rad_per_s
        self.gripper_speeds_m_per_s: npt.NDArray[np.float64] = gripper_speeds_m_per_s
        self.t: float = 0.0

        assert len(self.pose_waypoints) == len(
            self.gripper_width_waypoints
        ), f"{len(self.pose_waypoints)=}, {len(self.gripper_width_waypoints)=}"
        assert (
            len(self.pos_speeds_m_per_s)
            == len(self.rot_speeds_rad_per_s)
            == len(self.gripper_speeds_m_per_s)
        ), f"{len(self.pos_speeds_m_per_s)=}, {len(self.rot_speeds_rad_per_s)=}, {len(self.gripper_speeds_m_per_s)=}"
        assert (
            len(self.pose_waypoints) == len(self.pos_speeds_m_per_s) + 1
        ), f"{len(self.pose_waypoints)=}, {len(self.pos_speeds_m_per_s)=}"

        self.achieved_waypoint_idx: int = 0
        self.single_step_interpolator = LinearActionInterpolator(
            start_pose_xyz_wxyz=self.pose_waypoints[0],
            end_pose_xyz_wxyz=self.pose_waypoints[1],
            start_gripper_width=self.gripper_width_waypoints[0],
            end_gripper_width=self.gripper_width_waypoints[1],
            pos_speed_m_per_s=self.pos_speeds_m_per_s[0],
            rot_speed_rad_per_s=self.rot_speeds_rad_per_s[0],
            gripper_speed_m_per_s=self.gripper_speeds_m_per_s[0],
        )

    def interpolate(
        self, dt: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if self.single_step_interpolator.is_finished:
            self.achieved_waypoint_idx += 1
            if self.achieved_waypoint_idx < len(self.pose_waypoints) - 1:
                self.single_step_interpolator = LinearActionInterpolator(
                    start_pose_xyz_wxyz=self.pose_waypoints[self.achieved_waypoint_idx],
                    end_pose_xyz_wxyz=self.pose_waypoints[
                        self.achieved_waypoint_idx + 1
                    ],
                    start_gripper_width=self.gripper_width_waypoints[
                        self.achieved_waypoint_idx
                    ],
                    end_gripper_width=self.gripper_width_waypoints[
                        self.achieved_waypoint_idx + 1
                    ],
                    pos_speed_m_per_s=self.pos_speeds_m_per_s[
                        self.achieved_waypoint_idx
                    ],
                    rot_speed_rad_per_s=self.rot_speeds_rad_per_s[
                        self.achieved_waypoint_idx
                    ],
                    gripper_speed_m_per_s=self.gripper_speeds_m_per_s[
                        self.achieved_waypoint_idx
                    ],
                )
            else:
                return self.pose_waypoints[-1], self.gripper_width_waypoints[-1]

        return self.single_step_interpolator.interpolate(dt)

    @property
    def is_finished(self) -> bool:
        return self.achieved_waypoint_idx >= len(self.pose_waypoints) - 1


class CubicInterpolator(ActionInterpolator):
    def __init__(
        self,
        pose_waypoints: npt.NDArray[np.float64],
        gripper_width_waypoints: npt.NDArray[np.float64],
        timestamps_s: npt.NDArray[np.float64],
    ):
        self.waypoints = np.concatenate(
            [pose_waypoints, gripper_width_waypoints], axis=1
        )
        self.timestamps = timestamps_s
        assert len(self.waypoints) == len(
            self.timestamps
        ), f"{len(self.waypoints)=}, {len(self.timestamps)=}"
        self.t: float = 0.0

        from scipy.interpolate import CubicSpline

        self.spline = CubicSpline(self.timestamps, self.waypoints, bc_type="clamped")
        # self.splines = []
        # for i in range(self.waypoints.shape[1]):
        #     self.splines.append(
        #         CubicSpline(self.timestamps, self.waypoints[:, i], bc_type="clamped")
        #     )

    def visualize_spline(self):

        point_num = 100
        poses_xz = np.zeros((point_num, 2))
        for i, t in enumerate(np.linspace(0, self.timestamps[-1], point_num)):
            pose = self.spline(t)
            poses_xz[i, :] = pose[[0, 2]]

        fig, ax = plt.subplots()
        ax.plot(poses_xz[:, 0], poses_xz[:, 1], ".")
        fig.show()

    def interpolate(
        self, dt: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        self.t += dt
        x = self.spline(self.t).astype(np.float64)
        pose_xyz_wxyz = x[:7]
        gripper_width = x[7:]
        return pose_xyz_wxyz, gripper_width

    @property
    def is_finished(self) -> bool:
        return self.t >= self.timestamps[-1]
