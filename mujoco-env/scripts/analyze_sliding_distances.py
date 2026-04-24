import copy
import os
from typing import Any, cast

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import zarr


def parse_results(root: zarr.Group) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    episode_num = len(root)
    for episode_idx in range(episode_num):
        episode_data = root[f"episode_{episode_idx}"]
        assert isinstance(episode_data, zarr.Group)
        # for key, value in episode_data.items():
        #     print(key, value.shape)
        # for key, value in episode_data.attrs.items():
        #     print(key, value)
        result = {}
        result["episode_idx"] = episode_idx
        result["sliding_friction"] = episode_data.attrs["episode_config"][
            "sliding_friction"
        ]
        result["push_vel"] = episode_data.attrs["episode_config"]["push_vels_m_per_s"][
            0
        ]
        result["object_final_x"] = episode_data["obj0_object_pose_xyz_wxyz"][-1][0]
        result["object_final_y"] = episode_data["obj0_object_pose_xyz_wxyz"][-1][1]
        result["reward"] = episode_data.attrs["reward"]

        results.append(copy.deepcopy(result))
    print(f"{len(results)=}")
    return results


def organize_results(
    results: list[dict[str, Any]], repeat_num: int
) -> dict[str, npt.NDArray[np.float64]]:
    # Sort results by seed
    raw_sliding_frictions = [result["sliding_friction"] for result in results]
    # Round to 0.00001
    sliding_frictions = np.sort(np.unique(np.round(raw_sliding_frictions, 5)))
    raw_push_vels = [result["push_vel"] for result in results]
    push_vels = np.sort(np.unique(np.round(raw_push_vels, 5)))

    push_vel_num = len(results) // (len(sliding_frictions) * repeat_num)
    print(len(sliding_frictions), len(push_vels), repeat_num, push_vel_num)

    data_dict = {}
    data_dict["sliding_frictions"] = sliding_frictions
    data_dict["push_vels"] = push_vels
    data_dict["final_x"] = np.nan * np.ones(
        (len(sliding_frictions), len(push_vels), repeat_num)
    )
    data_dict["final_y"] = np.nan * np.ones(
        (len(sliding_frictions), len(push_vels), repeat_num)
    )
    data_dict["reward"] = np.zeros(
        (len(sliding_frictions), len(push_vels), repeat_num), dtype=bool
    )

    for result in results:
        episode_idx = result["episode_idx"]
        sliding_friction = np.round(result["sliding_friction"], 5)
        sliding_friction_idx = np.where(sliding_frictions == sliding_friction)[0]
        assert np.allclose(
            sliding_friction, sliding_frictions[sliding_friction_idx]
        ), f"{result['sliding_friction']}, {sliding_frictions[sliding_friction_idx]}"
        push_vel = np.round(result["push_vel"], 5)
        push_vel_idx = np.where(push_vels == push_vel)[0]
        assert np.allclose(
            push_vel, push_vels[push_vel_idx]
        ), f"{result['push_vel']}, {push_vels[push_vel_idx]}"
        repeat_idx = episode_idx % repeat_num
        data_dict["final_x"][sliding_friction_idx, push_vel_idx, repeat_idx] = result[
            "object_final_x"
        ]
        data_dict["final_y"][sliding_friction_idx, push_vel_idx, repeat_idx] = result[
            "object_final_y"
        ]
        data_dict["reward"][sliding_friction_idx, push_vel_idx, repeat_idx] = (
            result["reward"]
        )

    # Check consistency
    if repeat_num > 1:
        for i, sliding_friction in enumerate(sliding_frictions):
            for j, push_vel in enumerate(push_vels):
                final_x: npt.NDArray[np.float64] = data_dict["final_x"][i, j]
                final_y: npt.NDArray[np.float64] = data_dict["final_y"][i, j]
                reward: npt.NDArray[np.bool_] = data_dict["reward"][i, j]
                assert np.allclose(final_x, final_x[:1].repeat(repeat_num)) or np.all(
                    np.isnan(final_x)
                ), f"{i}, {j}, {final_x}"
                assert np.allclose(final_y, final_y[:1].repeat(repeat_num)) or np.all(
                    np.isnan(final_y)
                ), f"{i}, {j}, {final_y}"
                assert np.allclose(
                    reward, reward[:1].repeat(repeat_num)
                ), f"{i}, {j}, {reward}"

    # Merge the repeat_num dimension
    data_dict["final_x"] = data_dict["final_x"][:, :, 0]
    data_dict["final_y"] = data_dict["final_y"][:, :, 0]
    data_dict["reward"] = data_dict["reward"][:, :, 0]
    return data_dict


def plot_results(data_dict: dict[str, npt.NDArray[np.float64]], save_dir: str):
    x_base = 0.1  # The center line of the table
    y_base = 0.15  # The center of the cube
    sliding_frictions = data_dict["sliding_frictions"]
    push_vels = data_dict["push_vels"]
    delta_x = data_dict["final_x"] - x_base
    delta_x_max = np.nanmax(np.abs(delta_x))
    delta_y = data_dict["final_y"] - y_base
    delta_y_max = np.nanmax(np.abs(delta_y))

    X, Y = np.meshgrid(sliding_frictions, push_vels, indexing="ij")

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        X, Y, delta_x, vmin=-delta_x_max, vmax=delta_x_max, cmap="RdBu_r"
    )
    cbar = ax.figure.colorbar(mesh)
    cbar.set_label("Sliding Distance")
    ax.set_xlabel("Sliding Friction")
    ax.set_ylabel("Push Velocity")
    ax.set_title("Sliding Distance (x-axis)")
    fig.savefig(os.path.join(save_dir, "sliding_distances_x.pdf"))

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        X, Y, delta_y, vmin=-delta_y_max, vmax=delta_y_max, cmap="RdBu_r"
    )
    cbar = ax.figure.colorbar(mesh)
    cbar.set_label("Sliding Distance")
    ax.set_xlabel("Sliding Friction")
    ax.set_ylabel("Push Velocity")
    ax.set_title("Sliding Distance (y-axis)")
    fig.savefig(os.path.join(save_dir, "sliding_distances_y.pdf"))

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        X, Y, data_dict["reward"], vmin=-1, vmax=1, cmap="RdBu_r"
    )
    cbar = ax.figure.colorbar(mesh)
    cbar.set_label("Success")
    ax.set_xlabel("Sliding Friction")
    ax.set_ylabel("Push Velocity")
    ax.set_title("Success")
    fig.savefig(os.path.join(save_dir, "success.pdf"))


def plot_thresholds(
    data_dict: dict[str, npt.NDArray[np.float64]],
    save_dir: str,
    friction_min: float | None = None,
    friction_max: float | None = None,
    step_num: int = 4,
):
    x_base = 0.1  # The center line of the table
    y_base = 0.15  # The center of the cube
    sliding_frictions = data_dict["sliding_frictions"]

    push_vels = data_dict["push_vels"]
    delta_y = data_dict["final_y"] - y_base
    delta_y_max = np.nanmax(np.abs(delta_y))

    # For each sliding friction, find the push velocity that minimizes the y-axis sliding distance
    optimal_push_vels = np.zeros_like(sliding_frictions)
    max_push_vels = np.zeros_like(sliding_frictions)
    min_push_vels = np.zeros_like(sliding_frictions)
    for i, sliding_friction in enumerate(sliding_frictions):
        optimal_idx = np.nanargmin(np.abs(delta_y[i]))
        optimal_push_vels[i] = push_vels[optimal_idx]
        # Find the minimum and maximum push velocities that are still successful
        for j in range(optimal_idx, len(push_vels)):
            if not data_dict["reward"][i, j]:
                max_push_vels[i] = push_vels[j - 1]
                break
        else:
            max_push_vels[i] = optimal_push_vels[i]
        for j in range(optimal_idx, -1, -1):
            if not data_dict["reward"][i, j]:
                min_push_vels[i] = push_vels[j + 1]
                break
        else:
            min_push_vels[i] = optimal_push_vels[i]

    if friction_min is None:
        min_idx = 0
    else:
        min_idx = np.where(sliding_frictions >= friction_min)[0][0]
    if friction_max is None:
        max_idx = len(sliding_frictions) - 1
    else:
        max_idx = np.where(sliding_frictions <= friction_max)[0][-1]

    vel_min = optimal_push_vels[min_idx]
    vel_max = optimal_push_vels[max_idx]
    print(
        f"{vel_min=}, {vel_max=}, {sliding_frictions[min_idx]=}, {sliding_frictions[max_idx]=}"
    )
    vel_step = (vel_max - vel_min) / (2**step_num - 2)
    sampled_vels = np.arange(vel_min, vel_max + 1e-6, vel_step)
    assert (
        len(sampled_vels) == 2**step_num - 1
    ), f"{len(sampled_vels)=}, {2**step_num - 1=}"
    # There should be 2**step_num - 1 velocities in total

    # For each sampled velocity, find the minimum and maximum sliding friction that is still successful
    sampled_min_frictions = np.zeros_like(sampled_vels)
    sampled_max_frictions = np.zeros_like(sampled_vels)
    sampled_optimal_frictions = np.zeros_like(sampled_vels)
    for i, sampled_vel in enumerate(sampled_vels):
        sampled_min_frictions[i] = np.min(
            sliding_frictions[max_push_vels >= sampled_vel]
        )
        sampled_max_frictions[i] = np.max(
            sliding_frictions[min_push_vels <= sampled_vel]
        )
        optimal_idx = np.nanargmin(np.abs(optimal_push_vels - sampled_vel))
        sampled_optimal_frictions[i] = sliding_frictions[optimal_idx]
    np.set_printoptions(precision=5, suppress=True, sign=" ")
    print(f"{sampled_optimal_frictions=}")
    print(f"{sampled_vels=}")

    fig, ax = plt.subplots()
    sqrt_sliding_frictions = np.sqrt(sliding_frictions)
    ax.plot(
        sliding_frictions,
        max_push_vels,
        "--",
        color="#1f77b4",
        label="Max",
        linewidth=1,
    )
    ax.plot(
        sliding_frictions,
        optimal_push_vels,
        "-",
        color="#1f77b4",
        label="Optimal",
        linewidth=1.5,
    )
    ax.plot(
        sliding_frictions,
        min_push_vels,
        "--",
        color="#1f77b4",
        label="Min",
        linewidth=1,
    )

    for i, sampled_vel in enumerate(sampled_vels):
        ax.plot(
            [sampled_min_frictions[i], sampled_max_frictions[i]],
            [sampled_vel, sampled_vel],
            "-",
            color="r",
            linewidth=0.5,
        )
        ax.plot(
            [sampled_optimal_frictions[i]],
            [sampled_vel],
            "o",
            color="r",
            markersize=1,
        )
    ax.set_xlabel("Sliding Friction")
    ax.set_ylabel("Push Velocity")
    ax.set_title("Push Velocity Thresholds for Each Sliding Friction")
    ax.legend()
    fig.savefig(os.path.join(save_dir, "push_vel_thresholds.pdf"))


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.argument("repeat_num", type=int)
def main(data_dir: str, repeat_num: int):

    processed_data_path = os.path.join(data_dir, "sliding_distances.npz")
    zarr_dir = os.path.join(data_dir, "episode_data.zarr")
    if not os.path.exists(processed_data_path):
        root = cast(zarr.Group, zarr.open(zarr_dir, mode="r"))
        results = parse_results(root)
        data_dict = organize_results(results, repeat_num)
        np.savez(processed_data_path, **data_dict)
    else:
        data_dict = np.load(processed_data_path)
    # plot_results(data_dict, data_dir)
    plot_thresholds(
        data_dict, data_dir, friction_min=0.005, friction_max=0.015, step_num=4
    )


if __name__ == "__main__":
    main()
