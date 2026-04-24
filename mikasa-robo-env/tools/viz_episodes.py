"""Visualize episodes as side-by-side MP4s (third_person | wrist camera).

Supports two input formats:
  --format zarr: zarr store with episode_N/{third_person_camera, robot0_wrist_camera}
  --format npz:  eval output directory with episode_NNNN_*/episode.npz (6-channel rgb key)

Usage:
  python tools/viz_episodes.py --path path/to/episode_data.zarr --format zarr
  python tools/viz_episodes.py --path eval_results/RememberColor3-v0/ --format npz --fps 30
  python tools/viz_episodes.py --path path/to/episode_data.zarr --episodes 0,1,5
"""

import argparse
import glob
import os
import subprocess

import numpy as np
from tqdm import tqdm


def _write_video(frames: np.ndarray, output_path: str, fps: int):
    """Pipe raw RGB frames to ffmpeg for H.264 encoding."""
    T, H, W, _ = frames.shape
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        "-movflags", "+faststart", "-an", output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.stdin.write(frames.tobytes())
    proc.stdin.close()
    proc.wait()


def _viz_zarr(zarr_path: str, episode_indices: set[int] | None, fps: int):
    """Render zarr episodes as side-by-side videos."""
    import zarr

    z = zarr.open_group(zarr_path, mode="r")
    ep_keys = sorted(z.keys(), key=lambda k: int(k.split("_")[1]))

    if episode_indices is not None:
        ep_keys = [k for k in ep_keys if int(k.split("_")[1]) in episode_indices]

    for key in tqdm(ep_keys, desc="Rendering zarr episodes"):
        ep = z[key]
        third = ep["third_person_camera"][:]
        wrist = ep["robot0_wrist_camera"][:]
        frames = np.concatenate([third, wrist], axis=2)

        out_path = os.path.join(zarr_path, key, "viz.mp4")
        _write_video(frames, out_path, fps)

    print(f"done: {len(ep_keys)} videos saved under {zarr_path}/<episode>/viz.mp4")


def _viz_npz(npz_dir: str, episode_indices: set[int] | None, fps: int):
    """Render eval NPZ episodes as side-by-side videos."""
    # Find episode directories containing episode.npz
    ep_dirs = sorted(glob.glob(os.path.join(npz_dir, "episode_*")))
    ep_dirs = [d for d in ep_dirs if os.path.isfile(os.path.join(d, "episode.npz"))]

    if episode_indices is not None:
        ep_dirs = [d for d in ep_dirs
                   if int(os.path.basename(d).split("_")[1]) in episode_indices]

    for ep_dir in tqdm(ep_dirs, desc="Rendering npz episodes"):
        d = np.load(os.path.join(ep_dir, "episode.npz"))
        rgb = d["rgb"]  # (T, H, W, 6)
        frames = np.concatenate([rgb[..., :3], rgb[..., 3:]], axis=2)

        out_path = os.path.join(ep_dir, "viz.mp4")
        _write_video(frames, out_path, fps)

    print(f"done: {len(ep_dirs)} videos saved")


def main():
    parser = argparse.ArgumentParser(description="Render episodes to side-by-side MP4s")
    parser.add_argument("--path", type=str, required=True,
                        help="zarr store path or eval results directory")
    parser.add_argument("--format", type=str, choices=["zarr", "npz"], default="zarr",
                        help="input format: zarr episode store or eval npz directory")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--episodes", type=str, default=None,
                        help="comma-separated episode indices (e.g. '0,1,5'). default: all")
    args = parser.parse_args()

    indices = None
    if args.episodes is not None:
        indices = {int(x) for x in args.episodes.split(",")}

    if args.format == "zarr":
        _viz_zarr(args.path, indices, args.fps)
    else:
        _viz_npz(args.path, indices, args.fps)


if __name__ == "__main__":
    main()
