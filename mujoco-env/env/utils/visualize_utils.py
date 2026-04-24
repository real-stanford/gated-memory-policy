import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import cv2
import natsort
import numpy as np
import zarr
from flask import Flask, render_template, send_from_directory
import imageio

from loguru import logger

def convert_single_episode_to_video(
    episode_store: zarr.Group,
    video_dir: str,
    episode_name: str,
    video_type: str,
    split_videos: bool,
    show_timestep: bool = True,
    freq_hz: int = 10,
    add_black_screen_in_the_end: bool = True,
):
    assert isinstance(episode_store, zarr.Group), "episode_store must be a zarr.Group"

    if split_videos:
        for key, value in episode_store.items():
            if len(value.shape) == 4 and value.shape[-1] == 3:
                logger.info(f"{key}: {value.shape}")
                # is a image, will convert to video
                video_path = os.path.join(video_dir, f"{episode_name}_{key}_timestep_{show_timestep}.mp4")
                logger.info(f"Converting {key} to video {video_path}")
                # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                # video_writer = cv2.VideoWriter(
                #     video_path, fourcc, freq_hz, value.shape[2:0:-1]
                # )
                video_writer = imageio.get_writer(video_path, fps=freq_hz)
                for i, frame in enumerate(value):
                    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # video_writer.write(frame)
                    if show_timestep:
                        # frame = cv2.putText(frame, f"Timestep: {i}", (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2) # For push_cube
                        frame = cv2.putText(frame, f"Timestep: {i}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2) # For match_color
                    video_writer.append_data(frame)
                if add_black_screen_in_the_end:
                    for _ in range(3):
                        # video_writer.write(np.zeros_like(frame))
                        video_writer.append_data(np.zeros_like(frame))
                # video_writer.release()
                video_writer.close()
    else:
        images = []
        # Each image: (T, H, W, 3)
        for key, value in episode_store.items():
            if len(value.shape) == 4 and value.shape[-1] == 3:
                images.append(value)
        video_path = os.path.join(video_dir, f"{episode_name}_timestep_{show_timestep}.mp4")
        logger.info(f"Converting {episode_name} to video {video_path}")
        # fourcc = cv2.VideoWriter_fourcc(*"H264")
        max_height = max([img.shape[1] for img in images])
        for i, img in enumerate(images):
            images[i] = np.concatenate(
                [
                    img,
                    np.zeros(
                        (img.shape[0], max_height - img.shape[1], img.shape[2], 3),
                        dtype=img.dtype,
                    ),
                ],
                axis=1,
            )

        merged_image = np.concatenate(images, axis=2)
        logger.info(merged_image.shape)
        # video_writer = cv2.VideoWriter(
        #     video_path, fourcc, freq_hz, merged_image.shape[2:0:-1]
        # )
        video_writer = imageio.get_writer(video_path, fps=freq_hz)
        if merged_image.dtype == np.float32:
            merged_image = (merged_image * 255).astype(np.uint8)
        for i, frame in enumerate(merged_image):
            if show_timestep:
               frame = cv2.putText(frame, f"Timestep: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # video_writer.write(frame)
            video_writer.append_data(frame)
        if add_black_screen_in_the_end:
            for _ in range(3):
                # video_writer.write(np.zeros_like(frame))
                video_writer.append_data(np.zeros_like(frame))
        # video_writer.release()
        video_writer.close()


def convert_data_to_video(
    data_dir: str,
    split_videos: bool,
    video_type: str,
    freq_hz: int = 10,
    video_num: Optional[int] = None,
    add_black_screen_in_the_end: bool = True,
    show_timestep: Optional[bool] = None,
):
    assert video_type in ["failure", "success", "all"]
    zarr_dir = os.path.join(data_dir, "episode_data.zarr")
    zarr_store = zarr.open(zarr_dir, mode="r")
    assert isinstance(zarr_store, zarr.Group), "zarr_store must be a zarr.Group"
    video_dir = os.path.join(data_dir, f"{video_type}_videos")
    os.makedirs(video_dir, exist_ok=True)
    rendered_video_num = 0
    for episode_name, episode_store in zarr_store.items():
        if video_num is not None and rendered_video_num >= video_num:
            break
        rendered_video_num += 1

        if video_type == "failure":
            if episode_store.attrs["is_successful"]:
                continue
        if video_type == "success":
            if not episode_store.attrs["is_successful"]:
                continue

        if show_timestep is None:
            show_timestep_options = [True, False]
        else:
            show_timestep_options = [show_timestep]
        for show_timestep in show_timestep_options:
            convert_single_episode_to_video(
                episode_store=episode_store,
                video_dir=video_dir,
                episode_name=episode_name,
                video_type=video_type,
                split_videos=split_videos,
                freq_hz=freq_hz,
                add_black_screen_in_the_end=add_black_screen_in_the_end,
                show_timestep=show_timestep,
            )


def convert_data_to_video_parallel(
    data_dir: str,
    split_videos: bool,
    video_type: str,
    worker_num: int = 20,
    freq_hz: int = 10,
    show_timestep: Optional[bool] = None,
    video_num: Optional[int] = None,
    add_black_screen_in_the_end: bool = True,
):
    assert video_type in ["failure", "success", "all"]
    zarr_dir = os.path.join(data_dir, "episode_data.zarr")
    zarr_store = zarr.open(zarr_dir, mode="r")
    assert isinstance(zarr_store, zarr.Group), "zarr_store must be a zarr.Group"
    video_dir = os.path.join(data_dir, f"{video_type}_videos")
    os.makedirs(video_dir, exist_ok=True)

    episode_names = list(zarr_store.keys())
    if video_num is not None:
        episode_names = episode_names[:video_num]
    episode_names = natsort.natsorted(episode_names, reverse=True)

    if video_num is None:
        video_num = len(episode_names)

    with ProcessPoolExecutor(worker_num) as executor:
        for episode_name in episode_names[:video_num]:
            if video_type == "failure":
                if zarr_store[episode_name].attrs["is_successful"]:
                    continue
            if video_type == "success":
                if not zarr_store[episode_name].attrs["is_successful"]:
                    continue
            episode_store = zarr_store[episode_name]
            assert isinstance(
                episode_store, zarr.Group
            ), "episode_store must be a zarr.Group"
            if show_timestep is None:
                show_timestep_options = [True, False]
            else:
                show_timestep_options = [show_timestep]
            for show_timestep_val in show_timestep_options:
                executor.submit(
                    convert_single_episode_to_video,
                    episode_store=episode_store,
                    video_dir=video_dir,
                    episode_name=episode_name,
                    video_type=video_type,
                    split_videos=split_videos,
                    freq_hz=freq_hz,
                    add_black_screen_in_the_end=add_black_screen_in_the_end,
                    show_timestep=show_timestep_val,
                )


class RolloutWebsiteServer:
    def __init__(self, root_dir: str, exclude_dirs: list[str] | None = None):
        """
        structure of root_dir:
        root_dir/date/run_name/epoch_X_success_rate_Y/success_videos/episode_Z.mp4
        """

        self.root_dir = os.path.join(os.getcwd(), root_dir)

        self.app = Flask(
            __name__,
            static_url_path="",
            template_folder=f"{self.root_dir}/templates",
            static_folder=None,
        )
        self.app.url_map.strict_slashes = False

        self.exclude_dirs: list[str] = exclude_dirs or []

        @self.app.route("/")
        def home():
            self.generate_list_template(self.root_dir)
            folders = [
                f
                for f in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, f))
            ]
            if "templates" in folders:
                folders.remove("templates")
            if "favicon.ico" in folders:
                folders.remove("favicon.ico")
            for exclude_dir in self.exclude_dirs:
                if exclude_dir in folders:
                    folders.remove(exclude_dir)
            folders = natsort.natsorted(folders, reverse=True)
            return render_template(
                "folder_list.html",
                folders=folders,
                current_dir=".",
                parent_dir="",
            )

        @self.app.route("/<path:dir_name>")
        def serve_dir(dir_name: str):
            # Automatically redirect to the directory without consecutive slashes
            while "//" in dir_name:
                dir_name = dir_name.replace("//", "/")
            if dir_name.endswith("/"):
                dir_name = dir_name[:-1]

            dir_path = os.path.join(self.root_dir, dir_name)
            if not os.path.exists(dir_path):
                return "Folder not found", 404
            if os.path.isfile(dir_path):
                return send_from_directory(self.root_dir, dir_name)
            elif (
                "episode_data.zarr" in os.listdir(dir_path)
                or "success_videos" in os.listdir(dir_path)
                or "failure_videos" in os.listdir(dir_path)
                or "all_videos" in os.listdir(dir_path)
            ):
                self.generate_static_website(self.root_dir, data_dir=dir_name)
                return send_from_directory(dir_path, "index.html")
            else:
                folders = [
                    f
                    for f in os.listdir(dir_path)
                    if os.path.isdir(os.path.join(dir_path, f))
                ]
                if "templates" in folders:
                    folders.remove("templates")
                if "favicon.ico" in folders:
                    folders.remove("favicon.ico")
                folders = natsort.natsorted(folders, reverse=True)
                folders.insert(0, "..")
                parent_dir = "/".join(dir_name.split("/")[:-1])
                logger.info(
                    f"parent_dir: {parent_dir}, dir_name: {dir_name}, folders: {folders}"
                )
                return render_template(
                    "folder_list.html",
                    folders=folders,
                    current_dir=dir_name,
                    parent_dir=parent_dir,
                )

    def start(self, port=5000):
        """Start the server in a background thread"""
        from threading import Thread

        self.server_thread = Thread(target=self._run_server, args=(port,), daemon=True)
        self.server_thread.start()
        logger.info(f"Website server started at http://localhost:{port}")

    def _run_server(self, port):
        """Internal method to run the Flask server"""
        self.app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    def generate_list_template(self, current_dir: str):
        os.makedirs(
            os.path.join(self.root_dir, current_dir, "templates"), exist_ok=True
        )
        with open(
            os.path.join(self.root_dir, current_dir, "templates", "folder_list.html"),
            "w",
        ) as f:
            f.write(
                f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rollout Folder Selection</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            color: #333;
        }}
        ul {{
            list-style-type: none;
            padding: 0;
        }}
        li {{
            margin: 10px 0;
            border-radius: 5px;
        }}
        li a {{
            display: block;
            padding: 10px;
            background-color: #f5f5f5;
            text-decoration: none;
            color: #0066cc;
            font-weight: bold;
            border-radius: 5px;
        }}
        li a:hover {{
            text-decoration: none;
            background-color: #e0e0e0;
        }}
        .back-button {{
            display: inline-block;
            margin-bottom: 20px;
            padding: 10px 15px;
            background-color: #0066cc;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
        }}
        .back-button:hover {{
            background-color: #0055aa;
        }}
    </style>
</head>
<body>
    <a href="/{{{{ parent_dir }}}}" class="back-button">← Back to Last Level</a>
    <h1>Select a Folder</h1>
    <ul>
        {{% for folder in folders %}}
        <li><a href="/{{{{ current_dir }}}}/{{{{ folder }}}}">{{{{ folder }}}}</a></li>
        {{% endfor %}}
    </ul>
</body>
</html>"""
            )

    def generate_static_website(self, root_dir: str, data_dir: str):
        data_dir_segments = data_dir.split("/")
        # HTML template parts
        html_start = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rollout Monitor</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        h1, h2 {{
            color: #333;
        }}
        .video-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }}
        .video-container {{
            width: 100%;
        }}
        video {{
            width: 100%;
            border-radius: 8px;
        }}
        .back-button {{
            display: inline-block;
            margin-bottom: 20px;
            padding: 10px 15px;
            background-color: #0066cc;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
        }}
        .back-button:hover {{
            background-color: #0055aa;
        }}
    </style>
</head>
<body>
    <a href="/{'/'.join(data_dir_segments[:-1])}" class="back-button">← Back to Last Level</a>
    <h1>{data_dir_segments[-2] if len(data_dir_segments) > 1 else ""}</h1>
    <h2>{data_dir_segments[-1]}</h2>
"""

        html_end = """
</body>
</html>"""

        def create_video_section(title, folder):

            if not os.path.exists(os.path.join(root_dir, data_dir, folder)):
                return ""

            video_paths = [
                f
                for f in os.listdir(os.path.join(root_dir, data_dir, folder))
                if f.endswith(".mp4")
            ]
            episode_indices = sorted(
                [
                    int(f.replace(".mp4", "").replace("episode_", "").split("_")[0])
                    for f in video_paths
                ]
            )

            if len(episode_indices) == 0:
                return ""

            section = f"""
    <div class="section">
        <h2>{title}</h2>
        <div class="video-grid">
"""
            for episode_idx in episode_indices:
                section += f"""            <div class="video-container">
                <p>Episode {episode_idx}</p>
                <video autoplay muted loop>
                    <source src="{'/'.join(data_dir.split('/')[-1:])}/{folder}/episode_{episode_idx}_timestep_False.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
"""
            section += "        </div>\n    </div>\n"
            return section

        # Generate the complete HTML
        html_content = html_start
        html_content += "<br><br><br><br>"
        html_content += create_video_section("Failed Rollouts", "failure_videos")
        # Add some space between sections
        html_content += "<br><br><br><br>"
        html_content += create_video_section("Successful Rollouts", "success_videos")
        html_content += "<br><br><br><br>"
        html_content += create_video_section("All Rollouts", "all_videos")
        html_content += html_end

        # Write the HTML file
        with open(f"{root_dir}/{data_dir}/index.html", "w") as f:
            f.write(html_content)
