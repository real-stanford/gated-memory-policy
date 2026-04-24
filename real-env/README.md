# Environment Setup

## Prerequisites

- Our repository is only tested on Ubuntu 24.04 Linux desktop/laptop.
- An NVIDIA GPU is recommended for hardware accelerated video recording (see [robologger](https://github.com/yihuai-gao/robologger))
- [Miniforge3](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) or any conda distribution

## Python Environment

```bash
# Inside the real-env directory
conda env create -f env.yaml
conda activate real-env
pip install -e . # Install real-env as a editable package so that all paths can be resolved
```

## Hardware Setup

### UR5 or UR5e

1. Connect UR5 to the computer through Ethernet, ensure the ip address on the computer is set to the same subnet as UR5 ip address, but the last section should be different.
  - For example, if the UR5 ip address is 192.168.2.124, your computer's ip address of the corresponding ethernet connection should be set to 192.168.2.x, where x is a different number than 124. To test connection, run `ping 192.168.2.124` on the computer.
2. Update the `robot_ip` in the `real_env/configs/controllers/ur5_cartesian.yaml`.
3. Enable the UR5 control box, drag the robot end-effector to a neutral position, record the home pose on the control box, and update `home_pose_xyz_wxyz` in `real_env/configs/tasks/umi_ur5.yaml`.
4. Based on which direction the robot end-effector is pointing at, refer to `real_env/configs/agents/spacemouse_agent.yaml` and update `spacemouse_agent.pose_orientation_types` in `real_env/configs/tasks/umi_ur5.yaml`. Incorrect setting will result in unaligned spacemouse control.
5. `conda activate real-env` and run `ur5`. Run again it it fails. You should see `UR5 is connected` if connection is successful.

### WSG50 with iPhUMI

1. Mount WSG50 on the UR5 using the following part: [iPhone Compatible WSG50 Mount](https://cad.onshape.com/documents/37aa71cf7a5e55f92ca5985a/w/2d50a600a3aff00c7fe1f43d/e/cbd5ddbf034f143a1ad05aa1?renderMode=0&uiState=69bcb858495342ff93401d29) and follow [UMI Hardware Guide](https://docs.google.com/document/d/1TPYwV9sNVPAi0ZlAupDMkXZ4CA1hsZx7YDMSmcEy6EU/edit?usp=sharing) to install fin-ray fingers.
2. Connect the WSG50 to the computer through Ethernet, using the same subnet rule as UR5.
3. Update the `robot_ip` in the `real_env/configs/controllers/wsg50.yaml`.
4. Open WSG50 config panel in the browser. In motion -> manual control, press `ACK` on the right side and then `Manual Homing` button. The gripper should move to the widest position and run into idle mode. In scripting -> interactive scripting, create a new script and paste the file from `real_env/controllers/wsg50_scripts.lua`. Save and run the script.
5. `conda activate real-env` and run `wsg50`. You should see `right_end_effector is connected` if connection is successful.

### ARX5 with iPhUMI

1. Follow [arx5-sdk](https://github.com/real-stanford/arx5-sdk) to setup ARX5 controller. Ensure you can run all [test scripts](https://github.com/real-stanford/arx5-sdk?tab=readme-ov-file#test-scripts) successfully.
2. Follow UMI-on-Legs [3D Printing Guide](https://github.com/real-stanford/umi-on-legs/blob/main/real-wbc/docs/3d_printing.md) and [Assembly Guide](https://github.com/real-stanford/umi-on-legs/blob/main/real-wbc/docs/assembly.md#umi-customization-for-arx5) to install UMI gripper on ARX5.
3. `conda activate real-env` and run `arx5 <model>`. `<model>` can be `X5_iphumi`, `X5_umi`, `L5_iphumi`, `L5_umi`. Please choose the correct model based on your actual hardware. The robot should reset to home pose if connection is successful. If unable to connect, please run `sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up` (SLCAN) or `sudo ip link set up can0 type can bitrate 1000000` (candleLight) again.
> Notice: Sometimes there might be a jump when the robot boots up. This behavior is not reliably reproducible in our setup. Please raise an issue or PR if you can reproduce it.

### iPhone

1. Follow [iPhUMI] [TODO] to install the iPhone app and print the iPhone mount.
2. Connect the iphone to the computer through a (at least) **USB3.0** cable. `conda activate real-env` then run `iphone wrist`. If all 3 videos pop up (main, ultrawide, depth), connection is successful. The iPhone UDID will be printed in the terminal.
3. To connect to multiple iPhones, record the UDID of each iphone and update the `iphone_udid` in `real_env/configs/peripherals/iphone_wrist.yaml` for wrist view and `real_env/configs/peripherals/iphone_third.yaml` for third person view.

### Webcam / GoPro

1. Find the device name of the webcam / GoPro. It should be something like `/dev/video0` or `/dev/video1`.
2. Run `v4l2-ctl --list-formats-ext -d /dev/videoX` to get the supported resolution, image format, fps etc. For example, when using the Elgato capture card, the image format `NV12` supports 3840x2160@30fps and 1920x1080@120fps.
3. Update the `camera_id`, `cam_img_format` and `camera_configs` in the corresponding yaml file, e.g. `real_env/configs/peripherals/gopro0.yaml`, `real_env/configs/peripherals/webcam0.yaml`.
4. `conda activate real-env` and run `webcam 0` or `gopro 0`. The image stream should pop up in a new window.

### SpaceMouse

1. Purchase the latest generation of [3Dconnexion SpaceMouse](https://a.co/d/03ecxtwN) and use a usb cable to connect to the computer. Earlier generations may have drifting issues.
2. Enable the spacenavd service:
    ```bash
    sudo apt install libspnav-dev spacenavd
    sudo systemctl enable spacenavd.service
    sudo systemctl start spacenavd.service
    ```
3. `conda activate real-env` and run `spacemouse_server`. You should see the SpaceMouse readout in the terminal.

## System Overview

### Architecture

We employ a decoupled architecture for different components. Each component (UR5, WSG50, iPhone, SpaceMouse, Policy, etc.) runs an independent server process in the background, as shown in the last section. We don't need to shut down the server process unless encountering connection issues.

The task script (e.g. `real_env/tasks/iphumi_ur5_task.py`) will initialize multiple clients to communicate with each server on demand. This ensures each component runs at their own frequency and avoids potential synchronization issues.

In each policy inference loop, the task script will:
1. Query the latest camera images and the robot states
2. Process the images and states and send to the policy server
3. Wait for the policy server to return the next action trajectory
4. Schedule the action trajectory on each robot controller

### Customized Packages

1. [robot-message-queue](https://github.com/yihuai-gao/robot-message-queue) A light-weight and flexible Robot-centric Message Queue for Python applications based on ZeroMQ.
    - Spawns an additional C++ thread on each server to handle communication so the client requests will not interrupt the server loop.
    - Supports shared memory in C++ for high-res video streaming.
    - Self-contained without additional dependencies (if you are suffering from the complicated ROS environment).
2. [robologger](https://github.com/yihuai-gao/robologger) Light-weight and efficient logging library for robot learning applications.
    - Best used with GPU accelerated video recording.
    - Synchronizes multi-process logging in the background.
3. [teleop-utils](https://github.com/yihuai-gao/teleop-utils) A collection of teleoperation utilities (spacemouse, keyboard, mocap, iphone).
4. [robot-utils](https://github.com/yihuai-gao/robot-utils) A collection of (hopefully) bug-free and frequently used utility functions for robot learning research.

### Config Aggregation

During experiments, we might accidentally change some configs that leads to different behaviors and forgot the original configs. To ensure each episode is reproducible, we use hydra to compose configs in each component and aggregate them in the task script. The task script will then dump all the configs to `<project_name>/<task_name>/<run_name>/episode_xxxxxx/metadata.zarr/.zattrs`. (Please see the robologger output)

## Run Experiments

After all the setup hardware components above, we provide a checklist to run experiments. Except for the policy server, all other scripts should be run in the `real-env` conda environment and each in a different terminal. When switching checkpoints/experiments, the robot controller, spacemouse, and camera servers does not need to be restarted. Shortcut executables are installed in the `real-env` conda environment, use `which xxx` to find the underlying python script.

1. Launch robot controllers: `ur5`, `wsg50` (or `arx5 <model>` where `<model>` can be `X5_iphumi`, `X5_umi`, `L5_iphumi`, `L5_umi`; if not connected, please run `sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up` (SLCAN) or `sudo ip link set up can0 type can bitrate 1000000` (candleLight) again.)
2. Launch spacemouse server: `spacemouse_server`.
3. Launch camera servers: `iphone wrist` (or `webcam`, `gopro 0`).
4. Run the policy server in the `imitation-learning-policies` codebase, for example, `cd ../imitation-learning-policies` `conda activate imitation` and then `shell_scripts/serve_policy_ckpt.sh iphumi_place_back_with_correction_diffusion_gated.ckpt` or `shell_scripts/serve_policy_ckpt.sh data/checkpoints/real/umi_multi_diffusion_transformer_large.ckpt`.
5. Run the **task script**: `iphumi_ur5 flip_and_place_back` or `iphumi_arx5 flip_and_place_back` or `umi_ur5 multi_task` or `umi_arx5 multi_task`. If everything is working, you should be able to teleop the robot with SpaceMouse. The task configs are stored in `real_env/configs/tasks/`.
6. To reset the robot, press `r` in the **task script terminal**.
7. To continue policy control, press `c` in the **task script terminal**.
8. **(Does not have to be in the task script terminal)** To stop policy control and mark success/failure, press `y`/`n` wherever which window is on top; to stop policy control, press `s` there will be a prompt to set success/failure.
9. To run a single trajectory with policy control, press `C` in the **task script terminal**.
10. Run `shell_scripts/kill_all_processes.sh` to kill all the processes after the experiment.