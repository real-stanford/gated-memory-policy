#! /bin/bash

root_dir=$(dirname $(dirname $(realpath $0)))
cd $root_dir

set -e

mamba create -y -n "mujoco-env" python=3.10.15 pybind11-stubgen hydra-core black transforms3d matplotlib pybind11 \
        flask flask-socketio pynput opencv zarr line_profiler ray-core numba \
        make gxx_linux-64 spdlog natsort isort loguru imageio

. $(conda info --base)/etc/profile.d/conda.sh # This is equivalent to source the conda profile
conda activate mujoco-env

pip_path=$(conda info --base)/envs/mujoco-env/bin/pip
echo "Using conda env: $CONDA_PREFIX; pip path: $pip_path"
$pip_path install dm_control==1.0.31
$pip_path install mink==0.0.12
$pip_path install mujoco==3.3.5
$pip_path install "imageio[ffmpeg]"
$pip_path install robotmq
$pip_path install teleop-utils
$pip_path install "qpsolvers[quadprog]"
$pip_path install -e .
$pip_path install https://github.com/cheng-chi/spnav/archive/c1c938ebe3cc542db4685e0d13850ff1abfdb943.tar.gz

pybind11-stubgen mujoco -o $(python -c "import mujoco, os; print(os.path.dirname(os.path.dirname(mujoco.__file__)))")
shell_scripts/install_symlinks.sh

# Optional: if you want to run robomimic tasks

# git submodule update --init --recursive
# $pip_path install -e third_party/robosuite
# $pip_path install -e third_party/robomimic
