# Environment Setup

## Prerequisites

- A **Linux** machine with NVIDIA GPU(s) — Windows and macOS will run into package compatibility issues
- A compatible NVIDIA driver installed on the host (the CUDA toolkit itself is managed by conda)
- [Miniforge3](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) or any conda distribution

## Installation

```bash
conda env create -f env.yaml
conda activate imitation
```

> **Note:** For the latest GPU (e.g. RTX 5090), you may need to install a nightly build of PyTorch.
```bash
pip install --pre torch torchvision torchaudio torchcodec --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
```

## Download Checkpoints and Datasets

Install the Hugging Face CLI:
```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

```bash
mkdir -p data/checkpoints

# Download all checkpoints
hf download yihuai-gao/gated-memory-policy --type model --local-dir ./data/checkpoints # 48.8GB

# Or download checkpoints for specific environments
hf download yihuai-gao/gated-memory-policy --type model --include "memmimic/**" --local-dir ./data/checkpoints # 19.1GB
hf download yihuai-gao/gated-memory-policy --type model --include "robomimic/**" --local-dir ./data/checkpoints # 26.5GB
hf download yihuai-gao/gated-memory-policy --type model --include "mikasa/**" --local-dir ./data/checkpoints # 
hf download yihuai-gao/gated-memory-policy --type model --include "real/**" --local-dir ./data/checkpoints # 3.2GB

# Or download specific models / tasks 
hf download yihuai-gao/gated-memory-policy --type model --include "memmimic/*gated*" --local-dir ./data/checkpoints # 4.4 GB
hf download yihuai-gao/gated-memory-policy --type model --include "memmimic/*push_cube*" --local-dir ./data/checkpoints
hf download yihuai-gao/gated-memory-policy --type model --include "memmimic/*push_cube*gated*" --local-dir ./data/checkpoints
```

```bash
mkdir -p data/datasets

# Download all datasets
hf download yihuai-gao/gated-memory-policy --type dataset --local-dir ./data/datasets # 325GB 

# Or download specific datasets
hf download yihuai-gao/gated-memory-policy --type dataset --include "memmimic/**" --local-dir ./data/datasets # 141GB
hf download yihuai-gao/gated-memory-policy --type dataset --include "robomimic/**" --local-dir ./data/datasets # 184GB
hf download yihuai-gao/gated-memory-policy --type dataset --include "iphumi/**" --local-dir ./data/datasets # 44.8GB
hf download yihuai-gao/gated-memory-policy --type dataset --include "real_world/**" --local-dir ./data/datasets # 12.6GB
```

## Serve a Checkpoint

```bash
shell_scripts/serve_policy_ckpt.sh <path/to/checkpoint.ckpt>
```

For example:

```bash
shell_scripts/serve_policy_ckpt.sh data/checkpoints/memmimic/push_cube_diffusion_gated.ckpt # Use GPU 0 (default)
shell_scripts/serve_policy_ckpt.sh data/checkpoints/memmimic/push_cube_diffusion_gated.ckpt 1 # Use GPU 1
```

## Rollout all Simulation Checkpoints

Please ensure all the simulation environments are working in [mujoco-env](../mujoco-env/README.md).

The following script will use all the GPUs to rollout all the RoboMimic / MemMimic simulation checkpoints. It will spawn 1 process to send commands to send checkpoint paths and gather results, N (number of GPUs) processes to run policy servers, and N processes to host simulation environments. The entire evaluation process (100 episodes per checkpoint) takes about 2hrs on a 8-GPU server. You can specify the filter string to rollout only specific checkpoints.

```bash
shell_scripts/rollout_policies.sh <checkpoints_dir> <filter_str>
```

## Train a Policy

### Config System

This repository uses the [Hydra](https://hydra.cc/) config system. We set multiple levels of configurations for different components, avoiding repeated configuration across different policies and tasks. Each checkpoint will keep the entire config file in the checkpoint directory, so you can easily reproduce the training process by simply loading the checkpoint and instantiating the config.

For regular training, the config overriding order is (from lowest to highest):

- `base_config.yaml`: defaults to `base_workspace.yaml`
- `workspace/policy/{policy_name}.yaml`: will be specified based on the policy name passed from the command line.
- `workspace/dataset/{dataset_name}.yaml`: dataset name is determined by both policy (using memory or not: single-traj or multi-traj) and task (which specific dataset)
- `task/{task_name}.yaml`: will apply task-specific overrides, for example, different keys
- `train_policy.yaml` and **shell script additional arguments**: please use `+workspace.xxx=xxx` to append an additional override to `train_policy.yaml`. This config will be applied last to ensure the highest priority.

### Simulation

Please go through all comments in `shell_scripts/train_sim.sh` for detailed instructions.

By default it runs on all GPUs. To use a different single GPU or multiple GPUs, pass the GPU IDs as a comma-separated argument:

```bash
shell_scripts/train_sim.sh 1      # single GPU (GPU 1)
shell_scripts/train_sim.sh 0,1    # multi-GPU
```

### Real-world

Please go through all comments in `shell_scripts/train_real.sh` for detailed instructions.

By default it runs on all GPUs. To use a different single GPU or multiple GPUs, pass the GPU IDs as a comma-separated argument:

```bash
shell_scripts/train_real.sh 1      # single GPU (GPU 1)
shell_scripts/train_real.sh 0,1    # multi-GPU
```

## Misc.

- The rotation 6d is following the UMI convention, which uses the first 2 **rows** of the rotation matrix rather than the first 2 columns proposed in [this original paper](https://arxiv.org/abs/1812.07035).
