# Gated Memory Policy

[Yihuai Gao](https://yihuai-gao.github.io/), [Jinyun Liu](), [Shuang Li](https://shuangli59.github.io/), [Shuran Song](https://shurans.github.io/)

Stanford University

[Project Website](https://gated-memory-policy.github.io/), [ArXiv](https://arxiv.org/abs/2604.18933), [Models](https://huggingface.co/yihuai-gao/gated-memory-policy), [Datasets](https://huggingface.co/datasets/yihuai-gao/gated-memory-policy)

This repository contains source code for gated memory policy training, simulation data collection and evaluation (Memimic & RoboMimic and Mikasa-Robo benchmarks), and **real-world robot deployment with [in-the-wild checkpoints](https://huggingface.co/yihuai-gao/gated-memory-policy/blob/main/real/iphumi_place_back_with_correction_diffusion_memory.ckpt)**.

**Major Update (Jun 29th, 2026): [iPhUMI](https://github.com/real-stanford/iPhUMI) is released, huge shout out to [Austin Patel](https://austinapatel.github.io/)! Please feel free to try the cup placement policy in-the-wild!** 

- Tarined cups purchase list
    - https://a.co/d/0cbmGF33
    - https://a.co/d/0ipbq8zH
    - https://a.co/d/01BH9ghQ
    - https://a.co/d/0gHmcf6r
    - https://a.co/d/0h9Vaisz

- Please refer to the [policy codebase](imitation-learning-policies/) for instructions to set up the conda environment and serve the [in-the-wild checkpoint](https://huggingface.co/yihuai-gao/gated-memory-policy/blob/main/real/iphumi_place_back_with_correction_diffusion_memory.ckpt).
- For the real-world deployment codebase, please checkout the [gated-memory-policy](https://github.com/real-stanford/real-env/tree/gated-memory-policy) branch in case there are compatibility issues.

| Repo                                                           | What it does                                                                    |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [`imitation-learning-policies/`](imitation-learning-policies/) | Policy training and inference serving                                           |
| [`real-env/`](https://github.com/real-stanford/real-env/tree/gated-memory-policy)                                       | Real-world robot deployment (Natively support UR5, ARX5)                        |
| [`mujoco-env/`](mujoco-env/)                                   | MuJoCo sim, data collection and evaluation for the **Memimic (Ours)** and [**RoboMimic**](https://github.com/ARISE-Initiative/robomimic) benchmarks        |
| [`mikasa-robo-env/`](mikasa-robo-env/)                         | ManiSkill sim, data collection and evaluation for the [**Mikasa-Robo**](https://github.com/CognitiveAISystems/MIKASA-Robo) benchmark |


> **Note:** We use the submodule `real-env` to manage the real-world robot deployment codebase (shared with other projects). Please use `git submodule update --init --recursive` to initialize the submodule.

## Table of Contents

- [Policy Training and Serving](./imitation-learning-policies/README.md)
    - [Installation](./imitation-learning-policies/README.md#installation)
    - [Download Checkpoints and Datasets](./imitation-learning-policies/README.md#download-checkpoints-and-datasets)
    - [Serve a Checkpoint](./imitation-learning-policies/README.md#serve-a-checkpoint)
    - [Rollout all Simulation Checkpoints](./imitation-learning-policies/README.md#rollout-all-simulation-checkpoints)
    - [Train a Policy](./imitation-learning-policies/README.md#train-a-policy)
        - [Config System](./imitation-learning-policies/README.md#config-system)
        - [Simulation](./imitation-learning-policies/README.md#simulation)
        - [Real-world](./imitation-learning-policies/README.md#real-world)
    - [Misc.](./imitation-learning-policies/README.md#misc)
- [Real-World Deployment](https://github.com/real-stanford/real-env/tree/gated-memory-policy/README.md)
    - [Python Environment](https://github.com/real-stanford/real-env/tree/gated-memory-policy/README.md#python-environment)
    - [Hardware Setup](https://github.com/real-stanford/real-env/tree/gated-memory-policy/README.md#hardware-setup)
        - [UR5 or UR5e](https://github.com/real-stanford/real-env/tree/gated-memory-policy/README.md#ur5-or-ur5e)
        - [WSG50 with iPhUMI](https://github.com/real-stanford/real-env/tree/gated-memory-policy/README.md#wsg50-with-iphumi)
        - [ARX5 with iPhUMI](https://github.com/real-stanford/real-env/tree/gated-memory-policy/README.md#arx5-with-iphumi)
        - [iPhone](https://github.com/real-stanford/real-env/tree/gated-memory-policy/README.md#iphone)
        - [Webcam / GoPro](https://github.com/real-stanford/real-env/tree/gated-memory-policy/README.md#webcam--gopro)
        - [SpaceMouse](https://github.com/real-stanford/real-env/tree/gated-memory-policy/README.md#spacemouse)
    - [System Overview](https://github.com/real-stanford/real-env/tree/gated-memory-policy/README.md#system-overview)
        - [Architecture](https://github.com/real-stanford/real-env/tree/gated-memory-policy/README.md#architecture)
        - [Customized Packages](https://github.com/real-stanford/real-env/tree/gated-memory-policy/README.md#customized-packages)
        - [Config Aggregation](https://github.com/real-stanford/real-env/tree/gated-memory-policy/README.md#config-aggregation)
    - [Run Experiments](https://github.com/real-stanford/real-env/tree/gated-memory-policy/README.md#run-experiments)
- [Memimic & RoboMimic Benchmark](./mujoco-env/README.md)
    - [Installation](./mujoco-env/README.md#installation)
    - [Usage](./mujoco-env/README.md#usage)
        - [Run heurisitic policies](./mujoco-env/README.md#run-heurisitic-policies)
        - [Run spacemouse teleop](./mujoco-env/README.md#run-spacemouse-teleop)
        - [Collect heuristic data](./mujoco-env/README.md#collect-heuristic-data)
        - [Rollout a policy](./mujoco-env/README.md#rollout-a-policy)
        - [Serve a remote environment](./mujoco-env/README.md#serve-a-remote-environment)
        - [Serve a website for viewing rollout results](./mujoco-env/README.md#serve-a-website-for-viewing-rollout-results)
    - [Adding a New Task](./mujoco-env/README.md#adding-a-new-task)
- [Mikasa-Robo Benchmark](./mikasa-robo-env)
    - [Quick Start](./mikasa-robo-env/README.md#quick-start)
    - [Evaluation](./mikasa-robo-env/README.md#evaluation)
        - [Local (single checkpoint)](./mikasa-robo-env/README.md#local-single-checkpoint)
        - [Distributed (multi-checkpoint sweep)](./mikasa-robo-env/README.md#distributed-multi-checkpoint-sweep)
    - [Camera Resolution](./mikasa-robo-env/README.md#camera-resolution)
    - [Repo Layout](./mikasa-robo-env/README.md#repo-layout)

## Code Acknowledgments

We are grateful to the following amazing open-sourced projects that made this work possible:

- [iPhUMI](https://github.com/real-stanford/iPhUMI) and [UMI](https://github.com/real-stanford/universal_manipulation_interface) for the portable data collection system.
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) and [RDT-1B](https://github.com/thu-ml/RoboticsDiffusionTransformer) for the policy model and training framework.
- [RoboMimic](https://github.com/ARISE-Initiative/robomimic), [RoboSuite](https://github.com/ARISE-Initiative/robosuite), [Mikasa-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo), and [ManiSkill](https://github.com/haosulab/maniskill) for the simulation benchmarks.


## Citation

If you find this work useful, please cite:

```bibtex
@misc{gao2026gatedmemorypolicy,
  title         = {Gated Memory Policy: In-Context Memorization and Adaptation},
  author        = {Yihuai Gao and Jeff Jinyun Liu and Shuang Li and Shuran Song},
  year          = {2026},
  eprint        = {2604.18933},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2604.18933},
}
```
