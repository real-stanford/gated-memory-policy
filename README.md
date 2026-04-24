# Gated Memory Policy

[Yihuai Gao](https://yihuai-gao.github.io/), [Jinyun Liu](), [Shuang Li](https://shuangli59.github.io/), [Shuran Song](https://shurans.github.io/)

Stanford University

[Project Website](https://gated-memory-policy.github.io/), [ArXiv](https://arxiv.org/abs/2604.18933), [Models](https://huggingface.co/yihuai-gao/gated-memory-policy), [Datasets](https://huggingface.co/datasets/yihuai-gao/gated-memory-policy)

This repository contains source code for gated memory policy training, simulation data collection and evaluation (Memimic & RoboMimic and Mikasa-Robo benchmarks), and **real-world robot deployment with in-the-wild checkpoints**.

We've organized our code as separate folders so you can easily take any component you need and plug it into your own system.

| Repo                                                           | What it does                                                                    |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [`imitation-learning-policies/`](imitation-learning-policies/) | Policy training and inference serving                                           |
| [`real-env/`](real-env/)                                       | Real-world robot deployment (Natively support UR5, ARX5)                        |
| [`mujoco-env/`](mujoco-env/)                                   | MuJoCo sim, data collection and evaluation for the **Memimic (Ours)** and [**RoboMimic**](https://github.com/ARISE-Initiative/robomimic) benchmarks        |
| [`mikasa-robo-env/`](mikasa-robo-env/)                         | ManiSkill sim, data collection and evaluation for the [**Mikasa-Robo**](https://github.com/CognitiveAISystems/MIKASA-Robo) benchmark |


## Table of Contents

- [Policy Training and Serving](./imitation-learning-policies/README.md)
    - [Installation](./imitation-learning-policies/README.md#installation)
    - [Download Checkpoints and Datasets](./imitation-learning-policies/README.md#download-checkpoints-and-datasets)
    - [Serve a Checkpoint](./imitation-learning-policies/README.md#serve-a-checkpoint) (serves checkpoints for any environment)
    - [Rollout all Simulation Checkpoints](./imitation-learning-policies/README.md#rollout-all-simulation-checkpoints) (distributed multi-GPU evaluation)
    - [Train a Policy](./imitation-learning-policies/README.md#train-a-policy)
        - [Config System](./imitation-learning-policies/README.md#config-system) (Hydra-based multi-level config overriding)
        - [Simulation](./imitation-learning-policies/README.md#simulation)
        - [Real-world](./imitation-learning-policies/README.md#real-world)
    - [Misc.](./imitation-learning-policies/README.md#misc)
- [Real-World Deployment](./real-env/README.md)
    - [Python Environment](./real-env/README.md#python-environment)
    - [Hardware Setup](./real-env/README.md#hardware-setup)
        - [UR5 or UR5e](./real-env/README.md#ur5-or-ur5e)
        - [WSG50 with iPhUMI](./real-env/README.md#wsg50-with-iphumi)
        - [ARX5 with iPhUMI](./real-env/README.md#arx5-with-iphumi)
        - [iPhone](./real-env/README.md#iphone)
        - [Webcam / GoPro](./real-env/README.md#webcam--gopro)
        - [SpaceMouse](./real-env/README.md#spacemouse) (teleoperation)
    - [System Overview](./real-env/README.md#system-overview)
        - [Architecture](./real-env/README.md#architecture) (decoupled component servers over robotmq)
        - [Customized Packages](./real-env/README.md#customized-packages) (robotmq, robologger, teleop-utils, robot-utils)
        - [Config Aggregation](./real-env/README.md#config-aggregation)
    - [Run Experiments](./real-env/README.md#run-experiments) (place_back, flip, casting, multi-task variants)
- [Memimic & RoboMimic Benchmark](./mujoco-env/README.md) (MuJoCo)
    - [Installation](./mujoco-env/README.md#installation)
    - [Usage](./mujoco-env/README.md#usage)
        - [Run heurisitic policies](./mujoco-env/README.md#run-heurisitic-policies) (pick-and-place, color matching, push, fling)
        - [Run spacemouse teleop](./mujoco-env/README.md#run-spacemouse-teleop)
        - [Collect heuristic data](./mujoco-env/README.md#collect-heuristic-data)
        - [Rollout a policy](./mujoco-env/README.md#rollout-a-policy) (single-process or parallel via Ray)
        - [Serve a remote environment](./mujoco-env/README.md#serve-a-remote-environment) (multi-checkpoint evaluation)
        - [Serve a website for viewing rollout results](./mujoco-env/README.md#serve-a-website-for-viewing-rollout-results)
    - [Adding a New Task](./mujoco-env/README.md#adding-a-new-task)
- [Mikasa-Robo Benchmark](./mikasa-robo-env) ([MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo), [ManiSkill](https://github.com/haosulab/maniskill))
    - [Quick Start](./mikasa-robo-env/README.md#quick-start)
    - [Evaluation](./mikasa-robo-env/README.md#evaluation)
        - [Local (single checkpoint)](./mikasa-robo-env/README.md#local-single-checkpoint) (memory-intensive tasks: shell games, color recall, interception, etc.)
        - [Distributed (multi-checkpoint sweep)](./mikasa-robo-env/README.md#distributed-multi-checkpoint-sweep) (policy server + environment server via robotmq)
    - [Camera Resolution](./mikasa-robo-env/README.md#camera-resolution)
    - [Repo Layout](./mikasa-robo-env/README.md#repo-layout)

## Code Acknowledgments

We are grateful to the following amazing open-sourced projects that made this work possible:

- [iPhUMI](TODO) and [UMI](https://github.com/real-stanford/universal_manipulation_interface) for the portable data collection system.
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) and [RDT-1B](https://github.com/thu-ml/RoboticsDiffusionTransformer) for the policy model and training framework.
- [RoboMimic](https://github.com/ARISE-Initiative/robomimic), [RoboSuite](https://github.com/ARISE-Initiative/robosuite), [Mikasa-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo), and [ManiSkill](https://github.com/haosulab/maniskill) for the simulation benchmarks.


## Citation

If you find this work useful, please cite:

```bibtex
@misc{gao2026gatedmemorypolicy,
  title         = {Gated Memory Policy},
  author        = {Yihuai Gao and Jinyun Liu and Shuang Li and Shuran Song},
  year          = {2026},
  eprint        = {2604.18933},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2604.18933},
}
```
