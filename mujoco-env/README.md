# MuJoCo Environment

A MuJoCo-based robot simulation environment for heuristic data collection and policy rollout evaluation. Supports efficient multi-process rollout for the proposed MemMimic and the third-party RoboMimic benchmarks.

Key features:
- Modularized design for flexible task and agent development
- Hydra configuration for best reproducibility
- Ray-based parallel environment for efficient rollout and data generation
- Environment server for multi-checkpoint evaluation

## Installation

```bash
git clone <repo-url>
cd mujoco-env
conda env create -f env.yaml
conda activate mujoco-env
pip install -e .
```

> **Note:** Shell scripts expect the conda environment to be named `mujoco-env` (by default). If you rename it, update `python_path` in the scripts accordingly.


## Usage

All following commands should be run under the `mujoco-env` conda environment.

```bash
conda activate mujoco-env
```

### Run heurisitic policies

`task_name` should be one of the following: `pick_and_match_color` (T1: Match Color), `pick_and_match_color_rand_delay` (T1': Match Color with Random Delay), `pick_and_place_back` (T2: Discrete Place Back), `push_cube` (T4: Iterative Pushing), `fling_cloth` (T5: Iterative Flinging)

In the first terminal, run the mujoco simulation with heuristic policy:
```bash
python scripts/run_heuristic_agent.py task_name=push_cube
```

In the second terminal, run a keyboard server that listens to all keyboard inputs into **this terminal**.
```bash
keyboard_server
```
| Key | Action |
| --- | ------ |
| `q` | End the current episode |
| `R` | Reset the full episode (resamples the episode config) |
| `r` | Reset robot joints only, keep the current episode config |
| `d` | Display current task states and the latest robot/env/action data |
| `s` | Toggle pause/resume of the simulation |
| `S` | Advance a single simulation step, then pause |
| `x` | Switch the robot being controlled (SpacemouseAgent only) |
| `C` | Print the viewer camera attributes (`lookat`, `distance`, `azimuth`, `elevation`) |

### Run spacemouse teleop

In the first terminal, run the mujoco simulation with spacemouse agent:
```bash
python scripts/run_spacemouse_teleop.py task_name=pick_and_place_back
```

In the second terminal, run a keyboard server that listens to all keyboard inputs into **this terminal**. Keyboard commands are the same as `run_heuristic_agent`.
```bash
keyboard_server
```

In the third terminal, run the spacemouse server that listens to spacemouse actions:
```bash
spacemouse_server
```

If it's your first time setting up 3Dconnexion SpaceMouse, please run:
```bash
sudo apt install libspnav-dev spacenavd
sudo systemctl enable spacenavd.service
sudo systemctl start spacenavd.service
```

### Collect heuristic data

To generate data for different tasks, set the `task_name` and `run_name` in `shell_scripts/collect_data.sh` to the desired task. `run_name` can be arbitrary (same as the task name is fine); 

```bash
shell_scripts/collect_data.sh          # defaults to GPU 0
shell_scripts/collect_data.sh 1        # run on GPU 1
```

To configure how many parallel workers to use, set the `parallel_workers` in `config/collect_heuristic_data.yaml`.

To adjust episode count, compression, or other settings, edit `config/collect_heuristic_data.yaml`. For environment and scene settings, edit `config/task/<task_name>.yaml`.

> **Note:** Pass `dbg` as a second argument to run with a single worker and no compression — useful for debugging and visualization:
>
> ```bash
> shell_scripts/collect_data.sh 0 dbg
> ```

### Rollout a policy

Requires a running policy server. See the companion repo `imitation-learning-policies` [here](../imitation-learning-policies/README.md#serve-a-checkpoint) for how to serve a checkpoint.

Run keyboard server in a separate terminal to pause/teleop. Key commands are the same as [run_heuristic_agent](./README.md#run-heurisitic-policies).
```bash
keyboard_server
```

Pass the task name as the first argument to `shell_scripts/rollout_policy.sh` to rollout the policy for the desired task that matches the checkpoint you are serving.


> **Note:** Robomimic tasks (`robomimic_square`, `robomimic_tool_hang`, `robomimic_transport`) require the submodule setup:
> **Important:** To ensure performance, you should strictly install the git submodule inside this repo. Using other versions leads to significant performance degradation.
```bash
git submodule update --init --recursive
pip install -e third_party/robosuite
pip install -e third_party/robomimic
```

Use the single-process script for quick evaluation (with mujoco viewer), or the parallel script for faster large-scale rollout. Both default to GPU 0; pass a GPU ID as the second argument to override:

```bash
shell_scripts/rollout_policy.sh push_cube                    # single process, GPU 0
shell_scripts/rollout_policy.sh push_cube 1                  # single process, GPU 1
shell_scripts/rollout_policy_parallel.sh push_cube           # multiprocess, GPU 0
shell_scripts/rollout_policy_parallel.sh push_cube 1         # multiprocess, GPU 1
```

Results are saved to:
- Single-process: `data/rollout_policy/<timestamp>_<task_name>/`
- Parallel: `data/rollout_policy_parallel/<timestamp>/`

To adjust episode count, timing limits, or other settings, edit the task-specific config:
- Single-process: `config/rollout_policy_<task_name>.yaml` (falls back to `config/rollout_policy.yaml`)
- Parallel: `config/rollout_policy_parallel.yaml` — also edit `env_num` here to control parallelism

### Serve a remote environment

This is used for multi-checkpoints evaluation: the environment server listens to the policy server. Once a new checkpoint is loaded, the environment server will automatically create the corresponding parallel environments and rollout the policy. 

```bash
shell_scripts/serve_remote_env.sh # Serve on GPU 0, port 18765 (default)
shell_scripts/serve_remote_env.sh 1 18766 # Serve on GPU 1, port 18766
```

### Serve a website for viewing rollout results

```bash
shell_scripts/serve_website.sh
```
Then open your browser and navigate to `http://localhost:5000`.

## Adding a New Task

Use `pick_and_place_back`, `push_cube`, and `fling_cloth` as references.

### 1. Task class

Create `env/modules/tasks/<task_name>.py`. Each task file follows the same structure:

**a) Module-level `_process_episode_config` function (required)**

`BaseTask` declares this method but raises `NotImplementedError`, so every task must provide it. All existing tasks define it as a plain module-level function (taking `self` as the first argument) and monkey-patch it onto the class at the bottom of the file:

```python
def _process_episode_config(self: "MyTask | MyTaskParallel", episode_config):
    self.rng = np.random.default_rng(episode_config["seed"])
    if "my_param" not in episode_config:
        episode_config["my_param"] = self.rng.uniform(self.param_min, self.param_max)
    return episode_config

MyTask._process_episode_config = _process_episode_config
MyTaskParallel._process_episode_config = _process_episode_config
```

Think of it as a **"fill in what's missing"** layer. The caller decides how much to pre-specify; this function fills the gaps:

- **Heuristic data collection** — the caller provides only `seed` and `episode_idx`, so `_process_episode_config` samples everything else (friction, object placement, etc.) randomly from the ranges stored on `self`.
- **Rollout** — the caller pre-fills the params it cares about (e.g. a specific friction value to evaluate), so `_process_episode_config` finds them already present and leaves them untouched.

The function itself has no branching on mode — the `if "param" not in episode_config` pattern handles both cases automatically. The task class doesn't know or care which mode it's running in.

**b) `MyTask(BaseTask)` class**

Store task-level config in `__init__` (everything that comes from the task YAML, passed via `**kwargs` through to `super().__init__`). `BaseTask.run_episode()` handles the full step loop — you do not need to override it.

Optionally override:

- `reset()` — if the agent needs to be configured based on the episode config after it is processed. `PushCube` does this to inject a pre-computed velocity sequence directly into the heuristic agent. (`FlingCloth` does not override `reset()` — instead, `_process_episode_config` places `speed_scales` into `episode_config`, and the agent's own `reset()` reads it from there.)
- `customized_obs_dict` property — to log extra task-specific observations into episode data. `PickAndPlaceBack` uses this to log relative TCP-to-object and TCP-to-bin poses.

**c) `MyTaskParallel(ParallelTask)` class**

A slimmer variant used by the parallel rollout script. It shares the same `_process_episode_config` but typically omits single-process-only state (e.g. precomputed lookup tables for binary search). Store only what `_process_episode_config` needs at runtime (e.g. `param_min`, `param_max`).

> **Important:** Defining `MyTaskParallel` is not enough on its own — you also have to wire it into `env/utils/config_utils.py::convert_task_to_parallel`, which is where `task_name` is mapped to the parallel class's `_target_`. See Step 4.

### 2. Task config

Create `config/task/<task_name>.yaml`. This file is how Hydra instantiates your task class — the connection to Step 1 is direct: the `_target_` field names the Python class, and every other top-level key in the YAML becomes a keyword argument passed to its `__init__`.

For example, `push_cube.yaml` has:

```yaml
_target_: env.modules.tasks.push_cube.PushCube
sliding_friction_min: 0.005
sliding_friction_max: 0.015
sampled_vels_m_per_s: [...]
sampled_sliding_frictions: [...]
```

These map directly to `PushCube.__init__(self, sliding_friction_min, sliding_friction_max, sampled_vels_m_per_s, sampled_sliding_frictions, **kwargs)`. Whatever task-specific parameters your `_process_episode_config` reads off `self` (e.g. `self.sliding_friction_min`) must be declared in `__init__` and provided here in the YAML.

The `defaults:` block at the top pulls in shared sub-configs — typically an `env/...` entry (which itself nests scene/robot/object configs) and `base_task`. Start by copying an existing task YAML, update `_target_`, and replace the task-specific parameters with your own.

### 3. Heuristic agent

Create the agent class and its config:

- `env/modules/agents/heuristic_agents/<task_name>_agent.py`
- `config/task/agent/heuristic/<task_name>.yaml`

All heuristic agents in this repo follow the same two patterns:

**Phase-based state machine** — `predict_actions` is called every step. The agent tracks a `self.phase` string and advances through phases (e.g. `waiting → grasping → placing → done`). Each phase runs until its motion is complete, then transitions to the next. This keeps the logic linear and readable.

**Interpolator-driven motion** — rather than computing raw actions, each phase sets up an `ActionInterpolator` (linear, cubic, etc.) and calls `.interpolate(dt)` each step to get a smooth trajectory. When `.is_finished` is true, the phase transitions.

The agents differ in their **episode structure**:

| Agent                       | Structure                                                                                            | Key episode param                                            |
| ---------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `PushCubeAgent`         | Multiple trials per episode — retries pushes with different velocities in binary-search order until the cube reaches the target | `push_vels_m_per_s` injected by `PushCube.reset()`           |
| `FlingClothAgent`       | Multiple trials per episode — repeats flings with different speed scales, controls two robots in sync | `speed_scales` read from `episode_config` in agent `reset()` (set by `_process_episode_config`; only needed for heuristic collection — a learned policy figures out the right speed itself from observations) |
| `PickAndPlaceBackAgent` | Single pass per episode — wait → pregrasp → descend → grasp → lift → keyframe → place → release → return. Optionally inserts a randomized `drift` phase before placing to generate diverse/erroneous trajectories for richer training data | No injected param; randomizes pose candidates on the fly     |

### 4. Register the task

The task name is hardcoded in several places. Add `<task_name>` to every one of them — missing any will cause a runtime `AssertionError` or `ValueError` in the corresponding entry point.

**Required for all tasks** — add `<task_name>` to the assert list in:

- `scripts/run_heuristic_agent.py`
- `scripts/rollout_policy.py`
- `scripts/rollout_policy_parallel.py`

**Required if you defined a `MyTaskParallel` class (Step 1c)** — add a new `elif task_cfg.name == "<task_name>":` branch to:

- `env/utils/config_utils.py::convert_task_to_parallel` — maps the task name to the parallel class's `_target_`. Without this, `rollout_policy_parallel.py` and `serve_remote_env.py` will raise `ValueError(f"Task {task_cfg.name} is not supported")` even though the class exists.

**Required if you want to use spacemouse teleop for this task** — add `<task_name>` to the assert list in:

- `scripts/run_spacemouse_teleop.py`

**Required if you want to use the remote env server (multi-checkpoint eval)** — add `<task_name>` to the assert list in:

- `env/modules/remote_env_server.py`

### 5. Rollout config _(optional)_

Create `config/rollout_policy_<task_name>.yaml` to customize rollout settings. If omitted, falls back to `config/rollout_policy.yaml`.
