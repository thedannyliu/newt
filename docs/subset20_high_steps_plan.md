# Subset20 High-Step Newt Experiment Plan

Date: 2026-06-09

## Goal

The next experiment should test whether flow-based world models are genuinely weak, or whether the previous 40-task/5M setting was too sparse per task. The main change is to reduce the task count and increase online training steps while keeping the original Newt training pipeline intact.

Primary question:

- If each task receives more online interaction, can residual flow world models close the gap to the MLP world model?

## Strict Pipeline Constraint

These runs must follow the original Newt pipeline except for the explicit architecture cells and task subset:

- Entry point remains `tdmpc2/train.py`.
- Task mode remains `task=soup`.
- Observation mode remains `obs=state`.
- Model size remains `model_size=L`.
- Demonstrations are loaded from `data_dir/{task}.pt`.
- Offline pretraining remains enabled via `demo_steps=50000`.
- Online RL remains real environment interaction, not model-generated rollout data.
- Replay remains the existing Newt `EnsembleBuffer` behavior with demonstration and online replay mixing.
- Planning, value learning, reward prediction, policy update, MPPI/CEM behavior, and evaluation logic remain the existing code path.
- Checkpointing remains local under `outputs/logs/soup/{seed}/{run_name}/models`.
- Resume uses the latest full checkpoint when available, otherwise the latest non-full checkpoint.
- W&B logging remains enabled, and resumes use the same `wandb_run_id`.

The only intended differences from the original baseline are:

- `task_subset_file=configs/task_subsets/mmbench_balanced_20.json`
- `steps=10000000`
- `dynamics_arch` / `policy_arch` according to the comparison cells below.

## Task Subset

The 20 tasks are selected as two representative tasks from each block of the prior 40-task subset:

| Domain block | Tasks |
| --- | --- |
| DMControl | `walker-walk`, `cup-catch` |
| DMControl-ext | `walker-run-backward`, `reacher-three-easy` |
| MetaWorld | `mw-door-open`, `mw-drawer-close` |
| ManiSkill | `ms-pick-cube`, `ms-reach` |
| MuJoCo | `mujoco-ant`, `mujoco-hopper` |
| Box2D | `bipedal-walker-flat`, `lunarlander-hover` |
| RoboDesk | `rd-push-red`, `rd-open-drawer` |
| OGBench | `og-point-maze`, `og-ant-circle` |
| PyGame | `pygame-coinrun`, `pygame-pong` |
| Atari | `atari-assault`, `atari-seaquest` |

Config:

- `configs/task_subsets/mmbench_balanced_20.json`

## Comparison Cells

Run the minimum set needed to test the hypothesis:

| Array | World model | Policy | Reason |
| ---: | --- | --- | --- |
| 0 | `mlp` | `gaussian` | Original-style MLP WM baseline. |
| 1 | `mlp` | `flow` | Best 40-task/5M cell so far. |
| 2 | `residual_flow` | `gaussian` | Best completed flow-WM family so far. |
| 3 | `residual_mean_flow_wm` | `gaussian` | Best new residual-flow variant so far. |

Do not include pure `flow` WM in this first high-step pass because prior 5M results were consistently weak.

## Initial Run Spec

Script:

- `scripts/slurm_flow_subset20_10m.sbatch`

Defaults:

- `steps=10000000`
- `demo_steps=50000`
- `batch_size=1024`
- `buffer_size=5000000`
- `checkpoint_freq=250000`
- `replay_checkpoint_freq=1000000`
- `eval_episodes=1`
- `qos=embers`

The first submissions should run the same four cells across suitable GPU partitions with distinct `RUN_TAG`s so outputs and W&B runs cannot race.

## Success Criteria

Primary metric:

- 20-task eval `avg_score` at 10M.

Secondary metrics:

- `train/episode_score`
- `train/episode_success`
- `eval/avg_score` by domain
- `steps_per_second`
- `action_time`
- `update_time`
- `num_updates`
- checkpoint continuity after preemption

Decision rule:

- If `residual_flow` or `residual_mean_flow_wm` approaches or beats the MLP WM baselines at 10M, extend the strongest flow-WM candidate to 20M and add seeds.
- If flow-WM cells remain clearly below MLP WM, prioritize MLP WM + flow policy and residual-flow ablations over additional new flow variants.
