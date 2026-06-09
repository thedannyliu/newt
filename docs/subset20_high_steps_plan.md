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

## Initial Submissions

Submitted on 2026-06-09 with `qos=embers`, W&B entity `danny010324`, project `newt-flow-2x2`, group `subset20-10m-highsteps`, and `REPLAY_CHECKPOINT_FREQ=500000`.

| Job | Partition | Array | Run tag | Batch size | Status at submission check |
| ---: | --- | --- | --- | ---: | --- |
| `9703778` | `gpu-h200` | `0-3` | `h200-r1` | 1024 | pending, `Priority` |
| `9703779` | `gpu-h100` | `0-3` | `h100-r1` | 1024 | pending, `Priority` |
| `9703780` | `gpu-a100` | `0-3` | `a100-r1` | 1024 | pending, `Priority` |
| `9703781` | `gpu-l40s` | `0-3` | `l40s-r1` | 512 | pending, `Priority` |

Array mapping:

| Array | World model | Policy |
| ---: | --- | --- |
| 0 | `mlp` | `gaussian` |
| 1 | `mlp` | `flow` |
| 2 | `residual_flow` | `gaussian` |
| 3 | `residual_mean_flow_wm` | `gaussian` |

Submission validation before queuing:

- `python -m json.tool configs/task_subsets/mmbench_balanced_20.json`
- `bash -n scripts/slurm_flow_subset20_10m.sbatch`
- Demo file existence check for all 20 tasks.

Startup check:

- A100 job `9703780_0` started first.
- It loaded the intended 20 tasks and 440 demonstration episodes.
- It entered the original Newt demonstration pretraining path with `demo_steps=50000`.
- The run reports `Steps: 10,000,000`, `World size: 1`, and run name `subset20-10m-a100-r1_wm-mlp_pi-gaussian_seed-1`.
- No Python traceback, missing data error, or CUDA OOM appeared at startup.

## 2026-06-09 Monitoring and Continuation

At the 2026-06-09 14:29 EDT check, the initial H200/H100/A100 jobs had been interrupted by `embers` preemption or time limit. The L40S jobs were still running.

No Python traceback, missing data error, or CUDA OOM was found in the checked stderr logs. Interruptions were Slurm preemption or time-limit events only.

Checkpoint progress:

| Run | Latest checkpoint | Latest full checkpoint | Status |
| --- | ---: | ---: | --- |
| H200 `mlp + gaussian` | 1.25M | 1.00M | preempted; resubmitted |
| H200 `mlp + flow` | 1.00M | 1.00M | preempted; resubmitted |
| H200 `residual_flow + gaussian` | 0.75M | 0.50M | preempted; resubmitted |
| H200 `residual_mean_flow_wm + gaussian` | 1.00M | 1.00M | preempted; resubmitted |
| H100 `mlp + gaussian` | 4.25M | 4.00M | timed out; resubmitted |
| H100 `mlp + flow` | 3.00M | 3.00M | timed out; resubmitted |
| H100 `residual_flow + gaussian` | 1.50M | 1.50M | preempted; resubmitted |
| H100 `residual_mean_flow_wm + gaussian` | 4.00M | 4.00M | preempted; resubmitted |
| A100 `mlp + gaussian` | 3.75M | 3.50M | timed out; resubmitted |
| A100 `mlp + flow` | 2.25M | 2.00M | timed out; resubmitted |
| A100 `residual_flow + gaussian` | 1.75M | 1.50M | preempted; resubmitted |
| A100 `residual_mean_flow_wm + gaussian` | 2.25M | 2.00M | preempted; resubmitted |
| L40S `mlp + gaussian` | 2.50M | 2.50M | still running |
| L40S `mlp + flow` | 1.50M | 1.50M | still running |
| L40S `residual_flow + gaussian` | 1.25M | 1.00M | still running |
| L40S `residual_mean_flow_wm + gaussian` | 1.00M | 1.00M | still running |

Continuation submissions:

| Job | Partition | Array | Run tag | Status at submission check |
| ---: | --- | --- | --- | --- |
| `9747574` | `gpu-h200` | `0-3` | `h200-r1` | pending, `Priority` |
| `9747575` | `gpu-h100` | `0-3` | `h100-r1` | pending, `Priority` |
| `9747577` | `gpu-a100` | `0-3` | `a100-r1` | pending, `Priority` |

The L40S initial jobs were left running and were not duplicated.

## 2026-06-09 16:57 EDT Monitoring

Current subset20 status:

- H200 continuation `9747574_[0-3]` is running all four cells.
- H100 continuation `9747575_0` and `9747575_1` are running; `9747575_[2-3]` remains pending.
- A100 continuation `9747577_[0-3]` remains pending.
- Initial L40S `9703781_1` and `9703781_2` are still running.
- Initial L40S `9703781_0` and `9703781_3` were preempted.

Latest checkpoint progress:

| Run | Latest checkpoint | Latest full checkpoint |
| --- | ---: | ---: |
| H200 `mlp + gaussian` | 2.50M | 2.50M |
| H200 `mlp + flow` | 2.00M | 2.00M |
| H200 `residual_flow + gaussian` | 1.25M | 1.00M |
| H200 `residual_mean_flow_wm + gaussian` | 2.00M | 2.00M |
| H100 `mlp + gaussian` | 4.25M | 4.00M |
| H100 `mlp + flow` | 3.00M | 3.00M |
| H100 `residual_flow + gaussian` | 1.50M | 1.50M |
| H100 `residual_mean_flow_wm + gaussian` | 4.00M | 4.00M |
| A100 `mlp + gaussian` | 3.75M | 3.50M |
| A100 `mlp + flow` | 2.25M | 2.00M |
| A100 `residual_flow + gaussian` | 1.75M | 1.50M |
| A100 `residual_mean_flow_wm + gaussian` | 2.25M | 2.00M |
| L40S `mlp + gaussian` | 3.00M | 3.00M |
| L40S `mlp + flow` | 2.50M | 2.50M |
| L40S `residual_flow + gaussian` | 1.75M | 1.50M |
| L40S `residual_mean_flow_wm + gaussian` | 1.25M | 1.00M |

No Python traceback, missing data error, or CUDA OOM was found in the checked stderr logs.

## 2026-06-09 17:30 EDT Quick Result Check

The training-internal eval rows already provide a useful early answer before the 10M target is reached.

Latest subset20 eval snapshot:

| Run | Eval step | `avg_score` | Interpretation |
| --- | ---: | ---: | --- |
| H200 `mlp + flow` | 2.4M | 0.49539 | Strong early cell. |
| H200 `mlp + gaussian` | 2.8M | 0.48470 | Strong original-style baseline. |
| H200 `residual_flow + gaussian` | 1.6M | 0.29604 | Still behind MLP WM. |
| H200 `residual_mean_flow_wm + gaussian` | 2.2M | 0.35676 | Better than plain residual flow here, but still behind MLP WM. |
| H100 `mlp + flow` | 3.2M | 0.46716 | Strong early cell. |
| H100 `mlp + gaussian` | 4.4M | 0.46468 | Strong original-style baseline. |
| H100 `residual_flow + gaussian` | 1.6M | 0.34264 | Best plain residual-flow snapshot so far. |
| H100 `residual_mean_flow_wm + gaussian` | 4.0M | 0.37050 | Improved over 40-task new-flow eval, but still below MLP WM. |
| A100 `mlp + flow` | 2.4M | 0.52114 | Best current subset20 eval. |
| A100 `mlp + gaussian` | 3.8M | 0.43019 | Strong but noisy. |
| A100 `residual_flow + gaussian` | 1.8M | 0.27400 | Still behind MLP WM. |
| A100 `residual_mean_flow_wm + gaussian` | 2.4M | 0.35244 | Better than 40-task new-flow eval, still below MLP WM. |
| L40S `mlp + flow` | 2.6M | 0.45579 | Strong early cell. |
| L40S `mlp + gaussian` | 2.8M | 0.42660 | Strong original-style baseline. |
| L40S `residual_flow + gaussian` | 2.0M | 0.29991 | Still behind MLP WM. |
| L40S `residual_mean_flow_wm + gaussian` | 1.4M | 0.33967 | Better than early residual flow, still below MLP WM. |

Early conclusion:

- The reduced-task, higher-step direction matches the original hypothesis: the same pipeline gets much stronger learning signal than the 40-task/5M setting, reaching roughly 0.43-0.52 for MLP WM cells before 5M.
- The architecture ranking has not flipped: MLP WM remains clearly stronger than flow WM variants.
- `residual_mean_flow_wm` is the most promising flow-WM cell in this subset20 setting, but the gap to MLP WM is still large enough that it should be treated as a secondary candidate, not the main path.
- `mlp + flow policy` remains competitive or best on subset20, but its value needs to be weighed against action-time overhead.

Operational action:

- Submitted L40S continuation job `9760035_[0,1,3]` for the preempted subset20 cells only. Array `2` was still running, so it was not duplicated.

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
