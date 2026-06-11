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
- After L40S array `9703781_2` reached the 8-hour `embers` limit and was marked preempted, submitted array `2` continuation as job `9760175_[2]`.

## 2026-06-09 20:06 EDT Monitoring

Current queue state:

- H200 continuation `9747574_[0-3]` is running all four cells.
- H100 continuation `9747575_[0-3]` is running all four cells.
- A100 continuation `9747577_[0-3]` is running all four cells.
- L40S continuation `9760035_[0,1,3]` and `9760175_[2]` are running all four cells.
- No Python traceback, missing data error, or CUDA OOM was found in the checked logs; interruptions remain Slurm preemption/time-limit only.

Latest subset20 snapshot:

| Run | Train step | Latest eval step | Latest eval `avg_score` | Notes |
| --- | ---: | ---: | ---: | --- |
| H100 `mlp + flow` | 4.38M | 4.20M | 0.52156 | Current strongest eval. |
| H100 `mlp + gaussian` | 5.86M | 5.80M | 0.51572 | Strongest original-style baseline. |
| H100 `residual_flow + gaussian` | 2.56M | 2.40M | 0.42300 | Best flow-WM snapshot so far, but at fewer steps. |
| H100 `residual_mean_flow_wm + gaussian` | 5.34M | 5.20M | 0.34056 | Not closing the MLP gap at comparable steps. |
| H200 `mlp + flow` | 3.90M | 3.80M | 0.41414 | Below H100/A100 replicate so far. |
| H200 `mlp + gaussian` | 5.00M | 5.00M | 0.42347 | Continuing toward 10M. |
| H200 `residual_flow + gaussian` | 2.90M | 2.80M | 0.41988 | Competitive with H200 MLP eval at this early checkpoint. |
| H200 `residual_mean_flow_wm + gaussian` | 4.00M | 4.00M | 0.32699 | Behind MLP/residual_flow. |
| A100 `mlp + flow` | 2.40M | 2.40M | 0.47109 | Strong early cell. |
| A100 `mlp + gaussian` | 4.42M | 4.40M | 0.45273 | Strong baseline. |
| A100 `residual_flow + gaussian` | 1.82M | 1.80M | 0.31001 | Still early. |
| A100 `residual_mean_flow_wm + gaussian` | 2.30M | 2.20M | 0.31782 | Still early and behind MLP. |
| L40S `mlp + flow` | 3.40M | 3.20M | 0.47116 | Strong early cell. |
| L40S `mlp + gaussian` | 4.28M | 4.20M | 0.44272 | Strong baseline. |
| L40S `residual_flow + gaussian` | 2.52M | 2.40M | 0.31528 | Behind MLP. |
| L40S `residual_mean_flow_wm + gaussian` | 1.98M | 1.80M | 0.28558 | Behind MLP. |

Current read:

- The subset20 high-step run continues to support the reduced-task/high-step hypothesis: MLP-WM cells are already around 0.44-0.52 before 10M.
- Flow-WM ranking is mixed: `residual_flow` has the best subset20 flow-WM snapshot so far, while `residual_mean_flow_wm` is not improving enough at comparable steps.
- The main conclusion remains unchanged until 10M evals: keep running, but MLP WM is still the reference path to beat.

## 2026-06-09 21:28 EDT Monitoring and Continuation

New Slurm interruptions:

- H100 array `9747575_3` was preempted; this is `residual_mean_flow_wm + gaussian`.
- A100 array `9747577_3` was preempted; this is `residual_mean_flow_wm + gaussian`.
- L40S array `9760035_0` was preempted; this is `mlp + gaussian`.
- Checked stderr indicates Slurm preemption only; no Python traceback, missing data error, or CUDA OOM was found for these interruptions.

Repair actions:

| New job | Partition | Array | Run tag | Cell |
| ---: | --- | ---: | --- | --- |
| `9768054` | `gpu-h100` | `3` | `h100-r1` | `residual_mean_flow_wm + gaussian` |
| `9768055` | `gpu-a100` | `3` | `a100-r1` | `residual_mean_flow_wm + gaussian` |
| `9768056` | `gpu-l40s` | `0` | `l40s-r1` | `mlp + gaussian` |

Latest subset20 eval snapshot:

| Run | Latest eval step | Latest eval `avg_score` |
| --- | ---: | ---: |
| H200 `mlp + flow` | 4.6M | 0.54616 |
| H100 `mlp + flow` | 4.8M | 0.54557 |
| L40S `mlp + flow` | 3.8M | 0.52275 |
| A100 `mlp + flow` | 2.8M | 0.49973 |
| H100 `mlp + gaussian` | 6.6M | 0.45537 |
| L40S `mlp + gaussian` | 4.4M | 0.45497 |
| A100 `mlp + gaussian` | 5.0M | 0.44205 |
| H200 `mlp + gaussian` | 6.0M | 0.42972 |
| H200 `residual_mean_flow_wm + gaussian` | 4.8M | 0.40596 |
| H100 `residual_flow + gaussian` | 3.0M | 0.39700 |
| H100 `residual_mean_flow_wm + gaussian` | 5.6M | 0.34291 |
| A100 `residual_flow + gaussian` | 2.0M | 0.33476 |
| A100 `residual_mean_flow_wm + gaussian` | 2.6M | 0.33399 |
| H200 `residual_flow + gaussian` | 3.4M | 0.32833 |
| L40S `residual_flow + gaussian` | 2.8M | 0.32219 |
| L40S `residual_mean_flow_wm + gaussian` | 2.4M | 0.30593 |

Current read:

- `mlp + flow` is the strongest subset20 cell so far across H200/H100/L40S.
- `mlp + gaussian` remains the strongest original-style baseline.
- Flow-WM cells have useful signal but are still below the top MLP-WM cells; no architecture flip yet.

## 2026-06-10 03:00 EDT Monitoring and Continuation

Current active jobs before repair:

- Running:
  - `9768054_3`: H100 `residual_mean_flow_wm + gaussian`
  - `9768055_3`: A100 `residual_mean_flow_wm + gaussian`
  - `9768056_0`: L40S `mlp + gaussian`
- Stopped by `embers` preemption or 8-hour walltime:
  - H200 `9747574_[0-3]`
  - H100 `9747575_[0-2]`
  - A100 `9747577_[0-2]`
  - L40S `9760035_[1,3]` and `9760175_2`

Checked logs indicate Slurm preemption/time-limit only for the newly stopped jobs; no Python traceback, missing data error, or CUDA OOM was found.

Latest checkpoint/eval progress:

| Run | Latest train step | Latest eval step | Latest eval `avg_score` | Latest checkpoint | Latest full checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| A100 `mlp + flow` | 4.66M | 4.60M | 0.48211 | 4.50M | 4.50M |
| A100 `mlp + gaussian` | 5.66M | 5.60M | 0.49391 | 5.50M | 5.50M |
| A100 `residual_flow + gaussian` | 3.64M | 3.60M | 0.36821 | 3.50M | 3.50M |
| A100 `residual_mean_flow_wm + gaussian` | 4.12M | 4.00M | 0.39958 | 4.00M | 4.00M |
| H100 `mlp + flow` | 6.44M | 6.40M | 0.51718 | 6.25M | 6.00M |
| H100 `mlp + gaussian` | 8.46M | 8.40M | 0.58430 | 8.25M | 8.00M |
| H100 `residual_flow + gaussian` | 4.40M | 4.40M | 0.38287 | 4.25M | 4.00M |
| H100 `residual_mean_flow_wm + gaussian` | 8.04M | 8.00M | 0.38443 | 8.00M | 8.00M |
| H200 `mlp + flow` | 5.50M | 5.40M | 0.48872 | 5.50M | 5.50M |
| H200 `mlp + gaussian` | 7.20M | 7.20M | 0.51096 | 7.00M | 7.00M |
| H200 `residual_flow + gaussian` | 3.86M | 3.80M | 0.33989 | 3.75M | 3.50M |
| H200 `residual_mean_flow_wm + gaussian` | 6.00M | 6.00M | 0.39679 | 6.00M | 6.00M |
| L40S `mlp + flow` | 5.34M | 5.20M | 0.52713 | 5.25M | 5.00M |
| L40S `mlp + gaussian` | 6.78M | 6.60M | 0.51172 | 6.75M | 6.50M |
| L40S `residual_flow + gaussian` | 4.14M | 4.00M | 0.41860 | 4.00M | 4.00M |
| L40S `residual_mean_flow_wm + gaussian` | 4.18M | 4.00M | 0.35884 | 4.00M | 4.00M |

Repair submissions:

| Job | Partition | Array | Run tag | Cells |
| ---: | --- | --- | --- | --- |
| `9779356` | `gpu-h200` | `0-3` | `h200-r1` | all H200 cells |
| `9779357` | `gpu-h100` | `0-2` | `h100-r1` | inactive H100 cells; array `3` already running |
| `9779360` | `gpu-a100` | `0-2` | `a100-r1` | inactive A100 cells; array `3` already running |
| `9779361` | `gpu-l40s` | `1-3` | `l40s-r1` | inactive L40S cells; array `0` already running |

Queue after repair:

- `9768054_3`, `9768055_3`, and `9768056_0` remain running.
- `9779356_[0-3]`, `9779357_[0-2]`, `9779360_[0-2]`, and `9779361_[1-3]` are pending with reason `None`.

Current read:

- H100 `mlp + gaussian` has the best current subset20 eval at `0.58430` around 8.4M.
- `mlp + flow` remains strong across GPU types, especially L40S/H100.
- Flow-WM cells still trail MLP-WM cells; no evidence yet that longer subset20 training flips the architecture ranking.

## 2026-06-10 04:11 EDT Monitoring and Continuation

Follow-up status:

- Running:
  - `9768054_3`: H100 `residual_mean_flow_wm + gaussian`
  - `9768055_3`: A100 `residual_mean_flow_wm + gaussian`
  - `9779357_[0-2]`: H100 inactive-cell continuations
- Pending:
  - `9779356_[0-3]`: H200 all-cell continuation
  - `9779360_[0-2]`: A100 inactive-cell continuation
  - `9779361_[1-3]`: L40S inactive-cell continuation
- Newly stopped:
  - `9768056_0`: L40S `mlp + gaussian`, preempted after reaching 7.22M train / 7.20M eval.

Repair action:

- Submitted L40S `mlp + gaussian` continuation as `9781549_[0]`.

Metrics note:

- Some resumed runs append lower-step rows after loading the latest full checkpoint, so monitoring should compare the highest eval step per run, not simply the last row in `metrics.jsonl`.

Highest-step eval snapshot:

| Run | Highest eval step | `avg_score` |
| --- | ---: | ---: |
| H100 `mlp + gaussian` | 8.4M | 0.58430 |
| L40S `mlp + flow` | 5.2M | 0.52713 |
| L40S `mlp + gaussian` | 7.2M | 0.52470 |
| H100 `mlp + flow` | 6.4M | 0.51718 |
| H200 `mlp + gaussian` | 7.2M | 0.51096 |
| A100 `mlp + gaussian` | 5.6M | 0.49391 |
| H200 `mlp + flow` | 5.4M | 0.48872 |
| A100 `mlp + flow` | 4.6M | 0.48211 |
| H100 `residual_mean_flow_wm + gaussian` | 8.6M | 0.45783 |
| A100 `residual_mean_flow_wm + gaussian` | 4.4M | 0.42541 |
| L40S `residual_flow + gaussian` | 4.0M | 0.41860 |
| H200 `residual_mean_flow_wm + gaussian` | 6.0M | 0.39679 |
| H100 `residual_flow + gaussian` | 4.4M | 0.38287 |
| A100 `residual_flow + gaussian` | 3.6M | 0.36821 |
| L40S `residual_mean_flow_wm + gaussian` | 4.0M | 0.35884 |
| H200 `residual_flow + gaussian` | 3.8M | 0.33989 |

Current read:

- MLP-WM cells still dominate the subset20 run.
- The best flow-WM signal is now H100 `residual_mean_flow_wm + gaussian` at `0.45783`, but it remains below the MLP-WM cells at comparable or higher eval steps.

## 2026-06-10 14:10 EDT Monitoring and Continuation

Current queue before repair:

- Running:
  - `9779356_3`: H200 `residual_mean_flow_wm + gaussian`
  - `9781549_0`: L40S `mlp + gaussian`
  - `9779361_[1-3]`: L40S `mlp + flow`, `residual_flow + gaussian`, and `residual_mean_flow_wm + gaussian`
- Stopped:
  - H200 arrays `0-2` were preempted.
  - H100 arrays `0-3` stopped; array `0` reached the 10M target and should not be resubmitted.
  - A100 arrays `0-3` were preempted or timed out.

Checked recent stderr logs show Slurm preemption/time-limit only; no Python traceback, missing data error, or CUDA OOM was found.

Highest-step eval snapshot:

| Run | Highest train step | Highest eval step | `avg_score` | Status |
| --- | ---: | ---: | ---: | --- |
| H100 `mlp + gaussian` | 10.0M | 10.0M | 0.53816 | complete |
| H200 `mlp + flow` | 9.06M | 9.0M | 0.58255 | resubmitted |
| H100 `mlp + flow` | 7.8M | 7.8M | 0.57361 | resubmitted |
| A100 `mlp + flow` | 6.5M | 6.4M | 0.55556 | resubmitted |
| L40S `mlp + flow` | 7.54M | 7.4M | 0.54459 | running |
| L40S `mlp + gaussian` | 9.90M | 9.8M | 0.48712 | running, near target |
| H200 `mlp + gaussian` | 8.54M | 8.4M | 0.45496 | resubmitted |
| A100 `mlp + gaussian` | 6.5M | 6.4M | 0.45485 | resubmitted |
| H100 `residual_mean_flow_wm + gaussian` | 9.22M | 9.2M | 0.43459 | resubmitted |
| L40S `residual_flow + gaussian` | 5.48M | 5.4M | 0.42921 | running |
| A100 `residual_mean_flow_wm + gaussian` | 5.62M | 5.6M | 0.36085 | resubmitted |
| A100 `residual_flow + gaussian` | 4.94M | 4.8M | 0.41356 | resubmitted |
| H200 `residual_flow + gaussian` | 6.12M | 6.0M | 0.40036 | resubmitted |
| H200 `residual_mean_flow_wm + gaussian` | 7.10M | 7.0M | 0.39461 | running |
| L40S `residual_mean_flow_wm + gaussian` | 6.18M | 6.0M | 0.32281 | running |
| H100 `residual_flow + gaussian` | 5.74M | 5.6M | 0.43162 | resubmitted |

Repair submissions:

| Job | Partition | Array | Run tag | Cells |
| ---: | --- | --- | --- | --- |
| `9796333` | `gpu-h200` | `0-2` | `h200-r1` | inactive H200 cells; array `3` still running |
| `9796334` | `gpu-h100` | `1-3` | `h100-r1` | inactive incomplete H100 cells; array `0` already reached 10M |
| `9796335` | `gpu-a100` | `0-3` | `a100-r1` | all inactive A100 cells |

Queue after repair:

- Running:
  - `9779356_3`: H200 `residual_mean_flow_wm + gaussian`
  - `9781549_0`: L40S `mlp + gaussian`
  - `9779361_1`: L40S `mlp + flow`
  - `9779361_2`: L40S `residual_flow + gaussian`
  - `9779361_3`: L40S `residual_mean_flow_wm + gaussian`
  - `9796335_0`: A100 `mlp + gaussian`
- Pending:
  - `9796333_[0-2]`: H200 incomplete cells
  - `9796334_[1-3]`: H100 incomplete cells
  - `9796335_[1-3]`: A100 incomplete cells

Current read:

- First 10M cell is complete: H100 `mlp + gaussian` reached `avg_score=0.53816`.
- Peak observed subset20 scores are still MLP-WM cells, led by H200/H100/A100 `mlp + flow`.
- Flow-WM cells improved with longer training but remain below the strongest MLP-WM cells.
- L40S `mlp + gaussian` is very close to the 10M target; if it times out before the final checkpoint, resume from its latest full checkpoint.

## 2026-06-10 15:04 EDT Monitoring and Continuation

Queue check:

- Running:
  - `9796335_0`: A100 `mlp + gaussian`
  - `9779361_2`: L40S `residual_flow + gaussian`
  - `9779361_3`: L40S `residual_mean_flow_wm + gaussian`
- Pending:
  - `9796333_[0-2]`: H200 incomplete cells
  - `9796334_[1-3]`: H100 incomplete cells
  - `9796335_[1-3]`: A100 incomplete cells

Stopped since the previous check:

- `9779356_3`: H200 `residual_mean_flow_wm + gaussian`, preempted after reaching 7.24M train / 7.20M eval.
- `9779361_1`: L40S `mlp + flow`, preempted after reaching 7.78M train / 7.60M eval.
- `9781549_0`: L40S `mlp + gaussian`, marked preempted in `sacct` but stdout shows `Training completed successfully`; it reached 10.0M train / 10.0M eval and should not be resubmitted.

Latest highest-step eval snapshot:

| Run | Highest train step | Highest eval step | `avg_score` | Checkpoint | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| H100 `mlp + gaussian` | 10.0M | 10.0M | 0.53816 | 10.0M full | complete |
| L40S `mlp + gaussian` | 10.0M | 10.0M | 0.48801 | 10.0M full | complete |
| H200 `mlp + flow` | 9.06M | 9.0M | 0.58255 | 9.0M full | pending continuation |
| H100 `mlp + flow` | 7.8M | 7.8M | 0.57361 | 7.5M full | pending continuation |
| L40S `mlp + flow` | 7.78M | 7.6M | 0.58263 | 7.5M full | resubmitted |
| A100 `mlp + flow` | 6.5M | 6.4M | 0.55556 | 6.5M full | pending continuation |
| A100 `mlp + gaussian` | 6.88M | 6.8M | 0.48545 | 6.5M full | running |
| H200 `residual_mean_flow_wm + gaussian` | 7.24M | 7.2M | 0.44473 | 7.0M full | resubmitted |
| H100 `residual_mean_flow_wm + gaussian` | 9.22M | 9.2M | 0.43459 | 9.0M full | pending continuation |
| L40S `residual_flow + gaussian` | 5.74M | 5.6M | 0.39965 | 5.5M full | running |
| L40S `residual_mean_flow_wm + gaussian` | 6.5M | 6.4M | 0.34956 | 6.5M full | running |

Repair submissions:

| Job | Partition | Array | Run tag | Cell |
| ---: | --- | --- | --- | --- |
| `9797383` | `gpu-h200` | `3` | `h200-r1` | `residual_mean_flow_wm + gaussian` |
| `9797385` | `gpu-l40s` | `1` | `l40s-r1` | `mlp + flow` |

Current read:

- The second 10M cell is complete: L40S `mlp + gaussian` reached `avg_score=0.48801`.
- L40S `mlp + flow` reached the strongest latest-step local score so far at `0.58263` at 7.6M eval, but still needs continuation to 10M.
- Flow-WM cells remain behind MLP-WM cells; H200 `residual_mean_flow_wm + gaussian` improved to `0.44473` at 7.2M but does not close the gap.

## 2026-06-10 17:04 EDT Monitoring and Continuation

Queue check:

- Running:
  - `9796335_0`: A100 `mlp + gaussian`
  - `9796334_1`: H100 `mlp + flow`
  - `9796334_2`: H100 `residual_flow + gaussian`
  - `9796334_3`: H100 `residual_mean_flow_wm + gaussian`
- Pending:
  - `9796333_[0-2]`: H200 `mlp + gaussian`, `mlp + flow`, and `residual_flow + gaussian`
  - `9797383_[3]`: H200 `residual_mean_flow_wm + gaussian`
  - `9796335_[1-3]`: A100 `mlp + flow`, `residual_flow + gaussian`, and `residual_mean_flow_wm + gaussian`
  - `9797385_[1]`: L40S `mlp + flow`

Stopped since the previous check:

- `9779361_2`: L40S `residual_flow + gaussian`, preempted after reaching 5.90M train / 5.80M eval.
- `9779361_3`: L40S `residual_mean_flow_wm + gaussian`, preempted after reaching 6.74M train / 6.60M eval.

Latest highest-step eval snapshot:

| Run | Highest train step | Highest eval step | `avg_score` | Checkpoint | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| H100 `mlp + gaussian` | 10.0M | 10.0M | 0.53816 | 10.0M full | complete |
| L40S `mlp + gaussian` | 10.0M | 10.0M | 0.48801 | 10.0M full | complete |
| H200 `mlp + flow` | 9.06M | 9.0M | 0.58255 | 9.0M full | pending continuation |
| L40S `mlp + flow` | 7.78M | 7.6M | 0.58263 | 7.5M full | pending continuation |
| H100 `mlp + flow` | 8.42M | 8.4M | 0.54411 | 8.0M full | running |
| A100 `mlp + flow` | 6.5M | 6.4M | 0.55556 | 6.5M full | pending continuation |
| A100 `mlp + gaussian` | 7.78M | 7.6M | 0.45290 | 7.5M full | running |
| H100 `residual_flow + gaussian` | 6.28M | 6.2M | 0.40176 | 6.0M full | running |
| H100 `residual_mean_flow_wm + gaussian` | 9.84M | 9.8M | 0.42021 | 9.5M full | running |
| H200 `residual_mean_flow_wm + gaussian` | 7.24M | 7.2M | 0.44473 | 7.0M full | pending continuation |
| L40S `residual_flow + gaussian` | 5.90M | 5.8M | 0.46335 | 5.5M full | resubmitted |
| L40S `residual_mean_flow_wm + gaussian` | 6.74M | 6.6M | 0.37317 | 6.5M full | resubmitted |

Repair submission:

| Job | Partition | Array | Run tag | Cells |
| ---: | --- | --- | --- | --- |
| `9798924` | `gpu-l40s` | `2-3` | `l40s-r1` | `residual_flow + gaussian`, `residual_mean_flow_wm + gaussian` |

Current read:

- H100 continuations are making progress; `mlp + flow` remains strong but its latest-step score dipped to `0.54411` while the peak remains `0.58825`.
- L40S `residual_flow + gaussian` improved to `0.46335`, making it the strongest current flow-WM latest-step L40S result, but it is still below MLP-WM latest/peak scores.
- No new code/data failure was found; stopped L40S jobs were preemptions and were resubmitted from checkpoints.

## 2026-06-10 20:30 EDT Monitoring and Continuation

Queue check:

- Running:
  - `9796335_[0-3]`: A100 all four subset20 cells.
  - `9799983_1`: H100 `mlp + flow`.
  - `9803431_2`: H100 `residual_flow + gaussian`.
  - `9796333_0`: H200 `mlp + gaussian`.
  - `9797383_3`: H200 `residual_mean_flow_wm + gaussian`.
- Pending:
  - `9796333_[1-2]`: H200 `mlp + flow` and `residual_flow + gaussian`.
  - `9798924_[2-3]`: L40S flow-WM continuations.
  - `9797385_1`: L40S `mlp + flow`.

Checked stderr logs show no Python traceback, missing data error, or CUDA OOM. The only notable warnings are W&B resume messages where repeated lower-step logs are ignored until the resumed run passes the existing W&B step. This is expected when resuming from the latest full checkpoint behind the latest logged local step; local `metrics.jsonl` remains the source of truth for monitoring.

Latest highest-step eval snapshot:

| Run | Highest train step | Highest eval step | `avg_score` | Peak `avg_score` | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| H100 `mlp + gaussian` | 10.0M | 10.0M | 0.53816 | 0.59102 | complete |
| L40S `mlp + gaussian` | 10.0M | 10.0M | 0.48801 | 0.55139 | complete |
| H100 `mlp + flow` | 9.60M | 9.6M | 0.53309 | 0.58825 | running |
| H200 `mlp + flow` | 9.06M | 9.0M | 0.58255 | 0.58663 | pending continuation |
| L40S `mlp + flow` | 7.78M | 7.6M | 0.58263 | 0.58263 | pending continuation |
| A100 `mlp + flow` | 7.54M | 7.4M | 0.53330 | 0.56850 | running |
| A100 `mlp + gaussian` | 9.32M | 9.2M | 0.53724 | 0.53724 | running |
| H200 `mlp + gaussian` | 9.50M | 9.4M | 0.43385 | 0.53666 | running |
| H100 `residual_flow + gaussian` | 7.52M | 7.4M | 0.46682 | 0.51065 | running |
| H200 `residual_mean_flow_wm + gaussian` | 7.36M | 7.2M | 0.40268 | 0.45594 | running |
| A100 `residual_mean_flow_wm + gaussian` | 6.08M | 6.0M | 0.39683 | 0.42541 | running |
| A100 `residual_flow + gaussian` | 4.94M | 4.8M | 0.42288 | 0.42288 | running |
| L40S `residual_flow + gaussian` | 5.90M | 5.8M | 0.46335 | 0.46335 | pending continuation |
| L40S `residual_mean_flow_wm + gaussian` | 6.74M | 6.6M | 0.37317 | 0.37962 | pending continuation |
| H200 `residual_flow + gaussian` | 6.12M | 6.0M | 0.40036 | 0.47019 | pending continuation |

Current read:

- No repair submission was needed at this check; active jobs are still progressing and preempted L40S/H200 backup continuations are already queued.
- `MLP + flow` still has the best peak architecture signal, but H100 latest-step score dipped near 10M. Final 10M comparison should use both final and peak scores.
- `residual_flow + gaussian` improved on H100 to `0.46682` at 7.4M, but it remains below MLP-WM cells and is slower at action/eval time.

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
