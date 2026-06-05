# Flow WM/Policy Next Experiments

Last updated: 2026-06-05

## Why We Are Changing the Experiment Scale

The first 200-task single-GPU screening was useful for debugging and timing, but it is not enough to decide whether flow world models are useful:

- The 2x2 screening used only `demo_steps=1000`, while the formal Newt setting uses `demo_steps=200000`.
- The online checkpoints used for eval are around `400k-500k` total env steps. With 200 tasks, this is only about `2k-2.5k` env steps per task.
- Official Newt reports its main state-only result at `100M` total env steps, or about `500k` env steps per task.
- Current screening shows the original `mlp + gaussian` baseline is strongest and fastest, while flow policy adds clear action-selection overhead.

Therefore, the next experiments should reduce task count and increase both demo pretraining and online steps. This gives each task more interaction while keeping wall time practical.

## New First-Round Plan

Use a balanced 40-task subset with 4 tasks from each MMBench domain:

```text
configs/task_subsets/mmbench_balanced_40.json
```

Run the original 2x2 comparison:

| World model | Policy |
| --- | --- |
| `mlp` | `gaussian` |
| `flow` | `gaussian` |
| `mlp` | `flow` |
| `flow` | `flow` |

Settings:

- `task=soup`
- `task_subset_file=configs/task_subsets/mmbench_balanced_40.json`
- `model_size=L`
- `obs=state`
- `demo_steps=50000`
- `steps=5000000`
- `checkpoint_freq=250000`
- `replay_checkpoint_freq=1000000`
- W&B group: `subset40-5m-highpretrain`

Purpose:

- Get a more robust learning signal than the 200-task 0.5M screening.
- Check whether flow world models improve with stronger demo pretraining and more online steps per task.
- Keep policy flow in the 2x2 for now, but treat it cautiously because it is the current main timing overhead.

## Flow Ablation Plan

Before implementing larger flow variants, run a minimal ablation that does not require new model code:

| Ablation | World model | Policy | Steps |
| --- | --- | --- | ---: |
| `flow_steps=1` | `flow` | `gaussian` | 500k |
| `flow_steps=2` | `flow` | `gaussian` | 500k |
| `flow_steps=4` | `flow` | `gaussian` | 500k |

Settings:

- Same 40-task subset.
- `demo_steps=50000`
- W&B group: `subset40-flowsteps-500k`

Purpose:

- Separate whether the current flow world model is weak because of modeling choice or because multi-step flow sampling is too expensive/unstable.
- Keep Gaussian policy fixed to avoid confounding with flow-policy overhead.

## Future Flow Variants

Implement these only after the 40-task high-pretrain results show a reason to continue:

- Endpoint flow dynamics: predict the final latent residual directly, with lower inference cost than multi-step Euler sampling.
- MLP + residual flow: keep the MLP dynamics path and use flow only for residual corrections.
- Trajectory flow: model a short latent rollout segment for planning, higher research value but larger implementation risk.

Do not prioritize full flow policy variants yet; current evidence shows policy flow is the largest action-time overhead and has no score advantage in the early screening.

## Submissions

Submitted on 2026-06-05:

| Job | Purpose | Array | Partition/GPU | Notes |
| --- | --- | --- | --- | --- |
| `9450979` | 40-task `flow_steps=1/2/4` ablation | `0-2` | `gpu-h200`, 1x H200 | 500k online steps, 50k demo pretrain, Gaussian policy. |
| `9450994` | 40-task 5M high-pretrain 2x2 | `0-3` | `gpu-h200`, 1x H200 | 5M online steps, 50k demo pretrain, original 2x2 cells. |

Operational notes:

- The first submission attempt used 12 CPUs per H200 and was rejected by the cluster CPU:GPU ratio policy. Scripts now request 8 CPUs per H200.
- The first 5M submission attempt requested 10 hours and was rejected by the `embers` walltime limit. The 5M script now requests 8 hours and relies on full checkpoints for continuation if needed.
- Both accepted jobs were pending with reason `Priority` at submission time.
- Status check on 2026-06-05: both `9450979` and `9450994` remain pending with reason `Priority`; no stderr files have been produced yet.

Backup submissions on 2026-06-05:

The original H200 submissions were still pending, so backup jobs were submitted across H100, A100, and L40S to get any viable GPU running first. Scripts now support `RUN_TAG` so backup jobs write separate output directories and W&B run IDs.

| Job | Purpose | Partition/GPU | Run tag | Notes |
| --- | --- | --- | --- | --- |
| `9456326` | 40-task `flow_steps=1/2/4` ablation | `gpu-h100`, 1x H100 | `h100` | Same settings as H200. |
| `9456330` | 40-task `flow_steps=1/2/4` ablation | `gpu-a100`, 1x A100 | `a100` | Same settings as H200. |
| `9456356` | 40-task `flow_steps=1/2/4` ablation | `gpu-l40s`, 1x L40S | `l40s` | Uses `BATCH_SIZE=512` and 4 CPUs to fit L40S policy limits. |
| `9456329` | 40-task 5M high-pretrain 2x2 | `gpu-h100`, 1x H100 | `h100` | Same settings as H200. |
| `9456357` | 40-task 5M high-pretrain 2x2 | `gpu-a100`, 1x A100 | `a100` | Same settings as H200. |
| `9456355` | 40-task 5M high-pretrain 2x2 | `gpu-l40s`, 1x L40S | `l40s` | Uses `BATCH_SIZE=512` and 4 CPUs to fit L40S policy limits. |

All backup jobs were pending with reason `Priority` immediately after submission.

## Repair Log

2026-06-05 subset launch repair:

- Failed jobs: H200 `9450979`, `9450994`; L40S `9456355`, `9456356`.
- Symptom: every array task exited during startup with `AssertionError: task_subset_file must contain a JSON list of task names`.
- Root cause: `parse_cfg` assigned the loaded JSON directly to `cfg.tasks`, then checked `isinstance(cfg.tasks, list)`. Hydra/OmegaConf converted the assigned value, so the assertion checked the converted config object instead of the plain JSON list.
- Fix: validate `subset_tasks` immediately after `json.load`, then assign `cfg.tasks = list(subset_tasks)`.
- Validation: `python -m py_compile tdmpc2/config.py`; local `parse_cfg` smoke confirmed `40` unique tasks from `configs/task_subsets/mmbench_balanced_40.json`.
- Cleanup: no subset training output directories or checkpoints were created by the failed startup jobs; only Slurm stderr records remain under `outputs/slurm/` for diagnosis.

Replacement submissions after the fix:

| Job | Purpose | Partition/GPU | Run tag | Notes |
| --- | --- | --- | --- | --- |
| `9464590` | 40-task `flow_steps=1/2/4` ablation | `gpu-h200`, 1x H200 | `h200-r1` | Replacement for failed H200 `9450979`. |
| `9464589` | 40-task 5M high-pretrain 2x2 | `gpu-h200`, 1x H200 | `h200-r1` | Replacement for failed H200 `9450994`. |
| `9464593` | 40-task `flow_steps=1/2/4` ablation | `gpu-l40s`, 1x L40S | `l40s-r1` | Replacement for failed L40S `9456356`, uses `BATCH_SIZE=512`. |
| `9464591` | 40-task 5M high-pretrain 2x2 | `gpu-l40s`, 1x L40S | `l40s-r1` | Replacement for failed L40S `9456355`, uses `BATCH_SIZE=512`. |

Current queue note: H100 `9456326`/`9456329` and A100 `9456330`/`9456357` backups were still pending when the code fix landed, so they should pick up the repaired repo code at runtime.

2026-06-05 follow-up status:

- The subset parsing failure did not recur after commit `bd0f318`.
- H200 flow-step replacements `9464590_[0-2]` completed successfully.
- H100 flow-step backups `9456326_0` and `9456326_1` completed successfully; `9456326_2` was still running at the status check.
- H200 5M subset jobs `9464589_1`, `9464589_2`, and `9464589_3` were running normally.
- H200 5M subset job `9464589_0` was preempted by `embers` after about 2 hours, not failed by code. It produced `1_000_000_full.pt`, plus later non-full checkpoints through `1_500_000.pt`.
- Repair action: resubmitted H200 array `0` as job `9476097` with the same `RUN_TAG=h200-r1`, W&B run id, and output directory. The job will resume from `outputs/logs/soup/1/subset40-5m-h200-r1_wm-mlp_pi-gaussian_seed-1/models/1_000_000_full.pt`.
- To reduce future preemption loss, the resubmission overrides `REPLAY_CHECKPOINT_FREQ=500000`.

Observed early metrics at this status check:

| Run | Step | Score metric | SPS | Action time |
| --- | ---: | ---: | ---: | ---: |
| `subset40-5m-h200-r1_wm-mlp_pi-gaussian_seed-1` | 1.6M | `avg_score=0.2734` | 224.84 | eval row |
| `subset40-5m-h200-r1_wm-mlp_pi-flow_seed-1` | 1.08M | `episode_score=0.2724` | 173.42 | 0.0869 |
| `subset40-5m-h200-r1_wm-flow_pi-gaussian_seed-1` | 0.96M | `episode_score=0.1035` | 154.68 | 0.1049 |
| `subset40-5m-h200-r1_wm-flow_pi-flow_seed-1` | 0.72M | `episode_score=0.0734` | 120.00 | 0.1379 |
| `subset40-flowsteps1-h200-r1_wm-flow_pi-gaussian_seed-1` | 0.48M | `episode_score=0.0675` | 185.42 | 0.0589 |
| `subset40-flowsteps2-h200-r1_wm-flow_pi-gaussian_seed-1` | 0.48M | `episode_score=0.0677` | 167.75 | 0.0743 |
| `subset40-flowsteps4-h200-r1_wm-flow_pi-gaussian_seed-1` | 0.48M | `episode_score=0.0839` | 143.80 | 0.1043 |

Interpretation is still preliminary because the 5M runs are incomplete and some rows are train episode metrics rather than full eval rows. The timing trend is already consistent: more flow steps increase action time and reduce SPS.
