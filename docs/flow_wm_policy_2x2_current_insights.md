# Flow WM/Policy 2x2 Current Insights

Last updated: 2026-06-04

## Scope

This note summarizes the results currently available from the single-GPU full-like H200 runs and the formal baseline. These are not yet final formal 8-GPU results because the replacement formal 2x2 job is still pending.

The local aggregate CSV is:

```text
outputs/analysis/single_full_2x2_summary.csv
```

## Current Checkpoints

| Run | Latest full checkpoint | Last train metric step |
| --- | ---: | ---: |
| `single-full_wm-mlp_pi-gaussian_seed-1` | 500,000 | 400,000 |
| `single-full_wm-flow_pi-gaussian_seed-1` | 500,000 | 400,000 |
| `single-full_wm-mlp_pi-flow_seed-1` | 500,000 | 400,000 |
| `single-full_wm-flow_pi-flow_seed-1` | 400,000 | 400,000 |
| `wm-mlp_pi-gaussian_seed-1` | 1,000,000 | 1,000,000 |

## Single-GPU Full-Like Timing

The table below uses the latest train row, which is at 400k steps for each single-GPU run.

| World model | Policy | Train score | SPS | Action time | Action time vs baseline | SPS vs baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MLP | Gaussian | 0.0367 | 939.4 | 0.00139 s | 1.00x | 1.00x |
| Flow | Gaussian | 0.0272 | 942.7 | 0.00127 s | 0.91x | 1.00x |
| MLP | Flow | 0.0319 | 920.3 | 0.00327 s | 2.35x | 0.98x |
| Flow | Flow | 0.0300 | 792.7 | 0.00348 s | 2.50x | 0.84x |

## Early Insights

- Flow policy is the clear action-selection overhead source in the single-GPU setting. The flow-policy cells have about 2.35-2.50x higher action time than the Gaussian-policy baseline.
- Flow world model alone does not currently show an action-time penalty because the timing metric measures action selection, not model update cost. Update timing is still sparse in these short runs because the trainer logs at large episode/update boundaries.
- End-to-end SPS remains close to baseline for `mlp + flow` despite the higher action time, likely because environment stepping dominates much of the wall time. `flow + flow` is the only cell with a visible SPS drop at this scale.
- Current train scores are early and low across all cells. The baseline has the highest 400k train score among the single-GPU runs, but the gap is not enough to draw final performance conclusions.
- The current `eval` rows in the original training logs are pretraining/initial eval rows, not post-checkpoint eval rows. Dedicated checkpoint eval jobs are needed before making a score-based comparison table.

## Formal Status

- Formal baseline `wm-mlp_pi-gaussian_seed-1` has reached 1M train steps with last train score 0.2083 and SPS 200.7.
- Formal replacement job `9431231` is still pending on H200. Formal flow cells are therefore not ready for final 2x2 comparison.

## Next Eval Step

Use `tdmpc2/eval_checkpoint.py` through `scripts/slurm_flow_single_gpu_eval.sbatch` to evaluate the latest single-GPU full-like checkpoints. This writes eval-only metrics under `outputs/logs/soup/1/eval-single-full_*` and logs to W&B without uploading checkpoint artifacts.

Submitted eval job:

| Job | Purpose | Partition/GPU | Array | Notes |
| --- | --- | --- | --- | --- |
| `9431612` | Single-GPU post-checkpoint eval for all four full-like cells | `gpu-h200`, 1x H200 | `0-3` | Uses latest `*_full.pt` checkpoint per cell and `EVAL_EPISODES=1`. |

Job `9431612` failed before evaluation because `eval_checkpoint.py` instantiated `WorldModel` before `make_env(cfg)`, leaving `cfg.action_dim` unset. The eval script now constructs the environment first so `cfg.obs_shape`, `cfg.action_dim`, and `cfg.episode_length` match the training path before model initialization.

Replacement eval job:

| Job | Purpose | Partition/GPU | Array | Notes |
| --- | --- | --- | --- | --- |
| `9432390` | Single-GPU post-checkpoint eval after config-init fix | `gpu-h200`, 1x H200 | `0-3` | Uses the same checkpoint selection as `9431612` with the fixed eval script. |

Job `9432390` failed while constructing the full 200-task async eval environment. All four array tasks were scheduled close together on the same node, so OGBench/MuJoCo workers attempted to allocate many EGL contexts at once and hit `EGL_BAD_ALLOC`, followed by async worker `BrokenPipeError`. The eval Slurm script now limits the array to one active task at a time (`0-3%1`) to keep the full 200-task eval protocol while avoiding concurrent EGL allocation pressure from sibling array tasks.

Replacement serialized eval job:

| Job | Purpose | Partition/GPU | Array | Notes |
| --- | --- | --- | --- | --- |
| `9432574` | Serialized single-GPU post-checkpoint eval | `gpu-h200`, 1x H200 | `0-3%1` | Full 200-task eval for each cell, one array task active at a time. |

Job `9432574` still failed with `EGL_BAD_ALLOC`, showing that even a single full 200-task eval job can over-allocate EGL/OpenGL contexts during environment construction. The eval script now runs task chunks instead of the full 200-task suite at once:

- Added `task_start` to the config so a soup run can evaluate a contiguous task slice.
- Updated checkpoint loading to slice `_task_emb.weight` and `_action_masks` by `task_start` when loading a full 200-task checkpoint into a smaller task-slice model.
- Updated `scripts/slurm_flow_single_gpu_eval.sbatch` to use `0-39%4`, mapping 4 cells x 10 chunks of 20 tasks.

Chunked eval preserves the full task coverage after aggregation while avoiding one job constructing all 200 environments at once.

Chunked eval submissions:

| Job | Cell(s) | Array | Status at submission |
| --- | --- | --- | --- |
| `9433194` | `mlp + gaussian` | `0` | Pending |
| `9433197` | `mlp + gaussian` | `1-9%2` | Pending |
| `9433199` | `flow + gaussian` | `10-19%2` | Pending |
| `9433200` | `mlp + flow` | `20-29%2` | Pending |

The `flow + flow` chunks (`30-39`) were not submitted yet because `QOSMaxSubmitJobPerUserLimit` was reached. Submit them after existing jobs leave the pending/running set.

Chunked eval repair on 2026-06-04:

| Job | Failed chunks | Root cause | Fix |
| --- | --- | --- | --- |
| `9433194`, `9433197` | `0-5` | Eval-only trainer had no replay buffer, but full checkpoint restore attempted to load replay state. ManiSkill chunks also hit CUDA reinitialization inside forked async env workers. | `eval_checkpoint.py` now restores only model weights, running scale, and trainer step. `scripts/slurm_flow_single_gpu_eval.sbatch` now sets `env_mode=sync` for checkpoint eval. |

Pending old-script eval arrays `9433197`, `9433199`, and `9433200` should be canceled before resubmission because Slurm array tasks keep the script snapshot from submission time.

Replacement chunked eval submissions after repair:

| Job | Cell(s) | Array | Status at submission |
| --- | --- | --- | --- |
| `9436034` | `mlp + gaussian` | `0-9%2` | Pending |
| `9436035` | `flow + gaussian` | `10-19%2` | Pending |
| `9436038` | `flow + flow` | `30-39%2` | Pending |

The `mlp + flow` chunks (`20-29`) were not resubmitted yet because `QOSMaxSubmitJobPerUserLimit` was reached during submission. Submit them after one of the pending arrays starts or completes.
