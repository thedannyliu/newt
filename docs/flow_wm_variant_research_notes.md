# Flow World-Model Variant Research Notes

Last updated: 2026-06-08

## Context

The current Newt/LeWM experiments compare world-model and policy-prior architecture changes under the original Newt multitask training pipeline. The strongest completed 40-task 5M eval results so far favor the original MLP world model, with `mlp + flow policy` scoring best among completed cells but with higher action-selection cost. Flow world models remain weaker; `residual_flow + gaussian` is the only flow-WM direction with a plausible signal.

This motivates a narrower next step: keep the original Newt training pipeline and Gaussian policy fixed, and test flow only as a latent residual correction on top of stable MLP dynamics.

## References

- Newt project page: <https://www.nicklashansen.com/NewtWM/>
- Newt paper: <https://arxiv.org/abs/2511.19584>
- Flow Matching: <https://arxiv.org/abs/2210.02747>
- Rectified Flow, ICLR 2023: <https://openreview.net/forum?id=XVjTT1nw5z>
- Mean Flows for One-step Generative Modeling, NeurIPS 2025: <https://papers.neurips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html>
- Mean Flows arXiv: <https://arxiv.org/abs/2505.13447>
- One Step Diffusion via Shortcut Models, ICLR 2025 oral: <https://proceedings.iclr.cc/paper_files/paper/2025/hash/559a0998fab1d19b80e7e43a5852401c-Abstract-Conference.html>
- Shortcut Models project page: <https://danijar.com/project/shortcut/>
- Dynamic Conditional Optimal Transport through Simulation-Free Flows, NeurIPS 2024: <https://neurips.cc/virtual/2024/poster/93315>
- Dynamic Conditional Optimal Transport arXiv: <https://arxiv.org/abs/2404.04240>
- Diffusion Forcing, NeurIPS 2024: <https://openreview.net/forum?id=yDo1ynArjj>

## Why These Variants

The most relevant flow direction for this repo is action-conditioned latent residual modeling:

```text
base = MLP(z_t, action_chunk, task)
residual ~ flow(delta_z | z_t, action_chunk, task)
z_{t+1} = SimNorm(base + residual)
```

This keeps the fast latent-planning surface of Newt/TD-MPC2. It does not introduce pixel/video generation, transformer sequence modeling, or a new action-selection algorithm.

### 1. `residual_mean_flow_wm + gaussian`

Purpose: test whether one-step average-velocity flow can keep the useful residual uncertainty/modeling signal while avoiding multi-step Euler rollout cost.

Implementation:

- Base path: same MLP next-latent predictor used by `residual_flow`.
- Residual head: `MeanFlow`, conditioned on current residual sample, task/action latent condition, and interval endpoints `[r, t]`.
- Training target: latent residual `target_z - base.detach()`.
- Rollout: one function evaluation from zero residual at interval `[0, 1]`.

Important limitation: this is a compact pilot inspired by MeanFlow. It does not yet implement the full MeanFlow average/instantaneous velocity identity with the original paper's complete training machinery. The goal is to test the practical one-step average-velocity idea inside Newt's latent residual dynamics with minimal pipeline change.

### 2. `shortcut_residual_flow_wm + gaussian`

Purpose: test whether step-size-conditioned shortcut flow can support either one-step or few-step residual rollout without training separate distilled models.

Implementation:

- Base path: same MLP next-latent predictor.
- Residual head: `ShortcutFlow`, conditioned on current flow sample, flow time `t`, and desired step size `dt`.
- Training target: straight-path velocity from Gaussian source to latent residual target with random valid `dt`.
- Rollout: uses existing `flow_steps` as the step budget; each step is conditioned on `dt=1/flow_steps`.

This is closest to the ICLR 2025 shortcut-model idea: a single network can be queried with different step budgets at inference time.

### 3. `ot_cfm_residual_wm + gaussian`

Purpose: test whether better source-target coupling improves the current rectified-flow residual training signal.

Implementation:

- Base path: same MLP next-latent predictor.
- Residual head: `OTConditionalFlow`, same inference as rectified flow.
- Training target: latent residual `target_z - base.detach()`.
- Coupling: a lightweight sliced-OT pairing reorders only Gaussian source samples along a random projection. The target residual and conditioning row remain fixed, so task/action conditioning is not detached from the true transition.

This is a practical approximation to OT-CFM style straighter coupling. It is cheaper and less invasive than implementing a full conditional OT solver inside every TD-MPC2 update.

## Strict Newt Pipeline Adherence Check

The new variants intentionally change only the world-model dynamics architecture selected by `dynamics_arch`.

Unchanged pipeline components:

- Training entry point remains `tdmpc2/train.py`.
- Multitask setup remains `task=soup` with `task_subset_file=configs/task_subsets/mmbench_balanced_40.json`.
- Demonstrations are loaded by `load_demos()` from `data_dir`, exactly as current Newt runs do.
- Demo pretraining remains in `Trainer.train()` before online interaction when `use_demos=true`, `checkpoint=None`, and `demo_steps>0`.
- Online data still comes from real simulator/environment interaction through `env.step(action)`. The world model is not used to generate replay-buffer training data.
- The 50/50 demo-online training mix remains `EnsembleBuffer`: half batch from the offline demo buffer and half from the online replay buffer.
- Action selection remains Newt/TD-MPC2 latent planning through `TDMPC2.plan()`, `_sample_pi_trajs()`, `_mppi()`, `_estimate_value()`, and `WorldModel.next()`.
- Reward loss, value/Q loss, policy-prior loss, entropy term, temporal weighting `rho`, and target-Q update remain unchanged.
- Policy is fixed to `policy_arch=gaussian` for these three jobs.
- Checkpointing and resume remain `Trainer.save_checkpoint()`, `TDMPC2.save/load()`, W&B stable run IDs, local model checkpoints, and replay full checkpoints.
- Slurm output/log/runtime paths remain under `outputs/`.

Changed architecture surface:

- `tdmpc2/common/flow.py`: adds `MeanFlow`, `ShortcutFlow`, `OTConditionalFlow`, and residual dynamics wrappers.
- `tdmpc2/common/world_model.py`: maps the new `dynamics_arch` values to the new dynamics modules.
- `tdmpc2/config.py`: allows the new `dynamics_arch` names.
- `scripts/slurm_flow_wm_variants_5m.sbatch`: extends the WM-variant array with the three new rows while preserving old indices.

## Experiment Spec

Run all three with the same formal-like 40-task high-pretrain setting as prior WM-variant jobs:

- `task=soup`
- `task_subset_file=configs/task_subsets/mmbench_balanced_40.json`
- `model_size=L`
- `obs=state`
- `demo_steps=50000`
- `steps=5000000`
- `batch_size=1024` on H200/H100/A100; `512` if using L40S
- `buffer_size=5000000`
- `policy_arch=gaussian`
- `eval_episodes=1`
- `checkpoint_freq=250000`
- `replay_checkpoint_freq=1000000`
- `enable_wandb=true`
- `wandb_upload_artifacts=false`
- `wandb_resume=allow`

Primary metrics to compare:

- Formal eval `episode_score` averaged across the 40-task subset.
- Training `episode_score` trend.
- `steps_per_second`.
- `action_time`.
- `update_time`.
- `num_updates`.
- `consistency_loss`, `reward_loss`, `value_loss`, `pi_prior_loss`, and `total_loss`.

## Expected Read

Useful outcome:

- Any new residual-flow WM approaches the MLP-WM score while keeping action/update time acceptable.
- `residual_mean_flow_wm` is especially interesting if it improves over `residual_flow` while reducing flow rollout cost.
- `shortcut_residual_flow_wm` is useful if it can match multi-step residual flow with fewer effective steps.
- `ot_cfm_residual_wm` is useful if better coupling improves the weak pure/residual CFM signal without changing inference cost.

Negative outcome:

- If all three remain well below MLP WM, the next practical direction should be stronger MLP/latent-sequence modeling or better policy/planner integration, not more pure flow steps.

## Validation and Submissions

Validation completed before GPU submission:

- `python -m py_compile tdmpc2/common/flow.py tdmpc2/common/world_model.py tdmpc2/config.py`
- `bash -n scripts/slurm_flow_wm_variants_5m.sbatch`
- CPU smoke instantiated `residual_mean_flow_wm`, `shortcut_residual_flow_wm`, and `ot_cfm_residual_wm`, then verified finite `WorldModel.next()` outputs and finite `dynamics_loss()` values.

Implementation commit:

- `8c09a68 Add residual flow WM variants`

Submitted on 2026-06-08 with `qos=embers`, `account=gts-agarg35`, W&B entity `danny010324`, project `newt-flow-2x2`, group `subset40-5m-new-flowwm-variants`, and `REPLAY_CHECKPOINT_FREQ=500000`:

| Job | Partition/GPU | Array | Run tag | Batch size | Status at submission check |
| --- | --- | --- | --- | ---: | --- |
| `9611864` | `gpu-h200`, 1x H200 | `2-4` | `h200-newflow1` | 1024 | `2`, `3`, and `4` running |
| `9611865` | `gpu-h100`, 1x H100 | `2-4` | `h100-newflow1` | 1024 | pending, `Priority` |
| `9611867` | `gpu-a100`, 1x A100 | `2-4` | `a100-newflow1` | 1024 | pending, `Priority` |
| `9611866` | `gpu-l40s`, 1x L40S | `2-4` | `l40s-newflow1` | 512 | pending, `Priority` |

Array mapping:

| Array index | Dynamics architecture | Policy |
| ---: | --- | --- |
| `2` | `residual_mean_flow_wm` | `gaussian` |
| `3` | `shortcut_residual_flow_wm` | `gaussian` |
| `4` | `ot_cfm_residual_wm` | `gaussian` |

2026-06-08 04:01 EDT monitoring:

- H200 `9611864_[2-4]` is running on separate H200 nodes. All three jobs loaded the 40-task demo set, initialized W&B, printed the expected Newt architecture, and entered demonstration pretraining from scratch.
- H100 `9611865_2` is running for `residual_mean_flow_wm`; `9611865_[3-4]` remains pending with reason `Priority`.
- L40S `9611866_2` is running for `residual_mean_flow_wm`; `9611866_3` is running for `shortcut_residual_flow_wm`; `9611866_4` remains pending with reason `Priority`.
- A100 `9611867_[2-4]` remains pending with reason `Priority`.
- Error scan over the active new-flow stderr files found no `Traceback`, `RuntimeError`, `AssertionError`, CUDA OOM, missing data, or W&B fatal error. Current stderr contains only known environment warnings.
- Slurm allocation check confirms each active array task has a single allocated GPU TRES and distinct output/W&B names.

2026-06-08 14:59 EDT monitoring:

- Previous new-flow submissions finished or were interrupted by `embers` walltime/preemption only; no Python traceback, assertion, CUDA OOM, missing data, or W&B fatal error was found.
- Completed 5M:
  - H200 `residual_mean_flow_wm + gaussian`, run `subset40-5m-wmvar-h200-newflow1_wm-residual_mean_flow_wm_pi-gaussian_seed-1`, wrote `5_000_000_full.pt`.
- Partial new-flow progress before preemption/time-limit:
  - H200 `shortcut_residual_flow_wm`: latest `2_250_000.pt`.
  - H200 `ot_cfm_residual_wm`: latest `2_750_000.pt`.
  - H100 `residual_mean_flow_wm`: latest `2_750_000.pt`.
  - H100 `shortcut_residual_flow_wm`: latest `2_750_000.pt`.
  - H100 `ot_cfm_residual_wm`: latest `3_000_000_full.pt`.
  - A100 `residual_mean_flow_wm`: latest `2_750_000.pt`.
  - A100 `shortcut_residual_flow_wm`: latest `2_250_000.pt`.
  - A100 `ot_cfm_residual_wm`: latest early checkpoint only; the job was preempted early.
  - L40S `residual_mean_flow_wm`: latest `2_750_000.pt`.
  - L40S `shortcut_residual_flow_wm`: latest `1_750_000.pt`.
  - L40S `ot_cfm_residual_wm`: latest `2_000_000_full.pt`.
- Eval submitted for the completed H200 `residual_mean_flow_wm` 5M checkpoint as eval array index `22` in job `9652308`.
- Resume submissions:
  - `9652347_[3-4]`: H200 resume for `shortcut_residual_flow_wm` and `ot_cfm_residual_wm`.
  - `9652348_[2-4]`: H100 resume for all three new-flow variants.
  - `9652349_[2-4]`: A100 resume for all three new-flow variants.
  - `9652350_[2-4]`: L40S resume for all three new-flow variants, with `BATCH_SIZE=512`.
