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

2026-06-05 H200 preemption repair:

- H200 5M subset jobs `9464589_1`, `9464589_2`, and `9464589_3` were also preempted by `embers` after about 2 hours.
- Available checkpoints:
  - `wm-flow_pi-gaussian`: `1_000_000_full.pt`
  - `wm-mlp_pi-flow`: `1_000_000_full.pt`
  - `wm-flow_pi-flow`: latest checkpoint `750_000.pt`, but no full replay checkpoint yet
- Script fix: `scripts/slurm_flow_subset_5m.sbatch` now falls back to the latest non-full `*.pt` when no `*_full.pt` exists. Non-full checkpoints still restore model, optimizer, scheduler, RNG, and trainer step; they do not restore online replay.
- Repair action: resubmitted H200 array `1-3` as job `9476424_[1-3]` with the same `RUN_TAG=h200-r1`, W&B run IDs, and output directories. `REPLAY_CHECKPOINT_FREQ=500000` remains enabled for the repair submissions.

## WM Variant Runs

2026-06-05 implementation:

- Added `dynamics_arch=endpoint_flow`.
  - Predicts the latent residual endpoint in one deterministic network call.
  - Avoids Euler sampling during dynamics rollout, so it should be cheaper than increasing `flow_steps`.
- Added `dynamics_arch=residual_flow`.
  - Keeps an MLP next-latent path and adds a flow residual correction.
  - Trains the MLP path with next-latent MSE and the flow path on the remaining residual.
  - This is the lower-risk flow variant because it preserves the stable MLP dynamics baseline.
- Added `scripts/slurm_flow_wm_variants_5m.sbatch`.
  - Array `0`: `endpoint_flow + gaussian`.
  - Array `1`: `residual_flow + gaussian`.
  - Same 40-task high-pretrain setting as the 5M subset runs: `demo_steps=50000`, `steps=5000000`, `model_size=L`, `obs=state`, checkpointing and W&B resume enabled.
  - Uses distinct run names under `subset40-5m-wmvar...` so these jobs do not collide with the existing 2x2 runs.

Validation before submission:

- `python -m py_compile tdmpc2/common/flow.py tdmpc2/common/world_model.py tdmpc2/config.py`
- `bash -n scripts/slurm_flow_wm_variants_5m.sbatch`
- CPU smoke instantiated `mlp`, `flow`, `endpoint_flow`, and `residual_flow` world models and verified `next()` plus `dynamics_loss()` shape/finite-loss behavior on small tensors.

Variant submissions:

| Job | Purpose | Partition/GPU | Run tag | Notes |
| --- | --- | --- | --- | --- |
| `9477139` | `endpoint_flow`/`residual_flow` WM variants | `gpu-h200`, 1x H200 | `h200-r1` | Primary submission. |
| `9477141` | `endpoint_flow`/`residual_flow` WM variants | `gpu-h100`, 1x H100 | `h100` | Backup submission with the same 5M setting. |
| `9477140` | `endpoint_flow`/`residual_flow` WM variants | `gpu-a100`, 1x A100 | `a100` | Backup submission with the same 5M setting. |
| `9477142` | `endpoint_flow`/`residual_flow` WM variants | `gpu-l40s`, 1x L40S | `l40s` | Backup submission with `BATCH_SIZE=512` to fit L40S capacity. |

2026-06-06 resume repair:

- Several `embers` runs were preempted or hit the 8-hour walltime. This is expected under `embers`; affected runs keep local checkpoints and W&B run IDs.
- True failure: H200 `flow + flow` repair job `9476424_3` failed after loading `750_000.pt` because the checkpoint step was not aligned to the vectorized episode update boundary:
  - error: `Step 790000 does not match expected episode-boundary offset 0 for update frequency 40000`.
- Root cause: non-full checkpoints do not preserve partial environment/episode tensor state. After reset, the expected boundary offset should be `checkpoint_step % update_freq`, not the old stored offset.
- Fix: `Trainer.load_state_dict` now recomputes `_episode_boundary_offset = self._step % self._update_freq` on resume.
- Validation: `python -m py_compile tdmpc2/trainer.py`; direct load-state smoke confirmed `step=750000` gives offset `30000` for update frequency `40000`.
- Repair submissions after the fix:
  - `9490091_[0-3]`: H200 40-task 5M 2x2 resume with `RUN_TAG=h200-r1`.
  - `9490093_[1-3]`: H100 40-task 5M 2x2 resume with `RUN_TAG=h100`; array `0` already reached 5M.
  - `9490092_[0-2]`: L40S 40-task 5M 2x2 resume with `RUN_TAG=l40s-r1`; array `3` was still running at the repair check.
  - `9490094_[1]`: H200 residual-flow WM variant resume with `RUN_TAG=h200-r1`; endpoint-flow H200 already had a 5M full checkpoint.

2026-06-06 second resume repair:

- Follow-up failures showed two additional resume issues:
  - `sort -V | tail -n 1` selected `500_000_full.pt` after higher-step full checkpoints in some directories, causing W&B step regressions and stale resumes.
  - Episode-boundary alignment can shift after restoring from a checkpoint because the partial env/episode tensor is not stored. The trainer should resync instead of asserting.
- Fixes:
  - `scripts/slurm_flow_subset_5m.sbatch` and `scripts/slurm_flow_wm_variants_5m.sbatch` now select checkpoints by parsing the numeric step from the filename.
  - `Trainer.train` now resyncs `_episode_boundary_offset` to the current offset when a boundary mismatch is observed after resume.
- Validation:
  - `python -m py_compile tdmpc2/trainer.py`
  - `bash -n scripts/slurm_flow_subset_5m.sbatch scripts/slurm_flow_wm_variants_5m.sbatch`
  - Checkpoint selector smoke confirmed H200 residual WM now selects `4_000_000_full.pt` instead of `500_000_full.pt`.
- Repair submissions after the second fix:
  - `9521259_[0-3]`: H200 40-task 5M 2x2 resume with `RUN_TAG=h200-r1`.
  - `9521261_[3]`: H100 40-task 5M `flow + flow` resume with `RUN_TAG=h100`.
  - `9521260_[0-3]`: A100 40-task 5M 2x2 resume with `RUN_TAG=a100`.
  - `9521262_[1-3]`: L40S 40-task 5M 2x2 resume with `RUN_TAG=l40s-r1`.
  - `9521297_[1]`: H200 residual-flow WM variant resume with `RUN_TAG=h200-r1`.
  - `9521299_[0-1]`: H100 WM variant resumes with `RUN_TAG=h100`.
  - `9521300_[0-1]`: A100 WM variant resumes with `RUN_TAG=a100`.
  - `9521301_[0-1]`: L40S WM variant resumes with `RUN_TAG=l40s`.

2026-06-06 22:45 EDT monitoring update:

- Queue status: all repaired `952*` submissions were either running or pending; no new failed Slurm state was observed.
- Running jobs:
  - H200 2x2: `9521259_[0-3]`
  - H200 residual WM variant: `9521297_1`
  - H100 `flow + flow`: `9521261_3`
  - A100 2x2: `9521260_[0-2]`
  - L40S 2x2: `9521262_[1-2]`
- Pending backup jobs: A100 `9521260_3`, A100 WM variants `9521300_[0-1]`, H100 WM variants `9521299_[0-1]`, L40S `9521262_3`, and L40S WM variants `9521301_[0-1]`.
- Error scan: recent stderr showed W&B monotonic-step warnings only; no `Traceback`, `AssertionError`, CUDA OOM, preemption, or time-limit crash in the active `952*` logs.
- W&B note: warnings such as `Tried to log to step X that is less than the current step Y` are expected for some resumed runs where the online W&B history is one step ahead of the local checkpoint. Local `metrics.jsonl` continues updating, and W&B should resume accepting rows once the run surpasses the recorded online step.

Latest local progress snapshot:

| Run | Latest local row | Score | SPS | Action time | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| H200 `mlp + flow` | 4.84M train | 0.3732 | 2630.85 | 0.0863 | Near 5M, strongest active H200 row. |
| H200 `mlp + gaussian` | 3.96M train | 0.3329 | 2080.06 | 0.0536 | Healthy resume. |
| H200 `flow + gaussian` | 3.28M train | 0.0779 | 1861.40 | 0.1043 | Still weak. |
| H200 `flow + flow` | 0.92M train | 0.0795 | 588.84 | 0.1374 | Healthy but far behind after prior resume trouble. |
| H200 `residual_flow + gaussian` | 4.24M train | 0.3116 | 2533.59 | 0.1166 | Best flow WM variant so far. |
| H200 `endpoint_flow + gaussian` | 5.00M train | 0.2815 | 224.25 | 0.0579 | Finished earlier; below residual and MLP baseline. |

Interim interpretation remains unchanged: MLP WM is still the strongest baseline, while `residual_flow` is the most promising flow WM variant. Pure rectified-flow WM remains much weaker at comparable steps.

2026-06-07 04:00 EDT monitoring and repair:

- Slurm status:
  - Running before repair: A100 2x2 `9521260_[1-3]`, A100 WM variants `9521300_[0-1]`, L40S 2x2 `9521262_[1,3]`, and L40S WM residual `9521301_1`.
  - Completed: H100 2x2 `9521261_3`, H100 WM endpoint `9521299_0`, H200 `mlp + flow` `9521259_2`, L40S `mlp + flow` `9521262_2`, and L40S WM endpoint `9521301_0`.
  - Preempted by `embers`: H200 2x2 `9521259_[0,1,3]`, A100 `mlp + gaussian` `9521260_0`, H200 WM residual `9521297_1`, and H100 WM residual `9521299_1`.
- Error scan: no code-level failure was found in the active/recent `952*` logs. Recent stderr contained W&B monotonic-step warnings and explicit Slurm preemption notices only.
- Checkpoint status:
  - Already at 5M full checkpoint: H100 2x2 all cells, H200 `mlp + gaussian` and `mlp + flow`, L40S `mlp + gaussian` and `mlp + flow`, H200 WM endpoint/residual, H100 WM endpoint, and L40S WM endpoint.
  - Still incomplete and needing resume: H200 `flow + gaussian` at 4.0M full, H200 `flow + flow` at 2.0M, A100 `mlp + gaussian` at 3.0M full, and H100 WM residual at 4.0M full.
- Repair submissions:
  - `9536281_[1,3]`: H200 40-task 5M 2x2 resume for `flow + gaussian` and `flow + flow`, `RUN_TAG=h200-r1`.
  - `9536282_0`: A100 40-task 5M 2x2 resume for `mlp + gaussian`, `RUN_TAG=a100`.
  - `9536283_1`: H100 WM variant resume for `residual_flow + gaussian`, `RUN_TAG=h100`.
- Submission note: the first retry attempt failed because the shell had no default Slurm account. Existing running jobs used `Account=gts-agarg35`, so the successful repair submissions explicitly used `--account=gts-agarg35` with `--qos=embers` inherited from the scripts.
- Verification after repair: `9536281_[1,3]`, `9536282_0`, and `9536283_1` were all running immediately after submission.

Latest local progress snapshot:

| Run | Latest local row | Score | SPS | Action time | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| H200 `mlp + gaussian` | 5.00M train | 0.3651 | 783.76 | 0.0537 | 5M full checkpoint. |
| H200 `mlp + flow` | 5.00M train | 0.3817 | 1894.46 | 0.0862 | 5M full checkpoint. |
| H200 `flow + gaussian` | 4.24M train | 0.0957 | 538.29 | 0.1042 | Resubmitted as `9536281_1`. |
| H200 `flow + flow` | 2.04M train | 0.1184 | 198.14 | 0.1376 | Resubmitted as `9536281_3`. |
| H200 `residual_flow + gaussian` | 5.00M train | 0.3075 | 739.05 | 0.1165 | 5M full checkpoint. |
| H200 `endpoint_flow + gaussian` | 5.00M train | 0.2815 | 224.25 | 0.0579 | 5M full checkpoint. |

Interim result: the strong ordering is stable so far. MLP WM remains clearly ahead; residual-flow WM is the best flow WM variant; endpoint-flow is cheaper than multi-step pure flow but still below residual and MLP; pure rectified-flow WM is still weak.

2026-06-07 05:03 EDT monitoring update:

- Queue status: all active training jobs were running; no new Slurm `FAILED`, CUDA OOM, assertion, or Python traceback was observed.
- Active jobs:
  - `9536281_[1,3]`: H200 resume for `flow + gaussian` and `flow + flow`, running for about 1 hour.
  - `9536282_0`: A100 resume for `mlp + gaussian`, running for about 1 hour.
  - `9536283_1`: H100 resume for WM `residual_flow + gaussian`, running for about 1 hour.
  - `9521260_[1-3]`: A100 2x2 cells still running, now near 6-7 hours of walltime.
  - `9521300_[0-1]`: A100 WM endpoint/residual variants still running, near 6 hours.
  - `9521262_[1,3]` and `9521301_1`: L40S 2x2/WM-residual jobs still running, near 5-7 hours.
- Error scan: current stderr contains W&B monotonic-step warnings only. These continue to be non-fatal; local metrics and checkpoints are advancing.
- No repair submission was made in this pass. Several active jobs are close to the 8-hour limit, but they are still writing to their intended output directories, so duplicate submissions with the same run names would risk file races. If they hit walltime/preemption before 5M, resume from their newest checkpoint in the next pass.

Latest local progress snapshot:

| Run | Latest local row | Score | SPS | Action time | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| H200 `flow + gaussian` | 4.52M train | 0.1147 | 1280.95 | 0.1047 | Running under `9536281_1`; latest checkpoint 4.5M full. |
| H200 `flow + flow` | 2.20M train | 0.1529 | 634.01 | 0.1378 | Running under `9536281_3`; still far from 5M. |
| A100 `mlp + gaussian` | 3.44M train | 0.2884 | 973.19 | 0.1405 | Running under `9536282_0`. |
| A100 `mlp + flow` | 4.44M train | 0.3792 | 181.43 | 0.2266 | Running; likely needs resume if walltime hits first. |
| A100 WM `endpoint_flow + gaussian` | 4.92M train | 0.2581 | 229.41 | 0.1221 | Running, close to 5M. |
| A100 WM `residual_flow + gaussian` | 3.80M train | 0.2779 | 175.77 | 0.2535 | Running. |
| H100 WM `residual_flow + gaussian` | 4.40M eval | 0.2378 | 1194.09 | n/a | Running under `9536283_1`. |
| L40S WM `residual_flow + gaussian` | 3.20M eval | 0.2158 | 175.57 | n/a | Running, latest checkpoint 3.0M full. |

2026-06-07 05:53 EDT monitoring and eval push:

- Active training status:
  - No new code-level failures were observed. Recent stderr still shows W&B monotonic-step warnings only.
  - H200 `flow + gaussian` completed successfully under `9536281_1` and wrote `5_000_000_full.pt`.
  - H200 `flow + flow` is still running under `9536281_3`, latest local row around 2.56M.
  - A100 `mlp + flow`, `flow + gaussian`, `flow + flow`, and A100 WM residual are still running; several are close to the 8-hour walltime and may need resume if preempted or timed out.
  - A100 WM endpoint completed and wrote `5_000_000_full.pt`.
- Added `scripts/slurm_flow_subset_5m_eval.sbatch` for formal 40-task subset evaluation.
  - It evaluates only `5_000_000_full.pt` checkpoints.
  - It uses the same `configs/task_subsets/mmbench_balanced_40.json` task list as training.
  - Eval logs use separate `eval-...` experiment names, so they do not race with training directories.
- Eval submissions:
  - `9541180_[0-11%4]`: H100 eval array for completed H100 2x2, completed H200 MLP cells, completed H200 WM variants, completed L40S MLP cells, and completed endpoint WM checkpoints.
  - `9541207_12`: H100 eval for newly completed H200 `flow + gaussian`.
- Verification after eval submission: `9541180_0-3` and `9541207_12` were running; `9541180_4-11` were pending only because of the `%4` array throttle.

Latest local progress snapshot:

| Run | Latest local row | Score | SPS | Action time | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| H200 `flow + gaussian` | 5.00M train | 0.136 | n/a | 0.1047 | Completed 5M full; eval submitted as `9541207_12`. |
| H200 `flow + flow` | 2.56M train | 0.1329 | 403.69 | 0.1378 | Still running under `9536281_3`. |
| A100 `mlp + flow` | 4.76M train | 0.3753 | 173.35 | 0.2261 | Still running, close to 5M. |
| A100 WM `endpoint_flow + gaussian` | 5.00M train | 0.2554 | 227.29 | 0.1220 | Completed 5M full; eval submitted in `9541180`. |
| A100 WM `residual_flow + gaussian` | 4.00M eval | 0.1966 | 165.55 | n/a | Still running. |
| H100 WM `residual_flow + gaussian` | 4.72M train | 0.2729 | 741.87 | 0.1376 | Still running under `9536283_1`. |

2026-06-08 02:58 EDT monitoring and continuation:

- Training status:
  - No active training jobs were running at the start of this check; old `9431231_[0-3]` remained pending on H200.
  - Completed since the previous note: H200 `flow + gaussian`, H200 `flow + flow`, A100 `mlp + gaussian`, H100 WM `residual_flow + gaussian`.
  - Time-limit without code failure: A100 `flow + gaussian`, A100 `mlp + flow`, A100 `flow + flow`, A100 WM `residual_flow`, L40S `flow + gaussian`, L40S `flow + flow`, and L40S WM `residual_flow`.
  - Error scan found only Slurm preemption/time-limit records; no Python traceback, assertion, or CUDA OOM was observed.
- Checkpoint status:
  - 5M checkpoint available: H100 2x2 all cells; H200 2x2 all cells; H200 endpoint/residual WM; A100 `mlp + gaussian`; A100 endpoint WM; H100 endpoint/residual WM; L40S MLP cells and endpoint WM.
  - Still below 5M and resumed: A100 `flow + gaussian`, A100 `mlp + flow`, A100 `flow + flow`, A100 WM `residual_flow`, L40S `flow + gaussian`, L40S `flow + flow`, and L40S WM `residual_flow`.
- Repair/continuation submissions:
  - `9609358_[1-3]`: A100 2x2 resume for `flow + gaussian`, `mlp + flow`, and `flow + flow`.
  - `9609357_1`: A100 WM `residual_flow + gaussian` resume.
  - `9609388_[1,3]`: L40S 2x2 resume for `flow + gaussian` and `flow + flow`.
  - `9609389_1`: L40S WM `residual_flow + gaussian` resume.
  - L40S submissions required `--cpus-per-task=4` because the cluster enforces a 4:1 CPU:GPU ratio on `gpu-l40s`.
- Eval script update:
  - `scripts/slurm_flow_subset_5m_eval.sbatch` now includes A100 `mlp + gaussian`, H100 WM `residual_flow`, and H200 `flow + flow`.
  - The eval script now accepts either `5_000_000_full.pt` or `5_000_000.pt`; eval does not need replay state, so non-full 5M checkpoints are valid for evaluation.
  - `9609416_[13-15]` submitted these additional eval jobs; `9609416_13` started running and `9609416_[14-15]` were pending at submission check.

Formal 40-task eval results available so far:

| Run | Eval avg_score | Notes |
| --- | ---: | --- |
| H200 `mlp + flow` | 0.3391 | Best completed eval so far. |
| L40S `mlp + flow` | 0.3151 | Close to H100 `mlp + flow`. |
| H100 `mlp + flow` | 0.3149 | Strong MLP WM + flow policy. |
| H100 `mlp + gaussian` | 0.3025 | Strong MLP WM baseline. |
| H200 `mlp + gaussian` | 0.2848 | Lower than H100/H200 MLP+flow eval. |
| L40S `mlp + gaussian` | 0.2679 | MLP WM remains stronger than flow WM. |
| H200 `residual_flow + gaussian` | 0.2026 | Best completed flow-WM variant eval so far. |
| A100 `endpoint_flow + gaussian` | 0.1981 | Endpoint flow is below residual flow. |
| H200 `endpoint_flow + gaussian` | 0.1947 | Similar to A100 endpoint. |
| L40S `endpoint_flow + gaussian` | 0.1872 | Similar but lower endpoint result. |
| H100 `flow + gaussian` | 0.1361 | Pure flow WM remains weak. |
| H200 `flow + gaussian` | 0.1036 | Pure flow WM remains weak. |
| H100 `flow + flow` | 0.1006 | Pure flow WM + flow policy remains weak. |

Current interpretation: formal eval confirms the train-metric pattern. MLP WM is clearly strongest; MLP WM + flow policy is the top cell so far despite higher action time; residual-flow WM is the only flow-WM variant with a plausible signal, but it is still well below MLP WM; pure flow WM remains poor.
