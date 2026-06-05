# Flow WM/Policy 2x2 Experiment Status

Last updated: 2026-06-04

## Scope

This record tracks the Newt 2x2 comparison where the world model and policy are varied independently:

- `wm=mlp, policy=gaussian`
- `wm=flow, policy=gaussian`
- `wm=mlp, policy=flow`
- `wm=flow, policy=flow`

Runs use the `newt-flow-2x2` W&B project and keep Slurm output under `outputs/slurm` and experiment logs/checkpoints under `outputs/logs`.

## Completed Validation

- Smoke 2x2 array `9416807` completed successfully.
- Resume smoke `9416847` completed successfully.
- H200 single-GPU pilot `9419685` completed all four cells.

Single-GPU pilot timing showed flow policy inference is the dominant added cost:

| Cell | Approx action time | Approx SPS |
| --- | ---: | ---: |
| `mlp + gaussian` | 0.0010 s | 666 |
| `flow + gaussian` | 0.0011 s | 590 |
| `mlp + flow` | 0.0028 s | 549 |
| `flow + flow` | 0.0028 s | 532 |

## Single-GPU Full-Like Status

The 100k full-like H200 runs completed enough to validate full single-GPU initialization, pretraining, evaluation, W&B logging, and checkpoint writing. Initial eval averages:

| Run | Avg score | Weighted avg score | Checkpoint |
| --- | ---: | ---: | --- |
| `single-full_wm-mlp_pi-gaussian_seed-1` | 0.0541 | 0.0616 | `100_000_full.pt` |
| `single-full_wm-flow_pi-gaussian_seed-1` | 0.0426 | 0.0509 | `100_000_full.pt` |
| `single-full_wm-mlp_pi-flow_seed-1` | 0.0597 | 0.0618 | `100_000_full.pt` |
| `single-full_wm-flow_pi-flow_seed-1` | 0.0481 | 0.0567 | `100_000_full.pt` |

Follow-up 400k continuation status from local metrics:

| Run | Last observed train step | Last train score | Approx action time | Approx SPS | Max full checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| `single-full_wm-flow_pi-flow_seed-1` | 400,000 | 0.0300 | 0.0035 s | 792.7 | 400,000 |
| `single-full_wm-mlp_pi-gaussian_seed-1` | n/a | n/a | n/a | n/a | 275,000 |
| `single-full_wm-flow_pi-gaussian_seed-1` | n/a | n/a | n/a | n/a | 275,000 |
| `single-full_wm-mlp_pi-flow_seed-1` | n/a | n/a | n/a | n/a | 275,000 |

The three interrupted continuations failed with:

```text
AssertionError: Step 300000 is not a multiple of update frequency 200000.
```

Root cause: after resuming from a non-update-frequency-aligned checkpoint, the first complete resumed episode block preserves the checkpoint step offset. The trainer assertion incorrectly required a zero modulo offset even though the resumed trajectory was internally consistent.

Fix: `Trainer` now stores an `episode_boundary_offset` in checkpoints and falls back to `step % update_freq` for older checkpoints. Episode-boundary checks compare against that offset instead of hard-coding zero.

## Formal 8-GPU Status

Formal jobs were submitted on H200/H100 with `embers` QOS:

| Job | Cell | Observed status |
| --- | --- | --- |
| `9417059_0` | `wm=mlp, policy=gaussian` | Produced train metrics through 1,000,000 steps. Last observed train score: 0.2083. Last action time: 0.00117 s. Last SPS: 200.7. |
| `9417060_1` | `wm=flow, policy=gaussian` | Failed after pretraining with TorchInductor cache path missing. |
| `9417061_2` | `wm=mlp, policy=flow` | Failed after pretraining with TorchInductor cache path missing. |
| `9417062_3` | `wm=flow, policy=flow` | Status not confirmed from Slurm DB during controller outage. |

The formal flow jobs hit:

```text
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/torchinductor_eliu354'
```

Fix: Slurm scripts now create a job-specific `TORCHINDUCTOR_CACHE_DIR` under `${TMPDIR:-/tmp}` before launching Python.

## Current Operational Notes

- PACE Slurm controller and SlurmDB were intermittently unavailable during the latest status check, so local logs and metrics were used as the source of truth.
- Existing outputs are intentionally kept under `outputs/slurm` and `outputs/logs`; no new experiment output should be written to the repository root.
- `AGENTS.md` and `Newt_pipeline_reference.md` are local/untracked context files and should not be committed unless explicitly requested.

## Next Submission Strategy

After the fixes above:

- Resubmit failed single-GPU continuations from their latest full checkpoints to verify offset-aware resume.
- Resubmit failed formal flow jobs with the TorchInductor cache fix.
- Prefer H200, then H100, then A100/L40S if queue availability requires fallback.
- Use distinct Slurm output names, W&B group/name, and run IDs for any duplicate or backup diagnostics to avoid mixing results.

## Repair Submissions

Submitted on 2026-06-04 after commit `c310a49`:

| Job | Purpose | Partition/GPU | Array | Notes |
| --- | --- | --- | --- | --- |
| `9428343` | Formal 2x2 replacement/continuation | `gpu-h200`, 8x H200 | `0-3` | Uses fixed TorchInductor cache setup and resumes from existing model checkpoints when present. |
| `9428344` | Single-GPU full-like repair | `gpu-h200`, 1x H200 | `0-2` | Uses `DIAG_MODE=full`, `DIAG_STEPS=500000`, and same W&B run IDs to verify offset-aware resume for failed continuations. |
| `9428345` | Single-GPU reduced-memory repair | `gpu-h200`, 1x H200 | `0-3` | Uses `DIAG_MODE=reduced`, `DIAG_STEPS=500000` for lower-memory coverage. |

Stale pending job `9417062_3` was canceled before these submissions because it was submitted with the older Slurm script snapshot.

## Weight Cleanup

On 2026-06-04, checkpoint directories for smoke, pilot, and previously failed runs were removed to keep storage organized:

- Removed `models/` under `pipeline_smoke_*` and `pipeline_resume_smoke_*`.
- Removed `models/` under `single-pilot_*`.
- Removed failed single-GPU continuation weights for `single-full_wm-mlp_pi-gaussian_seed-1`, `single-full_wm-flow_pi-gaussian_seed-1`, and `single-full_wm-mlp_pi-flow_seed-1`.
- Removed failed formal pretrain/checkpoint weights for `wm-flow_pi-gaussian_seed-1` and `wm-mlp_pi-flow_seed-1`.

Metrics, W&B records, and Slurm logs were kept. Remaining local model directories after cleanup:

- `outputs/logs/soup/1/single-full_wm-flow_pi-flow_seed-1/models`
- `outputs/logs/soup/1/wm-mlp_pi-gaussian_seed-1/models`

The cleanup reduced `outputs/logs` from approximately 108 GB to 19 GB.

## Runtime Cache Fix

On 2026-06-04, single-GPU repair jobs `9428344` and `9428345` failed after startup because W&B artifact staging and temporary files still used the login-node HOME filesystem quota. HOME was at its 20 GB quota, while project storage had available capacity.

Fixes applied:

- Slurm scripts now set `TMPDIR`, `XDG_CACHE_HOME`, `WANDB_CACHE_DIR`, `WANDB_DATA_DIR`, `WANDB_ARTIFACT_DIR`, and `TORCHINDUCTOR_CACHE_DIR` under `outputs/runtime/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}`.
- Added `wandb_upload_artifacts` config. Slurm jobs set `wandb_upload_artifacts=false`, so W&B metrics continue to sync while checkpoints remain in local `outputs/logs` without a second artifact-staging copy.
- Cleared HOME cache directories `.cache/stable-pretraining` and `.cache/wandb`, reducing HOME usage from the 20 GB quota to about 11.2 GB.

The old pending formal job `9428343` should be canceled and resubmitted with the fixed Slurm script snapshot.

Repair submissions after runtime-cache fix:

| Job | Purpose | Partition/GPU | Array | Notes |
| --- | --- | --- | --- | --- |
| `9430627` | Formal 2x2 replacement/continuation | `gpu-h200`, 8x H200 | `0-3` | Submitted after canceling stale pending job `9428343`; uses project-local runtime cache paths and no W&B artifact staging. |
| `9430628` | Single-GPU full repair | `gpu-h200`, 1x H200 | `0-2` | Uses `DIAG_MODE=full`, `DIAG_STEPS=500000`, project-local cache paths, and no W&B artifact staging. |
| `9430629` | Single-GPU reduced debug | `gpu-h200`, 1x H200 | `0` | Uses `DIAG_MODE=reduced`, `DIAG_STEPS=20000`, and `CUDA_LAUNCH_BLOCKING=1` to locate the reduced-mode scatter/index failure. |

Failed repair weights created by `9428344` and `9428345` were removed again, leaving only `single-full_wm-flow_pi-flow_seed-1/models` and `wm-mlp_pi-gaussian_seed-1/models`.

Jobs `9430628` and `9430629` then failed at the post-pretraining checkpoint save with:

```text
AttributeError: 'Logger' object has no attribute 'cfg'
```

Root cause: the new W&B artifact-upload flag was read through `self.cfg` inside `Logger.save_agent()`, but `Logger` does not keep the full config object. Fix: `Logger` now stores `_wandb_upload_artifacts` during initialization and uses that field when deciding whether to upload checkpoint artifacts.

Repair submissions after the logger fix:

| Job | Purpose | Partition/GPU | Array | Notes |
| --- | --- | --- | --- | --- |
| `9431231` | Formal 2x2 replacement/continuation | `gpu-h200`, 8x H200 | `0-3` | Submitted after canceling stale pending job `9430627`; includes runtime cache and logger artifact fixes. |
| `9431232` | Single-GPU full repair | `gpu-h200`, 1x H200 | `0-2` | Uses `DIAG_MODE=full`, `DIAG_STEPS=500000`, and W&B metrics without artifact staging. |
| `9431233` | Single-GPU reduced debug | `gpu-h200`, 1x H200 | `0` | Uses `DIAG_MODE=reduced`, `DIAG_STEPS=20000`, and `CUDA_LAUNCH_BLOCKING=1`. |

Failed short checkpoints created by `9430628` and `9430629` were removed before these submissions.

Outcome observed on 2026-06-04:

| Job | Outcome |
| --- | --- |
| `9431231` | Formal 2x2 remains pending on `gpu-h200` with reason `Priority`. |
| `9431232` | Single-GPU full repair completed for arrays `0-2` with no stderr errors. Each run reached local checkpoints through `500_000_full.pt`. |
| `9431233` | Reduced debug completed with no stderr errors; no reduced-mode CUDA scatter/index failure reproduced after the logger fix. |

Single-GPU full repair metrics:

| Run | Last train step | Last train score | Approx SPS | Latest full checkpoint |
| --- | ---: | ---: | ---: | --- |
| `single-full_wm-mlp_pi-gaussian_seed-1` | 400,000 | 0.0367 | 939.4 | `500_000_full.pt` |
| `single-full_wm-flow_pi-gaussian_seed-1` | 400,000 | 0.0272 | 942.7 | `500_000_full.pt` |
| `single-full_wm-mlp_pi-flow_seed-1` | 400,000 | 0.0319 | 920.3 | `500_000_full.pt` |

Reduced debug checkpoints were removed after completion. Remaining model directories:

- `outputs/logs/soup/1/single-full_wm-flow_pi-flow_seed-1/models`
- `outputs/logs/soup/1/single-full_wm-flow_pi-gaussian_seed-1/models`
- `outputs/logs/soup/1/single-full_wm-mlp_pi-flow_seed-1/models`
- `outputs/logs/soup/1/single-full_wm-mlp_pi-gaussian_seed-1/models`
- `outputs/logs/soup/1/wm-mlp_pi-gaussian_seed-1/models`

## Checkpoint Eval Repair

On 2026-06-04, chunked checkpoint eval arrays `9433194` and `9433197` began failing:

- Chunks `0-3` failed because `tdmpc2/eval_checkpoint.py` loaded full checkpoint trainer state, including replay, while eval-only `Trainer` is constructed with `buffer=None`.
- ManiSkill-heavy chunks `4-5` additionally failed while creating async vector-env workers with `RuntimeError: Cannot re-initialize CUDA in forked subprocess`, followed by worker `BrokenPipeError`.

Fixes applied:

- `tdmpc2/eval_checkpoint.py` now performs eval-only restore: model weights, running scale, and trainer counters only. It does not restore optimizer, scheduler, replay, or RNG state.
- CUDA setup and seeding now happen after environment construction in `eval_checkpoint.py`.
- `scripts/slurm_flow_single_gpu_eval.sbatch` now passes `env_mode=sync` for checkpoint eval chunks to avoid async worker CUDA fork failures.

Pending old-script chunk arrays `9433197`, `9433199`, and `9433200` should be canceled and replaced after committing these fixes. Formal replacement job `9431231` remains pending on `gpu-h200` with reason `Priority`.

Old-script pending chunk arrays `9433197`, `9433199`, and `9433200` were canceled after the repair commit.

Replacement chunked checkpoint eval submissions:

| Job | Cell(s) | Array | Status at submission |
| --- | --- | --- | --- |
| `9436034` | `mlp + gaussian` | `0-9%2` | Pending |
| `9436035` | `flow + gaussian` | `10-19%2` | Pending |
| `9436038` | `flow + flow` | `30-39%2` | Pending |

The `mlp + flow` chunks (`20-29`) were blocked by `QOSMaxSubmitJobPerUserLimit` during resubmission and should be submitted once the pending/running job count drops.

Status checked on 2026-06-05:

- Repair is working for completed chunks: `9436034_0`, `9436034_1`, `9436034_2`, and `9436034_3` completed successfully and produced W&B/local eval metrics.
- `9436034_4-9` remains pending with `JobArrayTaskLimit` even though chunks `0-3` are complete. `scontrol show job 9436034_4` reports no dependency and no application-level error; this appears to be Slurm array throttle/bookkeeping rather than a Newt failure.
- `9436035_10-19`, `9436038_30-39`, and formal job `9431231_0-3` remain pending on `gpu-h200`.

Additional `mlp + flow` chunks submitted individually:

| Job | Chunk | Status at submission |
| --- | --- | --- |
| `9438508` | `20` | Pending |
| `9438513` | `21` | Pending |
| `9438512` | `22` | Pending |
| `9438514` | `24` | Pending |

Chunks `23` and `25-29` are still blocked by `QOSMaxSubmitJobPerUserLimit`.

Status checked on 2026-06-05 later:

- No new eval or formal stderr logs were produced after the previous check.
- Formal job `9431231_0-3` remains pending on `gpu-h200` with `Priority,Resources`.
- Eval arrays `9436035_10-19` and `9436038_30-39` remain pending with `Priority`.
- `9436034_4-9` was still pending with `JobArrayTaskLimit` even though chunks `0-3` had completed. This was repaired operationally with `scontrol update JobId=9436034 ArrayTaskThrottle=8`; the remaining chunks now show pending reason `Priority`.
- Additional attempts to submit `mlp + flow` chunks `23`, `25`, and `26` were still blocked by `QOSMaxSubmitJobPerUserLimit`.

Status checked again on 2026-06-05:

- `9436034_4-9`, `9436035_10-19`, `9436038_30-39`, and `9438508_20`, `9438512_22`, `9438513_21`, `9438514_24` all completed with exit code `0`.
- No eval stderr indicates an application failure; stderr files contain expected W&B/Gym warnings and run summaries.
- Formal 8-GPU job `9431231_0-3` is still pending on `gpu-h200` with reason `Resources`.

Completed checkpoint eval coverage:

| Cell | Completed chunks | Covered tasks |
| --- | ---: | ---: |
| `mlp + gaussian` | 10/10 | 200/200 |
| `flow + gaussian` | 10/10 | 200/200 |
| `mlp + flow` | 4/10 | 80/200 |
| `flow + flow` | 10/10 | 200/200 |

Additional `mlp + flow` missing chunks were submitted:

| Job | Chunk | Status at submission |
| --- | --- | --- |
| `9448058` | `3` | Pending |
| `9448060` | `5` | Pending |
| `9448061` | `6` | Pending |
| `9448059` | `7` | Pending |
| `9448068` | `8` | Pending |
| `9448069` | `9` | Pending |
