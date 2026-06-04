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
