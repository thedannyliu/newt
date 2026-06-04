# Newt Flow 2x2 Development Plan

## Summary

- Create development branch `flow-wm-policy-2x2`, while preserving existing untracked reference files.
- Follow the Newt pipeline exactly: official MMBench demos pretraining, online multitask RL, 50:50 demo/online replay, MPPI planning, and formal W&B logging.
- Use state-based 20M soup as the first formal comparison: `obs=state task=soup model_size=L steps=100_000_000 demo_steps=200_000 seed=1`.
- Run a 2x2 ablation: `{MLP WM, Flow WM} x {Gaussian policy, Flow policy}`.

## Key Changes

- Add configurable architecture switches:
  - `dynamics_arch=mlp|flow`
  - `policy_arch=gaussian|flow`
  - `flow_steps=4`, `flow_t_dim=64`, `checkpoint_freq=500_000`, `replay_checkpoint_freq=2_000_000`, `save_replay=True`
  - W&B defaults read from `WANDB_PROJECT` and `WANDB_ENTITY`; formal runs fail fast if entity is missing.
- Flow WM uses conditional rectified flow matching:
  - condition: `[z_t, action_t, task_embedding]`
  - target: `stopgrad(z_{t+1}) - z_t`
  - training loss: velocity MSE
  - inference: deterministic Euler integration from zero base noise, then `SimNorm(z_t + residual)`
- Flow policy uses conditional action flow:
  - condition: `[z_t, task_embedding]`
  - target: masked valid action dimensions
  - BC prior loss becomes masked flow-matching loss
  - Max-Q actor loss still backpropagates through sampled actions
  - exact entropy/log-prob is not used; flow-policy entropy is logged as zero
- Add full checkpoint/resume:
  - save model, optimizers, scheduler, running scale, trainer step, episode index, update tokens, RNG state, and online replay buffer
  - reload offline demos from `data_dir` instead of embedding them in checkpoints
  - keep local checkpointing independent from W&B
  - upload only lightweight model artifacts to W&B; record full replay checkpoint paths locally
- Data and weights:
  - download demos from `nicklashansen/mmbench`
  - use official `nicklashansen/newt` checkpoints for sanity/eval reference only, not as 2x2 formal initialization
  - ignore `data/`, `checkpoints/`, `wandb/`, and `*.pt`

## Experiment Plan

- Smoke:
  - run all four cells on a small task/small-step configuration
  - verify forward, update, planning, checkpoint, resume, W&B/local metrics
  - use `/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602/bin/python`
- Formal:
  - `wm-mlp_pi-gaussian_seed-1`
  - `wm-flow_pi-gaussian_seed-1`
  - `wm-mlp_pi-flow_seed-1`
  - `wm-flow_pi-flow_seed-1`
  - W&B group: `state20m-soup-2x2`
  - PACE-Phoenix Slurm with QOS `embers`; do not use `inferno`
  - give every run a distinct output root, checkpoint root, and W&B name
- Results:
  - add local `metrics.jsonl` and eval CSV logging
  - aggregate average score, weighted domain score, per-domain CSV, and per-task CSV into a 2x2 table

## Test Plan

- Import and shape tests:
  - baseline MLP/Gaussian behavior remains available
  - Flow WM and Flow policy pass fake-batch shape, mask, and gradient checks
- Checkpoint tests:
  - train for a few updates, save full checkpoint, reload, and continue
  - verify step, optimizer, scheduler, and buffer sampling resume
- Smoke Slurm:
  - each 2x2 cell completes a mini pretrain, mini online run, eval, checkpoint, and W&B/local metrics
- Formal acceptance:
  - all four cells have W&B runs, local checkpoints, resume support, and a generated 2x2 table

## Assumptions

- Flow family is Flow Matching / rectified flow.
- First formal table uses single seed `seed=1`; multi-seed is a later extension.
- RGB is excluded from the first formal table to match the state-based Newt main pipeline.
- Primary references:
  - https://www.nicklashansen.com/NewtWM/
  - https://arxiv.org/abs/2511.19584
  - https://huggingface.co/datasets/nicklashansen/mmbench
  - https://huggingface.co/nicklashansen/newt
  - https://arxiv.org/abs/2210.02747
  - https://arxiv.org/abs/2509.25756
