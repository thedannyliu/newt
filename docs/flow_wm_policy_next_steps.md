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
