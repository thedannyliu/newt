# Newt Progress Report Revision Suggestions

These are suggested English additions for `docs/Newt_progress.pdf`, focused on the empty `Conclusion & Insights` and `Problems & Next Steps` sections. They are written to complement the existing tables rather than repeat them.

## Quick Fixes Before Finalizing

- In Key Results, the `Our subset40 best completed 2x2` note should say `best single run 0.339`, not `0.256`.
- Replace `randomly choosen` with `randomly chosen`.
- Clarify that current `mean +/- std` values are mostly across completed GPU/replicate runs, not true independent random seeds, unless a table explicitly uses seeds.
- Keep the caveat that subset20/subset40 scores are not directly comparable to the official 200-task Newt score, because the task distribution and number of tasks differ.

## Suggested: Conclusion & Insights

- **We preserved the Newt pipeline and isolated architecture changes.**
  - The experiments keep the original offline-pretrain + online-RL + MPC evaluation structure.
  - The main changes are only the latent world-model dynamics and the policy prior.
  - This makes the results interpretable as architecture ablations, not a new pipeline.

- **The main signal is asymmetric: flow policy is more promising than flow WM.**
  - `MLP WM + flow policy` gives the strongest peak-score signal.
  - Replacing the MLP world model with flow dynamics has not reliably improved performance.
  - Interpretation: the policy prior may benefit from richer action distributions, while the world model must remain stable and cheap for repeated MPC rollouts.

- **The original MLP world model is a strong inductive bias for Newt.**
  - Closed-loop MPC repeatedly queries the latent dynamics during planning.
  - A small deterministic MLP predictor can work well if it is locally accurate, stable, and fast.
  - Flow dynamics adds expressivity, but that expressivity is not useful unless the planner can exploit uncertainty or multi-modal predictions.

- **Flow-WM works better as a residual correction than as a full replacement.**
  - Pure flow dynamics is clearly weak.
  - Residual-flow variants are consistently better than pure flow, but still below MLP-WM baselines.
  - Research implication: flow is more plausible as an uncertainty/correction layer on top of MLP dynamics than as the entire dynamics model.

- **Higher per-task interaction gives clearer architecture signals.**
  - subset40 5M is mainly a fast screen: about 125k steps per task.
  - subset20 10M reaches about 500k steps per task, matching the per-task budget of the official 100M/200-task setting.
  - Scores become much stronger in subset20, suggesting architecture comparisons need enough online data to be meaningful.

- **Compute cost changes the conclusion.**
  - `MLP WM + Gaussian policy` remains the strongest Pareto baseline: reliable, stable, and fast.
  - `MLP WM + flow policy` has the best peak signal but is slower and less stable.
  - `Residual-flow WM + Gaussian` is currently dominated: slower than MLP baselines and lower in score.

- **Current research direction.**
  - Do not prioritize making the whole world model a flow.
  - Prioritize flow where distributional modeling can affect planning or policy warm-starting.
  - Most promising directions: cheaper flow policy, residual/uncertainty-aware flow WM, and planner-aware flow rollouts.

## Suggested: Problems & Next Steps

- **Current comparison is not yet official-grade.**
  - Most averages are over completed GPU/replicate runs, not true random seeds.
  - subset20/subset40 are useful architecture screens, but they are not directly comparable to the official 200-task Newt benchmark.
  - Next: rerun the strongest cells with fixed tasks, fixed hardware, and at least 3 true seeds.

- **Training stability is still unresolved.**
  - Some high-performing runs show strong peak scores but lower final scores, especially `MLP WM + flow policy` and residual-flow WM.
  - Final score alone may miss useful transient gains; peak score alone may overstate robustness.
  - Next: report final score, peak score, score-vs-step curves, and area under the learning curve.

- **Average score is too coarse to explain why flow helps or fails.**
  - Flow may help specific domains, horizons, or contact-heavy tasks while hurting others.
  - A single averaged MMBench score cannot distinguish global failure from domain-specific tradeoffs.
  - Next: add domain-level curves and a small number of task-level visualizations; keep the full task breakdown in the appendix.

- **Flow-WM may have a planner/objective mismatch.**
  - The flow dynamics are trained for next-latent prediction, but MPC needs stable multi-step rollouts under candidate action sequences.
  - If flow uncertainty is not used by the planner, extra expressivity can become sampling noise rather than useful planning information.
  - Next: test multi-step latent rollout losses, uncertainty-aware MPC scoring, and residual corrections that only model MLP prediction error.

- **Flow policy has a real signal, but its compute cost must be justified.**
  - `MLP WM + flow policy` gives the strongest peak signal, but action selection is slower than the Gaussian prior.
  - The key question is whether the score gain survives true-seed repeats and remains Pareto-efficient.
  - Next: try cheaper flow-policy variants, including fewer flow steps, flow-to-Gaussian distillation, or using flow only to initialize CEM candidates.

- **Recommended next experiments.**
  - Run controlled subset20 10M true-seed repeats for `MLP+Gaussian`, `MLP+flow policy`, `residual_flow+Gaussian`, and `residual_mean_flow+Gaussian`.
  - Extend the best two subset20 cells to 20M to test whether peak-to-final drops are noise or real plateaus.
  - Run subset40 10M for `MLP+Gaussian` and `MLP+flow policy` to test whether the flow-policy signal survives more tasks.
  - Add world-model diagnostics: one-step latent error, multi-step rollout error, reward prediction error, Q-value consistency, and flow-sample calibration.
  - Test a planner-aware flow-WM where MPC evaluates multiple latent transition samples or uses uncertainty/risk penalties.
  - Scale toward the full 200-task official setting only after the subset20/subset40 signal is robust.

## Optional Ablation Mini-Conclusions

### Architecture Ablation

The 2x2 result should be read as evidence that the MLP world model is currently the anchor of performance. Flow policy can help when attached to the MLP world model, but flow dynamics does not rescue performance when the MLP dynamics is removed.

### Flow-WM Variants

The variant comparison shows that residualizing flow is necessary but not sufficient. Pure flow is clearly weak, endpoint flow is better but still not competitive, and residual/shortcut/OT-CFM variants are close to each other. This means the current limitation is probably not just the exact flow parameterization; it is how the flow prediction is used by the planner and trained for rollout utility.

### Compute / Runtime Ablation

Runtime should be treated as part of the scientific result. A variant that slightly improves peak score but substantially slows action selection may not be attractive for MPC-based online RL, where the world model is queried many times per environment step.

### Training Budget / Scaling Ablation

The scaling ablation supports using fewer tasks with higher per-task steps as an intermediate research regime. It gives a clearer learning signal while preserving the same offline-pretrain plus online-RL pipeline, but it should be presented as an architecture-development setting rather than an official benchmark replacement.
