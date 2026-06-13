# Ablations and Insights

Generated: 2026-06-11 EDT

Source metrics:

- Local run logs: `outputs/logs/soup/1/*/metrics.jsonl`
- Generated tables/figures: `docs/assets/ablation_insights_20260610/`
- W&B project: https://wandb.ai/danny010324/newt-flow-2x2
- Latest figure refresh: 2026-06-12 20:17 EDT. The subset20 high-step CSVs and figures were regenerated after additional A100/L40S residual_flow continuation metrics arrived.

Important caveat: `mean +/- std` below is over completed local runs across GPU/replicate labels, not over independent random seeds unless explicitly stated.

## Key Results

| Experiment name | # Tasks | Budget / protocol | WM | Policy | Eval step | Avg_score | Peak_score | Notes |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| Official Newt paper | 200 | 100M total, state obs | Newt MLP WM | Gaussian prior + MPC | 100M | 0.438 | n/a | Official full MMBench reference. |
| Official Newt paper | 200 | 20M total, state obs | Newt MLP WM | Gaussian prior + MPC | 20M | 0.310 | n/a | Official early reference. |
| Official TD-MPC2 single-task | 200 | 1B total across 200 single-task agents | MLP WM | Gaussian + MPC | 1B | 0.800 | n/a | Upper reference, not a multitask agent. |
| Our subset40 best completed 2x2 | 40 | 5M + 50k demo | MLP WM | Flow policy | 5M | 0.321 +/- 0.012 | 0.339 | Mean +/- std over 4 completed runs; best single run 0.339. Best completed 2x2 architecture. |
| Our subset40 best flow-WM | 40 | 5M + 50k demo | Residual_flow WM | Gaussian | 5M | 0.232 +/- 0.027 | 0.256 | Mean +/- std over 4 completed runs; best single run 0.256. Best flow-WM variant, still below MLP WM. |
| Our subset20 strongest signal | 20 | 10M target + 50k demo | MLP WM | Flow policy | 10.0M | 0.500-0.578 | 0.605 | All four hardware/replicate labels reached 10M; A100 is best with 0.578 final and 0.605 peak. |
| Our subset20 promising flow-WM | 20 | 10M target + 50k demo | Residual_flow WM | Gaussian | 7.8M-10.0M current | 0.386-0.463 | 0.511 | H100 residual_flow peak is still the strongest flow-WM signal; H100/H200 10M finals are 0.453/0.455 and remain below MLP-WM. |

## Architecture Ablation

### WM x Policy 2x2

Protocol: subset40, 5M online steps, 50k demo pretraining, state observations, closed-loop planning eval.

| WM | Policy | Runs | Final avg_score | Peak single run | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| MLP | Flow | 4 | 0.321 +/- 0.012 | 0.339 | Best subset40 2x2 architecture. |
| MLP | Gaussian | 4 | 0.291 +/- 0.018 | 0.308 | Strong baseline, below MLP+flow. |
| Best Flow-WM example: residual_flow | Gaussian | 4 | 0.232 +/- 0.027 | 0.256 | Best flow-WM family result, but still below MLP WM. |
| Best Flow-WM example: residual_flow | Flow | 0 | not run | not run | Missing cell; current flow-policy evidence is only with MLP WM or pure flow WM. |

Figure:

![Subset40 2x2 heatmap](assets/ablation_insights_20260610/subset40_2x2_heatmap.png)

The heatmap above is the literal original 2x2 with pure `flow` WM. The table uses `residual_flow` as the more informative Flow-WM example because pure flow was clearly dominated.

Training curves:

![Subset40 2x2 training curves](assets/ablation_insights_20260610/subset40_2x2_training_curves.png)

This curve plot includes the literal 2x2 cells plus `residual_flow + Gaussian` as the best Flow-WM reference line.

Conclusion:

- The strongest signal is not replacing the world model with pure flow.
- The best complete 2x2 result is **MLP WM + flow policy**, suggesting the policy prior is a better first target than fully replacing the latent dynamics.
- The best Flow-WM example so far is `residual_flow + Gaussian`, but it is still below both MLP policy variants on subset40.

### Flow-WM Variants

Protocol: subset40, 5M online steps, 50k demo pretraining, Gaussian policy, state observations, closed-loop planning eval.

| WM variant | Runs | Final avg_score | Peak single run | Interpretation |
| --- | ---: | ---: | ---: | --- |
| residual_flow | 4 | 0.232 +/- 0.027 | 0.256 | Best flow-WM family result, but still below MLP WM. |
| shortcut_residual_flow_wm | 4 | 0.232 +/- 0.016 | 0.245 | Similar to residual_flow, no clear win. |
| residual_mean_flow_wm | 4 | 0.230 +/- 0.016 | 0.245 | Similar to shortcut/residual, no clear win. |
| ot_cfm_residual_wm | 3 | 0.228 +/- 0.006 | 0.232 | Stable but not better than residual_flow. |
| endpoint_flow | 3 | 0.193 +/- 0.006 | 0.198 | Better than pure flow, still weak. |
| flow | 4 | 0.106 +/- 0.023 | 0.136 | Direct pure flow WM is weakest. |

Variant definitions:

- `flow`: directly models the next latent with a conditional rectified-flow dynamics network.
- `endpoint_flow`: predicts the endpoint/average velocity toward the next latent instead of integrating a standard multi-step flow.
- `residual_flow`: keeps an MLP next-latent predictor and adds a rectified-flow residual correction.
- `residual_mean_flow_wm`: keeps the MLP predictor and adds a one-step mean-flow residual correction.
- `shortcut_residual_flow_wm`: keeps the MLP predictor and uses a step-size-conditioned shortcut flow for the residual.
- `ot_cfm_residual_wm`: keeps the MLP predictor and trains the residual flow with sliced-OT conditional flow matching.

Figure:

![Subset40 flow-WM variants](assets/ablation_insights_20260610/subset40_flow_wm_variants.png)

Training curves:

![Subset40 flow-WM variant training curves](assets/ablation_insights_20260610/subset40_flow_wm_variants_training_curves.png)

Conclusion:

- Residualizing flow over an MLP-like base is necessary; direct flow dynamics is not competitive.
- The residual variants cluster tightly around 0.23, so none currently justify replacing the MLP WM.
- Further flow work should focus on uncertainty/calibration or policy/planner integration, not more direct next-latent flow variants.

## Compute / Runtime Ablation

Protocol: subset20 high-step runs on **H100 only**. This avoids mixing architecture effects with H100/H200/A100/L40S runtime differences. Scores are current latest/peak local metrics; not all rows are complete at 10M.

Definitions:

- `mean action_time`: average logged time to select/plan one environment action at the latest train step.
- `mean update_time`: average logged optimizer/world-model update time per update at the latest train step.
- `mean eval time`: log-derived estimate from `eval.elapsed_time - previous train.elapsed_time`; useful for relative comparison, but not a profiler-grade measurement.

| Architecture | H100 eval step | Current avg_score | Peak score | Mean action_time | Mean update_time | Mean eval time | Compute read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MLP + flow policy | 10.0M | 0.500 | 0.588 | 0.060s | 0.046s | 86.8s | Best current peak score signal, but final score dipped while action planning remains slower than Gaussian. |
| MLP + Gaussian | 10.0M | 0.538 | 0.591 | 0.035s | 0.038s | 61.7s | Original Newt setting; fastest strong baseline on H100. |
| residual_flow + Gaussian | 10.0M | 0.453 | 0.511 | 0.083s | 0.074s | 103.4s | Best flow-WM peak so far, but 10M final remains below MLP choices and action/update time is higher. |
| residual_mean_flow_wm + Gaussian | 10.0M | 0.489 | 0.492 | 0.049s | 0.048s | 71.8s | Cheaper than residual_flow but still below MLP WM. |

Figure:

![Subset20 H100 runtime bars](assets/ablation_insights_20260610/subset20_h100_runtime_bars.png)

Conclusion:

- `MLP + flow policy` is the best score candidate but increases action/planning-time cost.
- `residual_flow` costs more action/update time while producing lower score, so it is currently dominated by MLP-based choices.
- Logged `steps_per_second` and wall-clock estimates are resume/preemption-sensitive; use them as rough diagnostics, not final claims. For final reporting, compute wall-clock from Slurm accounting after jobs finish.

## Training Budget / Scaling Ablation

Best setting here intentionally uses the original Newt architecture, **MLP WM + Gaussian policy**, so this isolates task/step scaling from architecture changes.

| Setting | Tasks | Total steps | Steps per task | Score | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| subset40 5M MLP+Gaussian | 40 | 5M | 125k | 0.291 | Original setting short-run baseline; per-task interaction is low. |
| subset20 10M MLP+Gaussian | 20 | 10M | 500k | 0.538 | Original setting; same per-task budget as official 100M/200-task. |
| Official Newt 20M | 200 | 20M | 100k | 0.310 | Official early 200-task reference. |
| Official Newt 100M | 200 | 100M | 500k | 0.438 | Official full 200-task reference. |

Figure:

![Scaling by steps per task](assets/ablation_insights_20260610/scaling_steps_per_task.png)

Subset20 high-step training curves, H100 only:

![Subset20 H100 training curves](assets/ablation_insights_20260610/subset20_h100_training_curves.png)

Conclusion:

- Reducing task count while increasing per-task steps gives a clearer architecture signal.
- The subset20 scores are not directly comparable to official 200-task scores, but they show that the implementation can learn strongly when per-task interaction is sufficient.
- Next scaling test should be either subset20/20M for stronger convergence or subset40/10M to test whether the signal survives more tasks.

## W&B Locations

Project:

- https://wandb.ai/danny010324/newt-flow-2x2

Useful groups/runs to inspect:

- `subset20-10m-highsteps`: current high-step subset20 runs.
- `subset40-5m`: original subset40 2x2 runs.
- `subset40-5m-new-flowwm-variants`: residual/shortcut/OT-CFM flow-WM variants.
- `subset40-flowsteps-500k`: flow step-count runtime ablation.

Recommended W&B panels:

- `eval/avg_score` vs `step`
- `train/episode_score` vs `step`
- `train/action_time` vs `step`
- `train/update_time` vs `step`
- `train/steps_per_second` vs `step`
- domain eval metrics such as `eval/avg_score_metaworld`, `eval/avg_score_maniskill`, and `eval/avg_score_dmcontrol`

## Generated Files

Tables:

- `docs/assets/ablation_insights_20260610/subset40_2x2_summary.csv`
- `docs/assets/ablation_insights_20260610/subset40_flow_wm_variants_summary.csv`
- `docs/assets/ablation_insights_20260610/subset20_highstep_summary.csv`
- `docs/assets/ablation_insights_20260610/subset20_runs.csv`
- `docs/assets/ablation_insights_20260610/subset20_h100_runtime_summary.csv`
- `docs/assets/ablation_insights_20260610/subset40_flowsteps_summary.csv`
- `docs/assets/ablation_insights_20260610/scaling_summary.csv`

Figures:

- `docs/assets/ablation_insights_20260610/subset40_2x2_heatmap.png`
- `docs/assets/ablation_insights_20260610/subset40_2x2_training_curves.png`
- `docs/assets/ablation_insights_20260610/subset40_flow_wm_variants.png`
- `docs/assets/ablation_insights_20260610/subset40_flow_wm_variants_training_curves.png`
- `docs/assets/ablation_insights_20260610/subset20_highstep_training_curves.png`
- `docs/assets/ablation_insights_20260610/subset20_h100_training_curves.png`
- `docs/assets/ablation_insights_20260610/subset20_h100_runtime_bars.png`
- `docs/assets/ablation_insights_20260610/subset20_compute_vs_score.png`
- `docs/assets/ablation_insights_20260610/scaling_steps_per_task.png`
