# Experiment Results Summary

Last updated: 2026-06-11 03:38 EDT

Metric: `avg_score`, the Newt/MMBench normalized score averaged over evaluated tasks. This is the same normalized-score definition used by the official Newt paper, but our local subset20/subset40 results are not directly comparable to the official 200-task protocol.

| Experiment family | Tasks | Budget / protocol | WM | Policy | GPU/source | Status | Eval step | avg_score | Notes |
| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |
| Official Newt paper | 200 | 100M total, state obs | Newt MLP WM | Gaussian prior + MPC | paper | complete | 100M | 0.438 | Official full MMBench reference. |
| Official Newt paper | 200 | 20M total, state obs | Newt MLP WM | Gaussian prior + MPC | paper | complete | 20M | 0.31 | Official early reference. |
| Official TD-MPC2 single-task | 200 | 1B total across 200 single-task agents | MLP WM | Gaussian + MPC | paper | complete | 1B | 0.80 | Upper reference, not a multitask agent. |
| Pipeline smoke | small subset | smoke / resume validation | MLP / flow | Gaussian / flow | Slurm | pass | n/a | n/a | Validated pipeline, W&B, checkpoint/resume, and subset parsing; not a scientific result. |
| Single-GPU pilot | 200 | pilot/reduced screen | flow | flow | local eval | screen | 0 | 0.09615 | Startup/perf screen; not final protocol. |
| Single-GPU pilot | 200 | pilot/reduced screen | flow | Gaussian | local eval | screen | 0 | 0.09871 | Startup/perf screen; not final protocol. |
| Single-GPU pilot | 200 | pilot/reduced screen | MLP | flow | local eval | screen | 0 | 0.09012 | Startup/perf screen; not final protocol. |
| Single-GPU pilot | 200 | pilot/reduced screen | MLP | Gaussian | local eval | screen | 0 | 0.07122 | Startup/perf screen; not final protocol. |
| Single-GPU reduced | 200 | reduced-memory screen | flow | flow | local eval | screen | 0 | 0.05017 | Startup/perf screen; not final protocol. |
| Single-GPU reduced | 200 | reduced-memory screen | MLP | flow | local eval | screen | 0 | 0.05092 | Startup/perf screen; not final protocol. |
| Single-GPU reduced | 200 | reduced-memory screen | MLP | Gaussian | local eval | screen | 0 | 0.05175 | Startup/perf screen; not final protocol. |
| Single-GPU full chunk eval | 200 | chunked eval screen | flow | flow | chunk mean | screen | 0 | 0.03150 | 10 chunks; not final training result. |
| Single-GPU full chunk eval | 200 | chunked eval screen | flow | Gaussian | chunk mean | screen | 0 | 0.03889 | 10 chunks; not final training result. |
| Single-GPU full chunk eval | 200 | chunked eval screen | MLP | flow | chunk mean | screen | 0 | 0.02379 | 10 chunks; not final training result. |
| Single-GPU full chunk eval | 200 | chunked eval screen | MLP | Gaussian | chunk mean | screen | 0 | 0.04799 | 10 chunks; not final training result. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | MLP | flow | H200 | complete | 5M | 0.33909 | Best completed subset40 2x2 cell. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | MLP | flow | A100 | complete | 5M | 0.31517 | Strong MLP-WM replicate. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | MLP | flow | L40S | complete | 5M | 0.31512 | Strong MLP-WM replicate. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | MLP | flow | H100 | complete | 5M | 0.31489 | Strong MLP-WM replicate. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | MLP | Gaussian | A100 | complete | 5M | 0.30762 | Best Gaussian-policy MLP replicate. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | MLP | Gaussian | H100 | complete | 5M | 0.30250 | MLP baseline. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | MLP | Gaussian | H200 | complete | 5M | 0.28482 | MLP baseline. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | MLP | Gaussian | L40S | complete | 5M | 0.26794 | MLP baseline. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | flow | Gaussian | H100 | complete | 5M | 0.13614 | Pure flow WM weak. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | flow | flow | A100 | complete | 5M | 0.13609 | Pure flow WM weak. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | flow | flow | H200 | complete | 5M | 0.12196 | Pure flow WM weak. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | flow | flow | L40S | complete | 5M | 0.11709 | Pure flow WM weak. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | flow | Gaussian | H200 | complete | 5M | 0.10356 | Pure flow WM weak. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | flow | Gaussian | A100 | complete | 5M | 0.10345 | Pure flow WM weak. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | flow | flow | H100 | complete | 5M | 0.10061 | Pure flow WM weak. |
| Subset40 5M 2x2 | 40 | 5M + 50k demo | flow | Gaussian | L40S | complete | 5M | 0.08094 | Weakest completed 2x2 cell. |
| Subset40 flow-step ablation | 40 | 400k eval + 50k demo | flow, `flow_steps=1` | Gaussian | H200/H100/A100/L40S | complete | 400k | 0.09300-0.12381 | Best L40S; action-time range 0.059-0.191s. |
| Subset40 flow-step ablation | 40 | 400k eval + 50k demo | flow, `flow_steps=2` | Gaussian | H200/H100/A100/L40S | complete | 400k | 0.08417-0.10998 | Best H100; action-time range 0.074-0.244s. |
| Subset40 flow-step ablation | 40 | 400k eval + 50k demo | flow, `flow_steps=4` | Gaussian | H200/H100/A100/L40S | complete | 400k | 0.09120-0.11430 | More flow steps increase action time; no clear score gain. |
| Subset40 WM variant | 40 | 5M + 50k demo | residual_flow | Gaussian | L40S | complete | 5M | 0.25576 | Best completed flow-WM family result. |
| Subset40 WM variant | 40 | 5M + 50k demo | residual_flow | Gaussian | A100 | complete | 5M | 0.25500 | Close to best flow-WM result. |
| Subset40 WM variant | 40 | 5M + 50k demo | residual_flow | Gaussian | H100 | complete | 5M | 0.21465 | Lower replicate. |
| Subset40 WM variant | 40 | 5M + 50k demo | residual_flow | Gaussian | H200 | complete | 5M | 0.20264 | Lower replicate. |
| Subset40 WM variant | 40 | 5M + 50k demo | shortcut_residual_flow_wm | Gaussian | A100 | complete | 5M | 0.24545 | Best shortcut replicate. |
| Subset40 WM variant | 40 | 5M + 50k demo | shortcut_residual_flow_wm | Gaussian | H100 | complete | 5M | 0.24033 | Comparable to residual_mean. |
| Subset40 WM variant | 40 | 5M + 50k demo | shortcut_residual_flow_wm | Gaussian | H200 | complete | 5M | 0.23225 | Below A100/H100 shortcut. |
| Subset40 WM variant | 40 | 5M + 50k demo | shortcut_residual_flow_wm | Gaussian | L40S | complete | 5M | 0.20876 | Direct eval `9764791_[32]`; weak shortcut replicate. |
| Subset40 WM variant | 40 | 5M + 50k demo | residual_mean_flow_wm | Gaussian | H100 | complete | 5M | 0.24538 | Best residual_mean replicate. |
| Subset40 WM variant | 40 | 5M + 50k demo | residual_mean_flow_wm | Gaussian | A100 | complete | 5M | 0.23934 | Below best residual_flow. |
| Subset40 WM variant | 40 | 5M + 50k demo | residual_mean_flow_wm | Gaussian | H200 | complete | 5M | 0.22482 | Below best residual_flow. |
| Subset40 WM variant | 40 | 5M + 50k demo | residual_mean_flow_wm | Gaussian | L40S | complete | 5M | 0.20985 | Weak residual_mean replicate. |
| Subset40 WM variant | 40 | 5M + 50k demo | ot_cfm_residual_wm | Gaussian | H200 | complete | 5M | 0.23227 | OT-CFM consistently below residual_flow. |
| Subset40 WM variant | 40 | 5M + 50k demo | ot_cfm_residual_wm | Gaussian | H100 | complete | 5M | 0.23161 | OT-CFM consistently below residual_flow. |
| Subset40 WM variant | 40 | 5M + 50k demo | ot_cfm_residual_wm | Gaussian | L40S | complete | 5M | 0.22143 | Weak OT-CFM replicate. |
| Subset40 WM variant | 40 | 5M + 50k demo | endpoint_flow | Gaussian | A100 | complete | 5M | 0.19807 | Endpoint flow not competitive. |
| Subset40 WM variant | 40 | 5M + 50k demo | endpoint_flow | Gaussian | H200 | complete | 5M | 0.19475 | Endpoint flow not competitive. |
| Subset40 WM variant | 40 | 5M + 50k demo | endpoint_flow | Gaussian | L40S | complete | 5M | 0.18724 | Endpoint flow not competitive. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | MLP | Gaussian | H100 | complete | 10.0M | 0.53816 | First completed 10M subset20 cell; peak observed eval was 0.59102 at 8.8M. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | MLP | flow | L40S | resubmitted | 8.8M | 0.46663 | Preempted again and resubmitted as `9820444_[1]`; peak observed eval was 0.55132 at 8.0M. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | MLP | Gaussian | L40S | complete | 10.0M | 0.48801 | Completed 10M despite Slurm preemption label; peak observed eval was 0.55139 at 7.4M. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | MLP | Gaussian | H200 | reached target | 10.0M | 0.49223 | Reached 10M while job `9809460_0` was still active; peak observed eval was 0.53666 at 7.4M. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | MLP | flow | H100 | complete | 10.0M | 0.50044 | Reached 10M; peak observed eval was 0.58825 at 4.0M. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | MLP | flow | H200 | complete | 10.0M | 0.53852 | Reached 10M; peak observed eval was 0.58663 at 8.8M. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | MLP | Gaussian | A100 | complete | 10.0M | 0.53570 | Completed 10M; peak observed eval was 0.56490 at 9.4M. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | MLP | flow | A100 | resubmitted | 9.0M | 0.60544 | Best subset20 MLP+flow snapshot so far; preempted and resubmitted as `9820443_[1]`. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | residual_flow | Gaussian | L40S | resubmitted | 6.6M | 0.43467 | Preempted again and resubmitted as `9820444_[2]`; peak observed eval was 0.44826 at 5.6M. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | residual_mean_flow_wm | Gaussian | A100 | resubmitted | 8.4M | 0.46637 | Preempted and resubmitted as `9820443_[3]`; peak observed eval was 0.48471 at 7.2M. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | residual_mean_flow_wm | Gaussian | H200 | running | 8.2M | 0.41040 | Active as `9813253_3`; peak observed eval was 0.46176 at 8.0M. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | residual_mean_flow_wm | Gaussian | H100 | complete | 10.0M | 0.48851 | Completed 10M despite Slurm preemption label; peak observed eval was 0.49207 at 9.2M. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | residual_flow | Gaussian | H100 | resubmitted | 8.8M | 0.44898 | Preempted again and resubmitted as `9820441_[2]`; peak observed eval was 0.51065 at 6.0M. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | residual_flow | Gaussian | A100 | resubmitted | 6.6M | 0.44067 | Timed out and resubmitted as `9820443_[2]`; peak observed eval was 0.47201 at 6.0M. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | residual_mean_flow_wm | Gaussian | L40S | resubmitted | 8.4M | 0.40662 | Preempted again and resubmitted as `9820444_[3]`; peak observed eval was 0.41503 at 7.4M. |
| Subset20 10M high-step | 20 | 10M target + 50k demo | residual_flow | Gaussian | H200 | resubmitted | 6.4M | 0.43120 | Preempted and resubmitted as `9820442_[2]`; peak observed eval was 0.47019 at 5.4M. |

Current read:

- Best completed subset40 result is `MLP WM + flow policy` on H200: `0.33909`.
- Best completed flow-WM result is `residual_flow + Gaussian` on L40S/A100: `0.25576` / `0.25500`.
- New residual variants (`shortcut`, `residual_mean`, `ot_cfm`) did not beat plain `residual_flow` on completed subset40 evals.
- The subset20 high-step direction is much stronger overall, with completed 10M rows for H100/L40S `MLP + Gaussian` and H100 `residual_mean_flow_wm + Gaussian`; these are still not directly comparable to official 200-task results.
- Current subset20 ranking still favors MLP WM. A100 `MLP + flow` reached the best peak so far, 0.60544 at 9.0M, but completed MLP+flow finals are lower.
- The strongest completed original-style baseline remains `MLP + Gaussian`, with H100 0.53816 and A100 0.53570 at 10M; H200 reached 10M at 0.49223.
- H100 `residual_flow + Gaussian` remains the strongest flow-WM signal by peak at 0.51065, but its latest score regressed to 0.44898 by 8.8M and it remains slower.
