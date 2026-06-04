#!/usr/bin/env bash
#SBATCH --job-name=newt-flow-smoke
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --array=0-3
#SBATCH --partition=gpu-h200,gpu-h100,gpu-a100,gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --qos=embers

set -euo pipefail
mkdir -p outputs/slurm

PYTHON_BIN="${PYTHON_BIN:-/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602/bin/python}"
DATA_DIR="${DATA_DIR:-/storage/project/r-agarg35-0/eliu354/external_data/newt/mmbench_demos}"
export WANDB_PROJECT="${WANDB_PROJECT:-newt-flow-2x2}"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

CELLS=(
  "mlp gaussian"
  "flow gaussian"
  "mlp flow"
  "flow flow"
)
read -r DYNAMICS POLICY <<< "${CELLS[${SLURM_ARRAY_TASK_ID}]}"
NAME="pipeline_smoke_wm-${DYNAMICS}_pi-${POLICY}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

WANDB_ARGS=(enable_wandb=false)
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ARGS=(enable_wandb=true wandb_group=state2task-smoke-2x2 wandb_name="${NAME}")
fi

"${PYTHON_BIN}" tdmpc2/train.py \
  task=soup \
  task_limit=2 \
  model_size=S \
  obs=state \
  steps=6000 \
  demo_steps=2 \
  seed=1 \
  save_agent=true \
  save_replay=true \
  checkpoint_freq=1000 \
  replay_checkpoint_freq=2000 \
  eval_episodes=1 \
  compile=false \
  dynamics_arch="${DYNAMICS}" \
  policy_arch="${POLICY}" \
  exp_name="${NAME}" \
  data_dir="${DATA_DIR}" \
  "${WANDB_ARGS[@]}"
