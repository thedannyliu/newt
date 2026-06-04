#!/usr/bin/env bash
#SBATCH --job-name=newt-flow-2x2
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --qos=embers

set -euo pipefail
mkdir -p outputs/slurm

PYTHON_BIN="${PYTHON_BIN:-/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602/bin/python}"
DATA_DIR="${DATA_DIR:-/storage/project/r-agarg35-0/eliu354/external_data/newt/mmbench_demos}"
SEED="${SEED:-1}"
WANDB_GROUP="${WANDB_GROUP:-state20m-soup-2x2}"
export WANDB_PROJECT="${WANDB_PROJECT:-newt-flow-2x2}"

if [[ -z "${WANDB_ENTITY:-}" ]]; then
  echo "WANDB_ENTITY must be set for formal runs." >&2
  exit 1
fi

CELLS=(
  "mlp gaussian"
  "flow gaussian"
  "mlp flow"
  "flow flow"
)
read -r DYNAMICS POLICY <<< "${CELLS[${SLURM_ARRAY_TASK_ID}]}"
NAME="wm-${DYNAMICS}_pi-${POLICY}_seed-${SEED}"

"${PYTHON_BIN}" tdmpc2/train.py \
  task=soup \
  model_size=L \
  obs=state \
  steps=100000000 \
  demo_steps=200000 \
  seed="${SEED}" \
  enable_wandb=true \
  save_agent=true \
  save_replay=true \
  checkpoint_freq=500000 \
  replay_checkpoint_freq=2000000 \
  dynamics_arch="${DYNAMICS}" \
  policy_arch="${POLICY}" \
  wandb_group="${WANDB_GROUP}" \
  wandb_name="${NAME}" \
  wandb_run_id="${NAME}" \
  wandb_resume=allow \
  exp_name="${NAME}" \
  data_dir="${DATA_DIR}"
