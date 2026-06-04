#!/usr/bin/env bash
#SBATCH --job-name=newt-flow-2x2
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --array=0-3
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --qos=embers

set -euo pipefail
mkdir -p outputs/slurm

PYTHON_BIN="${PYTHON_BIN:-/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602/bin/python}"
DATA_DIR="${DATA_DIR:-/storage/project/r-agarg35-0/eliu354/external_data/newt/mmbench_demos}"
SEED="${SEED:-1}"
WANDB_GROUP="${WANDB_GROUP:-state20m-soup-2x2}"
export WANDB_PROJECT="${WANDB_PROJECT:-newt-flow-2x2}"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

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
MODEL_DIR="outputs/logs/soup/${SEED}/${NAME}/models"
CHECKPOINT_ARGS=()
if [[ -d "${MODEL_DIR}" ]]; then
  LATEST_CHECKPOINT="$(find "${MODEL_DIR}" -maxdepth 1 -name '*_full.pt' -print | sort -V | tail -n 1)"
  if [[ -n "${LATEST_CHECKPOINT}" ]]; then
    CHECKPOINT_ARGS=(checkpoint="${LATEST_CHECKPOINT}")
    echo "Resuming ${NAME} from ${LATEST_CHECKPOINT}"
  fi
fi

"${PYTHON_BIN}" tdmpc2/train.py \
  task=soup \
  model_size=L \
  obs=state \
  multiproc=true \
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
  data_dir="${DATA_DIR}" \
  "${CHECKPOINT_ARGS[@]}"
