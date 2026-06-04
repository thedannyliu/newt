#!/usr/bin/env bash
#SBATCH --job-name=newt-single-eval
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --array=0-3
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --qos=embers

set -euo pipefail
mkdir -p outputs/slurm

PYTHON_BIN="${PYTHON_BIN:-/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602/bin/python}"
DATA_DIR="${DATA_DIR:-/storage/project/r-agarg35-0/eliu354/external_data/newt/mmbench_demos}"
SEED="${SEED:-1}"
WANDB_GROUP="${WANDB_GROUP:-single-gpu-h200-eval}"
export WANDB_PROJECT="${WANDB_PROJECT:-newt-flow-2x2}"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
RUNTIME_ROOT="${RUNTIME_ROOT:-outputs/runtime/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/xdg_cache"
export WANDB_CACHE_DIR="${RUNTIME_ROOT}/wandb_cache"
export WANDB_DATA_DIR="${RUNTIME_ROOT}/wandb_data"
export WANDB_ARTIFACT_DIR="${RUNTIME_ROOT}/wandb_artifacts"
export TORCHINDUCTOR_CACHE_DIR="${RUNTIME_ROOT}/torchinductor"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${WANDB_ARTIFACT_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

if [[ -z "${WANDB_ENTITY:-}" ]]; then
  echo "WANDB_ENTITY must be set for eval runs." >&2
  exit 1
fi

CELLS=(
  "mlp gaussian"
  "flow gaussian"
  "mlp flow"
  "flow flow"
)
read -r DYNAMICS POLICY <<< "${CELLS[${SLURM_ARRAY_TASK_ID}]}"
SOURCE_NAME="single-full_wm-${DYNAMICS}_pi-${POLICY}_seed-${SEED}"
NAME="eval-${SOURCE_NAME}"
MODEL_DIR="outputs/logs/soup/${SEED}/${SOURCE_NAME}/models"
CHECKPOINT="$(find "${MODEL_DIR}" -maxdepth 1 -name '*_full.pt' -print | sort -V | tail -n 1)"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "No full checkpoint found under ${MODEL_DIR}" >&2
  exit 2
fi

"${PYTHON_BIN}" tdmpc2/eval_checkpoint.py \
  task=soup \
  model_size=L \
  obs=state \
  steps=1 \
  seed="${SEED}" \
  enable_wandb=true \
  wandb_upload_artifacts=false \
  save_agent=false \
  save_replay=false \
  eval_episodes="${EVAL_EPISODES:-1}" \
  compile=false \
  dynamics_arch="${DYNAMICS}" \
  policy_arch="${POLICY}" \
  wandb_group="${WANDB_GROUP}" \
  wandb_name="${NAME}" \
  wandb_run_id="${NAME}" \
  wandb_resume=allow \
  exp_name="${NAME}" \
  data_dir="${DATA_DIR}" \
  checkpoint="${CHECKPOINT}"
