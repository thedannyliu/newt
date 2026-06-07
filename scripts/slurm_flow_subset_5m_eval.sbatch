#!/usr/bin/env bash
#SBATCH --job-name=newt-subset5m-eval
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --array=0-12
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
TASK_SUBSET_FILE="${TASK_SUBSET_FILE:-configs/task_subsets/mmbench_balanced_40.json}"
SEED="${SEED:-1}"
WANDB_GROUP="${WANDB_GROUP:-subset40-5m-eval}"
EVAL_TAG="${EVAL_TAG:-}"
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
  echo "WANDB_ENTITY must be set for subset eval runs." >&2
  exit 1
fi

RUNS=(
  "subset40-5m-h100_wm-mlp_pi-gaussian_seed-1 mlp gaussian"
  "subset40-5m-h100_wm-flow_pi-gaussian_seed-1 flow gaussian"
  "subset40-5m-h100_wm-mlp_pi-flow_seed-1 mlp flow"
  "subset40-5m-h100_wm-flow_pi-flow_seed-1 flow flow"
  "subset40-5m-h200-r1_wm-mlp_pi-gaussian_seed-1 mlp gaussian"
  "subset40-5m-h200-r1_wm-mlp_pi-flow_seed-1 mlp flow"
  "subset40-5m-wmvar-h200-r1_wm-endpoint_flow_pi-gaussian_seed-1 endpoint_flow gaussian"
  "subset40-5m-wmvar-h200-r1_wm-residual_flow_pi-gaussian_seed-1 residual_flow gaussian"
  "subset40-5m-l40s-r1_wm-mlp_pi-gaussian_seed-1 mlp gaussian"
  "subset40-5m-l40s-r1_wm-mlp_pi-flow_seed-1 mlp flow"
  "subset40-5m-wmvar-l40s_wm-endpoint_flow_pi-gaussian_seed-1 endpoint_flow gaussian"
  "subset40-5m-wmvar-a100_wm-endpoint_flow_pi-gaussian_seed-1 endpoint_flow gaussian"
  "subset40-5m-h200-r1_wm-flow_pi-gaussian_seed-1 flow gaussian"
)

read -r SOURCE_NAME DYNAMICS POLICY <<< "${RUNS[${SLURM_ARRAY_TASK_ID}]}"
NAME="eval-${SOURCE_NAME}${EVAL_TAG:+-${EVAL_TAG}}"
MODEL_DIR="outputs/logs/soup/${SEED}/${SOURCE_NAME}/models"

latest_full_checkpoint() {
  find "${MODEL_DIR}" -maxdepth 1 -name '*_full.pt' -print | while read -r fp; do
    base="$(basename "${fp}" .pt)"
    base="${base%_full}"
    step="${base//_/}"
    printf '%012d %s\n' "${step}" "${fp}"
  done | sort -n | tail -n 1 | cut -d' ' -f2-
}

CHECKPOINT="$(latest_full_checkpoint)"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "No full checkpoint found under ${MODEL_DIR}" >&2
  exit 2
fi
if [[ "$(basename "${CHECKPOINT}")" != "5_000_000_full.pt" ]]; then
  echo "Latest full checkpoint is not 5M for ${SOURCE_NAME}: ${CHECKPOINT}" >&2
  exit 3
fi

"${PYTHON_BIN}" tdmpc2/eval_checkpoint.py \
  task=soup \
  task_subset_file="${TASK_SUBSET_FILE}" \
  model_size=L \
  obs=state \
  steps=1 \
  seed="${SEED}" \
  enable_wandb=true \
  wandb_upload_artifacts=false \
  save_agent=false \
  save_replay=false \
  eval_episodes="${EVAL_EPISODES:-1}" \
  env_mode=sync \
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
