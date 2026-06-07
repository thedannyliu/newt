#!/usr/bin/env bash
#SBATCH --job-name=newt-wmvariants
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --array=0-1
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=220G
#SBATCH --time=08:00:00
#SBATCH --qos=embers

set -euo pipefail
mkdir -p outputs/slurm

PYTHON_BIN="${PYTHON_BIN:-/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602/bin/python}"
DATA_DIR="${DATA_DIR:-/storage/project/r-agarg35-0/eliu354/external_data/newt/mmbench_demos}"
TASK_SUBSET_FILE="${TASK_SUBSET_FILE:-configs/task_subsets/mmbench_balanced_40.json}"
SEED="${SEED:-1}"
WANDB_GROUP="${WANDB_GROUP:-subset40-5m-wm-variants}"
RUN_TAG="${RUN_TAG:-}"
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
  echo "WANDB_ENTITY must be set for WM variant runs." >&2
  exit 1
fi

DYNAMICS_LIST=(endpoint_flow residual_flow)
DYNAMICS="${DYNAMICS_LIST[${SLURM_ARRAY_TASK_ID}]}"
POLICY="gaussian"
NAME="subset40-5m-wmvar${RUN_TAG:+-${RUN_TAG}}_wm-${DYNAMICS}_pi-${POLICY}_seed-${SEED}"

MODEL_DIR="outputs/logs/soup/${SEED}/${NAME}/models"
CHECKPOINT_ARGS=()
latest_checkpoint() {
  local pattern="$1"
  find "${MODEL_DIR}" -maxdepth 1 -name "${pattern}" -print | while read -r fp; do
    base="$(basename "${fp}" .pt)"
    base="${base%_full}"
    step="${base//_/}"
    printf '%012d %s\n' "${step}" "${fp}"
  done | sort -n | tail -n 1 | cut -d' ' -f2-
}
if [[ -d "${MODEL_DIR}" ]]; then
  LATEST_CHECKPOINT="$(latest_checkpoint '*_full.pt')"
  if [[ -n "${LATEST_CHECKPOINT}" ]]; then
    CHECKPOINT_ARGS=(checkpoint="${LATEST_CHECKPOINT}")
    echo "Resuming ${NAME} from ${LATEST_CHECKPOINT}"
  else
    LATEST_CHECKPOINT="$(latest_checkpoint '*.pt')"
    if [[ -n "${LATEST_CHECKPOINT}" ]]; then
      CHECKPOINT_ARGS=(checkpoint="${LATEST_CHECKPOINT}")
      echo "Resuming ${NAME} from non-full checkpoint ${LATEST_CHECKPOINT}; online replay state is not included."
    fi
  fi
fi

"${PYTHON_BIN}" tdmpc2/train.py \
  task=soup \
  task_subset_file="${TASK_SUBSET_FILE}" \
  model_size=L \
  obs=state \
  steps="${STEPS:-5000000}" \
  demo_steps="${DEMO_STEPS:-50000}" \
  batch_size="${BATCH_SIZE:-1024}" \
  buffer_size="${BUFFER_SIZE:-5000000}" \
  seed="${SEED}" \
  enable_wandb=true \
  wandb_upload_artifacts=false \
  save_agent=true \
  save_replay=true \
  eval_episodes=1 \
  compile=false \
  checkpoint_freq="${CHECKPOINT_FREQ:-250000}" \
  replay_checkpoint_freq="${REPLAY_CHECKPOINT_FREQ:-1000000}" \
  dynamics_arch="${DYNAMICS}" \
  policy_arch="${POLICY}" \
  wandb_group="${WANDB_GROUP}" \
  wandb_name="${NAME}" \
  wandb_run_id="${NAME}" \
  wandb_resume=allow \
  exp_name="${NAME}" \
  data_dir="${DATA_DIR}" \
  "${CHECKPOINT_ARGS[@]}"
