#!/usr/bin/env bash
#SBATCH --job-name=newt-single
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --array=0-3
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --qos=embers

set -euo pipefail
mkdir -p outputs/slurm

PYTHON_BIN="${PYTHON_BIN:-/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602/bin/python}"
DATA_DIR="${DATA_DIR:-/storage/project/r-agarg35-0/eliu354/external_data/newt/mmbench_demos}"
SEED="${SEED:-1}"
DIAG_MODE="${DIAG_MODE:-pilot}"
WANDB_GROUP="${WANDB_GROUP:-single-gpu-h200-diagnostics}"
export WANDB_PROJECT="${WANDB_PROJECT:-newt-flow-2x2}"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export TMPDIR="${TMPDIR:-/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${TMPDIR}/torchinductor_${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

if [[ -z "${WANDB_ENTITY:-}" ]]; then
  echo "WANDB_ENTITY must be set for diagnostic runs." >&2
  exit 1
fi

CELLS=(
  "mlp gaussian"
  "flow gaussian"
  "mlp flow"
  "flow flow"
)
read -r DYNAMICS POLICY <<< "${CELLS[${SLURM_ARRAY_TASK_ID}]}"
NAME="single-${DIAG_MODE}_wm-${DYNAMICS}_pi-${POLICY}_seed-${SEED}"

COMMON_ARGS=(
  task=soup
  model_size=L
  obs=state
  seed="${SEED}"
  enable_wandb=true
  save_agent=true
  save_replay=true
  eval_episodes=1
  compile=false
  dynamics_arch="${DYNAMICS}"
  policy_arch="${POLICY}"
  wandb_group="${WANDB_GROUP}"
  wandb_name="${NAME}"
  wandb_run_id="${NAME}"
  wandb_resume=allow
  exp_name="${NAME}"
  data_dir="${DATA_DIR}"
)

case "${DIAG_MODE}" in
  pilot)
    MODE_ARGS=(
      task_limit="${DIAG_TASK_LIMIT:-16}"
      steps="${DIAG_STEPS:-20000}"
      demo_steps="${DIAG_DEMO_STEPS:-200}"
      batch_size="${DIAG_BATCH_SIZE:-256}"
      buffer_size="${DIAG_BUFFER_SIZE:-1000000}"
      checkpoint_freq="${DIAG_CHECKPOINT_FREQ:-5000}"
      replay_checkpoint_freq="${DIAG_REPLAY_CHECKPOINT_FREQ:-10000}"
    )
    ;;
  reduced)
    MODE_ARGS=(
      steps="${DIAG_STEPS:-100000}"
      demo_steps="${DIAG_DEMO_STEPS:-1000}"
      batch_size="${DIAG_BATCH_SIZE:-256}"
      buffer_size="${DIAG_BUFFER_SIZE:-1000000}"
      checkpoint_freq="${DIAG_CHECKPOINT_FREQ:-10000}"
      replay_checkpoint_freq="${DIAG_REPLAY_CHECKPOINT_FREQ:-25000}"
    )
    ;;
  full)
    MODE_ARGS=(
      steps="${DIAG_STEPS:-100000}"
      demo_steps="${DIAG_DEMO_STEPS:-1000}"
      batch_size="${DIAG_BATCH_SIZE:-1024}"
      buffer_size="${DIAG_BUFFER_SIZE:-10000000}"
      checkpoint_freq="${DIAG_CHECKPOINT_FREQ:-10000}"
      replay_checkpoint_freq="${DIAG_REPLAY_CHECKPOINT_FREQ:-25000}"
    )
    ;;
  *)
    echo "Unknown DIAG_MODE=${DIAG_MODE}; expected pilot, reduced, or full." >&2
    exit 2
    ;;
esac

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
  "${COMMON_ARGS[@]}" \
  "${MODE_ARGS[@]}" \
  "${CHECKPOINT_ARGS[@]}"
