#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/storage/project/r-agarg35-0/eliu354/envs/newt_official_20260602/bin/python}"
DATA_DIR="${DATA_DIR:-/storage/project/r-agarg35-0/eliu354/external_data/newt/mmbench_demos}"
BASE_EXP="${BASE_EXP:-flow_2x2_smoke}"

for DYNAMICS in mlp flow; do
  for POLICY in gaussian flow; do
    EXP_NAME="${BASE_EXP}_wm-${DYNAMICS}_pi-${POLICY}"
    "${PYTHON_BIN}" tdmpc2/train.py \
      task=walker-walk \
      model_size=B \
      obs=state \
      use_demos=false \
      enable_wandb=false \
      save_agent=true \
      save_replay=false \
      compile=false \
      steps=1000 \
      demo_steps=0 \
      eval_episodes=1 \
      checkpoint_freq=500 \
      dynamics_arch="${DYNAMICS}" \
      policy_arch="${POLICY}" \
      exp_name="${EXP_NAME}" \
      data_dir="${DATA_DIR}"
  done
done
