#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_bootstrapped

num_envs="${NUM_ENVS:-4096}"
max_iterations="${MAX_ITERATIONS:-1000}"
run_name="${RUN_NAME:-backflip}"
task_name="${TASK_NAME:-Unitree-Go2-Backflip-Isaac-v0}"
resume="${RESUME:-0}"
load_run="${LOAD_RUN:-}"
checkpoint="${CHECKPOINT:-${CHECKPOINT_PATH:-}}"

if [[ "${resume}" == "1" || "${resume}" == "true" || "${resume}" == "yes" ]]; then
  if [[ -z "${load_run}" && -z "${checkpoint}" ]]; then
    load_run="${RESUME_LOAD_RUN:-.*_backflip(-resume)?$}"
  fi
fi

cd "${ROOT_DIR}"
args=(
  "${ROOT_DIR}/scripts/rsl_rl/train.py"
  --headless
  --task "${task_name}"
  --num_envs "${num_envs}"
  --max_iterations "${max_iterations}"
  --run_name "${run_name}"
)

if [[ "${resume}" == "1" || "${resume}" == "true" || "${resume}" == "yes" ]]; then
  args+=(--resume)
fi

if [[ -n "${load_run}" ]]; then
  args+=(--load_run "${load_run}")
fi

if [[ -n "${checkpoint}" ]]; then
  args+=(--checkpoint "${checkpoint}")
fi

isaac_python "${args[@]}"
