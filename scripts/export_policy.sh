#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_bootstrapped

task_name="${TASK_NAME:-Unitree-Go2-Backflip-Isaac-v0}"
num_envs="${PLAY_NUM_ENVS:-1}"
load_run="${LOAD_RUN:-.*_backflip(-resume)?$}"
checkpoint="${CHECKPOINT:-${CHECKPOINT_PATH:-}}"
policy_path="${POLICY_PATH:-}"

cd "${ROOT_DIR}"
args=(
  "${ROOT_DIR}/scripts/rsl_rl/export_policy.py"
  --headless
  --task "${task_name}"
  --num_envs "${num_envs}"
  --load_run "${load_run}"
)

if [[ -n "${checkpoint}" ]]; then
  args+=(--checkpoint "${checkpoint}")
fi

if [[ -n "${policy_path}" ]]; then
  args+=(--output "${policy_path}")
fi

isaac_python "${args[@]}"
