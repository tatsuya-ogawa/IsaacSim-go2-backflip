#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_bootstrapped

task_name="${TASK_NAME:-Unitree-Go2-Backflip-Isaac-v0}"
num_envs="${PLAY_NUM_ENVS:-1}"
load_run="${LOAD_RUN:-.*_backflip(-resume)?$}"
video_length="${VIDEO_LENGTH:-240}"
rendering_mode="${VIDEO_RENDERING_MODE:-quality}"
camera_tracking="${PLAY_CAMERA_TRACKING:-1}"
camera_eye_offset="${PLAY_CAMERA_EYE_OFFSET:-1.2,-1.7,0.75}"
camera_target_offset="${PLAY_CAMERA_TARGET_OFFSET:-0.0,0.0,0.25}"
camera_smoothing="${PLAY_CAMERA_SMOOTHING:-0.35}"

cd "${ROOT_DIR}"
echo "[INFO] Isaac video rendering mode: ${rendering_mode}"
args=(
  "${ROOT_DIR}/scripts/rsl_rl/play.py"
  --headless
  --video
  --video_length "${video_length}"
  --rendering_mode "${rendering_mode}"
  --task "${task_name}"
  --num_envs "${num_envs}"
  --load_run "${load_run}"
  --camera_tracking "${camera_tracking}"
  --camera_eye_offset "${camera_eye_offset}"
  --camera_target_offset "${camera_target_offset}"
  --camera_smoothing "${camera_smoothing}"
)

if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  args+=(--checkpoint "${CHECKPOINT_PATH}")
fi

isaac_python "${args[@]}"
