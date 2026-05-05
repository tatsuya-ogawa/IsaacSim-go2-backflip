#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/backflip}"
GO2_BACKFLIP_PYTHONPATH="${ROOT_DIR}/source/go2_backflip"

isaac_python() {
  export PYTHONPATH="${GO2_BACKFLIP_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
  if [[ -x /workspace/isaaclab/isaaclab.sh ]]; then
    /workspace/isaaclab/isaaclab.sh -p "$@"
  elif [[ -x /workspace/IsaacLab/isaaclab.sh ]]; then
    /workspace/IsaacLab/isaaclab.sh -p "$@"
  else
    python "$@"
  fi
}

ensure_bootstrapped() {
  export PYTHONPATH="${GO2_BACKFLIP_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
}
