#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_bootstrapped
echo "Local go2_backflip package is available through PYTHONPATH=${GO2_BACKFLIP_PYTHONPATH}."
echo "Isaac Lab's built-in Unitree Go2 USD asset is used by default."
