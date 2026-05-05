#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_bootstrapped

cd "${ROOT_DIR}"
isaac_python "${ROOT_DIR}/scripts/list_envs.py"
