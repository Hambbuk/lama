#!/usr/bin/env bash
# Wrapper for inference that avoids manual env var setup.
# Example:
#   ./scripts/inference.sh model.path=./big-lama indir=./test_images outdir=./output

set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( realpath "$SCRIPT_DIR/.." )"

DEFAULT_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}"
export TORCH_HOME="${TORCH_HOME:-$DEFAULT_CACHE_ROOT/lama}"
mkdir -p "$TORCH_HOME"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"

python "$PROJECT_ROOT/bin/predict.py" "$@"