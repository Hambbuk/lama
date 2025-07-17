#!/usr/bin/env bash
# Wrapper for inference that avoids manual env var setup.
# Example:
#   ./scripts/inference.sh model.path=./big-lama indir=./test_images outdir=./output

set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( realpath "$SCRIPT_DIR/.." )"

export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT}"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"

python "$PROJECT_ROOT/bin/predict.py" "$@"