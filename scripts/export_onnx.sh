#!/usr/bin/env bash
# Export a trained LaMa model to ONNX conveniently.
# Example:
#   ./scripts/export_onnx.sh --model-dir ./big-lama --checkpoint best.ckpt --output lama.onnx

set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( realpath "$SCRIPT_DIR/.." )"

export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT}"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"

python "$SCRIPT_DIR/export_onnx.py" "$@"