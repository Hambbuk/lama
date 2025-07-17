#!/usr/bin/env bash
# Thin wrapper around scripts/train.py that sets sensible defaults for TORCH_HOME and PYTHONPATH.
# Usage:  ./scripts/train.sh model=small trainer.max_epochs=1 out_dir=./runs

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( realpath "$SCRIPT_DIR/.." )"

# Use XDG cache if available, else fallback to ~/.cache/lama. Override with TORCH_HOME if set.
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/cache}"
mkdir -p "$TORCH_HOME"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"

python "$SCRIPT_DIR/train.py" "$@"