#!/usr/bin/env bash
# Thin wrapper around scripts/train.py that sets sensible defaults for TORCH_HOME and PYTHONPATH.
# Usage:  ./scripts/train.sh model=small trainer.max_epochs=1 out_dir=./runs

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( realpath "$SCRIPT_DIR/.." )"

export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/.cache/torch}"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"

python "$SCRIPT_DIR/train.py" "$@"