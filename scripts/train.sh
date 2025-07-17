#!/usr/bin/env bash
# Thin wrapper around scripts/train.py that sets sensible defaults for TORCH_HOME and PYTHONPATH.
# Usage:  ./scripts/train.sh model=small trainer.max_epochs=1 out_dir=./runs

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( realpath "$SCRIPT_DIR/.." )"

# Use XDG cache if available, else fallback to ~/.cache/lama. Override with TORCH_HOME if set.
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/pretrained}"
mkdir -p "$TORCH_HOME"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"

# ----- Optional convenience: --data-root DIR -----
DATA_ROOT=""
ARGS=()
for arg in "$@"; do
  case $arg in
    --data-root=*)
      DATA_ROOT="${arg#*=}";
      ;;
    --data-root)
      shift; DATA_ROOT="$1";;
    *) ARGS+=("$arg");;
  esac
done

# Build Hydra overrides if data root specified
if [[ -n "$DATA_ROOT" ]]; then
  ARGS+=("location.data_root_dir=$DATA_ROOT")
  ARGS+=("location.out_root_dir=$PROJECT_ROOT/experiments")
  ARGS+=("location.tb_dir=$PROJECT_ROOT/tb_logs")
fi

python "$SCRIPT_DIR/train.py" "${ARGS[@]}"