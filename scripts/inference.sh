#!/usr/bin/env bash
#
# inference.sh — Run LaMa inference
#
# Example:
#   ./scripts/inference.sh \
#       -m ./experiments/hanbin_2025-07-17_22-57-48_train_lama-fourier_ \
#       -i ./examples \
#       -o ./outputs
#
# Flags:
#   -m  Trained model path (required)
#   -i  Input directory       (default: ./demo)
#   -o  Output directory      (default: ./outputs)
#   -h  Show help & exit
#

set -euo pipefail

MODEL_PATH=""
# Default folder with demo images
INPUT_DIR="./demo"
OUTPUT_DIR="./outputs"

usage() { grep '^#' "$0" | cut -c4-; exit "${1:-0}"; }

while getopts ":m:i:o:h" opt; do
  case "$opt" in
    m) MODEL_PATH="$OPTARG" ;;
    i) INPUT_DIR="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    h) usage 0 ;;
    *) usage 1 ;;
  esac
done
shift $((OPTIND - 1))

[ -n "$MODEL_PATH" ] || { echo "[ERROR] -m <model_path> is required"; usage 1; }
INPUT_DIR="${INPUT_DIR:-$(pwd)/demo}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/outputs}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( realpath "$SCRIPT_DIR/.." )"

export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/pretrained}"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"
mkdir -p "$TORCH_HOME" "$OUTPUT_DIR"

PY="${PYTHON:-python3}"

exec "$PY" bin/predict.py \
      model.path="$MODEL_PATH" \
      indir="$INPUT_DIR" \
      outdir="$OUTPUT_DIR"