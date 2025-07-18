#!/usr/bin/env bash
#
# export_to_onnx.sh — Export a trained LaMa model to ONNX (옵션: simplify)
#
# Example:
#   ./scripts/export_to_onnx.sh \
#       -m ./experiments/hanbin_2025-07-17_22-57-48_train_lama-fourier_ \
#       -c best.ckpt \
#       -o lama.onnx \
#       -s          # ← 추가: onnx-simplifier 적용
#
# Flags:
#   -m  Model directory (required)
#   -c  Checkpoint filename inside <model_dir>/models/   (default: best.ckpt)
#   -o  Output ONNX file                                 (default: <model_dir>/lama.onnx)
#   -s  Simplify with onnx-simplifier after export
#   -h  Show help & exit
#

set -euo pipefail

MODEL_DIR=""
CHECKPOINT="best.ckpt"
OUTPUT=""
SIMPLIFY=false

usage() { grep '^#' "$0" | cut -c4-; exit "${1:-0}"; }

while getopts ":m:c:o:sh" opt; do
  case "$opt" in
    m) MODEL_DIR="$OPTARG" ;;
    c) CHECKPOINT="$OPTARG" ;;
    o) OUTPUT="$OPTARG" ;;
    s) SIMPLIFY=true ;;
    h) usage 0 ;;
    *) usage 1 ;;
  esac
done
shift $((OPTIND - 1))

[ -n "$MODEL_DIR" ] || { echo "[ERROR] -m <model_dir> is required"; usage 1; }
OUTPUT="${OUTPUT:-${MODEL_DIR%/}/lama.onnx}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( realpath "$SCRIPT_DIR/.." )"

export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/pretrained}"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"
mkdir -p "$TORCH_HOME"

PY="${PYTHON:-python3}"

echo "[INFO] Exporting to ONNX → $OUTPUT"
"$PY" "$PROJECT_ROOT/bin/export_to_onnx.py" \
      --model-dir "$MODEL_DIR" \
      --checkpoint "$CHECKPOINT" \
      --output "$OUTPUT"

if $SIMPLIFY; then
  echo "[INFO] Simplifying ONNX with onnx-simplifier…"
  TMP="${OUTPUT%.onnx}_sim.tmp.onnx"
  python -m onnxsim "$OUTPUT" "$TMP" --overwrite-input-shape image:1,3,256,256 mask:1,1,256,256
  mv "$TMP" "$OUTPUT"
  echo "[INFO] Simplified model saved to $OUTPUT"
fi