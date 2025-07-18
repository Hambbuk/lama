#!/usr/bin/env bash
# onnx_inference.sh — Run ONNX model on a single image+mask pair
#
# Example:
#   ./scripts/onnx_inference.sh \
#       -m lama.onnx \
#       -i ./examples/image.png \
#       -k ./examples/image_mask.png \
#       -o ./outputs/out.png
#
# Flags:
#   -m  ONNX model path (required)
#   -i  Input image  (default: ./examples/image.png)
#   -k  Input mask   (default: ./examples/image_mask.png)
#   -o  Output file  (default: ./outputs/out.png)
#   -n  inflation pixels  (default: 0)
#   -h  Show help & exit
#
set -euo pipefail

MODEL=""
IMG="./examples/image.png"
MASK="./examples/image_mask.png"
OUT="./outputs/out.png"
INFLATE=0

usage() { grep '^#' "$0" | cut -c4-; exit "${1:-0}"; }

while getopts ":m:i:k:o:n:h" opt; do
  case "$opt" in
    m) MODEL="$OPTARG" ;;
    i) IMG="$OPTARG" ;;
    k) MASK="$OPTARG" ;;
    o) OUT="$OPTARG" ;;
    n) INFLATE="$OPTARG" ;;
    h) usage 0 ;;
    *) usage 1 ;;
  esac
done
shift $((OPTIND - 1))

[ -n "$MODEL" ] || { echo "[ERROR] -m <model.onnx> is required"; usage 1; }

PY="${PYTHON:-python3}"
$PY bin/onnx_predict.py --model "$MODEL" --image "$IMG" --mask "$MASK" --output "$OUT" --inflate "$INFLATE"