#!/usr/bin/env bash
#
# export_to_onnx.sh — Export LaMa checkpoint to ONNX
#
# Example:
#   ./scripts/export_to_onnx.sh \
#       -m ./experiments/2025-07-17_train_lama-fourier_ \
#       -c best.ckpt \
#       -o lama.onnx \
#       -s       # onnx-simplifier
#       -t       # ${PROJECT_ROOT}/examples
#
# Flags:
#   -m  <dir>   Model run directory (required)
#   -c  <file>  Checkpoint filename          (default: best.ckpt)
#   -o  <file>  Output ONNX path             (default: <dir>/lama.onnx)
#   -s          Run onnx-simplifier afterwards
#   -t [dir]    Sanity-test ONNX with image.png & mask.png
#               (default: ${PROJECT_ROOT}/examples)
#   -h          Show help & exit
#

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
MODEL_DIR=""
CHECKPOINT="best.ckpt"
OUTPUT=""
SIMPLIFY=true
TEST=false
TEST_DATA_DIR="./examples"
# ────────────────────────────────────────────────────────────────────────────

usage() { grep '^#' "$0" | cut -c4-; exit "${1:-0}"; }

while getopts ":m:c:o:st::h" opt; do
  case "$opt" in
    m) MODEL_DIR="$OPTARG" ;;
    c) CHECKPOINT="$OPTARG" ;;
    o) OUTPUT="$OPTARG" ;;
    s) SIMPLIFY=true ;;
    t) TEST=true ; TEST_DATA_DIR="${OPTARG:-$TEST_DATA_DIR}" ;;
    h) usage 0 ;;
    *) usage 1 ;;
  esac
done
shift $((OPTIND - 1))

[ -n "$MODEL_DIR" ] || { echo "[ERROR] -m <model_dir> is required"; usage 1; }

# ── Project paths ───────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( realpath "$SCRIPT_DIR/.." )"

OUTPUT="${OUTPUT:-${MODEL_DIR%/}/lama.onnx}"

export PROJECT_ROOT
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/pretrained}"
mkdir -p "$TORCH_HOME"

# ── Export (and optional simplify / test) ───────────────────────────────────
PY="${PYTHON:-python3}"

echo "[INFO] Exporting to ONNX → $OUTPUT"
$PY "$PROJECT_ROOT/bin/export_to_onnx.py" \
    --model-dir "$MODEL_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    $( $TEST && echo "--test" ) \
    --test-data "$TEST_DATA_DIR"

if $SIMPLIFY; then
  echo "[INFO] Simplifying with onnx-simplifier…"
  SIM_OUTPUT="${OUTPUT%.onnx}_sim.onnx"
  python -m onnxsim "$OUTPUT" "$SIM_OUTPUT" \
    --overwrite-input-shape image:1,3,256,256 mask:1,1,256,256
  echo "[INFO] Simplified model saved to $SIM_OUTPUT"
fi