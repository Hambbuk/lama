#!/usr/bin/env bash
#
# train.sh — Launch LaMa training
#
# Example:
#   ./scripts/train.sh \
#       -m lama-fourier \
#       -l thumbnail \
#       -d abl-04-256-mh-dist-hand_mask-multi
#
# Flags:
#   -m  Model config YAML (default: lama-fourier)
#   -l  Location config YAML (default: thumbnail)
#   -d  Data config YAML (default: abl-04-256-mh-dist-hand_mask-multi)
#   -h  Show help & exit
#

set -euo pipefail

# ── defaults ───────────────────────────────────────────────────────────────
MODEL_CFG="lama-fourier"
LOCATION_CFG="thumbnail"
DATA_CFG="abl-04-256-mh-dist-hand_mask-multi"
# ───────────────────────────────────────────────────────────────────────────

usage() { grep '^#' "$0" | cut -c4-; exit "${1:-0}"; }

while getopts ":m:l:d:h" opt; do
  case "$opt" in
    m) MODEL_CFG="$OPTARG" ;;
    l) LOCATION_CFG="$OPTARG" ;;
    d) DATA_CFG="$OPTARG" ;;
    h) usage 0 ;;
    *) usage 1 ;;
  esac
done
shift $((OPTIND - 1))

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( realpath "$SCRIPT_DIR/.." )"

export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/pretrained}"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"
mkdir -p "$TORCH_HOME"

PY="${PYTHON:-python3}"

exec "$PY" -m bin.train \
      -cn "$MODEL_CFG" \
      location="$LOCATION_CFG" \
      data="$DATA_CFG"