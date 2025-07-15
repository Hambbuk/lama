#!/usr/bin/env bash

# Detect available CUDA version from nvidia-smi. Installs matching PyTorch build (or CPU fallback).
# Usage: source post_install.sh  # after `conda env create -f environment.yml`

set -euo pipefail

CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -n 1 | cut -d'.' -f1,2 || echo "")

# Map CUDA version to the extra-index-url tag used by PyTorch wheels
case "$CUDA_VERSION" in
    12*) TAG="cu121";;
    11.8) TAG="cu118";;
    11.7) TAG="cu117";;
    11.6) TAG="cu116";;
    11.3) TAG="cu113";;
    11.1) TAG="cu111";;
    10.2) TAG="cu102";;
    *) TAG="cpu";;
esac

echo "→ Installing PyTorch build for CUDA tag: $TAG"

pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$TAG

echo "✅ PyTorch installation finished"