#!/usr/bin/env bash
# One-click environment setup.
# Usage: ./setup.sh
set -euo pipefail

ENV_NAME="lama"

# Create env if it doesn't exist
if ! conda info --envs | grep -q "^$ENV_NAME\s"; then
  echo "→ Creating conda environment '$ENV_NAME'"
  conda env create -f environment.yml
else
  echo "✓ Conda environment '$ENV_NAME' already exists – skipping creation"
fi

# Activate env (works in login and non-login shells)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "→ Installing GPU-optimised PyTorch"
source scripts/post_install.sh

echo "✅ Environment ready. Run:  conda activate $ENV_NAME"