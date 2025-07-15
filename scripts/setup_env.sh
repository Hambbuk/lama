#!/usr/bin/env bash
# Simplest cross-platform environment bootstrap for the inpaint project.
# 1. Creates and activates a Python env (conda > virtualenv > venv)
# 2. Detects GPU / CUDA and installs the matching torch build
# 3. Installs the rest of the requirements
#
# Usage:
#   bash scripts/setup_env.sh [env_name]
set -e
ENV_NAME=${1:-inpaint}

command_exists() { command -v "$1" >/dev/null 2>&1; }

activate() { # shellcheck disable=SC1090
  source "$1"
  echo "Activated environment: $(python -V)"
}

# 1. Create env -------------------------------------------------------------
if command_exists conda; then
  echo "[SETUP] Using conda to create environment $ENV_NAME"
  conda create -y -n "$ENV_NAME" python=3.10
  activate "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$ENV_NAME"
elif command_exists virtualenv; then
  echo "[SETUP] Using virtualenv ($ENV_NAME)"
  virtualenv "$ENV_NAME" --python=python3
  activate "$ENV_NAME/bin/activate"
else
  echo "[SETUP] Falling back to python -m venv ($ENV_NAME)"
  python3 -m venv "$ENV_NAME"
  activate "$ENV_NAME/bin/activate"
fi

pip install --upgrade pip wheel

# 2. Detect CUDA ------------------------------------------------------------
CUDA_TAG="cpu"
if command_exists nvidia-smi; then
  DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  MAJOR_MINOR=$(echo "$DRIVER_VER" | awk -F. '{print $1 $2}')  # e.g. 535.113 -> 53113 → too long, use first two comps
  CUDA_DIGITS=$(echo "$DRIVER_VER" | cut -d'.' -f1,2 | tr -d '.') # 12.2 → 122 , 11.8 → 118
  CUDA_TAG="cu${CUDA_DIGITS}"
  echo "[SETUP] Detected NVIDIA driver $DRIVER_VER → Torch tag $CUDA_TAG"
fi

# 3. Install Torch ----------------------------------------------------------
TORCH_URL="https://download.pytorch.org/whl"
if [ "$CUDA_TAG" = "cpu" ]; then
  pip install torch torchvision --extra-index-url "$TORCH_URL/cpu"
else
  echo "[SETUP] Installing torch build from $TORCH_URL/$CUDA_TAG"
  if ! pip install torch torchvision --extra-index-url "$TORCH_URL/$CUDA_TAG"; then
    echo "[WARN] Specific wheel for tag $CUDA_TAG not found, falling back to CPU build"
    pip install torch torchvision --extra-index-url "$TORCH_URL/cpu"
  fi
fi

# 4. Install project requirements ------------------------------------------
pip install -r requirements.txt

echo "[SETUP] Environment ready. You can now run:"
echo "  python -m inpaint --help"