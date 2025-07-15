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
  CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d'.' -f1,2)
  case "$CUDA_VER" in
    12*) CUDA_TAG="cu121";;
    11*) CUDA_TAG="cu118";;
    10*) CUDA_TAG="cu117";;
    *) CUDA_TAG="cpu";;
  esac
  echo "[SETUP] Detected NVIDIA driver, selecting torch build: $CUDA_TAG"
else
  echo "[SETUP] No NVIDIA GPU detected, installing CPU torch build"
fi

# 3. Install Torch ----------------------------------------------------------
if [ "$CUDA_TAG" = "cpu" ]; then
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
  pip install torch torchvision --index-url https://download.pytorch.org/whl/$CUDA_TAG
fi

# 4. Install project requirements ------------------------------------------
pip install -r requirements.txt

echo "[SETUP] Environment ready. You can now run:"
echo "  python -m inpaint --help"