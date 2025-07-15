#!/usr/bin/env bash
set -euo pipefail
# ---------------------------------------------------------------------------
# Automatic environment bootstrap for the inpaint project.
#  • Creates a Python env (conda > virtualenv > venv)
#  • Detects CUDA driver and installs the correct torch+tv wheel,
#    falling back to CPU build if wheel unavailable
#  • Installs project deps via poetry / pip-tools / pip (in that order)
#  • Runs an import-walk smoke test; aborts on the first failure
# Usage:  bash scripts/setup_env.sh [env_name]  (default: inpaint)
# ---------------------------------------------------------------------------
ENV_NAME=${1:-inpaint}
PY_VER=${PY_VER:-3.10}
TORCH_REPO=https://download.pytorch.org/whl
REQ=requirements.txt

log(){ printf "\033[1;32m[SETUP]\033[0m %s\n" "$*"; }
cmd(){ log "$*"; "$@"; }

# 1 ─ create env ------------------------------------------------------------
if command -v conda &>/dev/null; then
  cmd conda create -y -n "$ENV_NAME" "python=$PY_VER"
  # shellcheck disable=SC1090
  . "$(conda info --base)/etc/profile.d/conda.sh" && cmd conda activate "$ENV_NAME"
elif command -v virtualenv &>/dev/null; then
  cmd virtualenv "$ENV_NAME" --python=python3
  # shellcheck disable=SC1091
  . "$ENV_NAME/bin/activate"
else
  cmd python3 -m venv "$ENV_NAME"
  # shellcheck disable=SC1091
  . "$ENV_NAME/bin/activate"
fi
log "Python: $(python -V)"

python -m pip install --upgrade pip wheel --quiet --progress-bar off

# simple retry wrapper for flaky networks
pip_retry() {
  local n=0
  until python -m pip install --quiet --progress-bar off "$@" && break; do
    n=$((n+1))
    [ $n -lt 5 ] || { echo "[pip] failed after 5 attempts"; exit 1; }
    echo "[pip] retry $n…" && sleep 4
  done
}

# Windows path: force known CUDA tag when uname says MINGW/MSYS
case "$(uname -s)" in
  MINGW*|MSYS*) TAG=${CUDA_TAG:-cu118};;
esac

# 2 ─ pick torch wheel tag --------------------------------------------------
TAG=cpu
if command -v nvidia-smi &>/dev/null; then
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  TAG=cu$(echo "$DRIVER" | cut -d'.' -f1,2 | tr -d '.')
  log "CUDA driver $DRIVER → torch tag $TAG"
fi
if ! python -m pip install --quiet --progress-bar off torch torchvision --extra-index-url "$TORCH_REPO/$TAG"; then
  log "Wheel for $TAG not found → installing CPU build"
  pip_retry torch torchvision --extra-index-url "$TORCH_REPO/cpu"
fi

# 3 ─ project deps ----------------------------------------------------------
if command -v poetry &>/dev/null && [ -f pyproject.toml ]; then
  log "Using Poetry for dependency install"
  cmd poetry install --without dev --no-interaction --no-root
elif command -v pip-compile &>/dev/null && [ -f requirements.in ]; then
  log "Using pip-tools (lock → install)"
  cmd pip-compile requirements.in -o requirements.lock
  cmd python -m pip install -r requirements.lock
else
  cmd python -m pip install -r "$REQ"
fi

# 4 ─ smoke test ------------------------------------------------------------
python - <<'PY'
import importlib, pkgutil, sys, inpaint
fail = []
for mod in pkgutil.walk_packages(inpaint.__path__, inpaint.__name__ + '.'):  # noqa
    try:
        importlib.import_module(mod.name)
    except Exception as e:  # noqa
        fail.append((mod.name, e))
if fail:
    for n,e in fail:
        print('❌', n, '->', e)
    raise SystemExit(f"{len(fail)} import failures")
print('✓ import-walk OK')
PY

log "Environment READY.  Try:  python -m inpaint --help"