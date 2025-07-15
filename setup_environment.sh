#!/bin/bash

# LaMa Environment Setup Script
# This script automatically sets up the environment for LaMa inpainting
# It detects CUDA availability and installs appropriate packages

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

print_header "🦙 LaMa Environment Setup"
print_header "==============================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_warning "Conda not found. Installing Miniconda..."
    
    # Download and install Miniconda
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    else
        print_error "Unsupported operating system"
        exit 1
    fi
    
    wget -O miniconda.sh "$MINICONDA_URL"
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    
    # Add conda to PATH
    export PATH="$HOME/miniconda/bin:$PATH"
    $HOME/miniconda/bin/conda init bash
    
    print_status "Miniconda installed successfully"
else
    print_status "Conda found"
fi

# Ensure conda is in PATH
if [[ ":$PATH:" != *":$HOME/miniconda/bin:"* ]]; then
    export PATH="$HOME/miniconda/bin:$PATH"
fi

# Update conda
print_status "Updating conda..."
conda update -n base -c defaults conda -y

# Check for CUDA availability
print_status "Checking for CUDA availability..."
CUDA_AVAILABLE=false

# Check for nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    if [[ -n "$CUDA_VERSION" ]]; then
        print_status "NVIDIA GPU detected with driver version: $CUDA_VERSION"
        CUDA_AVAILABLE=true
    fi
fi

# Check for NVIDIA devices
if [[ -e /dev/nvidia0 ]]; then
    print_status "NVIDIA device detected"
    CUDA_AVAILABLE=true
fi

# Check for CUDA runtime
if [[ -d /usr/local/cuda ]]; then
    print_status "CUDA runtime detected"
    CUDA_AVAILABLE=true
fi

# Remove existing environment if it exists
ENV_NAME="lama"
if [[ "$CUDA_AVAILABLE" == true ]]; then
    ENV_NAME="lama-gpu"
fi

print_status "Checking for existing environment: $ENV_NAME"
if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment $ENV_NAME already exists. Removing it..."
    conda env remove -n $ENV_NAME -y
fi

# Create environment based on CUDA availability
if [[ "$CUDA_AVAILABLE" == true ]]; then
    print_status "Creating GPU-enabled environment..."
    conda env create -f conda_env_gpu.yml
    print_status "GPU environment created successfully!"
    print_status "To activate: conda activate lama-gpu"
else
    print_status "Creating CPU-only environment..."
    conda env create -f conda_env.yml
    print_status "CPU environment created successfully!"
    print_status "To activate: conda activate lama"
fi

# Activate the environment and run additional setup
print_status "Activating environment and running post-setup..."
source $HOME/miniconda/bin/activate $ENV_NAME

# Install additional packages if needed
print_status "Installing additional packages..."
pip install --upgrade pip

# Verify installation
print_status "Verifying installation..."
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torchvision; print('TorchVision version:', torchvision.__version__)"

if [[ "$CUDA_AVAILABLE" == true ]]; then
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
fi

python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
python -c "import PIL; print('Pillow version:', PIL.__version__)"

# Download model weights if they don't exist
print_status "Checking for model weights..."
if [[ ! -d "models" ]]; then
    mkdir -p models
fi

# Create a simple test script
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3

import torch
import numpy as np
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import sklearn
import hydra
from omegaconf import DictConfig

def test_installation():
    print("🦙 LaMa Environment Test")
    print("=" * 30)
    
    # Test basic imports
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ NumPy: {np.__version__}")
    print(f"✓ OpenCV: {cv2.__version__}")
    print(f"✓ Pillow: {PIL.__version__}")
    print(f"✓ Matplotlib: {plt.__version__}")
    print(f"✓ Scikit-learn: {sklearn.__version__}")
    
    # Test CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA: Available (Version: {torch.version.cuda})")
        print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ CUDA: Not available (using CPU)")
    
    # Test tensor operations
    x = torch.randn(2, 3)
    y = torch.randn(3, 4)
    z = torch.mm(x, y)
    print(f"✓ Tensor operations: Working (shape: {z.shape})")
    
    # Test image operations
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)
    print(f"✓ Image operations: Working (size: {pil_img.size})")
    
    print("\n🎉 All tests passed! Environment is ready for LaMa inpainting.")

if __name__ == "__main__":
    test_installation()
EOF

chmod +x test_installation.py

print_status "Running installation test..."
python test_installation.py

print_header ""
print_header "🎉 Environment Setup Complete!"
print_header "=============================="
print_status "Environment name: $ENV_NAME"
print_status "To activate: conda activate $ENV_NAME"
print_status "To test: python test_installation.py"
print_status "To deactivate: conda deactivate"
print_header ""
print_status "You can now use LaMa for inpainting tasks!"
print_status "Check the README.md for usage instructions."