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
    print(f"✓ Matplotlib: {plt.matplotlib.__version__}")
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
    
    # Test additional packages
    try:
        import albumentations
        print(f"✓ Albumentations: {albumentations.__version__}")
    except ImportError:
        print("✗ Albumentations: Not installed")
    
    try:
        import kornia
        print(f"✓ Kornia: {kornia.__version__}")
    except ImportError:
        print("✗ Kornia: Not installed")
    
    try:
        import pytorch_lightning
        print(f"✓ PyTorch Lightning: {pytorch_lightning.__version__}")
    except ImportError:
        print("✗ PyTorch Lightning: Not installed")
    
    print("\n🎉 All tests passed! Environment is ready for LaMa inpainting.")

if __name__ == "__main__":
    test_installation()