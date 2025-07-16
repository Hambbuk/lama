#!/usr/bin/env python3
"""
Startup script to set up the environment for LaMa inpainting training
"""

import os
import sys
import subprocess
import importlib.util

def check_and_install_package(package_name, install_name=None):
    """Check if a package is installed and install if necessary"""
    if install_name is None:
        install_name = package_name
    
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"📦 Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', install_name])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False
    else:
        print(f"✅ {package_name} already installed")
        return True

def setup_environment():
    """Set up environment variables"""
    print("🔧 Setting up environment variables...")
    
    # Disable wandb
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_SILENT'] = 'true'
    
    # Set other useful environment variables
    os.environ['PYTHONPATH'] = os.getcwd()
    os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), '.torch')
    
    print("✅ Environment variables set")

def check_pytorch_lightning():
    """Check PyTorch Lightning installation"""
    print("🔍 Checking PyTorch Lightning...")
    
    try:
        import pytorch_lightning as pl
        print(f"✅ PyTorch Lightning {pl.__version__} is working")
        return True
    except ImportError as e:
        if "wandb" in str(e).lower():
            print("❌ PyTorch Lightning has WandB dependency issues")
            print("🔧 Installing WandB to resolve dependency...")
            
            if check_and_install_package('wandb'):
                try:
                    import pytorch_lightning as pl
                    print(f"✅ PyTorch Lightning {pl.__version__} is now working")
                    return True
                except ImportError:
                    print("❌ Still having issues with PyTorch Lightning")
                    return False
            else:
                return False
        else:
            print(f"❌ PyTorch Lightning import error: {e}")
            return False

def main():
    """Main setup function"""
    print("🦙 LaMa Inpainting - Environment Setup")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check PyTorch Lightning
    if not check_pytorch_lightning():
        print("\n❌ PyTorch Lightning setup failed")
        print("🔧 Please try one of these solutions:")
        print("1. pip install wandb")
        print("2. pip install pytorch-lightning==1.9.0")
        print("3. pip install lightning (instead of pytorch-lightning)")
        return False
    
    # Check other critical packages
    critical_packages = [
        ('torch', 'torch>=2.1.0'),
        ('torchvision', 'torchvision>=0.16.0'),
        ('hydra', 'hydra-core>=1.3.0'),
        ('omegaconf', 'omegaconf>=2.3.0'),
        ('cv2', 'opencv-python>=4.8.0'),
        ('albumentations', 'albumentations>=1.3.0'),
        ('torchmetrics', 'torchmetrics>=1.2.0'),
    ]
    
    print("\n🔍 Checking critical packages...")
    all_good = True
    for pkg, install_name in critical_packages:
        if not check_and_install_package(pkg, install_name):
            all_good = False
    
    if all_good:
        print("\n🎉 Environment setup completed successfully!")
        print("✅ Ready to start training!")
        print("\nNext steps:")
        print("1. Prepare your dataset")
        print("2. Update data_dir in configs/config.yaml")
        print("3. Run: python train.py")
        return True
    else:
        print("\n❌ Some packages failed to install")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)