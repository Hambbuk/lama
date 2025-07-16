#!/usr/bin/env python3
"""
Fix script for WandB-related issues with PyTorch Lightning
"""

import os
import subprocess
import sys

def fix_wandb_issue():
    """Fix WandB import issues"""
    print("🔧 Fixing WandB import issues...")
    
    # Method 1: Set environment variables
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_SILENT'] = 'true'
    
    print("✅ Environment variables set")
    
    # Method 2: Try to install wandb as a minimal dependency
    try:
        import wandb
        print("✅ WandB already installed")
    except ImportError:
        print("📦 Installing minimal WandB...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'wandb'])
            print("✅ WandB installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install WandB")
            
    # Method 3: Alternative - downgrade pytorch-lightning
    print("\n📋 Alternative solutions:")
    print("1. Install wandb (minimal impact):")
    print("   pip install wandb")
    print("\n2. Use older pytorch-lightning:")
    print("   pip install pytorch-lightning==1.9.0")
    print("\n3. Use lightning instead of pytorch-lightning:")
    print("   pip uninstall pytorch-lightning")
    print("   pip install lightning")

if __name__ == '__main__':
    fix_wandb_issue()