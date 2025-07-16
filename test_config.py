#!/usr/bin/env python3
"""Test script to verify configuration loading"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_config_loading():
    """Test that hydra configuration can be loaded"""
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        
        # Clear any existing hydra instance
        GlobalHydra.instance().clear()
        
        config_dir = os.path.join(os.path.dirname(__file__), "configs")
        
        # Initialize hydra
        with initialize_config_dir(config_dir=config_dir):
            cfg = compose(config_name="training/lama_small_train_masks")
            print("✓ Configuration loaded successfully")
            print(f"  - Config name: {cfg.get('run_title', 'N/A')}")
            print(f"  - Training model kind: {cfg.training_model.kind}")
            print(f"  - Max epochs: {cfg.trainer.kwargs.max_epochs}")
            return True
            
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_trainer_import():
    """Test that the trainer module can be imported"""
    try:
        from saicinpainting.training.trainers import make_training_model
        print("✓ Trainer module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Trainer import failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing LaMa configuration and imports...")
    
    config_ok = test_config_loading()
    trainer_ok = test_trainer_import()
    
    if config_ok and trainer_ok:
        print("\n✓ All tests passed! The training setup should work.")
    else:
        print("\n✗ Some tests failed. Please check the configuration.")
        sys.exit(1)