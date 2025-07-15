from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Union, Dict, Any


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    # Set defaults if not present
    defaults = {
        'model': {
            'ngf': 64,
            'n_downsampling': 3,
            'n_blocks': 18,
            'ratio': 0.75,
            'ndf': 64,
            'n_discriminator_layers': 3
        },
        'training': {
            'batch_size': 8,
            'epochs': 100,
            'lr': 0.0002,
            'beta1': 0.0,
            'beta2': 0.9,
            'lr_decay_steps': 1000,
            'lr_decay_rate': 0.1,
            'gan_mode': 'lsgan',
            'val_check_interval': 1.0,
            'early_stopping_patience': 10
        },
        'losses': {
            'l1': 1.0,
            'perceptual': 10.0,
            'style': 250.0,
            'adversarial': 0.1
        },
        'data': {
            'image_size': 512,
            'train_dir': './data/train/images',
            'train_masks_dir': './data/train/masks',
            'val_dir': './data/val/images',
            'val_masks_dir': './data/val/masks'
        }
    }
    
    # Merge with defaults
    config = OmegaConf.merge(defaults, config)
    
    return config


def merge_configs(*configs: Union[DictConfig, Dict[str, Any]]) -> DictConfig:
    """Merge multiple configurations"""
    return OmegaConf.merge(*configs)


def save_config(config: DictConfig, save_path: Union[str, Path]):
    """Save configuration to YAML file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, save_path)