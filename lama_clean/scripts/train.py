#!/usr/bin/env python3
"""
LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions
Training Script

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --gpus 4 --batch-size 16
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lama.training.trainers import make_training_model


def parse_args():
    parser = argparse.ArgumentParser(description='LaMa Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size per GPU (overrides config)')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Maximum epochs (overrides config)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode')
    return parser.parse_args()


def setup_callbacks(config, checkpoint_dir):
    """Setup training callbacks"""
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='lama-{epoch:04d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if config.trainer.get('early_stopping', True):
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=config.trainer.get('early_stopping_patience', 10),
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    return callbacks


def main():
    args = parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override config with command line args
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.max_epochs is not None:
        config.trainer.kwargs.max_epochs = args.max_epochs
    
    # Setup experiment directory
    if args.exp_name is None:
        args.exp_name = f"lama_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    exp_dir = Path('experiments') / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = exp_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save config
    OmegaConf.save(config, exp_dir / 'config.yaml')
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Setup logging
    logger = TensorBoardLogger(
        save_dir=str(exp_dir),
        name='logs',
        version=''
    )
    
    # Create model
    model = make_training_model(config)
    
    # Setup callbacks
    callbacks = setup_callbacks(config, checkpoint_dir)
    
    # Create trainer
    trainer_kwargs = {
        'gpus': args.gpus,
        'max_epochs': config.trainer.kwargs.get('max_epochs', 100),
        'callbacks': callbacks,
        'logger': logger,
        'precision': 16 if args.fp16 else 32,
        'deterministic': True,
        'benchmark': True,
        'gradient_clip_val': config.trainer.kwargs.get('gradient_clip_val', 1.0),
        'val_check_interval': config.trainer.kwargs.get('val_check_interval', 1.0),
        'log_every_n_steps': config.trainer.kwargs.get('log_every_n_steps', 50),
    }
    
    # Multi-GPU settings
    if args.gpus > 1:
        trainer_kwargs['accelerator'] = 'ddp'
        trainer_kwargs['sync_batchnorm'] = True
    
    # Debug settings
    if args.debug:
        trainer_kwargs['fast_dev_run'] = True
        trainer_kwargs['num_sanity_val_steps'] = 0
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train
    print(f"\nStarting training experiment: {args.exp_name}")
    print(f"Results will be saved to: {exp_dir}\n")
    
    if args.resume_from:
        trainer.fit(model, ckpt_path=args.resume_from)
    else:
        trainer.fit(model)
    
    print(f"\nTraining completed! Results saved to: {exp_dir}")


if __name__ == '__main__':
    main()