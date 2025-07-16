#!/usr/bin/env python3
"""
LaMa Inpainting Training Script
Usage: python train.py --config config.yaml --gpus 1 --data_path /path/to/dataset
"""

import os
import argparse
import logging
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from saicinpainting.training.trainers import make_training_model
from saicinpainting.utils import register_debug_signal_handlers

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LaMa inpainting model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    config.data.train.dataloader.batch_size = args.batch_size
    config.trainer.max_epochs = args.epochs
    config.optimizers.generator.lr = args.learning_rate
    config.data.train.dataloader.num_workers = min(args.gpus * 4, 8)
    
    # Set data path
    config.data.train.path = args.data_path
    config.data.val.path = args.data_path
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    OmegaConf.save(config, output_dir / "config.yaml")
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="lama-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Setup logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name="lama_training",
        version=None,
    )
    
    # Create training model
    model = make_training_model(config)
    
    # Setup trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        strategy="ddp" if args.gpus > 1 else None,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        precision=16 if args.gpus > 0 else 32,
        gradient_clip_val=1.0,
        resume_from_checkpoint=args.resume,
    )
    
    # Start training
    logger.info(f"Starting training with {args.gpus} GPUs")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {args.config}")
    
    trainer.fit(model)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()