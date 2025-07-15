#!/usr/bin/env python3
"""
LaMa Inpainting Model Training Script

사용법:
    python train.py --config configs/train.yaml --gpus 4
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lama import LamaInpaintingModel
from src.data.dataloader import get_data_loader
from src.utils.config import load_config, merge_configs
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='LaMa Inpainting Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use (default: 1)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size per GPU (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    parser.add_argument('--output-dir', type=str, default='./experiments',
                        help='Output directory for experiments')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers per GPU')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping value')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 실험 이름 생성
    if args.exp_name is None:
        args.exp_name = f"lama_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 출력 디렉토리 설정
    exp_dir = Path(args.output_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 로거 설정
    logger = setup_logger('train', exp_dir / 'train.log')
    logger.info(f"Starting training experiment: {args.exp_name}")
    logger.info(f"Using {args.gpus} GPU(s)")
    
    # 설정 로드
    config = load_config(args.config)
    
    # 명령줄 인자로 설정 오버라이드
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.lr = args.lr
    
    # 설정 저장
    OmegaConf.save(config, exp_dir / 'config.yaml')
    
    # 시드 설정
    pl.seed_everything(args.seed)
    
    # 데이터 로더 준비
    logger.info("Preparing data loaders...")
    train_loader = get_data_loader(
        config.data.train_dir,
        config.data.train_masks_dir,
        batch_size=config.training.batch_size * args.gpus,
        num_workers=args.num_workers * args.gpus,
        shuffle=True,
        augment=True,
        image_size=config.data.image_size
    )
    
    val_loader = get_data_loader(
        config.data.val_dir,
        config.data.val_masks_dir,
        batch_size=config.training.batch_size * args.gpus,
        num_workers=args.num_workers,
        shuffle=False,
        augment=False,
        image_size=config.data.image_size
    )
    
    # 모델 준비
    logger.info("Initializing model...")
    model = LamaInpaintingModel(config)
    
    # 체크포인트에서 재개
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    
    # 콜백 설정
    callbacks = [
        ModelCheckpoint(
            dirpath=exp_dir / 'checkpoints',
            filename='lama-{epoch:03d}-{val_loss:.4f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            save_last=True,
            verbose=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.training.early_stopping_patience,
            mode='min',
            verbose=True
        )
    ]
    
    # 로거 설정
    tb_logger = TensorBoardLogger(
        save_dir=exp_dir,
        name='tensorboard',
        version=''
    )
    
    # Trainer 설정
    trainer_kwargs = {
        'max_epochs': config.training.epochs,
        'gpus': args.gpus,
        'accelerator': 'ddp' if args.gpus > 1 else None,
        'precision': 16 if args.fp16 else 32,
        'gradient_clip_val': args.gradient_clip,
        'callbacks': callbacks,
        'logger': tb_logger,
        'log_every_n_steps': 50,
        'val_check_interval': config.training.val_check_interval,
        'deterministic': True,
        'benchmark': True,
        'num_sanity_val_steps': 2,
    }
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # 학습 시작
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # 최종 모델 저장
    logger.info("Saving final model...")
    final_model_path = exp_dir / 'final_model.ckpt'
    trainer.save_checkpoint(final_model_path)
    
    logger.info(f"Training completed! Results saved to: {exp_dir}")


if __name__ == '__main__':
    main()