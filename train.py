"""
Training script for LaMa inpainting model using PyTorch Lightning and Hydra
"""

import os
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from omegaconf import DictConfig, OmegaConf
import torch

from src.lightning_module import InpaintingLightningModule
from src.data.dataset import get_datamodule


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    """Main training function"""
    
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    
    # Print configuration
    print("=" * 50)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 50)
    
    # Initialize data module
    print("Initializing data module...")
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # Initialize model
    print("Initializing model...")
    model = hydra.utils.instantiate(cfg.model)
    
    # Setup logger
    logger = None
    if 'logger' in cfg and cfg.logger is not None:
        if cfg.logger.get('type') == 'tensorboard':
            logger = TensorBoardLogger(
                save_dir=cfg.output_dir,
                name=cfg.experiment_name
            )
            print(f"Using {cfg.logger.type} logger")
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        filename=f'{cfg.experiment_name}-{{epoch:02d}}-{{val_gen_loss:.4f}}',
        monitor='val/gen_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val/gen_loss',
        mode='min',
        patience=20,
        verbose=True,
        strict=False
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator='auto',
        devices=cfg.trainer.get('gpus', 'auto'),
        precision=cfg.trainer.get('precision', 32),
        gradient_clip_val=cfg.trainer.get('gradient_clip_val', None),
        accumulate_grad_batches=cfg.trainer.get('accumulate_grad_batches', 1),
        check_val_every_n_epoch=cfg.trainer.get('check_val_every_n_epoch', 1),
        log_every_n_steps=cfg.trainer.get('log_every_n_steps', 50),
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # Set to True for reproducibility but slower training
        benchmark=True,  # Optimize for consistent input sizes
    )
    
    # Print model summary
    print("\nModel Summary:")
    print(f"Generator parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Start training
    print(f"\nStarting training for {cfg.trainer.max_epochs} epochs...")
    print(f"Output directory: {os.getcwd()}")
    
    try:
        trainer.fit(model, datamodule)
        
        # Test best model
        print("\nTesting best model...")
        trainer.test(model, datamodule, ckpt_path='best')
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    
    print("Training completed!")
    
    # Save final model
    final_model_path = os.path.join(cfg.checkpoint_dir, f"{cfg.experiment_name}_final.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    train()