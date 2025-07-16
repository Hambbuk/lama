"""
PyTorch Lightning Module for LaMa Inpainting
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from typing import Any, Dict, List, Optional, Tuple, Union
import torchmetrics
import wandb

from models.generator import get_generator
from models.discriminator import get_discriminator
from losses.losses import get_losses


class InpaintingLightningModule(pl.LightningModule):
    """Lightning module for inpainting training"""
    
    def __init__(
        self,
        generator_config: Dict,
        discriminator_config: Dict,
        loss_config: Dict,
        optimizer_config: Dict,
        scheduler_config: Optional[Dict] = None,
        automatic_optimization: bool = False
    ):
        super().__init__()
        
        # Store configs
        self.save_hyperparameters()
        self.automatic_optimization = automatic_optimization
        
        # Initialize models
        self.generator = get_generator(generator_config)
        self.discriminator = get_discriminator(discriminator_config)
        
        # Initialize losses
        self.generator_loss, self.discriminator_loss, self.gradient_penalty = get_losses(loss_config)
        
        # Initialize metrics
        self._init_metrics()
        
        # Training step count for discriminator/generator ratio
        self.global_step_count = 0
        self.discriminator_update_ratio = loss_config.get('discriminator_update_ratio', 1)
        
    def _init_metrics(self):
        """Initialize metrics for monitoring"""
        # PSNR and SSIM for validation
        self.val_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.val_ssim = torchmetrics.StructuralSimilarityIndexMeasure()
        
        # FID for evaluation (computed externally)
        self.val_metrics = torchmetrics.MetricCollection({
            'psnr': torchmetrics.PeakSignalNoiseRatio(),
            'ssim': torchmetrics.StructuralSimilarityIndexMeasure()
        })
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference"""
        return self.generator(x)
        
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        opt_config = self.hparams.optimizer_config
        
        # Generator optimizer
        if opt_config.get('optimizer_type', 'adam') == 'adamw':
            gen_optimizer = AdamW(
                self.generator.parameters(),
                lr=opt_config.get('generator_lr', 1e-4),
                betas=opt_config.get('betas', (0.5, 0.999)),
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        else:
            gen_optimizer = Adam(
                self.generator.parameters(),
                lr=opt_config.get('generator_lr', 1e-4),
                betas=opt_config.get('betas', (0.5, 0.999))
            )
            
        # Discriminator optimizer
        if opt_config.get('optimizer_type', 'adam') == 'adamw':
            disc_optimizer = AdamW(
                self.discriminator.parameters(),
                lr=opt_config.get('discriminator_lr', 1e-4),
                betas=opt_config.get('betas', (0.5, 0.999)),
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        else:
            disc_optimizer = Adam(
                self.discriminator.parameters(),
                lr=opt_config.get('discriminator_lr', 1e-4),
                betas=opt_config.get('betas', (0.5, 0.999))
            )
            
        optimizers = [gen_optimizer, disc_optimizer]
        
        # Schedulers
        schedulers = []
        if self.hparams.scheduler_config:
            sch_config = self.hparams.scheduler_config
            scheduler_type = sch_config.get('type', 'step')
            
            for optimizer in optimizers:
                if scheduler_type == 'step':
                    scheduler = StepLR(
                        optimizer,
                        step_size=sch_config.get('step_size', 30),
                        gamma=sch_config.get('gamma', 0.1)
                    )
                elif scheduler_type == 'cosine':
                    scheduler = CosineAnnealingLR(
                        optimizer,
                        T_max=sch_config.get('T_max', 100),
                        eta_min=sch_config.get('eta_min', 1e-6)
                    )
                elif scheduler_type == 'plateau':
                    scheduler = ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=sch_config.get('factor', 0.5),
                        patience=sch_config.get('patience', 10)
                    )
                else:
                    raise ValueError(f"Unknown scheduler type: {scheduler_type}")
                    
                schedulers.append(scheduler)
                
        if schedulers:
            return optimizers, schedulers
        else:
            return optimizers
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step"""
        if self.automatic_optimization:
            return self._automatic_training_step(batch, batch_idx)
        else:
            return self._manual_training_step(batch, batch_idx)
            
    def _automatic_training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Automatic optimization training step"""
        image = batch['image']
        mask = batch['mask']
        generator_input = batch['generator_input']
        
        # Generate fake image
        fake_image = self.generator(generator_input)
        
        # Discriminator predictions
        real_pred = self.discriminator(image)
        fake_pred = self.discriminator(fake_image.detach())
        
        # Discriminator loss
        disc_real_loss = self.discriminator_loss(real_pred, True)
        disc_fake_loss = self.discriminator_loss(fake_pred, False)
        disc_loss = (disc_real_loss + disc_fake_loss) * 0.5
        
        # Gradient penalty if enabled
        if self.gradient_penalty is not None:
            real_image = image.requires_grad_(True)
            real_pred_gp = self.discriminator(real_image)
            gp_loss = self.gradient_penalty(real_pred_gp, real_image)
            disc_loss += gp_loss
            
        # Generator loss
        fake_pred_gen = self.discriminator(fake_image)
        gen_loss, gen_losses = self.generator_loss(
            fake_image, image, mask, fake_pred_gen
        )
        
        # Log losses
        self.log('train/disc_loss', disc_loss, on_step=True, on_epoch=True)
        self.log('train/gen_loss', gen_loss, on_step=True, on_epoch=True)
        
        for key, value in gen_losses.items():
            self.log(f'train/gen_{key}', value, on_step=True, on_epoch=True)
            
        # Return loss for the current optimizer
        if self.optimizers()[self.trainer.optimizer_step % 2] == self.optimizers()[0]:
            return gen_loss
        else:
            return disc_loss
            
    def _manual_training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Manual optimization training step"""
        gen_opt, disc_opt = self.optimizers()
        
        image = batch['image']
        mask = batch['mask']
        generator_input = batch['generator_input']
        
        # Generate fake image
        fake_image = self.generator(generator_input)
        
        # Update discriminator
        if self.global_step_count % self.discriminator_update_ratio == 0:
            disc_opt.zero_grad()
            
            # Real and fake predictions
            real_pred = self.discriminator(image)
            fake_pred = self.discriminator(fake_image.detach())
            
            # Discriminator losses
            disc_real_loss = self.discriminator_loss(real_pred, True)
            disc_fake_loss = self.discriminator_loss(fake_pred, False)
            disc_loss = (disc_real_loss + disc_fake_loss) * 0.5
            
            # Gradient penalty
            if self.gradient_penalty is not None:
                real_image = image.requires_grad_(True)
                real_pred_gp = self.discriminator(real_image)
                gp_loss = self.gradient_penalty(real_pred_gp, real_image)
                disc_loss += gp_loss
                self.log('train/gp_loss', gp_loss, on_step=True, on_epoch=True)
                
            self.manual_backward(disc_loss)
            disc_opt.step()
            
            self.log('train/disc_loss', disc_loss, on_step=True, on_epoch=True)
            self.log('train/disc_real_loss', disc_real_loss, on_step=True, on_epoch=True)
            self.log('train/disc_fake_loss', disc_fake_loss, on_step=True, on_epoch=True)
        
        # Update generator
        gen_opt.zero_grad()
        
        fake_pred_gen = self.discriminator(fake_image)
        
        # Extract features for feature matching loss if needed
        fake_features = None
        real_features = None
        if hasattr(self.discriminator, 'getIntermFeat') and self.discriminator.getIntermFeat:
            # For multiscale discriminator with intermediate features
            real_features = self.discriminator(image)
            fake_features = fake_pred_gen
            
        gen_loss, gen_losses = self.generator_loss(
            fake_image, image, mask, fake_pred_gen, real_features, fake_features
        )
        
        self.manual_backward(gen_loss)
        gen_opt.step()
        
        # Log generator losses
        self.log('train/gen_loss', gen_loss, on_step=True, on_epoch=True)
        for key, value in gen_losses.items():
            if key != 'total':
                self.log(f'train/gen_{key}', value, on_step=True, on_epoch=True)
                
        self.global_step_count += 1
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step"""
        image = batch['image']
        mask = batch['mask']
        generator_input = batch['generator_input']
        
        # Generate prediction
        with torch.no_grad():
            fake_image = self.generator(generator_input)
            
        # Compute generator loss for monitoring
        fake_pred = self.discriminator(fake_image)
        gen_loss, gen_losses = self.generator_loss(fake_image, image, mask, fake_pred)
        
        # Denormalize images for metric computation
        fake_image_denorm = (fake_image + 1) / 2
        real_image_denorm = (image + 1) / 2
        
        # Compute metrics
        self.val_metrics.update(fake_image_denorm, real_image_denorm)
        
        # Log validation losses
        self.log('val/gen_loss', gen_loss, on_step=False, on_epoch=True)
        for key, value in gen_losses.items():
            if key != 'total':
                self.log(f'val/gen_{key}', value, on_step=False, on_epoch=True)
                
        # Log sample images
        if batch_idx == 0:
            self._log_images(image, mask, fake_image, 'val')
            
        return gen_loss
        
    def validation_epoch_end(self, outputs):
        """End of validation epoch"""
        # Compute and log metrics
        metrics = self.val_metrics.compute()
        for key, value in metrics.items():
            self.log(f'val/{key}', value)
            
        self.val_metrics.reset()
        
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step"""
        return self.validation_step(batch, batch_idx)
        
    def _log_images(self, real: torch.Tensor, mask: torch.Tensor, fake: torch.Tensor, stage: str):
        """Log sample images to wandb"""
        if isinstance(self.logger, pl.loggers.WandbLogger):
            # Take first 4 images from batch
            n_samples = min(4, real.size(0))
            
            # Denormalize images
            real_denorm = ((real[:n_samples] + 1) / 2).clamp(0, 1)
            fake_denorm = ((fake[:n_samples] + 1) / 2).clamp(0, 1)
            mask_viz = mask[:n_samples].repeat(1, 3, 1, 1)
            
            # Create masked images for visualization
            masked_images = real_denorm * (1 - mask_viz)
            
            # Create grid
            images = []
            for i in range(n_samples):
                images.extend([
                    wandb.Image(real_denorm[i], caption=f"{stage}_real_{i}"),
                    wandb.Image(masked_images[i], caption=f"{stage}_masked_{i}"),
                    wandb.Image(fake_denorm[i], caption=f"{stage}_fake_{i}"),
                ])
                
            self.logger.experiment.log({f"{stage}_images": images})
            
    def on_train_epoch_end(self):
        """End of training epoch"""
        # Update learning rate schedulers
        if hasattr(self, 'lr_schedulers'):
            schedulers = self.lr_schedulers()
            if schedulers:
                for scheduler in schedulers:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        # Use validation loss for plateau scheduler
                        val_loss = self.trainer.callback_metrics.get('val/gen_loss')
                        if val_loss is not None:
                            scheduler.step(val_loss)
                    else:
                        scheduler.step()
                        
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Prediction step for inference"""
        generator_input = batch['generator_input']
        return self.generator(generator_input)
        
    def configure_callbacks(self):
        """Configure callbacks"""
        callbacks = []
        
        # Model checkpoint
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val/gen_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            filename='lama-{epoch:02d}-{val_gen_loss:.4f}'
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val/gen_loss',
            mode='min',
            patience=20,
            verbose=True
        )
        callbacks.append(early_stop_callback)
        
        # Learning rate monitor
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        return callbacks