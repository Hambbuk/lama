import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any
from omegaconf import OmegaConf

from .fourier_unit import FourierUnit
from .discriminator import NLayerDiscriminator
from ..losses import PerceptualLoss, StyleLoss, AdversarialLoss
from ..utils.metrics import PSNR, SSIM


class FFC(nn.Module):
    """Fast Fourier Convolution"""
    
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 activation_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 padding_mode='zeros'):
        super().__init__()
        
        self.stride = stride
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        
        # Local connections
        self.convl2l = nn.Conv2d(in_cl, out_cl, kernel_size, stride, padding, 
                                 dilation, groups, bias, padding_mode) if in_cl > 0 and out_cl > 0 else None
        self.convl2g = nn.Conv2d(in_cl, out_cg, kernel_size, stride, padding,
                                 dilation, groups, bias, padding_mode) if in_cl > 0 and out_cg > 0 else None
        self.convg2l = nn.Conv2d(in_cg, out_cl, kernel_size, stride, padding,
                                 dilation, groups, bias, padding_mode) if in_cg > 0 and out_cl > 0 else None
        
        # Global connections
        self.convg2g = FourierUnit(in_cg, out_cg, groups) if in_cg > 0 and out_cg > 0 else None
        
        self.activation = activation_layer(inplace=True)
        
        # Normalization
        self.norm_l = norm_layer(out_cl) if out_cl > 0 else None
        self.norm_g = norm_layer(out_cg) if out_cg > 0 else None

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        
        out_xl, out_xg = 0, 0
        
        if self.convl2l is not None:
            out_xl = self.convl2l(x_l)
        if self.convg2l is not None:
            out_xl = out_xl + self.convg2l(x_g)
        if self.convl2g is not None:
            out_xg = self.convl2g(x_l)
        if self.convg2g is not None:
            out_xg = out_xg + self.convg2g(x_g)
            
        out_xl = self.activation(self.norm_l(out_xl)) if self.norm_l else out_xl
        out_xg = self.activation(self.norm_g(out_xg)) if self.norm_g else out_xg
        
        return out_xl, out_xg


class FFCResnetBlock(nn.Module):
    """FFC ResNet Block"""
    
    def __init__(self, dim, padding_mode='zeros', norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.ReLU, dilation=1, ratio_gin=0.75, ratio_gout=0.75):
        super().__init__()
        
        self.conv1 = FFC(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                         norm_layer=norm_layer, activation_layer=activation_layer,
                         padding_mode=padding_mode, ratio_gin=ratio_gin, ratio_gout=ratio_gout)
        self.conv2 = FFC(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                         norm_layer=norm_layer, activation_layer=activation_layer,
                         padding_mode=padding_mode, ratio_gin=ratio_gout, ratio_gout=ratio_gin)

    def forward(self, x):
        if isinstance(x, tuple):
            x_l, x_g = x
        else:
            x_l, x_g = x, 0
            
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        
        x_l = x_l + id_l
        x_g = x_g + id_g
        
        return x_l, x_g


class LamaGenerator(nn.Module):
    """LaMa Inpainting Generator"""
    
    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_downsampling=3,
                 n_blocks=18, norm_layer=nn.BatchNorm2d, padding_mode='zeros',
                 activation_layer=nn.ReLU, ratio=0.75):
        super().__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3, padding_mode=padding_mode),
            norm_layer(ngf),
            activation_layer(True)
        )
        
        # Downsampling
        self.down_layers = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** i
            self.down_layers.append(nn.Sequential(
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                         stride=2, padding=1, padding_mode=padding_mode),
                norm_layer(ngf * mult * 2),
                activation_layer(True)
            ))
        
        # FFC ResNet blocks
        mult = 2 ** n_downsampling
        self.ffc_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.ffc_blocks.append(FFCResnetBlock(
                ngf * mult, padding_mode=padding_mode, norm_layer=norm_layer,
                activation_layer=activation_layer, ratio_gin=ratio, ratio_gout=ratio
            ))
        
        # Upsampling
        self.up_layers = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            self.up_layers.append(nn.Sequential(
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3,
                                  stride=2, padding=1, output_padding=1),
                norm_layer(ngf * mult // 2),
                activation_layer(True)
            ))
        
        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3, padding_mode=padding_mode),
            nn.Tanh()
        )
        
        self.ratio = ratio

    def forward(self, x):
        # Initial convolution
        x = self.initial(x)
        
        # Downsampling
        for down in self.down_layers:
            x = down(x)
        
        # FFC blocks
        x_l, x_g = x, 0
        for block in self.ffc_blocks:
            x_l, x_g = block((x_l, x_g))
        
        # Combine local and global features
        x = x_l + x_g
        
        # Upsampling
        for up in self.up_layers:
            x = up(x)
        
        # Final convolution
        x = self.final(x)
        
        return x


class LamaInpaintingModel(pl.LightningModule):
    """LaMa Inpainting Lightning Model"""
    
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Generator
        self.generator = LamaGenerator(
            input_nc=4,  # RGB + mask
            output_nc=3,  # RGB
            ngf=config.model.get('ngf', 64),
            n_downsampling=config.model.get('n_downsampling', 3),
            n_blocks=config.model.get('n_blocks', 18),
            ratio=config.model.get('ratio', 0.75)
        )
        
        # Discriminator (for training only)
        if not config.get('predict_only', False):
            self.discriminator = NLayerDiscriminator(
                input_nc=3,
                ndf=config.model.get('ndf', 64),
                n_layers=config.model.get('n_discriminator_layers', 3)
            )
            
            # Losses
            self.l1_loss = nn.L1Loss()
            self.perceptual_loss = PerceptualLoss()
            self.style_loss = StyleLoss()
            self.adversarial_loss = AdversarialLoss(config.training.get('gan_mode', 'lsgan'))
            
            # Loss weights
            self.lambda_l1 = config.losses.get('l1', 1.0)
            self.lambda_perceptual = config.losses.get('perceptual', 10.0)
            self.lambda_style = config.losses.get('style', 250.0)
            self.lambda_adversarial = config.losses.get('adversarial', 0.1)
        
        # Metrics
        self.psnr = PSNR()
        self.ssim = SSIM()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        image = batch['image']
        mask = batch['mask']
        
        # Prepare input
        masked_image = image * (1 - mask)
        input_tensor = torch.cat([masked_image, mask], dim=1)
        
        # Generate
        output = self.generator(input_tensor)
        
        # Combine with unmasked regions
        inpainted = masked_image + output * mask
        
        return {
            'inpainted': inpainted,
            'generated': output
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, 
                      optimizer_idx: int) -> torch.Tensor:
        """Training step"""
        image = batch['image']
        mask = batch['mask']
        
        # Generator training
        if optimizer_idx == 0:
            # Forward
            output = self(batch)
            inpainted = output['inpainted']
            
            # Reconstruction losses
            loss_l1 = self.l1_loss(inpainted, image) * self.lambda_l1
            loss_perceptual = self.perceptual_loss(inpainted, image) * self.lambda_perceptual
            loss_style = self.style_loss(inpainted, image) * self.lambda_style
            
            # Adversarial loss
            pred_fake = self.discriminator(inpainted)
            loss_g_adversarial = self.adversarial_loss(pred_fake, True) * self.lambda_adversarial
            
            # Total generator loss
            loss_g = loss_l1 + loss_perceptual + loss_style + loss_g_adversarial
            
            # Logging
            self.log('train/loss_g', loss_g)
            self.log('train/loss_l1', loss_l1)
            self.log('train/loss_perceptual', loss_perceptual)
            self.log('train/loss_style', loss_style)
            self.log('train/loss_g_adversarial', loss_g_adversarial)
            
            return loss_g
        
        # Discriminator training
        elif optimizer_idx == 1:
            # Forward
            with torch.no_grad():
                output = self(batch)
                inpainted = output['inpainted']
            
            # Real loss
            pred_real = self.discriminator(image)
            loss_d_real = self.adversarial_loss(pred_real, True)
            
            # Fake loss
            pred_fake = self.discriminator(inpainted.detach())
            loss_d_fake = self.adversarial_loss(pred_fake, False)
            
            # Total discriminator loss
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            
            # Logging
            self.log('train/loss_d', loss_d)
            self.log('train/loss_d_real', loss_d_real)
            self.log('train/loss_d_fake', loss_d_fake)
            
            return loss_d

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step"""
        image = batch['image']
        
        # Forward
        output = self(batch)
        inpainted = output['inpainted']
        
        # Compute metrics
        psnr = self.psnr(inpainted, image)
        ssim = self.ssim(inpainted, image)
        
        # Compute loss
        loss_l1 = self.l1_loss(inpainted, image)
        
        # Logging
        self.log('val/loss', loss_l1)
        self.log('val/psnr', psnr)
        self.log('val/ssim', ssim)
        
        return {'val_loss': loss_l1}

    def configure_optimizers(self):
        """Configure optimizers"""
        # Generator optimizer
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.training.lr,
            betas=(self.config.training.get('beta1', 0.0), 
                   self.config.training.get('beta2', 0.9))
        )
        
        # Discriminator optimizer
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.training.lr,
            betas=(self.config.training.get('beta1', 0.0), 
                   self.config.training.get('beta2', 0.9))
        )
        
        # Schedulers
        scheduler_g = torch.optim.lr_scheduler.StepLR(
            opt_g, 
            step_size=self.config.training.get('lr_decay_steps', 1000),
            gamma=self.config.training.get('lr_decay_rate', 0.1)
        )
        
        scheduler_d = torch.optim.lr_scheduler.StepLR(
            opt_d,
            step_size=self.config.training.get('lr_decay_steps', 1000),
            gamma=self.config.training.get('lr_decay_rate', 0.1)
        )
        
        return [opt_g, opt_d], [scheduler_g, scheduler_d]