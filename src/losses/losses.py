"""
Loss functions for LaMa inpainting model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import lpips


class AdversarialLoss(nn.Module):
    """GAN adversarial loss"""
    
    def __init__(self, type_: str = 'lsgan', target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        
        self.type = type_
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        
        if type_ == 'lsgan':
            self.loss = nn.MSELoss()
        elif type_ == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif type_ == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError(f'GAN type [{type_}] is not implemented')
            
    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Create label tensors with the same size as the input"""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return torch.full_like(prediction, target_tensor)

    def __call__(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Calculate loss given Discriminator's output and ground truth labels"""
        if self.type == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        return loss


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for better training stability"""
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
        
    def __call__(self, fake_features: List[torch.Tensor], real_features: List[torch.Tensor]) -> torch.Tensor:
        loss = 0
        for fake_feat, real_feat in zip(fake_features, real_features):
            loss += self.criterion(fake_feat, real_feat.detach())
        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG network"""
    
    def __init__(self, net: str = 'alex'):
        super().__init__()
        self.loss_network = lpips.LPIPS(net=net, spatial=False)
        self.loss_network.eval()
        
        # Freeze parameters
        for param in self.loss_network.parameters():
            param.requires_grad = False
            
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_network(pred, target).mean()


class ResNetPLLoss(nn.Module):
    """ResNet-based perceptual loss"""
    
    def __init__(self, weights_path: Optional[str] = None):
        super().__init__()
        
        from torchvision import models
        
        # Load pre-trained ResNet
        self.net = models.resnet50(pretrained=weights_path is None)
        if weights_path:
            self.net.load_state_dict(torch.load(weights_path))
            
        # Remove final layers
        self.net = nn.Sequential(*list(self.net.children())[:-2])
        self.net.eval()
        
        # Freeze parameters
        for param in self.net.parameters():
            param.requires_grad = False
            
        self.criterion = nn.L1Loss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self.net(pred)
        target_features = self.net(target)
        return self.criterion(pred_features, target_features)


class R1GradientPenalty(nn.Module):
    """R1 gradient penalty for discriminator regularization"""
    
    def __init__(self, gamma: float = 10.0):
        super().__init__()
        self.gamma = gamma
        
    def __call__(self, real_pred: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
        grad_real = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )[0]
        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        return self.gamma / 2 * grad_penalty


class MaskedL1Loss(nn.Module):
    """L1 loss applied only to masked regions"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]  
            mask: Binary mask [B, 1, H, W] (1 for masked regions)
        """
        diff = torch.abs(pred - target)
        masked_diff = diff * mask
        loss = masked_diff.sum() / (mask.sum() + 1e-8)
        return loss


class HighFrequencyLoss(nn.Module):
    """High frequency loss to preserve details"""
    
    def __init__(self):
        super().__init__()
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculate high frequency loss"""
        # Convert to grayscale if needed
        if pred.size(1) == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            pred_gray = pred
            target_gray = target
            
        # Apply Sobel filters
        pred_grad_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        
        target_grad_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        # Apply mask
        diff = torch.abs(pred_grad - target_grad) * mask
        loss = diff.sum() / (mask.sum() + 1e-8)
        
        return loss


class InpaintingLoss(nn.Module):
    """Combined loss for inpainting"""
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        adversarial_weight: float = 0.1,
        feature_matching_weight: float = 10.0,
        resnet_pl_weight: float = 30.0,
        hf_weight: float = 1.0,
        use_lpips: bool = True,
        use_resnet_pl: bool = True,
        resnet_pl_path: Optional[str] = None,
        adversarial_loss_type: str = 'lsgan'
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.feature_matching_weight = feature_matching_weight
        self.resnet_pl_weight = resnet_pl_weight
        self.hf_weight = hf_weight
        
        # Loss functions
        self.l1_loss = MaskedL1Loss()
        self.adversarial_loss = AdversarialLoss(adversarial_loss_type)
        self.feature_matching_loss = FeatureMatchingLoss()
        self.hf_loss = HighFrequencyLoss()
        
        if use_lpips:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None
            
        if use_resnet_pl:
            self.resnet_pl_loss = ResNetPLLoss(resnet_pl_path)
        else:
            self.resnet_pl_loss = None
            
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        discriminator_pred: Optional[torch.Tensor] = None,
        discriminator_real_features: Optional[List[torch.Tensor]] = None,
        discriminator_fake_features: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        losses = {}
        total_loss = 0
        
        # L1 loss
        if self.l1_weight > 0:
            l1_loss = self.l1_loss(pred, target, mask)
            losses['l1'] = l1_loss
            total_loss += self.l1_weight * l1_loss
            
        # Perceptual loss (LPIPS)
        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(pred, target)
            losses['perceptual'] = perceptual_loss
            total_loss += self.perceptual_weight * perceptual_loss
            
        # ResNet perceptual loss
        if self.resnet_pl_loss is not None and self.resnet_pl_weight > 0:
            resnet_pl_loss = self.resnet_pl_loss(pred, target)
            losses['resnet_pl'] = resnet_pl_loss
            total_loss += self.resnet_pl_weight * resnet_pl_loss
            
        # High frequency loss
        if self.hf_weight > 0:
            hf_loss = self.hf_loss(pred, target, mask)
            losses['hf'] = hf_loss
            total_loss += self.hf_weight * hf_loss
            
        # Adversarial loss
        if discriminator_pred is not None and self.adversarial_weight > 0:
            if isinstance(discriminator_pred, list):
                adv_loss = 0
                for pred_i in discriminator_pred:
                    adv_loss += self.adversarial_loss(pred_i[-1], True)
                adv_loss = adv_loss / len(discriminator_pred)
            else:
                adv_loss = self.adversarial_loss(discriminator_pred, True)
            losses['adversarial'] = adv_loss
            total_loss += self.adversarial_weight * adv_loss
            
        # Feature matching loss
        if (discriminator_real_features is not None and 
            discriminator_fake_features is not None and 
            self.feature_matching_weight > 0):
            fm_loss = self.feature_matching_loss(discriminator_fake_features, discriminator_real_features)
            losses['feature_matching'] = fm_loss
            total_loss += self.feature_matching_weight * fm_loss
            
        losses['total'] = total_loss
        return total_loss, losses


def get_losses(config: dict) -> Tuple[InpaintingLoss, AdversarialLoss, Optional[R1GradientPenalty]]:
    """Factory function to create losses based on config"""
    
    generator_loss = InpaintingLoss(
        l1_weight=config.get('l1_weight', 1.0),
        perceptual_weight=config.get('perceptual_weight', 1.0),
        adversarial_weight=config.get('adversarial_weight', 0.1),
        feature_matching_weight=config.get('feature_matching_weight', 10.0),
        resnet_pl_weight=config.get('resnet_pl_weight', 30.0),
        hf_weight=config.get('hf_weight', 1.0),
        use_lpips=config.get('use_lpips', True),
        use_resnet_pl=config.get('use_resnet_pl', True),
        resnet_pl_path=config.get('resnet_pl_path'),
        adversarial_loss_type=config.get('adversarial_loss_type', 'lsgan')
    )
    
    discriminator_loss = AdversarialLoss(config.get('adversarial_loss_type', 'lsgan'))
    
    gradient_penalty = None
    if config.get('use_r1_penalty', False):
        gradient_penalty = R1GradientPenalty(config.get('r1_gamma', 10.0))
        
    return generator_loss, discriminator_loss, gradient_penalty