import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PSNR(nn.Module):
    """Peak Signal-to-Noise Ratio metric"""
    
    def __init__(self, max_value: float = 1.0):
        super().__init__()
        self.max_value = max_value
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute PSNR
        
        Args:
            pred: Predicted image
            target: Target image
            mask: Optional mask to compute PSNR only on masked regions
        
        Returns:
            PSNR value
        """
        if mask is not None:
            # Compute MSE only on masked regions
            mse = F.mse_loss(pred * mask, target * mask, reduction='sum')
            mse = mse / mask.sum()
        else:
            mse = F.mse_loss(pred, target)
        
        if mse == 0:
            return torch.tensor(float('inf'))
        
        return 20 * torch.log10(self.max_value / torch.sqrt(mse))


class SSIM(nn.Module):
    """Structural Similarity Index metric"""
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 data_range: float = 1.0, k1: float = 0.01, k2: float = 0.03,
                 channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.channel = channel
        
        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, channel, sigma))
    
    def _create_window(self, window_size: int, channel: int, sigma: float) -> torch.Tensor:
        """Create Gaussian window"""
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / (2.0 * sigma ** 2)))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        
        return window
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM
        
        Args:
            pred: Predicted image
            target: Target image
        
        Returns:
            SSIM value
        """
        channel = pred.size(1)
        window = self.window.type_as(pred)
        
        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim_map.mean()


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity metric"""
    
    def __init__(self, net: str = 'alex'):
        super().__init__()
        # Note: This is a placeholder. For actual LPIPS, you would need to install lpips package
        # and use: self.model = lpips.LPIPS(net=net)
        self.model = None
        print("Warning: LPIPS not available. Install lpips package for perceptual metrics.")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            return torch.tensor(0.0)
        return self.model(pred, target).mean()