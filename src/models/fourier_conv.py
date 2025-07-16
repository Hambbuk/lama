"""
Fourier Convolutions for LaMa inpainting model.
Based on "Resolution-robust Large Mask Inpainting with Fourier Convolutions"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpectralTransform(nn.Module):
    """Spectral Transform for Fourier Convolution"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        
        # Fourier domain weights
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, 1, 1, 2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        
        # Apply FFT
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        # Convert to real representation for multiplication
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        
        # Reshape for group convolution
        x_ft = x_ft.view(batch, self.groups, channels // self.groups, height, width // 2 + 1, 2)
        
        # Apply spectral transform
        weight = self.weight.view(self.out_channels, 1, channels // self.groups, 1, 1, 2)
        
        # Complex multiplication in real representation
        out_ft_real = x_ft[..., 0] * weight[..., 0] - x_ft[..., 1] * weight[..., 1]
        out_ft_imag = x_ft[..., 0] * weight[..., 1] + x_ft[..., 1] * weight[..., 0]
        
        # Combine real and imaginary parts
        out_ft = torch.complex(out_ft_real.sum(dim=2), out_ft_imag.sum(dim=2))
        
        # Apply inverse FFT
        out = torch.fft.irfft2(out_ft, s=(height, width), dim=(-2, -1), norm='ortho')
        
        return out


class FourierUnit(nn.Module):
    """Fourier Unit combining spectral and spatial processing"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        groups: int = 1,
        spatial_scale_factor: float = 0.5,
        spectral_pos_encoding: bool = False
    ):
        super().__init__()
        
        self.groups = groups
        inter_channels = max(1, int(out_channels * spatial_scale_factor))
        
        # Spectral transform
        self.spectral_transform = SpectralTransform(
            in_channels, out_channels, groups=groups
        )
        
        # Spatial convolution branch
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, 1, groups=groups, bias=False)
        
        # Combine spectral and spatial
        self.conv_out = nn.Conv2d(out_channels * 2, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectral branch
        spectral_out = self.spectral_transform(x)
        
        # Spatial branch
        spatial_out = self.conv1(x)
        spatial_out = self.bn1(spatial_out)
        spatial_out = self.relu(spatial_out)
        spatial_out = self.conv2(spatial_out)
        
        # Combine
        combined = torch.cat([spectral_out, spatial_out], dim=1)
        out = self.conv_out(combined)
        out = self.bn_out(out)
        
        return out


class FFC_BN_ACT(nn.Module):
    """Fast Fourier Convolution with BatchNorm and Activation"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        ratio_gin: float = 0.5,
        ratio_gout: float = 0.5,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation_layer: nn.Module = nn.ReLU
    ):
        super().__init__()
        
        assert stride == 1 or stride == 2, "stride should be 1 or 2."
        
        self.stride = stride
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        
        # Local convolution
        if in_cl > 0 and out_cl > 0:
            self.convl2l = nn.Conv2d(
                in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias
            )
            self.bnl2l = norm_layer(out_cl) if norm_layer else nn.Identity()
            
        # Local to global
        if in_cl > 0 and out_cg > 0:
            self.convl2g = nn.Conv2d(
                in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias
            )
            self.bnl2g = norm_layer(out_cg) if norm_layer else nn.Identity()
            
        # Global convolution
        if in_cg > 0 and out_cg > 0:
            self.convg2g = FourierUnit(in_cg, out_cg, groups)
            self.bng2g = norm_layer(out_cg) if norm_layer else nn.Identity()
            
        # Global to local
        if in_cg > 0 and out_cl > 0:
            self.convg2l = nn.Conv2d(
                in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias
            )
            self.bng2l = norm_layer(out_cl) if norm_layer else nn.Identity()
            
        self.activation = activation_layer(inplace=True) if activation_layer else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l, x_g = self.split_channels(x)
        
        out_xl, out_xg = 0, 0
        
        # Local to local
        if hasattr(self, 'convl2l'):
            out_xl += self.bnl2l(self.convl2l(x_l))
            
        # Local to global  
        if hasattr(self, 'convl2g'):
            out_xg += self.bnl2g(self.convl2g(x_l))
            
        # Global to global
        if hasattr(self, 'convg2g'):
            out_xg += self.bng2g(self.convg2g(x_g))
            
        # Global to local
        if hasattr(self, 'convg2l'):
            out_xl += self.bng2l(self.convg2l(x_g))
            
        # Combine
        out = torch.cat([out_xl, out_xg], dim=1) if (hasattr(self, 'convl2l') or hasattr(self, 'convg2l')) and (hasattr(self, 'convl2g') or hasattr(self, 'convg2g')) else (out_xl if hasattr(self, 'convl2l') or hasattr(self, 'convg2l') else out_xg)
        
        return self.activation(out)
    
    def split_channels(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split input into local and global channels"""
        if self.ratio_gin == 0:
            return x, None
        elif self.ratio_gin == 1:
            return None, x
        else:
            split_point = int(x.size(1) * self.ratio_gin)
            return x[:, split_point:], x[:, :split_point]


class FFCResnetBlock(nn.Module):
    """FFC ResNet Block for the generator"""
    
    def __init__(
        self,
        dim: int,
        padding_type: str = 'reflect',
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU,
        dilation: int = 1,
        ratio_gin: float = 0.5,
        ratio_gout: float = 0.5
    ):
        super().__init__()
        
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, activation, dilation, ratio_gin, ratio_gout
        )
        
    def build_conv_block(self, dim, padding_type, norm_layer, activation, dilation, ratio_gin, ratio_gout):
        conv_block = []
        
        # First FFC layer
        conv_block += [
            FFC_BN_ACT(
                dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                norm_layer=norm_layer, activation_layer=activation,
                ratio_gin=ratio_gin, ratio_gout=ratio_gout
            )
        ]
        
        # Second FFC layer
        conv_block += [
            FFC_BN_ACT(
                dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                norm_layer=norm_layer, activation_layer=None,
                ratio_gin=ratio_gin, ratio_gout=ratio_gout
            )
        ]
        
        return nn.Sequential(*conv_block)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.conv_block(x)
        return out