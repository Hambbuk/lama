import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FourierUnit(nn.Module):
    """Fourier Unit for global feature extraction"""
    
    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None,
                 spatial_scale_mode='bilinear', spectral_pos_encoding=False,
                 use_se=False, se_kwargs=None, ffc3d=False, inplace=True):
        super().__init__()
        
        self.groups = groups
        self.in_channels = in_channels // groups
        self.out_channels = out_channels // groups
        
        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.use_se = use_se
        
        # Convolution layers
        self.conv_layer = torch.nn.Conv2d(
            self.in_channels * 2 + (2 if spectral_pos_encoding else 0),
            self.out_channels * 2,
            kernel_size=1, bias=False, groups=self.groups
        )
        self.bn = torch.nn.BatchNorm2d(self.out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=inplace)
        
        # Squeeze and Excitation
        if use_se:
            se_kwargs = se_kwargs or {}
            self.se = SELayer(self.out_channels, **se_kwargs)
        else:
            self.se = None

    def forward(self, x):
        batch = x.shape[0]
        
        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor,
                            mode=self.spatial_scale_mode, align_corners=False)
        
        # FFT
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm="ortho")
        ffted = torch.stack([ffted.real, ffted.imag], dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        
        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(
                batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(
                batch, 1, height, width).to(ffted)
            ffted = torch.cat([ffted, coords_vert, coords_hor], dim=1)
        
        # Convolution in frequency domain
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        
        # Separate real and imaginary parts
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        
        # Inverse FFT
        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm="ortho")
        
        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode,
                                 align_corners=False)
        
        if self.use_se and self.se is not None:
            output = self.se(output)
        
        return output


class SELayer(nn.Module):
    """Squeeze and Excitation Layer"""
    
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)