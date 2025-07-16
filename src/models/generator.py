"""
Generator for LaMa inpainting model using Fourier Convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import functools

from .fourier_conv import FFC_BN_ACT, FFCResnetBlock


class FFCGenerator(nn.Module):
    """
    FFC-based Generator for image inpainting
    Uses Fourier convolutions for large receptive field
    """
    
    def __init__(
        self,
        input_nc: int = 4,  # RGB + mask
        output_nc: int = 3,  # RGB
        ngf: int = 64,
        n_downsampling: int = 3,
        n_blocks: int = 9,
        norm_layer: nn.Module = nn.BatchNorm2d,
        padding_type: str = 'reflect',
        activation_layer: nn.Module = nn.ReLU,
        up_norm_layer: nn.Module = nn.BatchNorm2d,
        up_activation: nn.Module = nn.ReLU,
        init_conv_kwargs: dict = None,
        downsample_conv_kwargs: dict = None,
        resnet_conv_kwargs: dict = None,
        upsample_conv_kwargs: dict = None,
        output_conv_kwargs: dict = None
    ):
        super().__init__()
        
        if init_conv_kwargs is None:
            init_conv_kwargs = {'ratio_gin': 0, 'ratio_gout': 0}
        if downsample_conv_kwargs is None:
            downsample_conv_kwargs = {'ratio_gin': 0, 'ratio_gout': 0.75}
        if resnet_conv_kwargs is None:
            resnet_conv_kwargs = {'ratio_gin': 0.75, 'ratio_gout': 0.75}
        if upsample_conv_kwargs is None:
            upsample_conv_kwargs = {'ratio_gin': 0.75, 'ratio_gout': 0}
        if output_conv_kwargs is None:
            output_conv_kwargs = {'ratio_gin': 0, 'ratio_gout': 0}
            
        self.n_downsampling = n_downsampling
        
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            FFC_BN_ACT(
                input_nc, ngf, kernel_size=7, padding=0,
                norm_layer=norm_layer, activation_layer=activation_layer,
                **init_conv_kwargs
            )
        ]
        
        # Downsampling
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                FFC_BN_ACT(
                    ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1,
                    norm_layer=norm_layer, activation_layer=activation_layer,
                    **downsample_conv_kwargs
                )
            ]
            
        # ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
                FFCResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    activation=activation_layer,
                    norm_layer=norm_layer,
                    **resnet_conv_kwargs
                )
            ]
            
        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                FFC_BN_ACT(
                    ngf * mult, ngf * mult // 2, kernel_size=3, padding=1,
                    norm_layer=up_norm_layer, activation_layer=up_activation,
                    **upsample_conv_kwargs
                )
            ]
            
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            FFC_BN_ACT(
                ngf, output_nc, kernel_size=7, padding=0,
                norm_layer=None, activation_layer=nn.Tanh,
                **output_conv_kwargs
            )
        ]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResnetBlock(nn.Module):
    """Define a Resnet block for baseline comparison"""

    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False), norm_layer(dim), activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class BaselineGenerator(nn.Module):
    """Baseline ResNet-based generator for comparison"""
    
    def __init__(
        self,
        input_nc: int = 4,
        output_nc: int = 3,
        ngf: int = 64,
        norm_layer: nn.Module = nn.BatchNorm2d,
        use_dropout: bool = False,
        n_blocks: int = 6,
        padding_type: str = 'reflect'
    ):
        super().__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=nn.ReLU(True), norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


def get_generator(config: dict) -> nn.Module:
    """Factory function to create generator based on config"""
    
    generator_type = config.get('type', 'ffc')
    
    if generator_type == 'ffc':
        return FFCGenerator(
            input_nc=config.get('input_nc', 4),
            output_nc=config.get('output_nc', 3),
            ngf=config.get('ngf', 64),
            n_downsampling=config.get('n_downsampling', 3),
            n_blocks=config.get('n_blocks', 9),
            norm_layer=getattr(nn, config.get('norm_layer', 'BatchNorm2d')),
            **config.get('kwargs', {})
        )
    elif generator_type == 'baseline':
        return BaselineGenerator(
            input_nc=config.get('input_nc', 4),
            output_nc=config.get('output_nc', 3),
            ngf=config.get('ngf', 64),
            n_blocks=config.get('n_blocks', 6),
            norm_layer=getattr(nn, config.get('norm_layer', 'BatchNorm2d')),
            use_dropout=config.get('use_dropout', False),
            padding_type=config.get('padding_type', 'reflect')
        )
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")