"""
Discriminator for LaMa inpainting model
Multi-scale discriminator for better training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import functools


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator"""
    
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        use_sigmoid: bool = False
    ):
        super().__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class MultiscaleDiscriminator(nn.Module):
    """Multi-scale discriminator that operates on different scales"""
    
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        use_sigmoid: bool = False,
        num_D: int = 3,
        getIntermFeat: bool = False
    ):
        super().__init__()
        
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid
            )
            setattr(self, f'discriminator_{i}', netD)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        
        for i in range(num_D):
            model = getattr(self, f'discriminator_{i}')
            result.append(self.singleD_forward(model, input_downsampled))
            
            if i != (num_D - 1):
                input_downsampled = F.avg_pool2d(
                    input_downsampled, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False
                )
                
        return result


class ProjectedDiscriminator(nn.Module):
    """
    Projected discriminator from StyleGAN2
    More efficient and better gradient flow
    """
    
    def __init__(
        self,
        diffaug=True,
        interp224=True,
        backbone_kwargs={}
    ):
        super().__init__()
        
        from timm import create_model
        
        backbone_kwargs.setdefault('pretrained', True)
        backbone_kwargs.setdefault('features_only', True)
        backbone_kwargs.setdefault('out_indices', (0, 1, 2, 3))
        
        self.backbone = create_model('resnet50', **backbone_kwargs)
        
        feature_dims = self.backbone.feature_info.channels()
        
        self.diffaug = diffaug
        self.interp224 = interp224
        
        # Projection heads
        self.heads = nn.ModuleList()
        for dim in feature_dims:
            head = nn.Sequential(
                nn.Conv2d(dim, 1, kernel_size=1),
            )
            self.heads.append(head)
            
    def forward(self, x):
        if self.interp224:
            x = F.interpolate(x, 224, mode='bilinear', align_corners=False)
            
        if self.diffaug:
            x = self.apply_augmentation(x)
            
        features = self.backbone(x)
        
        outputs = []
        for feat, head in zip(features, self.heads):
            outputs.append(head(feat))
            
        return outputs
    
    def apply_augmentation(self, x):
        """Simple augmentation for better training"""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            x = torch.flip(x, [3])
        return x


def get_discriminator(config: dict) -> nn.Module:
    """Factory function to create discriminator based on config"""
    
    discriminator_type = config.get('type', 'multiscale')
    
    if discriminator_type == 'nlayer':
        return NLayerDiscriminator(
            input_nc=config.get('input_nc', 3),
            ndf=config.get('ndf', 64),
            n_layers=config.get('n_layers', 3),
            norm_layer=getattr(nn, config.get('norm_layer', 'BatchNorm2d')),
            use_sigmoid=config.get('use_sigmoid', False)
        )
    elif discriminator_type == 'multiscale':
        return MultiscaleDiscriminator(
            input_nc=config.get('input_nc', 3),
            ndf=config.get('ndf', 64),
            n_layers=config.get('n_layers', 3),
            norm_layer=getattr(nn, config.get('norm_layer', 'BatchNorm2d')),
            use_sigmoid=config.get('use_sigmoid', False),
            num_D=config.get('num_D', 3),
            getIntermFeat=config.get('getIntermFeat', False)
        )
    elif discriminator_type == 'projected':
        return ProjectedDiscriminator(
            diffaug=config.get('diffaug', True),
            interp224=config.get('interp224', True),
            backbone_kwargs=config.get('backbone_kwargs', {})
        )
    else:
        raise ValueError(f"Unknown discriminator type: {discriminator_type}")