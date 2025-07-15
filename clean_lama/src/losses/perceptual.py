import torch
import torch.nn as nn
import torchvision.models as models
from typing import List


class VGG19Features(nn.Module):
    """VGG19 feature extractor for perceptual loss"""
    
    def __init__(self, feature_layers: List[str] = None):
        super().__init__()
        
        if feature_layers is None:
            feature_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        
        self.feature_layers = feature_layers
        
        # Load pretrained VGG19
        vgg19 = models.vgg19(pretrained=True)
        
        # Build feature extractor
        self.features = nn.ModuleDict()
        
        # VGG19 layer names
        layer_names = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
        ]
        
        # Extract layers
        features = list(vgg19.features.children())
        
        for i, (name, layer) in enumerate(zip(layer_names, features)):
            self.features[name] = layer
            
            # Stop when we have all required layers
            if name == feature_layers[-1]:
                break
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x):
        # Normalize input
        x = (x - self.mean) / self.std
        
        features = {}
        
        for name, layer in self.features.items():
            x = layer(x)
            if name in self.feature_layers:
                features[name] = x
        
        return features


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features"""
    
    def __init__(self, feature_layers: List[str] = None, weights: List[float] = None):
        super().__init__()
        
        self.vgg = VGG19Features(feature_layers)
        
        if weights is None:
            weights = [1.0] * len(self.vgg.feature_layers)
        
        assert len(weights) == len(self.vgg.feature_layers)
        self.weights = weights
        
        self.criterion = nn.L1Loss()
    
    def forward(self, pred, target):
        # Extract features
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        
        # Compute loss
        loss = 0.0
        
        for layer, weight in zip(self.vgg.feature_layers, self.weights):
            loss += weight * self.criterion(pred_features[layer], target_features[layer])
        
        return loss