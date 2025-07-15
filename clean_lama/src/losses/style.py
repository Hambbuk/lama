import torch
import torch.nn as nn
from .perceptual import VGG19Features
from typing import List


def gram_matrix(x):
    """Compute Gram matrix"""
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


class StyleLoss(nn.Module):
    """Style loss using Gram matrices of VGG19 features"""
    
    def __init__(self, feature_layers: List[str] = None, weights: List[float] = None):
        super().__init__()
        
        if feature_layers is None:
            feature_layers = ['relu2_2', 'relu3_4', 'relu4_4', 'relu5_2']
        
        self.vgg = VGG19Features(feature_layers)
        
        if weights is None:
            weights = [1.0] * len(feature_layers)
        
        assert len(weights) == len(feature_layers)
        self.weights = weights
        
        self.criterion = nn.L1Loss()
    
    def forward(self, pred, target):
        # Extract features
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        
        # Compute style loss
        loss = 0.0
        
        for layer, weight in zip(self.vgg.feature_layers, self.weights):
            # Compute Gram matrices
            pred_gram = gram_matrix(pred_features[layer])
            target_gram = gram_matrix(target_features[layer])
            
            # Compute loss
            loss += weight * self.criterion(pred_gram, target_gram)
        
        return loss