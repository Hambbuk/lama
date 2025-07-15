import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training"""
    
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        
        self.gan_mode = gan_mode
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.loss = None
        else:
            raise ValueError(f'Unsupported GAN mode: {gan_mode}')
    
    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input"""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def forward(self, prediction, target_is_real):
        """Calculate loss given discriminator's output and ground truth labels"""
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'hinge':
            if target_is_real:
                loss = F.relu(1.0 - prediction).mean()
            else:
                loss = F.relu(1.0 + prediction).mean()
        else:
            raise ValueError(f'Unsupported GAN mode: {self.gan_mode}')
        
        return loss