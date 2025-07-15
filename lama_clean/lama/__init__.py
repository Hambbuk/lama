"""
LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions
"""

__version__ = '1.0.0'

# Make key modules easily accessible
from .training.trainers import make_training_model, load_checkpoint
from .training.data.datasets import make_default_train_dataloader, make_default_val_dataloader
from .evaluation.utils import move_to_device
from .evaluation.refinement import refine_predict

__all__ = [
    'make_training_model',
    'load_checkpoint', 
    'make_default_train_dataloader',
    'make_default_val_dataloader',
    'move_to_device',
    'refine_predict',
]