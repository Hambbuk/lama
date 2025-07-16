"""
Dataset and DataModule for inpainting training
"""

import os
import random
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MaskGenerator:
    """Base class for mask generation"""
    
    def __call__(self, height: int, width: int) -> np.ndarray:
        raise NotImplementedError


class RandomMaskGenerator(MaskGenerator):
    """Random rectangular mask generator"""
    
    def __init__(
        self,
        min_size: float = 0.02,
        max_size: float = 0.4,
        min_aspect: float = 0.3,
        max_aspect: float = 3.0,
        max_masks: int = 1
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.max_masks = max_masks
        
    def __call__(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        
        num_masks = random.randint(1, self.max_masks)
        
        for _ in range(num_masks):
            # Random mask size
            mask_size = random.uniform(self.min_size, self.max_size)
            mask_area = mask_size * height * width
            
            # Random aspect ratio
            aspect = random.uniform(self.min_aspect, self.max_aspect)
            
            # Calculate dimensions
            mask_h = int(np.sqrt(mask_area / aspect))
            mask_w = int(mask_area / mask_h)
            
            # Ensure mask fits in image
            mask_h = min(mask_h, height - 1)
            mask_w = min(mask_w, width - 1)
            
            # Random position
            start_y = random.randint(0, height - mask_h)
            start_x = random.randint(0, width - mask_w)
            
            # Apply mask
            mask[start_y:start_y + mask_h, start_x:start_x + mask_w] = 1
            
        return mask


class IrregularMaskGenerator(MaskGenerator):
    """Irregular mask generator using random walk"""
    
    def __init__(
        self,
        min_num_vertex: int = 4,
        max_num_vertex: int = 12,
        min_width: int = 10,
        max_width: int = 40,
        max_angle: float = 4.0,
        max_len: int = 60
    ):
        self.min_num_vertex = min_num_vertex
        self.max_num_vertex = max_num_vertex
        self.min_width = min_width
        self.max_width = max_width
        self.max_angle = max_angle
        self.max_len = max_len
        
    def __call__(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        
        num_vertex = np.random.randint(self.min_num_vertex, self.max_num_vertex)
        
        for _ in range(num_vertex):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            
            for _ in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(self.max_angle)
                if np.random.randint(2) == 0:
                    angle = 2 * np.pi - angle
                    
                length = 10 + np.random.randint(self.max_len)
                brush_w = self.min_width + np.random.randint(self.max_width - self.min_width)
                
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1, brush_w)
                
                start_x, start_y = end_x, end_y
                
        return np.clip(mask, 0, 1)


class OutpaintingMaskGenerator(MaskGenerator):
    """Generate masks for outpainting (extending image boundaries)"""
    
    def __init__(self, min_crop: float = 0.2, max_crop: float = 0.8):
        self.min_crop = min_crop
        self.max_crop = max_crop
        
    def __call__(self, height: int, width: int) -> np.ndarray:
        mask = np.ones((height, width), dtype=np.uint8)
        
        # Random crop size
        crop_ratio = random.uniform(self.min_crop, self.max_crop)
        
        crop_h = int(height * crop_ratio)
        crop_w = int(width * crop_ratio)
        
        # Center crop
        start_y = (height - crop_h) // 2
        start_x = (width - crop_w) // 2
        
        mask[start_y:start_y + crop_h, start_x:start_x + crop_w] = 0
        
        return mask


class InpaintingDataset(Dataset):
    """Dataset for inpainting training"""
    
    def __init__(
        self,
        data_root: str,
        mask_generators: List[MaskGenerator],
        image_size: int = 256,
        augment: bool = True,
        load_size: Optional[int] = None
    ):
        self.data_root = Path(data_root)
        self.mask_generators = mask_generators
        self.image_size = image_size
        self.load_size = load_size or image_size
        
        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(list(self.data_root.glob(f'**/{ext}')))
            self.image_paths.extend(list(self.data_root.glob(f'**/{ext.upper()}')))
            
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_root}")
            
        print(f"Found {len(self.image_paths)} images in {data_root}")
        
        # Setup augmentations
        if augment:
            self.transform = A.Compose([
                A.Resize(self.load_size, self.load_size),
                A.RandomCrop(self.image_size, self.image_size) if self.load_size > self.image_size else A.NoOp(),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
            
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image = transformed['image']
        
        # Generate mask
        mask_generator = random.choice(self.mask_generators)
        mask = mask_generator(self.image_size, self.image_size)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        # Create masked image
        masked_image = image * (1 - mask)
        
        # Combine image and mask for generator input
        generator_input = torch.cat([masked_image, mask], dim=0)
        
        return {
            'image': image,
            'mask': mask,
            'masked_image': masked_image,
            'generator_input': generator_input,
            'image_path': str(image_path)
        }


class InpaintingDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for inpainting"""
    
    def __init__(
        self,
        train_data_root: str,
        val_data_root: Optional[str] = None,
        test_data_root: Optional[str] = None,
        image_size: int = 256,
        load_size: Optional[int] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        mask_config: Optional[Dict] = None
    ):
        super().__init__()
        
        self.train_data_root = train_data_root
        self.val_data_root = val_data_root or train_data_root
        self.test_data_root = test_data_root or val_data_root or train_data_root
        
        self.image_size = image_size
        self.load_size = load_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Setup mask generators
        if mask_config is None:
            mask_config = {'types': ['random', 'irregular']}
            
        self.mask_generators = self._create_mask_generators(mask_config)
        
    def _create_mask_generators(self, config: Dict) -> List[MaskGenerator]:
        generators = []
        
        for mask_type in config.get('types', ['random']):
            if mask_type == 'random':
                generators.append(RandomMaskGenerator(
                    min_size=config.get('random_min_size', 0.02),
                    max_size=config.get('random_max_size', 0.4),
                    max_masks=config.get('random_max_masks', 1)
                ))
            elif mask_type == 'irregular':
                generators.append(IrregularMaskGenerator(
                    min_num_vertex=config.get('irregular_min_vertex', 4),
                    max_num_vertex=config.get('irregular_max_vertex', 12),
                    min_width=config.get('irregular_min_width', 10),
                    max_width=config.get('irregular_max_width', 40)
                ))
            elif mask_type == 'outpainting':
                generators.append(OutpaintingMaskGenerator(
                    min_crop=config.get('outpainting_min_crop', 0.2),
                    max_crop=config.get('outpainting_max_crop', 0.8)
                ))
            else:
                raise ValueError(f"Unknown mask type: {mask_type}")
                
        return generators
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages"""
        
        if stage == 'fit' or stage is None:
            self.train_dataset = InpaintingDataset(
                data_root=self.train_data_root,
                mask_generators=self.mask_generators,
                image_size=self.image_size,
                load_size=self.load_size,
                augment=True
            )
            
            self.val_dataset = InpaintingDataset(
                data_root=self.val_data_root,
                mask_generators=self.mask_generators,
                image_size=self.image_size,
                load_size=self.image_size,  # No crop for validation
                augment=False
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = InpaintingDataset(
                data_root=self.test_data_root,
                mask_generators=self.mask_generators,
                image_size=self.image_size,
                load_size=self.image_size,
                augment=False
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )


def get_datamodule(config: Dict) -> InpaintingDataModule:
    """Factory function to create data module based on config"""
    
    return InpaintingDataModule(
        train_data_root=config['train_data_root'],
        val_data_root=config.get('val_data_root'),
        test_data_root=config.get('test_data_root'),
        image_size=config.get('image_size', 256),
        load_size=config.get('load_size'),
        batch_size=config.get('batch_size', 4),
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        mask_config=config.get('mask_config')
    )