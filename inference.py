"""
Inference script for LaMa inpainting model
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.lightning_module import InpaintingLightningModule


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    model = InpaintingLightningModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    return model


def preprocess_image(image_path: str, target_size: int = 256):
    """Preprocess input image"""
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Transform
    transform = A.Compose([
        A.Resize(target_size, target_size),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0)
    
    return image_tensor, image


def preprocess_mask(mask_path: str, target_size: int = 256):
    """Preprocess mask image"""
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure binary mask
    mask = (mask > 128).astype(np.uint8)
    
    # Transform
    transform = A.Compose([
        A.Resize(target_size, target_size),
        ToTensorV2()
    ])
    
    transformed = transform(image=mask)
    mask_tensor = transformed['image'].unsqueeze(0).float()
    
    return mask_tensor


def postprocess_output(output: torch.Tensor):
    """Convert model output back to image"""
    # Denormalize
    output = (output + 1) / 2
    output = output.clamp(0, 1)
    
    # Convert to numpy
    output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_np = (output_np * 255).astype(np.uint8)
    
    return output_np


def create_result_grid(original: np.ndarray, masked: np.ndarray, result: np.ndarray):
    """Create a grid showing original, masked, and result images"""
    h, w = original.shape[:2]
    grid = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    grid[:, :w] = original
    grid[:, w:2*w] = masked
    grid[:, 2*w:] = result
    
    return grid


def inpaint_image(
    model, 
    image_path: str, 
    mask_path: str, 
    output_path: str,
    device: str = 'cuda',
    target_size: int = 256,
    save_grid: bool = True
):
    """Perform inpainting on a single image"""
    
    # Preprocess inputs
    image_tensor, original_image = preprocess_image(image_path, target_size)
    mask_tensor = preprocess_mask(mask_path, target_size)
    
    # Move to device
    image_tensor = image_tensor.to(device)
    mask_tensor = mask_tensor.to(device)
    
    # Create masked image
    masked_image = image_tensor * (1 - mask_tensor)
    
    # Create generator input
    generator_input = torch.cat([masked_image, mask_tensor], dim=1)
    
    # Inference
    with torch.no_grad():
        result = model(generator_input)
    
    # Post-process
    result_np = postprocess_output(result)
    
    # Create masked image for visualization
    masked_vis = image_tensor * (1 - mask_tensor)
    masked_vis_np = postprocess_output(masked_vis)
    
    # Resize original image to match result
    original_resized = cv2.resize(original_image, (target_size, target_size))
    
    if save_grid:
        # Create and save grid
        grid = create_result_grid(original_resized, masked_vis_np, result_np)
        output_grid_path = output_path.replace('.png', '_grid.png').replace('.jpg', '_grid.jpg')
        cv2.imwrite(output_grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"Result grid saved to {output_grid_path}")
    
    # Save result
    cv2.imwrite(output_path, cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))
    print(f"Result saved to {output_path}")
    
    return result_np


def main():
    parser = argparse.ArgumentParser(description='LaMa Inpainting Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--mask', type=str, required=True,
                        help='Path to mask image (white=masked region)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save result')
    parser.add_argument('--size', type=int, default=256,
                        help='Image size for inference (default: 256)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--no-grid', action='store_true',
                        help='Do not save result grid')
    
    args = parser.parse_args()
    
    # Check inputs
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"Mask not found: {args.mask}")
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, device)
    
    # Run inference
    result = inpaint_image(
        model=model,
        image_path=args.image,
        mask_path=args.mask,
        output_path=args.output,
        device=device,
        target_size=args.size,
        save_grid=not args.no_grid
    )
    
    print("Inference completed!")


if __name__ == '__main__':
    main()