#!/usr/bin/env python3
"""
LaMa Inference Script

Usage:
    # Single image
    python predict.py --model path/to/checkpoint --image input.jpg --mask mask.jpg --output output.jpg
    
    # Batch processing
    python predict.py --model path/to/checkpoint --input-dir ./images --output-dir ./results
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lama.training.trainers import load_checkpoint
from lama.training.data.datasets import make_default_val_dataset
from lama.evaluation.utils import move_to_device
from lama.evaluation.refinement import refine_predict


def parse_args():
    parser = argparse.ArgumentParser(description='LaMa Inference')
    # Model
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint directory')
    parser.add_argument('--checkpoint', type=str, default='last.ckpt',
                        help='Checkpoint filename')
    
    # Single image mode
    parser.add_argument('--image', type=str, default=None,
                        help='Input image path')
    parser.add_argument('--mask', type=str, default=None,
                        help='Mask image path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path')
    
    # Batch mode
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Input directory with images and masks')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    
    # Options
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--refine', action='store_true',
                        help='Use refinement for better quality')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    
    return parser.parse_args()


def load_model(model_path, checkpoint_name, device):
    """Load model from checkpoint"""
    # Load config
    config_path = Path(model_path) / 'config.yaml'
    with open(config_path, 'r') as f:
        config = OmegaConf.create(yaml.safe_load(f))
    
    # Set to evaluation mode
    config.training_model.predict_only = True
    config.visualizer.kind = 'noop'
    
    # Load checkpoint
    checkpoint_path = Path(model_path) / 'checkpoints' / checkpoint_name
    model = load_checkpoint(config, str(checkpoint_path), strict=False, map_location='cpu')
    model.freeze()
    model.to(device)
    
    return model, config


def predict_single(model, image_path, mask_path, device, refine=False):
    """Predict single image"""
    # Load images
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8) * 255
    
    # Prepare batch
    batch = {
        'image': torch.from_numpy(image).float() / 255.0,
        'mask': torch.from_numpy(mask).float() / 255.0
    }
    
    # Add batch dimension and permute
    batch['image'] = batch['image'].permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = batch['mask'].unsqueeze(0).unsqueeze(0)
    
    # Move to device
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1
    
    # Predict
    if refine:
        # Store original size for refinement
        batch['unpad_to_size'] = [image.shape[0], image.shape[1]]
        result = refine_predict(batch, model)
        result = result[0].permute(1, 2, 0).detach().cpu().numpy()
    else:
        with torch.no_grad():
            output = model(batch)
            result = output['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
    
    # Convert to uint8
    result = np.clip(result * 255, 0, 255).astype('uint8')
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    return result


def predict_batch(model, config, input_dir, output_dir, device, refine=False):
    """Predict batch of images"""
    # Create dataset
    dataset = make_default_val_dataset(input_dir, **config.dataset)
    
    # Process each image
    for i in tqdm(range(len(dataset)), desc="Processing images"):
        # Get mask filename
        mask_fname = dataset.mask_filenames[i]
        out_fname = Path(output_dir) / Path(mask_fname).name
        out_fname.parent.mkdir(parents=True, exist_ok=True)
        
        # Get batch
        batch = dataset[i]
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1
        
        # Predict
        if refine:
            if 'unpad_to_size' in batch:
                result = refine_predict(batch, model)
                result = result[0].permute(1, 2, 0).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    output = model(batch)
                    result = output['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
        else:
            with torch.no_grad():
                output = model(batch)
                result = output['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
        
        # Handle padding if necessary
        if 'unpad_to_size' in batch:
            orig_height, orig_width = batch['unpad_to_size']
            result = result[:orig_height, :orig_width]
        
        # Save result
        result = np.clip(result * 255, 0, 255).astype('uint8')
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_fname), result)


def main():
    args = parse_args()
    
    # Setup device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model}")
    model, config = load_model(args.model, args.checkpoint, device)
    
    # Single image mode
    if args.image and args.mask and args.output:
        print(f"Processing single image: {args.image}")
        result = predict_single(model, args.image, args.mask, device, args.refine)
        cv2.imwrite(args.output, result)
        print(f"Result saved to: {args.output}")
    
    # Batch mode
    elif args.input_dir and args.output_dir:
        print(f"Processing batch from: {args.input_dir}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        predict_batch(model, config, args.input_dir, args.output_dir, device, args.refine)
        print(f"Results saved to: {args.output_dir}")
    
    else:
        print("Please provide either:")
        print("  --image, --mask, --output for single image mode")
        print("  --input-dir, --output-dir for batch mode")
        sys.exit(1)


if __name__ == '__main__':
    main()