#!/usr/bin/env python3
"""
LaMa Inpainting Inference Script
Usage: python inference.py --checkpoint path/to/checkpoint --input input_dir --output output_dir
"""

import os
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LaMa inpainting inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing images and masks")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--refine", action="store_true", help="Use refinement for better quality")
    parser.add_argument("--pad_modulo", type=int, default=8, help="Pad input to modulo")
    parser.add_argument("--img_suffix", type=str, default=".png", help="Image suffix")
    parser.add_argument("--out_ext", type=str, default=".png", help="Output extension")
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    if os.path.isfile(checkpoint_path):
        # Single checkpoint file
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
        model_path = checkpoint_path
    else:
        # Directory containing checkpoint and config
        config_path = os.path.join(checkpoint_path, "config.yaml")
        model_path = os.path.join(checkpoint_path, "checkpoints", "last.ckpt")
        
        # Find best checkpoint if last doesn't exist
        if not os.path.exists(model_path):
            checkpoint_dir = os.path.join(checkpoint_path, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
                if checkpoints:
                    model_path = os.path.join(checkpoint_dir, checkpoints[0])
    
    # Load config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = OmegaConf.create(yaml.safe_load(f))
    
    config.training_model.predict_only = True
    config.visualizer.kind = 'noop'
    
    # Load model
    model = load_checkpoint(config, model_path, strict=False, map_location=device)
    model.freeze()
    model.to(device)
    
    return model, config


def process_image(model, batch, device, refine=False, refine_config=None):
    """Process a single image through the model."""
    if refine and refine_config:
        assert 'unpad_to_size' in batch, "Unpadded size is required for refinement"
        result = refine_predict(batch, model, **refine_config)
        result = result[0].permute(1, 2, 0).detach().cpu().numpy()
    else:
        with torch.no_grad():
            batch = move_to_device(batch, device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = model(batch)
            result = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
            
            # Unpad if necessary
            unpad_to_size = batch.get('unpad_to_size', None)
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                result = result[:orig_height, :orig_width]
    
    return result


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    
    # Setup dataset
    input_dir = args.input
    if not input_dir.endswith('/'):
        input_dir += '/'
    
    dataset = make_default_val_dataset(
        input_dir,
        kind="default",
        img_suffix=args.img_suffix,
        pad_out_to_modulo=args.pad_modulo
    )
    
    logger.info(f"Found {len(dataset)} images to process")
    
    # Refine config if needed
    refine_config = None
    if args.refine:
        refine_config = {
            'gpu_ids': [0],
            'modulo': args.pad_modulo,
            'n_iters': 15,
            'lr': 0.002,
            'min_side': 512,
            'max_scales': 3,
            'px_budget': 1800000
        }
    
    # Process images
    for img_i in tqdm(range(len(dataset)), desc="Processing images"):
        mask_fname = dataset.mask_filenames[img_i]
        output_fname = output_dir / (
            Path(mask_fname).relative_to(input_dir).with_suffix(args.out_ext)
        )
        
        # Create output subdirectory if needed
        output_fname.parent.mkdir(parents=True, exist_ok=True)
        
        # Process image
        batch = default_collate([dataset[img_i]])
        result = process_image(model, batch, device, args.refine, refine_config)
        
        # Save result
        result = np.clip(result * 255, 0, 255).astype('uint8')
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_fname), result)
    
    logger.info(f"Inference completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()