#!/usr/bin/env python3
"""
LaMa ONNX Export Script
Usage: python export_onnx.py --checkpoint path/to/checkpoint --output model.onnx
"""

import os
import argparse
import logging
from pathlib import Path

import torch
import torch.onnx
import yaml
from omegaconf import OmegaConf

from saicinpainting.training.trainers import load_checkpoint

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Export LaMa model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX file path")
    parser.add_argument("--input_size", type=int, nargs=2, default=[512, 512], help="Input size (height, width)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--opset_version", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--dynamic_axes", action="store_true", help="Enable dynamic axes")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    return parser.parse_args()


def load_model(checkpoint_path):
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
    model = load_checkpoint(config, model_path, strict=False, map_location='cpu')
    model.eval()
    
    return model


class ONNXExportWrapper(torch.nn.Module):
    """Wrapper for ONNX export that handles input/output format."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, image, mask):
        """Forward pass for ONNX export."""
        batch = {
            'image': image,
            'mask': mask
        }
        
        # Process through model
        with torch.no_grad():
            batch['mask'] = (batch['mask'] > 0) * 1
            result = self.model(batch)
            
        return result['inpainted']


def export_to_onnx(model, output_path, input_size, batch_size, opset_version, dynamic_axes):
    """Export model to ONNX format."""
    height, width = input_size
    
    # Create dummy inputs
    dummy_image = torch.randn(batch_size, 3, height, width)
    dummy_mask = torch.randn(batch_size, 1, height, width)
    
    # Wrap model for ONNX export
    wrapper = ONNXExportWrapper(model)
    
    # Define input/output names
    input_names = ['image', 'mask']
    output_names = ['inpainted']
    
    # Define dynamic axes if requested
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'image': {0: 'batch_size', 2: 'height', 3: 'width'},
            'mask': {0: 'batch_size', 2: 'height', 3: 'width'},
            'inpainted': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    
    # Export to ONNX
    logger.info(f"Exporting model to ONNX...")
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_mask),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False
    )
    
    logger.info(f"ONNX model exported to: {output_path}")


def verify_onnx_model(onnx_path, input_size, batch_size):
    """Verify the exported ONNX model."""
    try:
        import onnx
        import onnxruntime as ort
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(onnx_path)
        
        # Test inference
        height, width = input_size
        dummy_image = torch.randn(batch_size, 3, height, width).numpy()
        dummy_mask = torch.randn(batch_size, 1, height, width).numpy()
        
        outputs = ort_session.run(
            None,
            {'image': dummy_image, 'mask': dummy_mask}
        )
        
        logger.info(f"ONNX model verification successful!")
        logger.info(f"Output shape: {outputs[0].shape}")
        
        return True
        
    except ImportError:
        logger.warning("ONNX runtime not installed, skipping verification")
        return False
    except Exception as e:
        logger.error(f"ONNX model verification failed: {e}")
        return False


def simplify_onnx_model(onnx_path):
    """Simplify ONNX model."""
    try:
        import onnx
        import onnxsim
        
        logger.info("Simplifying ONNX model...")
        
        # Load and simplify
        onnx_model = onnx.load(onnx_path)
        simplified_model, check = onnxsim.simplify(onnx_model)
        
        if check:
            # Save simplified model
            simplified_path = onnx_path.replace('.onnx', '_simplified.onnx')
            onnx.save(simplified_model, simplified_path)
            logger.info(f"Simplified model saved to: {simplified_path}")
            return simplified_path
        else:
            logger.warning("Model simplification failed")
            return onnx_path
            
    except ImportError:
        logger.warning("onnx-simplifier not installed, skipping simplification")
        return onnx_path
    except Exception as e:
        logger.error(f"Model simplification failed: {e}")
        return onnx_path


def main():
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint)
    
    # Export to ONNX
    export_to_onnx(
        model,
        str(output_path),
        args.input_size,
        args.batch_size,
        args.opset_version,
        args.dynamic_axes
    )
    
    # Verify ONNX model
    verify_onnx_model(str(output_path), args.input_size, args.batch_size)
    
    # Simplify if requested
    if args.simplify:
        simplified_path = simplify_onnx_model(str(output_path))
        logger.info(f"Final model: {simplified_path}")
    
    logger.info("ONNX export completed successfully!")


if __name__ == "__main__":
    main()